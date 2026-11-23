# SPDX-License-Identifier: BSD-3-Clause

"""
Train a small model with LoRA on MLX (Apple Silicon), with tasks drawn by a task
sampler, and evaluate it on the test split while it trains.

Start a `reasoning_gym_env` server that serves the Task API, then, from this folder:

    python train_mlx.py --base-url http://localhost:8001 --sampler curriculum \
        --out runs/curriculum-seed0

The folder gets `config.json`, `train.jsonl` (one record per step), `evals.jsonl`
(one record per evaluation, with the sampler's counters), `evals/run-NNNN/` (the
episodes of each evaluation) and, at the end, `adapters.safetensors`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx_lm
from curriculum_sampler import CurriculumSampler
from mlx.utils import tree_flatten, tree_map
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import linear_to_lora_layers
from openenv import __version__
from openenv.core.evals import EnvEvalHarness, EvalConfig, EvalResult, EvalSchedule
from openenv.core.harness import MCPHarnessAdapter, ModelStepResult
from openenv.core.llm_client import LLMResponse
from openenv.core.task_api_client import TaskAPIClient
from openenv.core.task_sampler import UniformSampler
from reasoning_gym_env import ReasoningGymEnv
from reasoning_gym_env.harness import ReasoningGymSessionFactory
from reasoning_gym_tasks import chain_sum_model_step, ReasoningGymTasks
from training_loop import Completion, train

SYSTEM_PROMPT = (
    "Solve the arithmetic problem. Work step by step, briefly, then give only the "
    "final number inside <answer></answer> tags, for example <answer>42</answer>."
)
# Some chat templates, Llama 3.2's among them, write today's date into every
# prompt; a fixed date keeps the prompts of two runs identical.
PROMPT_DATE = "26 Sep 2026"
LORA_KEYS = {
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
}


def _loss(model: nn.Module, inputs, targets, mask, advantages, norm):
    logits = model(inputs).astype(mx.float32)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    token_logp = mx.take_along_axis(logp, targets[..., None], axis=-1).squeeze(-1)
    # One on-policy update per batch, so the GRPO ratio is 1 and the loss is the
    # advantage-weighted log-likelihood, normalized by the answer tokens.
    return -(token_logp * mask * advantages[:, None]).sum() / norm


class MlxPolicy:
    """
    A model trained with LoRA adapters on MLX, as the training loop's policy.

    Args:
        model (`str`):
            Hugging Face repository of an MLX model.
        lora_rank (`int`):
            Rank of the adapters on the attention and MLP projections.
        lr (`float`):
            Learning rate of Adam.
        micro_batch (`int`):
            Answers per forward pass of an update.
        seed (`int`):
            Seed of MLX's random generator, set just before the model loads.
    """

    def __init__(
        self, model: str, *, lora_rank: int, lr: float, micro_batch: int, seed: int
    ) -> None:
        mx.random.seed(seed)
        self.model, self.tokenizer = mlx_lm.load(model)
        self.model.freeze()
        linear_to_lora_layers(
            self.model,
            len(self.model.layers),
            {"rank": lora_rank, "scale": 2.0, "dropout": 0.0, "keys": LORA_KEYS},
        )
        self.optimizer = optim.Adam(learning_rate=lr)
        self.micro_batch = micro_batch
        pad_id = self.tokenizer.pad_token_id
        self.pad_id = (
            pad_id if pad_id is not None else min(self.tokenizer.eos_token_ids)
        )

    def encode_prompt(self, question: str) -> list[int]:
        """Return the chat prompt of a question, as token ids."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        # mlx_lm's tokenizer wrapper returns the token ids as a list.
        return list(
            self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, date_string=PROMPT_DATE
            )
        )

    def generate(
        self, prompts: list[list[int]], max_tokens: int, temperature: float
    ) -> list[Completion]:
        """Generate one answer per prompt in one batch; a temperature of 0 is greedy."""
        generator = BatchGenerator(
            self.model,
            stop_tokens=[[token] for token in self.tokenizer.eos_token_ids],
            sampler=make_sampler(temp=temperature) if temperature > 0 else None,
            completion_batch_size=max(len(prompts), 1),
            prefill_batch_size=min(len(prompts), 16),
        )
        uids = generator.insert(prompts, [max_tokens] * len(prompts))
        tokens: dict[int, list[int]] = {uid: [] for uid in uids}
        stopped = dict.fromkeys(uids, False)
        while responses := generator.next_generated():
            for response in responses:
                tokens[response.uid].append(response.token)
                stopped[response.uid] |= response.finish_reason == "stop"
        generator.close()
        return [
            Completion(
                tokens=tokens[uid],
                text=self.tokenizer.decode(
                    tokens[uid][:-1] if stopped[uid] else tokens[uid]
                ),
            )
            for uid in uids
        ]

    def update(self, samples: list[tuple[list[int], list[int], float]]) -> float:
        """Take one Adam step on the samples, in micro-batches; return the loss."""
        loss_and_grad = nn.value_and_grad(self.model, _loss)
        norm = mx.array(float(sum(len(answer) for _, answer, _ in samples)))
        grads, total = None, 0.0
        for start in range(0, len(samples), self.micro_batch):
            batch = samples[start : start + self.micro_batch]
            sequences = [prompt + answer for prompt, answer, _ in batch]
            width = max(len(sequence) for sequence in sequences)
            inputs, targets, mask = [], [], []
            for (prompt, answer, _), sequence in zip(batch, sequences):
                pad = width - len(sequence)
                inputs.append(sequence[:-1] + [self.pad_id] * pad)
                targets.append(sequence[1:] + [self.pad_id] * pad)
                mask.append(
                    [0.0] * (len(prompt) - 1) + [1.0] * len(answer) + [0.0] * pad
                )
            advantages = mx.array([a for _, _, a in batch], dtype=mx.float32)
            loss, g = loss_and_grad(
                self.model,
                mx.array(inputs),
                mx.array(targets),
                mx.array(mask),
                advantages,
                norm,
            )
            grads = g if grads is None else tree_map(mx.add, grads, g)
            mx.eval(grads, loss)
            total += loss.item()
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state)
        return total

    def model_step(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any],
        sampling: dict[str, Any],
    ) -> ModelStepResult:
        """Answer the last user message in text, as a model step for evaluation."""
        prompt = self.encode_prompt(messages[-1]["content"])
        [completion] = self.generate(
            [prompt], sampling["max_tokens"], sampling["temperature"]
        )
        return ModelStepResult(
            response=LLMResponse(content=completion.text),
            prompt_ids=prompt,
            completion_ids=completion.tokens,
        )

    def save(self, path: Path) -> None:
        """Write the adapters to `path`."""
        adapters = dict(tree_flatten(self.model.trainable_parameters()))
        mx.save_safetensors(str(path), adapters)


def _append(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--base-url", required=True, help="reasoning_gym_env server")
    parser.add_argument("--out", type=Path, required=True, help="a new folder")
    parser.add_argument(
        "--sampler", choices=["uniform", "curriculum"], default="curriculum"
    )
    parser.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--prompts-per-step", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=100, help="0 turns it off")
    parser.add_argument(
        "--eval-limit", type=int, default=None, help="first N test tasks (default all)"
    )
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True)  # an existing folder is refused
    versions = {
        "openenv": __version__,
        "mlx": mx.__version__,
        "mlx_lm": mlx_lm.__version__,
    }
    (args.out / "config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "out": str(args.out),
                "prompt_date": PROMPT_DATE,
                **versions,
            },
            indent=2,
        )
        + "\n"
    )

    client = TaskAPIClient(args.base_url, env_name="reasoning_gym_env")
    train_tasks = list(client.iter_tasks("train"))
    test_tasks = list(client.iter_tasks("test"))
    if args.sampler == "curriculum":
        sampler = CurriculumSampler(
            train_tasks, group_size=args.group_size, rng_seed=args.seed
        )
    else:
        sampler = UniformSampler(train_tasks, rng_seed=args.seed)
    policy = MlxPolicy(
        args.model,
        lora_rank=args.lora_rank,
        lr=args.lr,
        micro_batch=args.micro_batch,
        seed=args.seed,
    )

    harness = EnvEvalHarness(
        ReasoningGymSessionFactory(
            lambda: ReasoningGymEnv(base_url=args.base_url), dataset_name="chain_sum"
        ),
        MCPHarnessAdapter(),
        test_tasks,
        model_step=chain_sum_model_step(policy.model_step),
        output_dir=args.out / "evals",
    )
    config = EvalConfig(
        harness_name=harness.name,
        harness_version=__version__,
        library_versions=versions,
        dataset="reasoning_gym_env",
        eval_parameters={
            "split": "test",
            "limit": args.eval_limit,
            "max_tokens": args.max_tokens,
            "model": args.model,
        },
    )

    def on_result(step: int, result: EvalResult) -> None:
        record = {"step": step, "sampler": sampler.metrics(), **result.scores}
        _append(args.out / "evals.jsonl", record)

    env = ReasoningGymTasks(args.base_url)
    try:
        train(
            policy,
            env,
            sampler,
            steps=args.steps,
            prompts_per_step=args.prompts_per_step,
            group_size=args.group_size,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            log=lambda record: _append(args.out / "train.jsonl", record),
            schedule=EvalSchedule(
                harness, config, every_n_steps=args.eval_every, on_result=on_result
            ),
        )
    finally:
        env.close()
    policy.save(args.out / "adapters.safetensors")


if __name__ == "__main__":
    main()
