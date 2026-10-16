"""
The model side of the trial, in MLX. The only module that imports `mlx`.

`MODELS`, `load_policy`, `prompt_tokens`, `generate`, `sequence_logp_loss` and
`train_step` are the pilot's code, copied from `pilot/smoke.py` and left as the
pilot wrote them; `prompt_tokens` takes the date as one more parameter. A test
compares each with its source. `MlxPolicy` wraps them for the training loop.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import linear_to_lora_layers
from trial.policy import Completion
from trial.tasks import PROMPT_DATE, SYSTEM_PROMPT

MODELS = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "llama3.2-1b": "mlx-community/Llama-3.2-1B-Instruct-bf16",
}


def load_policy(name: str, lora_rank: int | None):
    model, tokenizer = load(MODELS.get(name, name))
    if lora_rank:
        model.freeze()
        keys = {
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        }
        linear_to_lora_layers(
            model,
            len(model.layers),
            {"rank": lora_rank, "scale": 2.0, "dropout": 0.0, "keys": keys},
        )
    return model, tokenizer


def prompt_tokens(tokenizer, question: str, date_string: str) -> list[int]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    # enable_thinking=False turns off Qwen3's long thinking mode; other templates ignore it.
    out = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        date_string=date_string,
    )
    if isinstance(out, dict) or hasattr(out, "keys"):
        out = out["input_ids"]
    return list(out)


def generate(model, tokenizer, prompts: list[list[int]], max_tokens: int, temp: float):
    """Batched generation. Returns (completion tokens incl. stop token, text, hit_stop) per prompt."""
    gen = BatchGenerator(
        model,
        stop_tokens=[[t] for t in tokenizer.eos_token_ids],
        sampler=make_sampler(temp=temp) if temp > 0 else None,
        completion_batch_size=max(len(prompts), 1),
        prefill_batch_size=min(len(prompts), 16),
    )
    uids = gen.insert(prompts, [max_tokens] * len(prompts))
    tokens = {u: [] for u in uids}
    stopped = {u: False for u in uids}
    while responses := gen.next_generated():
        for r in responses:
            tokens[r.uid].append(r.token)
            if r.finish_reason == "stop":
                stopped[r.uid] = True
    gen.close()
    out = []
    for u in uids:
        toks = tokens[u]
        text_toks = toks[:-1] if stopped[u] else toks
        out.append((toks, tokenizer.decode(text_toks), stopped[u]))
    return out


def sequence_logp_loss(model, inputs, targets, mask, adv, norm):
    logits = model(inputs).astype(mx.float32)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tok_logp = mx.take_along_axis(logp, targets[..., None], axis=-1).squeeze(-1)
    # One on-policy update per batch, so the GRPO ratio is 1 and the loss reduces to
    # advantage-weighted log-likelihood, normalized by completion tokens (DAPO style).
    return -(tok_logp * mask * adv[:, None]).sum() / norm


def train_step(model, optimizer, samples, micro_batch: int, pad_id: int) -> float:
    """samples: list of (prompt_tokens, completion_tokens, advantage)."""
    loss_and_grad = nn.value_and_grad(model, sequence_logp_loss)
    norm = mx.array(float(sum(len(c) for _, c, _ in samples)))
    grads, total_loss = None, 0.0
    for start in range(0, len(samples), micro_batch):
        mb = samples[start : start + micro_batch]
        seqs = [p + c for p, c, _ in mb]
        width = max(len(s) for s in seqs)
        inp, tgt, msk = [], [], []
        for (p, c, _), s in zip(mb, seqs):
            pad = width - len(s)
            inp.append(s[:-1] + [pad_id] * pad)
            tgt.append(s[1:] + [pad_id] * pad)
            msk.append([0.0] * (len(p) - 1) + [1.0] * len(c) + [0.0] * pad)
        adv = mx.array([a for _, _, a in mb], dtype=mx.float32)
        loss, g = loss_and_grad(
            model, mx.array(inp), mx.array(tgt), mx.array(msk), adv, norm
        )
        grads = g if grads is None else tree_map(mx.add, grads, g)
        mx.eval(grads, loss)
        total_loss += loss.item()
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    return total_loss


class MlxPolicy:
    """
    A LoRA-trained MLX model behind the `trial.policy.Policy` contract.

    Seeds MLX's random generator immediately before the model loads, then
    builds the optimizer and the padding id, in the pilot's order.

    Args:
        model (`str`):
            A name from `MODELS`, or a Hugging Face repository.
        lora_rank (`int`, *optional*):
            Rank of the LoRA adapters; `None` loads the model without them.
        lr (`float`):
            Learning rate of Adam.
        micro_batch (`int`):
            Number of answers per forward pass of an update.
        seed (`int`):
            Seed of MLX's random generator.
        prompt_date (`str`, *optional*, defaults to `PROMPT_DATE`):
            The date the chat template writes into every prompt.
    """

    def __init__(
        self,
        *,
        model: str,
        lora_rank: int | None,
        lr: float,
        micro_batch: int,
        seed: int,
        prompt_date: str = PROMPT_DATE,
    ) -> None:
        mx.random.seed(seed)
        self.model, self.tokenizer = load_policy(model, lora_rank)
        self.optimizer = optim.Adam(learning_rate=lr)
        self.pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else min(self.tokenizer.eos_token_ids)
        )
        self.repo = MODELS.get(model, model)
        self.micro_batch = micro_batch
        self.prompt_date = prompt_date

    def encode_prompt(self, question: str) -> list[int]:
        """Return the prompt of a question, as token ids."""
        return prompt_tokens(self.tokenizer, question, self.prompt_date)

    def prompt_text(self, question: str) -> str:
        """Return the prompt of a question as the model receives it."""
        return self.tokenizer.decode(self.encode_prompt(question))

    def generate(
        self, prompts: list[list[int]], max_tokens: int, temperature: float
    ) -> list[Completion]:
        """Generate one answer per prompt; a temperature of 0 is greedy."""
        outs = generate(self.model, self.tokenizer, prompts, max_tokens, temperature)
        return [Completion(tokens, text, hit_stop) for tokens, text, hit_stop in outs]

    def update(self, samples: list[tuple[list[int], list[int], float]]) -> float:
        """Take one optimizer step and return the loss."""
        return train_step(
            self.model, self.optimizer, samples, self.micro_batch, self.pad_id
        )

    def save(self, path: Path) -> None:
        """Write the adapters to exactly this file, through a temporary one."""
        path = Path(path)
        # MLX appends ".safetensors" to any other ending, so the temporary name has it.
        temporary = path.with_name("adapters.tmp.safetensors")
        adapters = dict(tree_flatten(self.model.trainable_parameters()))
        mx.save_safetensors(str(temporary), adapters)
        os.replace(temporary, path)

    def describe(self) -> dict[str, Any]:
        """Return `trainable_params` and the local `model_snapshot` hash."""
        from huggingface_hub import snapshot_download

        try:
            snapshot = Path(snapshot_download(self.repo, local_files_only=True)).name
        except Exception:  # the hub raises several unrelated types here
            snapshot = None
        parameters = tree_flatten(self.model.trainable_parameters())
        return {
            "trainable_params": sum(value.size for _, value in parameters),
            "model_snapshot": snapshot,
        }

    def peak_memory_gb(self) -> float:
        """Return MLX's peak memory so far, in gigabytes."""
        return mx.get_peak_memory() / 1e9


__all__ = ["MODELS", "MlxPolicy"]
