"""Smoke test for RFC 013: GRPO + task sampling on OpenEnv's reasoning_gym_env, in MLX.

Two commands:

  baseline  Greedy accuracy per difficulty tier on the fixed test split,
            plus generation throughput. Use it to compare models.
  train     A short GRPO run (LoRA) with a uniform or curriculum sampler,
            logging time per step and evaluating the test split every N steps.

The environment runs in-process (no Docker): each task maps to a
reset(dataset_name, dataset_config, seed, size=1) of ReasoningGymEnvironment,
and the reward is whatever the environment's step() returns.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import linear_to_lora_layers

PILOT_DIR = Path(__file__).resolve().parent.parent / "pilot"
sys.path.insert(0, str(PILOT_DIR))
sys.path.insert(0, str(PILOT_DIR / "vendor"))
from reasoning_gym_env.models import ReasoningGymAction  # noqa: E402
from reasoning_gym_env.server.reasoning_gym_environment import (  # noqa: E402
    ReasoningGymEnvironment,
)

from curriculum import CurriculumSampler, UniformSampler  # noqa: E402

MODELS = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "llama3.2-1b": "mlx-community/Llama-3.2-1B-Instruct-bf16",
}

# chain_sum difficulty tiers: more terms, more digits, then negative numbers.
DATASET = "chain_sum"
TIERS = [
    dict(min_terms=2, max_terms=2, min_digits=1, max_digits=1),
    dict(min_terms=3, max_terms=3, min_digits=2, max_digits=2),
    dict(min_terms=4, max_terms=4, min_digits=3, max_digits=3),
    dict(min_terms=5, max_terms=5, min_digits=4, max_digits=4, allow_negation=True),
    dict(min_terms=6, max_terms=6, min_digits=5, max_digits=5, allow_negation=True),
]
SPLIT_SEED_OFFSET = {"train": 0, "test": 500_000_000}
PROMPT_DATE = "26 Sep 2026"  # the day of the pilot; the template would write today's date

SYSTEM_PROMPT = (
    "Solve the arithmetic problem. Work step by step, briefly, then give only the "
    "final number inside <answer></answer> tags, for example <answer>42</answer>."
)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\{\s*([^{}]*?)\s*\}")
NUMBER_RE = re.compile(r"-?\d[\d,]*")


def extract_answer(completion: str) -> str:
    """Turn a completion into the env action: <answer> tags, then \\boxed{}, then the last number.

    Lenient on purpose, so the experiment measures arithmetic rather than how fast
    the model learns one output format. Strict tag usage is still reported as
    ``format_rate``. The reward itself always comes from the environment.
    """
    for pattern in (ANSWER_RE, BOXED_RE, NUMBER_RE):
        found = pattern.findall(completion)
        if found:
            return found[-1].replace(",", "").strip()
    return ""


# --------------------------------------------------------------------------- tasks

def make_tasks(split: str, per_tier: int) -> list[dict]:
    return [
        {"id": f"{DATASET}-{split}-t{tier}-{i}", "split": split, "group": DATASET,
         "difficulty": tier, "index": i,
         "seed": SPLIT_SEED_OFFSET[split] + tier * 1_000_003 + i}
        for tier in range(len(TIERS)) for i in range(per_tier)
    ]


class TaskEnv:
    """Runs one task per episode on an in-process ReasoningGymEnvironment."""

    def __init__(self):
        self.env = ReasoningGymEnvironment()
        self._questions: dict[str, str] = {}

    def _reset(self, task: dict):
        return self.env.reset(dataset_name=DATASET, dataset_config=TIERS[task["difficulty"]],
                              seed=task["seed"], size=1)

    def question(self, task: dict) -> str:
        if task["id"] not in self._questions:
            self._questions[task["id"]] = self._reset(task).question
        return self._questions[task["id"]]

    def reward(self, task: dict, completion: str) -> float:
        self._reset(task)
        answer = extract_answer(completion)
        return float(self.env.step(ReasoningGymAction(answer=answer)).reward or 0.0)


# ---------------------------------------------------------------------- model utils

def load_policy(name: str, lora_rank: int | None):
    model, tokenizer = load(MODELS.get(name, name))
    if lora_rank:
        model.freeze()
        keys = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
        linear_to_lora_layers(model, len(model.layers),
                              {"rank": lora_rank, "scale": 2.0, "dropout": 0.0, "keys": keys})
    return model, tokenizer


def prompt_tokens(tokenizer, question: str) -> list[int]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    # enable_thinking=False turns off Qwen3's long thinking mode; other templates ignore it.
    out = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                        enable_thinking=False, date_string=PROMPT_DATE)
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


# ------------------------------------------------------------------------ evaluation

def evaluate(model, tokenizer, env: TaskEnv, tasks: list[dict], max_tokens: int,
             batch_size: int = 64) -> dict:
    t0 = time.time()
    by_tier: dict[int, list[float]] = {}
    formatted, gen_tokens = 0, 0
    for start in range(0, len(tasks), batch_size):
        chunk = tasks[start:start + batch_size]
        outs = generate(model, tokenizer, [prompt_tokens(tokenizer, env.question(t)) for t in chunk],
                        max_tokens, temp=0.0)
        for task, (toks, text, _) in zip(chunk, outs):
            r = env.reward(task, text)
            by_tier.setdefault(task["difficulty"], []).append(r)
            formatted += bool(ANSWER_RE.search(text))
            gen_tokens += len(toks)
    elapsed = time.time() - t0
    all_r = [r for rs in by_tier.values() for r in rs]
    return {
        "success_rate": sum(r >= 1.0 for r in all_r) / len(all_r),
        "by_tier": {str(k): sum(r >= 1.0 for r in v) / len(v) for k, v in sorted(by_tier.items())},
        "format_rate": formatted / len(all_r),
        "mean_completion_tokens": gen_tokens / len(all_r),
        "eval_seconds": round(elapsed, 1),
        "gen_tokens_per_sec": round(gen_tokens / elapsed, 1),
    }


# -------------------------------------------------------------------------- training

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
        mb = samples[start:start + micro_batch]
        seqs = [p + c for p, c, _ in mb]
        width = max(len(s) for s in seqs)
        inp, tgt, msk = [], [], []
        for (p, c, _), s in zip(mb, seqs):
            pad = width - len(s)
            inp.append(s[:-1] + [pad_id] * pad)
            tgt.append(s[1:] + [pad_id] * pad)
            msk.append([0.0] * (len(p) - 1) + [1.0] * len(c) + [0.0] * pad)
        adv = mx.array([a for _, _, a in mb], dtype=mx.float32)
        loss, g = loss_and_grad(model, mx.array(inp), mx.array(tgt), mx.array(msk), adv, norm)
        grads = g if grads is None else tree_map(mx.add, grads, g)
        mx.eval(grads, loss)
        total_loss += loss.item()
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    return total_loss


def cmd_train(args):
    run_dir = Path(args.out) / f"{args.model}-{args.sampler}-s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / "log.jsonl").open("x")

    def emit(record):
        print(json.dumps(record), flush=True)
        log.write(json.dumps(record) + "\n")
        log.flush()

    mx.random.seed(args.seed)
    model, tokenizer = load_policy(args.model, args.lora_rank)
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    optimizer = optim.Adam(learning_rate=args.lr)
    env = TaskEnv()
    train_tasks = make_tasks("train", args.train_per_tier)
    test_tasks = make_tasks("test", args.eval_per_tier)
    if args.sampler == "curriculum":
        sampler = CurriculumSampler(train_tasks, group_size=args.group_size, seed=args.seed)
    else:
        sampler = UniformSampler(train_tasks, seed=args.seed)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else min(tokenizer.eos_token_ids)
    emit({"event": "config", **{k: v for k, v in vars(args).items() if k != "fn"}, "trainable_params": n_train, "prompt_date": PROMPT_DATE})

    train_elapsed, gen_tokens_total = 0.0, 0

    def run_eval(step):
        result = evaluate(model, tokenizer, env, test_tasks, args.max_tokens)
        emit({"event": "eval", "step": step, "train_elapsed_s": round(train_elapsed, 1),
              "gen_tokens_total": gen_tokens_total, **result})
        save_checkpoint(model, sampler, run_dir)

    if args.eval_at_start:
        run_eval(0)

    done_event = {"event": "done", "stop_reason": "steps", "step": 0}
    for step in range(1, args.steps + 1):
        t0 = time.time()
        batch = sampler.sample(args.prompts_per_step)
        prompts = [prompt_tokens(tokenizer, env.question(t)) for t in batch]
        outs = generate(model, tokenizer, [p for p in prompts for _ in range(args.group_size)],
                        args.max_tokens, temp=args.temperature)
        t_gen = time.time() - t0

        samples, rewards_all, tier_rewards = [], [], {}
        for i, task in enumerate(batch):
            group = outs[i * args.group_size:(i + 1) * args.group_size]
            rewards = [env.reward(task, text) for _, text, _ in group]
            sampler.update(task, rewards)
            rewards_all += rewards
            tier_rewards.setdefault(task["difficulty"], []).extend(rewards)
            mean = sum(rewards) / len(rewards)
            std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
            if std < 1e-6:
                continue  # zero-signal group: generated, but contributes no gradient
            for (toks, _, _), r in zip(group, rewards):
                samples.append((prompts[i], toks, (r - mean) / (std + 1e-4)))
        t1 = time.time()
        loss = train_step(model, optimizer, samples, args.micro_batch, pad_id) if samples else 0.0
        t_train = time.time() - t1

        train_elapsed += time.time() - t0
        gen_tokens_total += sum(len(t) for t, _, _ in outs)
        record = {
            "event": "step", "step": step, "train_elapsed_s": round(train_elapsed, 1),
            "seconds": round(time.time() - t0, 2), "gen_seconds": round(t_gen, 2),
            "train_seconds": round(t_train, 2),
            "gen_tokens": sum(len(t) for t, _, _ in outs),
            "mean_reward": round(sum(rewards_all) / len(rewards_all), 4),
            "groups_with_signal": len(samples) // args.group_size,
            "zero_signal_fraction_step": round(1 - len(samples) / len(outs), 3),
            "zero_signal_fraction_recent": round(sampler.zero_signal_fraction, 3),
            "tier_mean_reward": {str(k): round(sum(v) / len(v), 3) for k, v in sorted(tier_rewards.items())},
            "loss": round(loss, 5),
            "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 2),
        }
        if isinstance(sampler, CurriculumSampler):
            record["p_hat"] = {str(k): round(sampler.p_hat(k), 3) for k in sampler.keys}
            record["sample_probs"] = {str(k): round(v, 3) for k, v in sampler.probabilities().items()}
        emit(record)
        done_event["step"] = step

        out_of_time = args.max_minutes and train_elapsed >= args.max_minutes * 60
        over_token_budget = args.token_budget and gen_tokens_total >= args.token_budget
        if (args.eval_every and step % args.eval_every == 0) or out_of_time or over_token_budget or step == args.steps:
            run_eval(step)
        if out_of_time or over_token_budget:
            done_event["stop_reason"] = "token_budget" if over_token_budget else "max_minutes"
            break
    emit({**done_event, "gen_tokens_total": gen_tokens_total})


def save_checkpoint(model, sampler, run_dir: Path):
    adapters = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(run_dir / "adapters.safetensors"), adapters)
    (run_dir / "sampler_state.json").write_text(json.dumps(sampler.state_dict()))


def cmd_baseline(args):
    env = TaskEnv()
    tasks = make_tasks("test", args.eval_per_tier)
    for name in args.models:
        model, tokenizer = load_policy(name, lora_rank=None)
        t0 = time.time()
        result = evaluate(model, tokenizer, env, tasks, args.max_tokens)
        print(json.dumps({"event": "baseline", "model": name, "n_tasks": len(tasks),
                          "load_and_eval_seconds": round(time.time() - t0, 1), **result}), flush=True)
        del model
        mx.clear_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("baseline")
    b.add_argument("--models", nargs="+", default=list(MODELS))
    b.add_argument("--eval-per-tier", type=int, default=40)
    b.add_argument("--max-tokens", type=int, default=384)
    b.set_defaults(fn=cmd_baseline)

    t = sub.add_parser("train")
    t.add_argument("--model", default="qwen3-0.6b")
    t.add_argument("--sampler", choices=["uniform", "curriculum"], default="curriculum")
    t.add_argument("--steps", type=int, default=20)
    t.add_argument("--max-minutes", type=float, default=0,
                   help="Stop after this much training time (generation + update, evals excluded).")
    t.add_argument("--token-budget", type=int, default=0,
                   help="Stop once this many tokens were generated in training; 0 turns token_budget off.")
    t.add_argument("--prompts-per-step", type=int, default=8)
    t.add_argument("--group-size", type=int, default=8)
    t.add_argument("--max-tokens", type=int, default=384)
    t.add_argument("--temperature", type=float, default=1.0)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--lora-rank", type=int, default=16)
    t.add_argument("--micro-batch", type=int, default=4)
    t.add_argument("--train-per-tier", type=int, default=2000)
    t.add_argument("--eval-per-tier", type=int, default=40)
    t.add_argument("--eval-every", type=int, default=10)
    t.add_argument("--eval-at-start", action="store_true")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default="runs")
    t.set_defaults(fn=cmd_train)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
