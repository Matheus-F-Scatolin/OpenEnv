# SPDX-License-Identifier: BSD-3-Clause

"""
The training loop of the trial, independent of the model.

It does what the pilot's loop does, in the same order, so that a run with the
pilot's settings draws the same random numbers. What it adds: a stop on
generated tokens, evaluations scheduled by tokens, and a detailed log.
"""

from __future__ import annotations

import math
import shutil
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

from samplers import CurriculumSampler, has_signal, TaskSampler, UniformSampler
from trial.policy import Policy
from trial.runlog import environment_info, RunLog, write_json_atomic
from trial.tasks import ANSWER_RE, LADDERS, PROMPT_DATE

if TYPE_CHECKING:
    from trial.env import TaskEnv

METHODS = (
    "uniform",
    "curriculum",
    "curriculum-tokens",
    "curriculum-tokens-mix",
    "curriculum-tokens-mix-decay",
)
# The settings each curriculum method adds to the RFC's defaults.
_CURRICULUM_SETTINGS: dict[str, dict[str, str]] = {
    "curriculum": {},
    "curriculum-tokens": {"cost": "tokens"},
    "curriculum-tokens-mix": {"cost": "tokens", "floor": "mixture"},
    "curriculum-tokens-mix-decay": {
        "cost": "tokens",
        "floor": "mixture",
        "prior": "decayed",
    },
}
EVAL_BATCH = 64
RECENT_GROUPS = 200
ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RunConfig:
    """
    The settings of one run: one field per command-line flag.

    Attributes:
        model (`str`):
            The model name.
        sampler (`str`):
            The sampling method, one of `METHODS` (or `baseline`).
        out (`str`):
            The folder of the experiment.
        cmd (`str`):
            `train` or `baseline`.
        steps (`int`):
            Stop after this many training steps; 0 turns it off.
        token_budget (`int`):
            Stop once this many tokens were generated in training; 0 turns it off.
        eval_every (`int`):
            Evaluate every this many steps; 0 turns it off.
        eval_every_tokens (`int`):
            Evaluate each time the generated tokens cross a multiple; 0 turns it off.
        snapshot_every_tokens (`int`):
            Keep a copy of the adapters each time the generated tokens cross a
            multiple; 0 turns it off.
        ladder (`str`):
            The name of the difficulty ladder.
    """

    model: str
    sampler: str
    out: str
    cmd: str = "train"
    steps: int = 0
    token_budget: int = 0
    prompts_per_step: int = 8
    group_size: int = 8
    max_tokens: int = 384
    temperature: float = 1.0
    lr: float = 2e-5
    lora_rank: int = 16
    micro_batch: int = 4
    train_per_tier: int = 2000
    eval_per_tier: int = 40
    eval_every: int = 0
    eval_every_tokens: int = 0
    eval_at_start: bool = False
    seed: int = 0
    ladder: str = "pilot"
    snapshot_every_tokens: int = 1_000_000


def build_sampler(
    method: str, tasks: list[dict[str, Any]], *, group_size: int, seed: int
) -> TaskSampler:
    """
    Build the sampler of a method name. The one place that knows the names.

    Args:
        method (`str`):
            One of `METHODS`.
        tasks (`list[dict[str, Any]]`):
            The training tasks.
        group_size (`int`):
            Number of answers per group.
        seed (`int`):
            The run's seed.

    Returns:
        `TaskSampler`: The sampler.

    Raises:
        `ValueError`: If the method name is unknown.
    """
    if method == "uniform":
        return UniformSampler(tasks, rng_seed=seed)
    if method in _CURRICULUM_SETTINGS:
        return CurriculumSampler(
            tasks,
            group_size=group_size,
            rng_seed=seed,
            **_CURRICULUM_SETTINGS[method],
        )
    raise ValueError(f"unknown method {method!r}: one of {METHODS}")


def stop_reason(
    step: int, gen_tokens_total: int, steps: int, token_budget: int
) -> str | None:
    """
    Say whether a run stops after a step, and why.

    Args:
        step (`int`):
            The step that just ended.
        gen_tokens_total (`int`):
            Tokens generated in training so far.
        steps (`int`):
            The step limit; 0 means none.
        token_budget (`int`):
            The token limit; 0 means none.

    Returns:
        `str` or `None`: `"token_budget"`, `"steps"`, or `None` to go on.
    """
    if token_budget and gen_tokens_total >= token_budget:
        return "token_budget"
    if steps and step >= steps:
        return "steps"
    return None


def eval_due(
    step: int,
    total_before: int,
    total_after: int,
    eval_every: int,
    eval_every_tokens: int,
) -> bool:
    """
    Say whether an evaluation is scheduled after a step.

    Args:
        step (`int`):
            The step that just ended.
        total_before (`int`):
            Tokens generated before the step.
        total_after (`int`):
            Tokens generated after the step.
        eval_every (`int`):
            Evaluate every this many steps; 0 means off.
        eval_every_tokens (`int`):
            Evaluate when the total crosses a multiple of this; 0 means off.

    Returns:
        `bool`: `True` when either schedule asks for it. A step that crosses
        two multiples still gets one evaluation.
    """
    if eval_every and step % eval_every == 0:
        return True
    return bool(
        eval_every_tokens
        and total_after // eval_every_tokens > total_before // eval_every_tokens
    )


def summarize_eval(records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute the totals of one evaluation from its per-question records.

    Args:
        records (`list[dict[str, Any]]`):
            One record per test question, with `tier`, `reward`, `tokens`,
            `truncated` and `formatted`.

    Returns:
        `dict[str, Any]`: The pilot's `success_rate`, `by_tier`, `format_rate`
        and `mean_completion_tokens`, then `n_tasks`,
        `mean_completion_tokens_by_tier` and `truncated_rate`.
    """
    by_tier: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_tier.setdefault(record["tier"], []).append(record)
    n = len(records)
    return {
        "success_rate": sum(r["reward"] >= 1.0 for r in records) / n,
        "by_tier": {
            str(tier): sum(r["reward"] >= 1.0 for r in rs) / len(rs)
            for tier, rs in sorted(by_tier.items())
        },
        "format_rate": sum(r["formatted"] for r in records) / n,
        "mean_completion_tokens": sum(r["tokens"] for r in records) / n,
        "n_tasks": n,
        "mean_completion_tokens_by_tier": {
            str(tier): sum(r["tokens"] for r in rs) / len(rs)
            for tier, rs in sorted(by_tier.items())
        },
        "truncated_rate": sum(r["truncated"] for r in records) / n,
    }


def evaluate(
    policy: Policy, env: TaskEnv, tasks: list[dict[str, Any]], max_tokens: int
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Evaluate the policy on fixed tasks: greedy, in task order.

    Args:
        policy (`Policy`):
            The model.
        env (`TaskEnv`):
            The environment that asks and scores.
        tasks (`list[dict[str, Any]]`):
            The test tasks.
        max_tokens (`int`):
            Length limit of each answer.

    Returns:
        `tuple[dict, list[dict], list[dict]]`: The totals (with `eval_seconds`
        and `gen_tokens_per_sec`), one record per question, and one text per
        question.
    """
    started = time.time()
    records, texts = [], []
    for start in range(0, len(tasks), EVAL_BATCH):
        chunk = tasks[start : start + EVAL_BATCH]
        prompts = [policy.encode_prompt(env.question(task)) for task in chunk]
        for task, completion in zip(chunk, policy.generate(prompts, max_tokens, 0.0)):
            records.append(
                {
                    "task": task["id"],
                    "tier": task["difficulty"],
                    "reward": env.reward(task, completion.text),
                    "tokens": len(completion.tokens),
                    "truncated": not completion.hit_stop,
                    "formatted": bool(ANSWER_RE.search(completion.text)),
                }
            )
            texts.append({"task": task["id"], "text": completion.text})
    elapsed = time.time() - started
    tokens = sum(record["tokens"] for record in records)
    summary = summarize_eval(records)
    summary["eval_seconds"] = round(elapsed, 1)
    summary["gen_tokens_per_sec"] = round(tokens / elapsed, 1) if elapsed else 0.0
    return summary, records, texts


def _config_event(
    config: RunConfig,
    policy: Policy,
    env: TaskEnv,
    sampler: TaskSampler | None,
    test_tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    settings = sampler.state_dict().get("settings", {}) if sampler else {}
    return {
        "event": "config",
        "schema_version": 2,
        **asdict(config),
        "tiers": LADDERS[config.ladder],
        **environment_info(ROOT, PROMPT_DATE),
        "prompt_example": policy.prompt_text(env.question(test_tasks[0])),
        "sampler_settings": settings,
        **policy.describe(),
    }


def _write_questions(
    log: RunLog, env: TaskEnv, test_tasks: list[dict[str, Any]]
) -> None:
    log.write_questions(
        [
            {
                "task": task["id"],
                "tier": task["difficulty"],
                "question": env.question(task),
                "correct_answer": env.correct_answer(task),
            }
            for task in test_tasks
        ]
    )


def _log_eval(
    log: RunLog,
    policy: Policy,
    env: TaskEnv,
    test_tasks: list[dict[str, Any]],
    config: RunConfig,
    step: int,
    train_elapsed: float,
    total: int,
) -> None:
    summary, records, texts = evaluate(policy, env, test_tasks, config.max_tokens)
    log.emit(
        {
            "event": "eval",
            "step": step,
            "train_elapsed_s": round(train_elapsed, 1),
            "gen_tokens_total": total,
            **summary,
        }
    )
    log.write_eval(
        [{"step": step, "gen_tokens_total": total, **r} for r in records],
        [{"step": step, **t} for t in texts],
    )


def _snapshot_name(tokens: int) -> str:
    label = f"{tokens // 1_000_000}M" if tokens % 1_000_000 == 0 else str(tokens)
    return f"adapters-{label}.safetensors"


def train_run(
    config: RunConfig,
    policy: Policy,
    env: TaskEnv,
    sampler: TaskSampler,
    log: RunLog,
    test_tasks: list[dict[str, Any]],
) -> None:
    """
    Train until the step limit or the token budget, and log everything.

    Args:
        config (`RunConfig`):
            The settings of the run.
        policy (`Policy`):
            The model.
        env (`TaskEnv`):
            The environment that asks and scores.
        sampler (`TaskSampler`):
            Chooses the training tasks.
        log (`RunLog`):
            The run's files.
        test_tasks (`list[dict[str, Any]]`):
            The fixed test tasks.

    Raises:
        `ValueError`: If neither `steps` nor `token_budget` is set.
    """
    if not config.steps and not config.token_budget:
        raise ValueError("give --steps or --token-budget: a run needs a stop rule")
    group_size = config.group_size
    adapters = log.run_dir / "adapters.safetensors"
    log.emit(_config_event(config, policy, env, sampler, test_tasks))
    _write_questions(log, env, test_tasks)

    train_elapsed, total, snapshots = 0.0, 0, 0
    recent_zero: deque[bool] = deque(maxlen=RECENT_GROUPS)

    def run_eval(step: int) -> None:
        nonlocal snapshots
        _log_eval(log, policy, env, test_tasks, config, step, train_elapsed, total)
        policy.save(adapters)
        write_json_atomic(log.run_dir / "sampler_state.json", sampler.state_dict())
        every = config.snapshot_every_tokens
        if every and total // every > snapshots:
            snapshots = total // every
            target = log.run_dir / _snapshot_name(snapshots * every)
            temporary = target.with_name(target.name + ".tmp")
            shutil.copyfile(adapters, temporary)
            temporary.replace(target)

    if config.eval_at_start:
        run_eval(0)

    step, reason = 0, None
    while reason is None:
        step += 1
        t0 = time.time()
        batch = sampler.sample(config.prompts_per_step)
        prompts = [policy.encode_prompt(env.question(task)) for task in batch]
        outs = policy.generate(
            [prompt for prompt in prompts for _ in range(group_size)],
            config.max_tokens,
            config.temperature,
        )
        t_gen = time.time() - t0

        samples, rewards_all, groups, texts = [], [], [], []
        tier_rewards: dict[int, list[float]] = {}
        for i, task in enumerate(batch):
            group = outs[i * group_size : (i + 1) * group_size]
            rewards = [env.reward(task, completion.text) for completion in group]
            tokens = [len(completion.tokens) for completion in group]
            sampler.update(task, rewards, costs=tokens)
            rewards_all += rewards
            tier_rewards.setdefault(task["difficulty"], []).extend(rewards)
            groups.append(
                {
                    "task": task["id"],
                    "tier": task["difficulty"],
                    "prompt_tokens": len(prompts[i]),
                    "rewards": rewards,
                    "tokens": tokens,
                    "truncated": [not completion.hit_stop for completion in group],
                    "formatted": [
                        bool(ANSWER_RE.search(completion.text)) for completion in group
                    ],
                }
            )
            texts += [
                {"step": step, "group": i, "answer": j, "text": completion.text}
                for j, completion in enumerate(group)
            ]
            signal = has_signal(rewards)
            recent_zero.append(not signal)
            if not signal:
                continue  # generated, but contributes no gradient
            mean = sum(rewards) / len(rewards)
            std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
            for completion, r in zip(group, rewards):
                samples.append(
                    (prompts[i], completion.tokens, (r - mean) / (std + 1e-4))
                )
        t1 = time.time()
        loss = policy.update(samples) if samples else 0.0
        t_train = time.time() - t1

        train_elapsed += time.time() - t0
        gen_tokens = sum(len(completion.tokens) for completion in outs)
        total_before, total = total, total + gen_tokens
        record: dict[str, Any] = {
            "event": "step",
            "step": step,
            "train_elapsed_s": round(train_elapsed, 1),
            "seconds": round(time.time() - t0, 2),
            "gen_seconds": round(t_gen, 2),
            "train_seconds": round(t_train, 2),
            "gen_tokens": gen_tokens,
            "mean_reward": round(sum(rewards_all) / len(rewards_all), 4),
            "groups_with_signal": len(samples) // group_size,
            "zero_signal_fraction_step": round(1 - len(samples) / len(outs), 3),
            "zero_signal_fraction_recent": round(
                sum(recent_zero) / len(recent_zero), 3
            ),
            "tier_mean_reward": {
                str(tier): round(sum(values) / len(values), 3)
                for tier, values in sorted(tier_rewards.items())
            },
            "loss": round(loss, 5),
            "peak_mem_gb": round(policy.peak_memory_gb(), 2),
        }
        if isinstance(sampler, CurriculumSampler):
            probabilities = sampler.probabilities()
            record["p_hat"] = {str(k): round(sampler.p_hat(k), 3) for k in sampler.keys}
            record["sample_probs"] = {
                str(k): round(value, 3) for k, value in probabilities.items()
            }
            record["sampler"] = {
                "p_hat": {str(k): sampler.p_hat(k) for k in sampler.keys},
                "cost_hat": {str(k): sampler.cost_hat(k) for k in sampler.keys},
                "probs": {str(k): value for k, value in probabilities.items()},
            }
        record["gen_tokens_total"] = total
        record["groups"] = groups
        log.emit(record)
        log.write_train_texts(texts)

        reason = stop_reason(step, total, config.steps, config.token_budget)
        if reason or eval_due(
            step, total_before, total, config.eval_every, config.eval_every_tokens
        ):
            run_eval(step)

    log.emit(
        {
            "event": "done",
            "stop_reason": reason,
            "step": step,
            "gen_tokens_total": total,
            "episodes": step * config.prompts_per_step * group_size,
        }
    )


def baseline_run(
    config: RunConfig,
    policy: Policy,
    env: TaskEnv,
    log: RunLog,
    test_tasks: list[dict[str, Any]],
) -> None:
    """
    Evaluate the policy once, without training.

    Args:
        config (`RunConfig`):
            The settings of the run, with `cmd="baseline"`.
        policy (`Policy`):
            The model.
        env (`TaskEnv`):
            The environment that asks and scores.
        log (`RunLog`):
            The run's files.
        test_tasks (`list[dict[str, Any]]`):
            The fixed test tasks.
    """
    log.emit(_config_event(config, policy, env, None, test_tasks))
    _write_questions(log, env, test_tasks)
    _log_eval(log, policy, env, test_tasks, config, 0, 0.0, 0)
    log.emit(
        {
            "event": "done",
            "stop_reason": "baseline",
            "step": 0,
            "gen_tokens_total": 0,
            "episodes": 0,
        }
    )


__all__ = [
    "METHODS",
    "RunConfig",
    "baseline_run",
    "build_sampler",
    "eval_due",
    "evaluate",
    "stop_reason",
    "summarize_eval",
    "train_run",
]
