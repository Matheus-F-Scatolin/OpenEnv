# SPDX-License-Identifier: BSD-3-Clause

"""
Say whether two run logs describe the same training.

Only fields that depend on the training alone are compared. Times, memory and
fields that one of the logs lacks are ignored.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from trial.runlog import read_log

STEP_FIELDS = (
    "gen_tokens",
    "mean_reward",
    "groups_with_signal",
    "zero_signal_fraction_step",
    "zero_signal_fraction_recent",
    "tier_mean_reward",
    "loss",
    "p_hat",
    "sample_probs",
)
EVAL_FIELDS = ("success_rate", "by_tier", "mean_completion_tokens")


def _load(
    path: str | Path, steps: int
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any]]:
    """Return the step events, the evaluation events and the config of a log."""
    events = read_log(path)
    configs = [event for event in events if event.get("event") == "config"]
    if len(configs) != 1:
        raise ValueError(f"{path}: {len(configs)} config events, expected 1")
    by_step: dict[int, dict[str, Any]] = {}
    for event in events:
        if event.get("event") != "step":
            continue
        if event["step"] in by_step:
            raise ValueError(f"{path}: step {event['step']} is logged twice")
        by_step[event["step"]] = event
    missing = [step for step in range(1, steps + 1) if step not in by_step]
    if missing:
        raise ValueError(
            f"{path}: has {len(by_step)} steps, cannot compare {steps} steps"
        )
    evals = {e["step"]: e for e in events if e.get("event") == "eval"}
    return by_step, evals, configs[0]


def _questions(evaluation: dict[str, Any], config: dict[str, Any]) -> int:
    """Number of test questions an evaluation event covers."""
    if "n_tasks" in evaluation:
        return evaluation["n_tasks"]
    return config["eval_per_tier"] * len(evaluation["by_tier"])


def compare_logs(
    old: str | Path, new: str | Path, steps: int
) -> tuple[int, int, list[str]]:
    """
    Compare the first training steps of two logs.

    Args:
        old (`str` or `Path`):
            The reference log.
        new (`str` or `Path`):
            The log to check against it.
        steps (`int`):
            Number of training steps to compare, from step 1.

    Returns:
        `tuple[int, int, list[str]]`: The number of steps that agree, the number
        of evaluations compared, and one line per difference.

    Raises:
        `ValueError`: If a log has more or fewer than one `config` event, logs a
            step number twice, or has fewer steps than asked.
    """
    old_steps, old_evals, old_config = _load(old, steps)
    new_steps, new_evals, new_config = _load(new, steps)
    differences: list[str] = []
    same = 0
    for step in range(1, steps + 1):
        before = len(differences)
        for field in STEP_FIELDS:
            a, b = old_steps[step], new_steps[step]
            if field in a and field in b and a[field] != b[field]:
                differences.append(f"step {step}: {field} {a[field]} != {b[field]}")
        same += len(differences) == before
    compared = 0
    for step in sorted(set(old_evals) & set(new_evals)):
        a, b = old_evals[step], new_evals[step]
        if step > steps or _questions(a, old_config) != _questions(b, new_config):
            continue
        compared += 1
        for field in EVAL_FIELDS:
            if a[field] != b[field]:
                differences.append(
                    f"eval at step {step}: {field} {a[field]} != {b[field]}"
                )
    return same, compared, differences


def main(argv: list[str] | None = None) -> int:
    """
    Run the comparison from the command line.

    Args:
        argv (`list[str]`, *optional*):
            The arguments; `sys.argv[1:]` when left out.

    Returns:
        `int`: 0 when the logs agree, 1 when they differ, 2 when a log is refused.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", help="the reference log.jsonl")
    parser.add_argument("new", help="the log.jsonl to check")
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        same, compared, differences = compare_logs(args.old, args.new, args.steps)
    except ValueError as error:
        print(f"refused: {error}")
        return 2
    for line in differences:
        print(line)
    if differences:
        print(f"different: {same}/{args.steps} steps agree")
        return 1
    print(f"identical in {same}/{args.steps} steps, {compared} evaluations compared")
    return 0


if __name__ == "__main__":
    sys.exit(main())
