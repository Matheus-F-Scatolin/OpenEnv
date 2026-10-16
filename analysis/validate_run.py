# SPDX-License-Identifier: BSD-3-Clause

"""
Check that run folders are complete and consistent.

  python -m analysis.validate_run <folder> [<folder> ...]

Given a run folder it checks that one; given any other folder it checks every
run folder inside it. It prints one line per run folder and exits with 1 when
any has a problem.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from trial.loop import stop_reason, summarize_eval
from trial.runlog import LOG_NAME, read_log
from trial.tasks import PROMPT_DATE

CONFIG_FIELDS = (
    "schema_version",
    "prompt_date",
    "versions",
    "git_commit",
    "seed",
    "cmd",
    "steps",
    "token_budget",
)
SAME_ACROSS_RUNS = ("model_snapshot", "versions", "platform")
SKIPPED = "skipped (incomplete, moved aside by the queue)"


def find_run_dirs(folder: str | Path) -> list[Path]:
    """
    List the run folders in a folder.

    Args:
        folder (`str` or `Path`):
            A run folder, or a folder that holds run folders at any depth.

    Returns:
        `list[Path]`: Every folder with a `log.jsonl`, sorted.
    """
    folder = Path(folder)
    if (folder / LOG_NAME).is_file():
        return [folder]
    return sorted(path.parent for path in folder.rglob(LOG_NAME))


def _check_steps(steps: list[dict[str, Any]]) -> str | None:
    numbers = [step["step"] for step in steps]
    if numbers != list(range(1, len(steps) + 1)):
        return "step numbers are not 1, 2, 3, ... without gaps or repeats"
    return None


def _check_version_1(events: list[dict[str, Any]]) -> str | None:
    problem = _check_steps([e for e in events if e["event"] == "step"])
    if problem:
        return problem
    total = 0
    for event in events:
        if event["event"] == "step":
            total += event["gen_tokens"]
        elif event["event"] == "eval" and "gen_tokens_total" in event:
            if event["gen_tokens_total"] != total:
                return f"eval at step {event['step']}: gen_tokens_total is not the running sum"
    return None


def _check_version_2(run_dir: Path, events: list[dict[str, Any]]) -> str | None:
    config, done = events[0], events[-1]
    for field in CONFIG_FIELDS:
        if config.get(field) is None:
            return f"config has no {field}"
    if f"Today Date: {PROMPT_DATE}" not in config.get("prompt_example", ""):
        return f"prompt_example does not contain 'Today Date: {PROMPT_DATE}'"
    steps = [e for e in events if e["event"] == "step"]
    problem = _check_steps(steps)
    if problem:
        return problem
    if config["cmd"] not in ("train", "baseline"):
        return f"unknown command in config: {config['cmd']}"
    if config["cmd"] == "baseline" and steps:
        return "a baseline has training steps"
    total = 0
    totals = {0: 0}
    for step in steps:
        number = step["step"]
        groups = step["groups"]
        if len(groups) != config["prompts_per_step"]:
            return f"step {number}: {len(groups)} groups, expected {config['prompts_per_step']}"
        for group in groups:
            sizes = {len(group["rewards"]), len(group["tokens"])}
            if sizes != {config["group_size"]}:
                return f"step {number}: a group does not have {config['group_size']} rewards and tokens"
        if sum(sum(group["tokens"]) for group in groups) != step["gen_tokens"]:
            return (
                f"step {number}: the tokens of the groups do not add up to gen_tokens"
            )
        total += step["gen_tokens"]
        if step["gen_tokens_total"] != total:
            return f"step {number}: gen_tokens_total is not the running sum"
        totals[number] = total
        reached = stop_reason(number, total, config["steps"], config["token_budget"])
        if reached and number < len(steps):
            return f"step {number}: training continued after the {reached} limit"

    expected_counters = {
        "step": len(steps),
        "gen_tokens_total": total,
        "episodes": sum(
            len(group["rewards"]) for step in steps for group in step["groups"]
        ),
    }
    for field, expected in expected_counters.items():
        if done.get(field) != expected:
            return f"done: {field} is {done.get(field)}, expected {expected}"
    expected_reason = (
        "baseline"
        if config["cmd"] == "baseline"
        else stop_reason(len(steps), total, config["steps"], config["token_budget"])
    )
    if expected_reason is None:
        return "run finished before reaching its step limit or token budget"
    if done.get("stop_reason") != expected_reason:
        return f"done: stop_reason is {done.get('stop_reason')}, expected {expected_reason}"

    evals_file = run_dir / "evals.jsonl"
    records = (
        [json.loads(line) for line in evals_file.read_text().splitlines()]
        if evals_file.is_file()
        else []
    )
    evaluations = [e for e in events if e["event"] == "eval"]
    if not any(e["step"] == len(steps) for e in evaluations):
        return f"no evaluation at the last step ({len(steps)})"
    for evaluation in evaluations:
        own = [r for r in records if r["step"] == evaluation["step"]]
        where = f"eval at step {evaluation['step']}"
        if evaluation["gen_tokens_total"] != totals.get(evaluation["step"]):
            return f"{where}: gen_tokens_total is not the running sum"
        if any(r["gen_tokens_total"] != evaluation["gen_tokens_total"] for r in own):
            return f"{where}: its records in evals.jsonl have another gen_tokens_total"
        if len(own) != evaluation["n_tasks"]:
            return f"{where}: {len(own)} records in evals.jsonl, expected {evaluation['n_tasks']}"
        summary = summarize_eval(own)
        for field in ("success_rate", "mean_completion_tokens"):
            if summary[field] != evaluation[field]:
                return f"{where}: {field} differs from its records"

    return None


def check_run(run_dir: str | Path) -> tuple[str, dict[str, Any] | None]:
    """
    Check one run folder.

    Args:
        run_dir (`str` or `Path`):
            The folder of one run.

    Returns:
        `tuple[str, dict | None]`: `"OK"`, `"OK (version 1)"` or the first
        problem found; and the run's `config` event when it has one.
    """
    run_dir = Path(run_dir)
    events = read_log(run_dir / LOG_NAME)
    if not events or events[0].get("event") != "config":
        return "the first event is not config", None
    config = events[0]
    if events[-1].get("event") != "done":
        return "the last event is not done: the run did not finish", config
    if not config.get("prompt_date"):
        return "config has no prompt_date", config
    if "schema_version" in config:
        problem = _check_version_2(run_dir, events)
        return problem or "OK", config
    problem = _check_version_1(events)
    return problem or "OK (version 1)", config


def validate(folders: list[str | Path]) -> list[str]:
    """
    Check every run folder in the given folders, and the runs against each other.

    Args:
        folders (`list[str | Path]`):
            Run folders, or folders that hold them.

    Returns:
        `list[str]`: One line per run folder, `<run_dir>: <status>`, then one
        line per field in which the runs differ from each other. Runs of the
        gate experiments are left out of that comparison.
    """
    lines: list[str] = []
    comparable: list[tuple[Path, dict[str, Any]]] = []
    for folder in folders:
        for run_dir in find_run_dirs(folder):
            if ".incomplete-" in run_dir.name:
                lines.append(f"{run_dir}: {SKIPPED}")
                continue
            status, config = check_run(run_dir)
            lines.append(f"{run_dir}: {status}")
            if config and not run_dir.parent.name.startswith("gate"):
                comparable.append((run_dir, config))
    if not lines:
        lines.append(f"{', '.join(map(str, folders))}: no run folder found")
    fields = [("prompt_date", comparable)]
    version_2 = [(d, c) for d, c in comparable if "schema_version" in c]
    fields += [(field, version_2) for field in SAME_ACROSS_RUNS]
    for field, runs in fields:
        values = {json.dumps(config.get(field), sort_keys=True) for _, config in runs}
        if len(values) > 1:
            lines.append(f"the runs differ in {field}: {sorted(values)}")
    return lines


def main(argv: list[str] | None = None) -> int:
    """
    Run the checker from the command line.

    Args:
        argv (`list[str]`, *optional*):
            The arguments; `sys.argv[1:]` when left out.

    Returns:
        `int`: 0 when every run is `OK` or skipped, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="+")
    args = parser.parse_args(argv)
    lines = validate(args.folders)
    print("\n".join(lines))
    good = (": OK", f": {SKIPPED}")
    return 0 if all(line.endswith(good) or ": OK (" in line for line in lines) else 1


if __name__ == "__main__":
    sys.exit(main())
