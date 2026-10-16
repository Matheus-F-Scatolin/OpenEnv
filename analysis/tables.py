# SPDX-License-Identifier: BSD-3-Clause

"""
Print the first tables from run folders, as Markdown.

  python -m analysis.tables ten-run <experiment folder>
  python -m analysis.tables ladder <run or experiment folder>
  python -m analysis.tables cost-probe <experiment folder>
  python -m analysis.tables shadow <run folder>

Only complete runs are read; folders the queue set aside are ignored.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from analysis.validate_run import find_run_dirs
from samplers import has_signal
from trial.loop import _CURRICULUM_SETTINGS, build_sampler
from trial.runlog import is_complete, LOG_NAME, read_log
from trial.tasks import LADDERS, make_tasks

TEN_RUN_EVALS = (0, 10, 20, 30)
TEN_RUN_WINDOWS = {"1-10": (1, 10), "11-20": (11, 20), "21-30": (21, 30)}
MARGIN = 20.0


def load_runs(folder: str | Path) -> list[dict[str, Any]]:
    """
    Read every complete run in a folder.

    Args:
        folder (`str` or `Path`):
            A run folder, or a folder that holds run folders.

    Returns:
        `list[dict[str, Any]]`: Per run: `dir`, `config`, `steps` and `evals`
        (by step), sorted by seed, then in the order of the method names.
    """
    runs = []
    for run_dir in find_run_dirs(folder):
        if ".incomplete-" in run_dir.name or not is_complete(run_dir):
            continue
        events = read_log(run_dir / LOG_NAME)
        runs.append(
            {
                "dir": run_dir,
                "config": events[0],
                "steps": [e for e in events if e["event"] == "step"],
                "evals": {e["step"]: e for e in events if e["event"] == "eval"},
            }
        )
    order = ["baseline", "uniform", *_CURRICULUM_SETTINGS]
    return sorted(
        runs,
        key=lambda r: (r["config"]["seed"], order.index(r["config"]["sampler"])),
    )


def load_pilot_runs(pilot: str | Path) -> list[dict[str, Any]]:
    """
    Read the pilot's two stored runs, which end without a `done` event.

    Args:
        pilot (`str` or `Path`):
            The pilot's folder; its runs are in `runs/pilot/`.

    Returns:
        `list[dict[str, Any]]`: Per run, the same keys as `load_runs`.
    """
    runs = []
    for log in sorted((Path(pilot) / "runs" / "pilot").glob(f"*/{LOG_NAME}")):
        events = read_log(log)
        runs.append(
            {
                "dir": log.parent,
                "config": events[0],
                "steps": [e for e in events if e["event"] == "step"],
                "evals": {e["step"]: e for e in events if e["event"] == "eval"},
            }
        )
    return runs


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _tokens_by_tier(steps: list[dict[str, Any]]) -> dict[str, float]:
    """Mean training tokens per answer, per level, over some steps."""
    by_tier: dict[int, list[int]] = {}
    for step in steps:
        for group in step.get("groups", []):
            by_tier.setdefault(group["tier"], []).extend(group["tokens"])
    return {str(tier): _mean(tokens) for tier, tokens in sorted(by_tier.items())}


def ten_run(folder: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Read the ten-run check: answer length on the test questions, per run.

    Args:
        folder (`str` or `Path`):
            The experiment folder.

    Returns:
        `tuple[list[dict], dict]`: One row per run (`method`, `seed`, `length`
        at steps 0, 10, 20 and 30, `length_by_tier` at step 30, `train_tokens`
        per level in three windows, `signal_groups` up to step 30; the last
        three are empty for a version 1 log). And the summary: the difference
        curriculum minus uniform at step 30 per seed, the new seeds (all but
        seed 0), how many of them are at least 20 tokens above, their mean and
        their range.
    """
    rows = []
    for run in load_runs(folder):
        steps = [s for s in run["steps"] if s["step"] <= 30]
        final = run["evals"].get(30, {})
        rows.append(
            {
                "method": run["config"]["sampler"],
                "seed": run["config"]["seed"],
                "length": {
                    step: run["evals"][step]["mean_completion_tokens"]
                    for step in TEN_RUN_EVALS
                    if step in run["evals"]
                },
                "length_by_tier": final.get("mean_completion_tokens_by_tier", {}),
                "train_tokens": {
                    name: _tokens_by_tier(
                        [s for s in steps if low <= s["step"] <= high]
                    )
                    for name, (low, high) in TEN_RUN_WINDOWS.items()
                },
                "signal_groups": sum(s["groups_with_signal"] for s in steps),
            }
        )
    at_30 = {(r["method"], r["seed"]): r["length"].get(30) for r in rows}
    difference = {
        seed: at_30[("curriculum", seed)] - at_30[("uniform", seed)]
        for method, seed in at_30
        if method == "curriculum"
        and at_30[("curriculum", seed)] is not None
        and at_30.get(("uniform", seed)) is not None
    }
    new = {seed: value for seed, value in difference.items() if seed != 0}
    summary: dict[str, Any] = {
        "difference_at_30": difference,
        "new_seeds": sorted(new),
        "new_seeds_above_20": sum(value >= MARGIN for value in new.values()),
        "mean": _mean(list(new.values())) if new else None,
        "range": (min(new.values()), max(new.values())) if new else None,
    }
    return rows, summary


def ladder(folder: str | Path) -> dict[str, list[dict[str, Any]]]:
    """
    Read the first evaluation of each run, level by level.

    Args:
        folder (`str` or `Path`):
            A run folder or an experiment folder.

    Returns:
        `dict[str, list[dict]]`: Per run folder name, one row per level:
        `tier`, `n`, `success`, `standard_error` and `mean_tokens`, from the
        records of the evaluation at step 0.
    """
    out = {}
    for run in load_runs(folder):
        evals_file = run["dir"] / "evals.jsonl"
        if not evals_file.is_file():
            continue
        by_tier: dict[int, list[dict[str, Any]]] = {}
        for line in evals_file.read_text().splitlines():
            record = json.loads(line)
            if record["step"] == 0:
                by_tier.setdefault(record["tier"], []).append(record)
        rows = []
        for tier, records in sorted(by_tier.items()):
            p = sum(r["reward"] >= 1.0 for r in records) / len(records)
            rows.append(
                {
                    "tier": tier,
                    "n": len(records),
                    "success": p,
                    "standard_error": math.sqrt(p * (1 - p) / len(records)),
                    "mean_tokens": _mean([r["tokens"] for r in records]),
                }
            )
        out[run["dir"].name] = rows
    return out


def cost_probe(folder: str | Path, first: int = 51, last: int = 100) -> list[dict]:
    """
    Read where each cost-aware formula sent its episodes.

    Args:
        folder (`str` or `Path`):
            The experiment folder of the probes.
        first (`int`, *optional*, defaults to `51`):
            First training step counted.
        last (`int`, *optional*, defaults to `100`):
            Last training step counted, and the step of the evaluation read.

    Returns:
        `list[dict]`: One row per run: `method`, `seed`, the share of episodes
        and of generated tokens per level, the mean sampling probability per
        level, groups with signal per 1,000 episodes and per million tokens,
        the training tokens per answer per level, and the mean answer length
        on the test questions at step `last`.
    """
    rows = []
    for run in load_runs(folder):
        steps = [s for s in run["steps"] if first <= s["step"] <= last]
        groups = [group for step in steps for group in step["groups"]]
        if not groups:
            continue
        tiers = sorted({str(group["tier"]) for group in groups}, key=int)
        tokens = {t: 0 for t in tiers}
        count = {t: 0 for t in tiers}
        for group in groups:
            tokens[str(group["tier"])] += sum(group["tokens"])
            count[str(group["tier"])] += 1
        episodes = sum(len(group["rewards"]) for group in groups)
        signal = sum(has_signal(group["rewards"]) for group in groups)
        logged = [s["sampler"]["probs"] for s in steps if "sampler" in s]
        rows.append(
            {
                "method": run["config"]["sampler"],
                "seed": run["config"]["seed"],
                "episode_share": {t: count[t] / len(groups) for t in tiers},
                "token_share": {t: tokens[t] / sum(tokens.values()) for t in tiers},
                "mean_probs": (
                    {t: _mean([p[t] for p in logged]) for t in logged[0]}
                    if logged
                    else {}
                ),
                "signal_per_1000_episodes": 1000 * signal / episodes,
                "signal_per_million_tokens": 1e6 * signal / sum(tokens.values()),
                "train_tokens": _tokens_by_tier(steps),
                "eval_length_at_100": run["evals"]
                .get(last, {})
                .get("mean_completion_tokens"),
            }
        )
    return rows


def shadow(
    run_dir: str | Path,
) -> tuple[dict[str, list[dict[str, float]]], dict[str, list[dict[str, float]]]]:
    """
    Feed the logged groups of one run to each curriculum formula.

    It costs no machine time: it shows the probabilities each formula would
    have used on the data this run saw. The run itself would have gone another
    way under another formula, so this is a reading of the formulas, not a
    result.

    Args:
        run_dir (`str` or `Path`):
            One complete run folder with a version 2 log.

    Returns:
        `tuple[dict, dict]`: Per method name, the probabilities per level
        after each step; and their mean in each quarter of the run.
    """
    run = load_runs(run_dir)[0]
    config = run["config"]
    tasks = make_tasks("train", config["train_per_tier"], LADDERS[config["ladder"]])
    traces: dict[str, list[dict[str, float]]] = {}
    quarters: dict[str, list[dict[str, float]]] = {}
    for method in _CURRICULUM_SETTINGS:
        sampler = build_sampler(
            method, tasks, group_size=config["group_size"], seed=config["seed"]
        )
        trace = []
        for step in run["steps"]:
            for group in step["groups"]:
                task = {"id": group["task"], "difficulty": group["tier"]}
                sampler.update(task, group["rewards"], costs=group["tokens"])
            trace.append({str(k): v for k, v in sampler.probabilities().items()})
        traces[method] = trace
        size = len(trace) / 4
        quarters[method] = [
            {
                tier: _mean([p[tier] for p in part])
                for tier in trace[0]
                if (part := trace[round(q * size) : round((q + 1) * size)])
            }
            for q in range(4)
        ]
    return traces, quarters


def markdown_table(header: list[str], rows: list[list[Any]]) -> str:
    """
    Format rows as a Markdown table.

    Args:
        header (`list[str]`):
            The column names.
        rows (`list[list[Any]]`):
            One list of cells per row. A float is written with one decimal and
            `None` as a dash.

    Returns:
        `str`: The table, without a final newline.
    """

    def cell(value: Any) -> str:
        if value is None:
            return "–"
        return f"{value:.1f}" if isinstance(value, float) else str(value)

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _percent(shares: dict[str, float], tiers: list[str]) -> list[str]:
    return [f"{100 * shares[t]:.1f}%" if t in shares else "–" for t in tiers]


def ten_run_text(folder: str | Path) -> str:
    """
    Format the tables of the ten-run check.

    Args:
        folder (`str` or `Path`):
            The experiment folder.

    Returns:
        `str`: The Markdown tables of `ten_run`, without a final newline.
    """
    rows, summary = ten_run(folder)
    parts = [
        "Mean answer length on the test questions, in tokens\n",
        markdown_table(
            ["method", "seed", *[f"step {s}" for s in TEN_RUN_EVALS], "signal groups"],
            [
                [
                    r["method"],
                    r["seed"],
                    *[r["length"].get(s) for s in TEN_RUN_EVALS],
                    r["signal_groups"],
                ]
                for r in rows
            ],
        ),
        "\nCurriculum minus uniform at step 30, per seed\n",
        markdown_table(
            ["seed", "difference"],
            [
                [f"seed {seed}" + (" (known)" if seed == 0 else ""), value]
                for seed, value in sorted(summary["difference_at_30"].items())
            ],
        ),
    ]
    if summary["new_seeds"]:
        low, high = summary["range"]
        parts.append(
            f"\nNew seeds {summary['new_seeds']}: {summary['new_seeds_above_20']} of "
            f"{len(summary['new_seeds'])} at least {MARGIN:.0f} tokens above; "
            f"mean {summary['mean']:.1f}, range {low:.1f} to {high:.1f}."
        )
    tiers = sorted({t for r in rows for t in r["length_by_tier"]}, key=int)
    if tiers:
        parts.append(
            "\nAt step 30, per level (test), then training tokens per answer\n"
        )
        header = ["method", "seed", "what", *[f"level {t}" for t in tiers]]
        body = []
        for r in rows:
            body.append(
                [r["method"], r["seed"], "test, step 30"]
                + [r["length_by_tier"].get(t) for t in tiers]
            )
            for name, values in r["train_tokens"].items():
                body.append(
                    [r["method"], r["seed"], f"train, steps {name}"]
                    + [values.get(t) for t in tiers]
                )
        parts.append(markdown_table(header, body))
    return "\n".join(parts)


def ladder_text(folder: str | Path) -> str:
    """
    Format the success per level of each run's first evaluation.

    Args:
        folder (`str` or `Path`):
            A run folder or an experiment folder.

    Returns:
        `str`: One Markdown table per run, each under the run folder's name.
    """
    parts = []
    for name, rows in ladder(folder).items():
        parts += [
            f"{name}\n",
            markdown_table(
                ["level", "questions", "success", "standard error", "mean tokens"],
                [
                    [
                        r["tier"],
                        r["n"],
                        f"{100 * r['success']:.1f}%",
                        f"{100 * r['standard_error']:.1f}",
                        r["mean_tokens"],
                    ]
                    for r in rows
                ],
            ),
            "",
        ]
    return "\n".join(parts)


def cost_probe_text(folder: str | Path) -> str:
    """
    Format where each cost-aware formula sent its episodes in the probes.

    Args:
        folder (`str` or `Path`):
            The experiment folder of the probes.

    Returns:
        `str`: The Markdown tables of `cost_probe`, without a final newline.
    """
    rows = cost_probe(folder)
    tiers = sorted({t for r in rows for t in r["episode_share"]}, key=int)
    header = ["method", "seed", "what", *[f"level {t}" for t in tiers]]
    body = []
    for r in rows:
        for what, key in (
            ("share of episodes", "episode_share"),
            ("share of tokens", "token_share"),
            ("mean probability", "mean_probs"),
        ):
            body.append([r["method"], r["seed"], what, *_percent(r[key], tiers)])
        body.append(
            [r["method"], r["seed"], "train tokens per answer"]
            + [r["train_tokens"].get(t) for t in tiers]
        )
    summary = markdown_table(
        [
            "method",
            "seed",
            "easiest level, share of episodes",
            "signal groups per 1,000 episodes",
            "per million tokens",
            "test answer length at step 100",
        ],
        [
            [
                r["method"],
                r["seed"],
                _percent(r["episode_share"], tiers[:1])[0],
                r["signal_per_1000_episodes"],
                r["signal_per_million_tokens"],
                r["eval_length_at_100"],
            ]
            for r in rows
        ],
    )
    return "\n".join(["Steps 51 to 100\n", markdown_table(header, body), "", summary])


def shadow_text(run_dir: str | Path) -> str:
    """
    Format the probabilities each formula would have used on one run's data.

    Args:
        run_dir (`str` or `Path`):
            One complete run folder with a version 2 log.

    Returns:
        `str`: The Markdown table of `shadow`, one row per formula and quarter.
    """
    _, quarters = shadow(run_dir)
    tiers = list(next(iter(quarters.values()))[0])
    return "\n".join(
        [
            "Probability per level each formula would have used on this run's data\n",
            markdown_table(
                ["formula", "quarter", *[f"level {t}" for t in tiers]],
                [
                    [method, number + 1, *_percent(quarter, tiers)]
                    for method, parts in quarters.items()
                    for number, quarter in enumerate(parts)
                ],
            ),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    """
    Print one table from the command line.

    Args:
        argv (`list[str]`, *optional*):
            The arguments; `sys.argv[1:]` when left out.

    Returns:
        `int`: 0.
    """
    formatters = {
        "ten-run": ten_run_text,
        "ladder": ladder_text,
        "cost-probe": cost_probe_text,
        "shadow": shadow_text,
    }
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("table", choices=sorted(formatters))
    parser.add_argument("folder")
    args = parser.parse_args(argv)
    print(formatters[args.table](args.folder))
    return 0


__all__ = [
    "cost_probe",
    "cost_probe_text",
    "ladder",
    "ladder_text",
    "load_pilot_runs",
    "load_runs",
    "main",
    "markdown_table",
    "shadow",
    "shadow_text",
    "ten_run",
    "ten_run_text",
]


if __name__ == "__main__":
    sys.exit(main())
