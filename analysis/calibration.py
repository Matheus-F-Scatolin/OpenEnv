# SPDX-License-Identifier: BSD-3-Clause

"""
Check the binomial model of groups with signal against the trial's logs, and
measure how graded the rewards are.

`python -m analysis.report` writes the result to `calibration.md`. It reads
the runs of seeds 1, 2 and 3 in `trial/`, `cost-full/` and `seed3/`; the
pilot's log does not record the level of each group.
"""

from __future__ import annotations

import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pvariance
from typing import Any

from analysis.figures import METHODS, SEEDS, trial_runs
from analysis.tables import markdown_table
from samplers import CurriculumSampler, has_signal

# The estimate of a level the curriculum has not seen yet, under either prior.
PRIOR_P_HAT = 0.5
SUCCESS_THRESHOLD = (
    inspect.signature(CurriculumSampler).parameters["success_threshold"].default
)
WINDOWS = ("whole run", "first quarter", "last quarter")
MIN_GROUPS = 10
EVERY_METHOD = "every method"


@dataclass
class _Group:
    tier: str
    rewards: list[float]
    successes: int
    # The curriculum's estimate for the level when the step was drawn.
    p_hat: float | None


def signal_chance(p: float, group_size: int) -> float:
    """
    Return the chance that a group has signal if its answers are independent.

    Args:
        p (`float`):
            The chance that one answer succeeds.
        group_size (`int`):
            The answers in a group, `G`.

    Returns:
        `float`: `1 - p**G - (1 - p)**G`, the chance of at least one success
        and one failure. The curriculum sampler builds its weights from it.
    """
    return 1.0 - p**group_size - (1.0 - p) ** group_size


def _threshold(config: dict[str, Any]) -> float:
    # Uniform has no success threshold; its runs are scored with the
    # curriculum's default.
    return config["sampler_settings"].get("success_threshold", SUCCESS_THRESHOLD)


def _group_size(runs: list[dict[str, Any]]) -> int:
    sizes = {run["config"]["group_size"] for run in runs}
    if len(sizes) != 1:
        raise ValueError(f"the runs have group sizes {sorted(sizes)}, not one")
    return sizes.pop()


def _windows(run: dict[str, Any]) -> dict[str, list[_Group]]:
    """
    Split the groups of one run into the windows of `WINDOWS`.

    The first quarter is the steps whose index `i` (from 0, of `n`) has
    `4 * i < n`; the last quarter those with `4 * i >= 3 * n`.
    """
    config, steps = run["config"], run["steps"]
    threshold = _threshold(config)
    drawn_with: list[dict[str, float] | None]
    if config["sampler"] == "uniform":
        drawn_with = [None] * len(steps)
    else:
        # A step samples, generates, then updates; the log records the state
        # after the updates, so a step is drawn with the previous step's state.
        logged = [step["sampler"]["p_hat"] for step in steps]
        drawn_with = [dict.fromkeys(logged[0], PRIOR_P_HAT), *logged[:-1]]
    per_step = [
        [
            _Group(
                tier=str(group["tier"]),
                rewards=group["rewards"],
                successes=sum(r >= threshold for r in group["rewards"]),
                p_hat=None if estimates is None else estimates[str(group["tier"])],
            )
            for group in step["groups"]
        ]
        for step, estimates in zip(steps, drawn_with)
    ]
    first, last = math.ceil(len(steps) / 4), math.ceil(3 * len(steps) / 4)
    return {
        "whole run": [g for groups in per_step for g in groups],
        "first quarter": [g for groups in per_step[:first] for g in groups],
        "last quarter": [g for groups in per_step[last:] for g in groups],
    }


def signal_calibration(
    runs: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, float]]:
    """
    Compare the groups with signal that the binomial model predicts with those
    that happened.

    For a curriculum method, `p` is the sampler's estimate for the level when
    it drew the step: the `p_hat` logged at the previous step, `PRIOR_P_HAT`
    at the first step. Uniform has no estimate: its `p` is the share of the
    level's answers that succeeded, in the same run and window.

    Args:
        runs (`list[dict[str, Any]]`):
            Runs as `load_runs` reads them, with version 2 logs.

    Returns:
        `dict[tuple[str, str, str], dict[str, float]]`: Per method, window (one
        of `WINDOWS`) and level, pooled over the runs: `groups`, `predicted`
        (the mean of `signal_chance(p, G)` over the groups) and `observed` (the
        share of the groups for which `has_signal` holds).
    """
    cells: dict[tuple[str, str, str], list[tuple[float, bool]]] = {}
    for run in runs:
        method, size = run["config"]["sampler"], run["config"]["group_size"]
        for window, groups in _windows(run).items():
            successes: dict[str, list[int]] = {}
            for g in groups:
                successes.setdefault(g.tier, []).append(g.successes)
            rate = {tier: sum(k) / (size * len(k)) for tier, k in successes.items()}
            for g in groups:
                p = rate[g.tier] if g.p_hat is None else g.p_hat
                cells.setdefault((method, window, g.tier), []).append(
                    (signal_chance(p, size), has_signal(g.rewards))
                )
    return {
        key: {
            "groups": len(values),
            "predicted": fmean(predicted for predicted, _ in values),
            "observed": fmean(observed for _, observed in values),
        }
        for key, values in cells.items()
    }


def group_correlation(
    runs: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, float | None]]:
    """
    Measure how much more alike the answers of one group are than the binomial
    model assumes, in the last quarter of each run.

    Args:
        runs (`list[dict[str, Any]]`):
            Runs as `load_runs` reads them, with version 2 logs and one group
            size.

    Returns:
        `dict[tuple[str, str], dict[str, float | None]]`: Per method and
        level, pooled over the runs: `groups`; `p`, the share of answers that
        succeeded; `rho`, `(Var(k) / (G p (1 - p)) - 1) / (G - 1)` with `k` the
        successes of a group and `Var` the population variance over the
        groups (`None` when `p` is 0 or 1, or below `MIN_GROUPS` groups);
        `all_fail`, the share of groups without a success; and
        `binomial_all_fail`, `(1 - p)**G`.

    Raises:
        `ValueError`: If the runs have different group sizes.
    """
    size = _group_size(runs)
    cells: dict[tuple[str, str], list[int]] = {}
    for run in runs:
        for g in _windows(run)["last quarter"]:
            cells.setdefault((run["config"]["sampler"], g.tier), []).append(g.successes)
    out: dict[tuple[str, str], dict[str, float | None]] = {}
    for key, k in cells.items():
        p = sum(k) / (size * len(k))
        binomial = size * p * (1 - p)
        out[key] = {
            "groups": len(k),
            "p": p,
            "rho": (
                (pvariance(k) / binomial - 1) / (size - 1)
                if binomial and len(k) >= MIN_GROUPS
                else None
            ),
            "all_fail": sum(successes == 0 for successes in k) / len(k),
            "binomial_all_fail": (1 - p) ** size,
        }
    return out


def graded_rewards(runs: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """
    Count the rewards between failure and success, in training and evaluation.

    Args:
        runs (`list[dict[str, Any]]`):
            Runs as `load_runs` reads them, each folder with its `evals.jsonl`.

    Returns:
        `dict[str, dict[str, int]]`: Per method, and for `EVERY_METHOD`: `runs`;
        `rollouts`, the training answers; `partial`, those with a reward
        strictly between 0 and 1; `groups`; `signal`, the groups for which
        `has_signal` holds; `signal_without_success`, those of them with no
        answer at the success threshold; `mixed_without_signal`, the groups
        with successes and failures but no signal; `answers`, the evaluation
        answers; and `partial_answers`, those with a reward strictly between 0
        and 1.
    """
    out: dict[str, dict[str, int]] = {}
    for run in runs:
        groups = _windows(run)["whole run"]
        signal = [has_signal(g.rewards) for g in groups]
        lines = (run["dir"] / "evals.jsonl").read_text().splitlines()
        answers = [json.loads(line)["reward"] for line in lines]
        counts = {
            "runs": 1,
            "rollouts": sum(len(g.rewards) for g in groups),
            "partial": sum(0 < r < 1 for g in groups for r in g.rewards),
            "groups": len(groups),
            "signal": sum(signal),
            "signal_without_success": sum(
                s and g.successes == 0 for g, s in zip(groups, signal)
            ),
            "mixed_without_signal": sum(
                not s and 0 < g.successes < len(g.rewards)
                for g, s in zip(groups, signal)
            ),
            "answers": len(answers),
            "partial_answers": sum(0 < r < 1 for r in answers),
        }
        for name in (run["config"]["sampler"], EVERY_METHOD):
            total = out.setdefault(name, dict.fromkeys(counts, 0))
            for key, value in counts.items():
                total[key] += value
    return out


def _fixed(value: float | None, digits: int) -> str | None:
    if value is None:
        return None
    text = f"{value:.{digits}f}"
    # A small negative value rounds to "-0.00"; write it as "0.00".
    return text.removeprefix("-") if float(text) == 0 else text


def _share(count: int, total: int) -> str | None:
    return f"{100 * count / total:.3f}%" if total else None


def calibration_text(runs_folder: str | Path) -> str:
    """
    Format the calibration of the signal model and the graded rewards.

    Args:
        runs_folder (`str` or `Path`):
            The folder that holds `trial/`, `cost-full/` and `seed3/`.

    Returns:
        `str`: The Markdown of `calibration.md`, without its title.

    Raises:
        `ValueError`: If the runs have different group sizes.
    """
    runs = [run for run in trial_runs(runs_folder) if run["config"]["seed"] in SEEDS]
    size = _group_size(runs)
    seeds = [str(seed) for seed in sorted({run["config"]["seed"] for run in runs})]
    which = (
        f"seed {seeds[0]}"
        if len(seeds) == 1
        else f"seeds {', '.join(seeds[:-1])} and {seeds[-1]}"
    )
    pooled = "" if len(seeds) == 1 else ", pooled over the seeds"
    thresholds = sorted({_threshold(run["config"]) for run in runs})
    present = {run["config"]["sampler"] for run in runs}
    methods = [method for method in METHODS if method in present]
    calibration = signal_calibration(runs)
    tiers = sorted({tier for _, _, tier in calibration}, key=int)
    parts = [
        f"The {len(runs)} {'run' if len(runs) == 1 else 'runs'} of {which} in "
        f"`trial/`, `cost-full/` and `seed3/`{pooled}. The pilot is left out: its "
        "log does not record the level of each group. A group is the "
        f"G = {size} answers to one task. It has signal when the population "
        "standard deviation of its rewards is at least 1e-6 "
        "(`samplers.has_signal`), in practice when they are not all equal: only "
        "such a group changes the model. "
        "An answer succeeds when its reward is at or above the "
        "`success_threshold` of its run's sampler settings; uniform logs no "
        "sampler settings, and is scored with the curriculum sampler's default, "
        f"{SUCCESS_THRESHOLD}. Thresholds in these runs: "
        f"{', '.join(map(str, thresholds))}. The first quarter of a run is the "
        "first quarter of its steps, rounded up, and the last quarter its last "
        "quarter, rounded down.",
        "\n## Predicted and observed groups with signal\n",
        f"If the {size} answers of a group were independent, each a success with "
        f"chance p, the group would have signal with chance 1 - p^{size} - "
        f"(1 - p)^{size}. The curriculum methods build their sampling weights "
        "from this chance. For them p is the sampler's own estimate for the "
        "level (`p_hat`) when it drew the step: a step samples, generates and "
        "then reports its groups, and the log records the estimate after the "
        "reports, so a step is drawn with the estimate logged at the previous "
        f"step, and the first step with {PRIOR_P_HAT}, the estimate of a level "
        "not seen yet. Uniform has no estimate: its p is the share of the "
        "level's answers that succeeded, in the same run and window. Predicted "
        "is the mean of that chance over the groups, observed the share of the "
        "groups that had signal.",
    ]
    for window in WINDOWS:
        rows = []
        for method in methods:
            source = (
                "success share in the window"
                if method == "uniform"
                else "`p_hat` before the step"
            )
            for tier in tiers:
                cell = calibration.get((method, window, tier))
                if cell is None:
                    continue
                ratio = (
                    cell["observed"] / cell["predicted"] if cell["predicted"] else None
                )
                rows.append(
                    [
                        method,
                        source,
                        tier,
                        cell["groups"],
                        _fixed(cell["predicted"], 3),
                        _fixed(cell["observed"], 3),
                        _fixed(ratio, 2),
                    ]
                )
        parts += [
            f"\n### {window.capitalize()}\n",
            markdown_table(
                [
                    "method",
                    "p",
                    "level",
                    "groups",
                    "predicted",
                    "observed",
                    "observed / predicted",
                ],
                rows,
            ),
        ]
    correlation = group_correlation(runs)
    parts += [
        "\n## Correlation of successes inside a group\n",
        "The last quarter of each run. p is the share of answers that succeeded "
        "and k the successes in a group. rho = (Var(k) / "
        f"({size} p (1 - p)) - 1) / {size - 1}, with Var the population variance "
        "over the groups, estimates the correlation between two answers to the "
        "same task: 0 if they were independent, as the binomial model assumes, "
        "and 1 if every group either all succeeded or all failed; it also "
        "absorbs any change in skill within the window, and any difference "
        "between the seeds, since both make some groups easier than others. It "
        f"is a dash when p is exactly 0 or 1, or below {MIN_GROUPS} groups; p is "
        "rounded, so a p written as 1.000 can still have a rho. The last "
        f"two columns are the share of groups in which all {size} answers "
        f"failed, observed and as the binomial model gives it, (1 - p)^{size}.\n",
        markdown_table(
            [
                "method",
                "level",
                "groups",
                "p",
                "rho",
                f"all {size} fail",
                f"(1 - p)^{size}",
            ],
            [
                [
                    method,
                    tier,
                    cell["groups"],
                    _fixed(cell["p"], 3),
                    _fixed(cell["rho"], 2),
                    _fixed(cell["all_fail"], 3),
                    _fixed(cell["binomial_all_fail"], 3),
                ]
                for method in methods
                for tier in tiers
                if (cell := correlation.get((method, tier))) is not None
            ],
        ),
    ]
    counts = graded_rewards(runs)
    graded = {name: counts[name] for name in [*methods, EVERY_METHOD]}
    parts += [
        "\n## How graded the rewards are\n",
        "Every training answer of the runs. A reward strictly between 0 and 1 is "
        "partial credit. A group can have signal without a success when its "
        "rewards differ below the threshold; the binomial model counts only "
        "successes, so it does not see such a group. The last column counts "
        "groups with successes and failures but no signal; it is 0 unless the "
        "rewards of such a group are so close that their population standard "
        "deviation is below 1e-6.\n",
        markdown_table(
            [
                "method",
                "runs",
                "training answers",
                "strictly between 0 and 1",
                "share",
                "groups",
                "with signal",
                "with signal, no success",
                "share of groups",
                "share of groups with signal",
                "successes and failures, no signal",
            ],
            [
                [
                    name,
                    c["runs"],
                    c["rollouts"],
                    c["partial"],
                    _share(c["partial"], c["rollouts"]),
                    c["groups"],
                    c["signal"],
                    c["signal_without_success"],
                    _share(c["signal_without_success"], c["groups"]),
                    _share(c["signal_without_success"], c["signal"]),
                    c["mixed_without_signal"],
                ]
                for name, c in graded.items()
            ],
        ),
        "\nThe greedy answers to the test questions, at every evaluation of the "
        "runs (`evals.jsonl`).\n",
        markdown_table(
            ["method", "evaluation answers", "strictly between 0 and 1", "share"],
            [
                [
                    name,
                    c["answers"],
                    c["partial_answers"],
                    _share(c["partial_answers"], c["answers"]),
                ]
                for name, c in graded.items()
            ],
        ),
    ]
    return "\n".join(parts)


__all__ = [
    "calibration_text",
    "EVERY_METHOD",
    "graded_rewards",
    "group_correlation",
    "MIN_GROUPS",
    "PRIOR_P_HAT",
    "signal_calibration",
    "signal_chance",
    "SUCCESS_THRESHOLD",
    "WINDOWS",
]
