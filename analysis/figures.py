# SPDX-License-Identifier: BSD-3-Clause

"""
Compare the methods of the trial and the cost-aware runs, as the expectations
define the comparison, and draw their curves.

  python -m analysis.figures <runs folder> <figures folder>

It reads the complete runs of `trial/`, `cost-full/` and `seed3/` under the
runs folder, prints the tables as Markdown and writes three figures.
`python -m analysis.report` calls the same functions with the pilot's two
runs as seed 0.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from analysis.tables import load_pilot_runs, load_runs, markdown_table

EXPERIMENTS = ("trial", "cost-full", "seed3")
TOKEN_MARKS = (4_000_000, 4_250_000, 4_500_000, 4_750_000, 5_000_000)
LAST_EPISODE_MARK = 700
QUESTION_SEEDS = (1, 2)
SEEDS = (1, 2, 3)
METHODS = {
    "uniform": ("uniform", "#2a78d6"),
    "curriculum": ("curriculum", "#eb6834"),
    "curriculum-tokens": ("cost-aware, RFC formula", "#1baf7a"),
    "curriculum-tokens-mix-decay": ("cost-aware, mix-decay", "#eda100"),
}
FIGURES = (
    "trial-success-per-token.png",
    "trial-success-per-episode.png",
    "trial-success-levels-2-4.png",
)
MEASURES = ("per_token", "per_token_2_4", "per_episode", "per_episode_2_4")


def levels_2_to_4(evaluation: dict[str, Any]) -> float:
    """
    Return the success of one evaluation on levels 2 to 4 only.

    Args:
        evaluation (`dict[str, Any]`):
            An `eval` event of a log.

    Returns:
        `float`: The mean of the success rates of levels 2, 3 and 4. They have
        the same number of questions, so this is their success rate together.
    """
    return sum(evaluation["by_tier"][tier] for tier in ("2", "3", "4")) / 3


def _interpolate(xs: list[float], ys: list[float], x: float) -> float:
    for x0, y0, x1, y1 in zip(xs, ys, xs[1:], ys[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    raise ValueError(f"{x} is outside the evaluations, {xs[0]} to {xs[-1]}")


def per_token(
    evals: list[dict[str, Any]],
    key: Callable[[dict[str, Any]], float] = lambda e: e["success_rate"],
) -> float:
    """
    Return the success per generated token of one run.

    Args:
        evals (`list[dict[str, Any]]`):
            The run's `eval` events, in order.
        key (`Callable`, *optional*):
            What to read from an evaluation; its success rate when left out.

    Returns:
        `float`: The mean at 4.00, 4.25, 4.50, 4.75 and 5.00 million generated
        tokens. An evaluation falls on the first step past a mark, so each
        mark is read by a straight line between the two nearest evaluations.

    Raises:
        `ValueError`: If a mark is outside the run's evaluations.
    """
    xs = [e["gen_tokens_total"] for e in evals]
    ys = [key(e) for e in evals]
    return sum(_interpolate(xs, ys, mark) for mark in TOKEN_MARKS) / len(TOKEN_MARKS)


def per_episode(
    evals: list[dict[str, Any]],
    last_mark: int,
    key: Callable[[dict[str, Any]], float] = lambda e: e["success_rate"],
) -> float:
    """
    Return the success per episode of one run.

    Args:
        evals (`list[dict[str, Any]]`):
            The run's `eval` events, in order.
        last_mark (`int`):
            The last step read: 700, or less when a run of the same seed
            stopped earlier.
        key (`Callable`, *optional*):
            What to read from an evaluation; its success rate when left out.

    Returns:
        `float`: The mean at steps 100, 200, ... `last_mark`, each read by a
        straight line between the two nearest evaluations.

    Raises:
        `ValueError`: If `last_mark` is below 100, or a mark is outside the
            run's evaluations.
    """
    marks = range(100, last_mark + 1, 100)
    if not marks:
        raise ValueError(f"no step mark up to {last_mark}: the window starts at 100")
    return sum(episode_marks(evals, last_mark, key).values()) / len(marks)


def episode_marks(
    evals: list[dict[str, Any]],
    last_mark: int,
    key: Callable[[dict[str, Any]], float] = lambda e: e["success_rate"],
) -> dict[int, float]:
    """
    Return the success of one run at steps 100, 200, ... `last_mark`.

    Args:
        evals (`list[dict[str, Any]]`):
            The run's `eval` events, in order.
        last_mark (`int`):
            The last step read.
        key (`Callable`, *optional*):
            What to read from an evaluation; its success rate when left out.

    Returns:
        `dict[int, float]`: Per step mark, the value read by a straight line
        between the two nearest evaluations.

    Raises:
        `ValueError`: If a mark is outside the run's evaluations.
    """
    xs = [e["step"] for e in evals]
    ys = [key(e) for e in evals]
    return {mark: _interpolate(xs, ys, mark) for mark in range(100, last_mark + 1, 100)}


def trial_runs(
    runs_folder: str | Path, pilot: str | Path | None = None
) -> list[dict[str, Any]]:
    """
    Read the complete runs of the trial and of the cost-aware runs.

    Args:
        runs_folder (`str` or `Path`):
            The folder that holds `trial/`, `cost-full/` and `seed3/`.
        pilot (`str` or `Path`, *optional*):
            The pilot's folder. Its two runs are then added, as seed 0.

    Returns:
        `list[dict[str, Any]]`: The runs as `load_runs` reads them, sorted by
        seed, then in the order of `METHODS`.
    """
    runs = [run for name in EXPERIMENTS for run in load_runs(Path(runs_folder) / name)]
    if pilot is not None:
        runs += load_pilot_runs(pilot)
    order = list(METHODS)
    return sorted(
        runs,
        key=lambda r: (r["config"]["seed"], order.index(r["config"]["sampler"])),
    )


def comparison(
    runs_folder: str | Path,
    pilot: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[int, float]]]]:
    """
    Measure every run, and each method against uniform within each seed.

    Args:
        runs_folder (`str` or `Path`):
            The folder that holds `trial/`, `cost-full/` and `seed3/`.
        pilot (`str` or `Path`, *optional*):
            The pilot's folder. Its two runs are then read as seed 0.

    Returns:
        `tuple[list[dict], dict]`: One row per run: `method`, `seed`,
        `last_step`, `episode_window` and the four measures (`per_token`,
        `per_token_2_4`, `per_episode`, `per_episode_2_4`), as fractions; and
        `own_window` with `episode_marks` and `episode_marks_2_4`, the success
        at every step mark the run itself reached, up to 700. And, per method
        other than uniform and per measure, the difference to uniform for each
        seed that has both runs, read in the window of the whole seed.
    """
    runs = trial_runs(runs_folder, pilot)
    last_step: dict[int, int] = {}
    for run in runs:
        seed, step = run["config"]["seed"], run["steps"][-1]["step"]
        last_step[seed] = min(last_step.get(seed, step), step)
    rows = []
    for run in runs:
        seed = run["config"]["seed"]
        evals = [run["evals"][step] for step in sorted(run["evals"])]
        window = min(LAST_EPISODE_MARK, last_step[seed] // 100 * 100)
        own = min(LAST_EPISODE_MARK, run["steps"][-1]["step"] // 100 * 100)
        rows.append(
            {
                "method": run["config"]["sampler"],
                "seed": seed,
                "last_step": run["steps"][-1]["step"],
                "episode_window": window,
                "per_token": per_token(evals),
                "per_token_2_4": per_token(evals, levels_2_to_4),
                "per_episode": per_episode(evals, window),
                "per_episode_2_4": per_episode(evals, window, levels_2_to_4),
                "own_window": own,
                "episode_marks": episode_marks(evals, own),
                "episode_marks_2_4": episode_marks(evals, own, levels_2_to_4),
            }
        )
    uniform = {r["seed"]: r for r in rows if r["method"] == "uniform"}
    differences: dict[str, dict[str, dict[int, float]]] = {}
    for row in rows:
        if row["method"] == "uniform" or row["seed"] not in uniform:
            continue
        by_measure = differences.setdefault(row["method"], {m: {} for m in MEASURES})
        for measure in MEASURES:
            by_measure[measure][row["seed"]] = (
                row[measure] - uniform[row["seed"]][measure]
            )
    return rows, differences


def _figure(
    runs: list[dict[str, Any]],
    path: Path,
    title: str,
    x: Callable[[dict[str, Any], dict[str, Any]], float],
    xlabel: str,
    y: Callable[[dict[str, Any]], float],
    ylabel: str,
    shade: tuple[float, float] | None = None,
) -> None:
    seeds = sorted({run["config"]["seed"] for run in runs})
    fig, axes = plt.subplots(
        1, len(seeds), figsize=(5 * len(seeds), 4.6), sharey=True, squeeze=False
    )
    for ax, seed in zip(axes[0], seeds):
        for run in runs:
            if run["config"]["seed"] != seed:
                continue
            label, color = METHODS[run["config"]["sampler"]]
            evals = [run["evals"][step] for step in sorted(run["evals"])]
            ax.plot(
                [x(run, e) for e in evals],
                [y(e) for e in evals],
                color=color,
                linewidth=2,
                label=label,
            )
        if shade:
            ax.axvspan(*shade, color="#f1f0ec", zorder=0)
        ax.set_title(f"seed {seed}", loc="left", fontsize=11, color="#0b0b0b")
        ax.set_xlabel(xlabel, fontsize=9, color="#52514e")
        ax.grid(True, color="#e6e5e0", linewidth=0.8)
        ax.tick_params(colors="#52514e", labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#e6e5e0")
    axes[0][0].set_ylabel(ylabel, fontsize=9, color="#52514e")
    # One legend for every method drawn, whichever panels they appear in.
    lines = {}
    for ax in axes[0]:
        for line in ax.get_lines():
            lines.setdefault(line.get_label(), line)
    axes[0][0].legend(list(lines.values()), list(lines), frameon=False, fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color="#0b0b0b")
    fig.tight_layout()
    temporary = path.with_name(path.name + ".tmp")
    fig.savefig(temporary, format="png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    os.replace(temporary, path)


def plot(
    runs_folder: str | Path, out: str | Path, pilot: str | Path | None = None
) -> None:
    """
    Draw the three figures, one panel per seed and one line per method.

    Args:
        runs_folder (`str` or `Path`):
            The folder that holds `trial/`, `cost-full/` and `seed3/`.
        out (`str` or `Path`):
            The folder the figures are written to; created if missing.
        pilot (`str` or `Path`, *optional*):
            The pilot's folder. Its two runs are then drawn as seed 0.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    runs = trial_runs(runs_folder, pilot)

    def tokens(run: dict[str, Any], e: dict[str, Any]) -> float:
        return e["gen_tokens_total"] / 1e6

    def episodes(run: dict[str, Any], e: dict[str, Any]) -> float:
        config = run["config"]
        return e["step"] * config["prompts_per_step"] * config["group_size"] / 1000

    token_axis = "generated tokens (millions)"
    window = (TOKEN_MARKS[0] / 1e6, TOKEN_MARKS[-1] / 1e6)
    _figure(
        runs,
        out / FIGURES[0],
        "Test success against generated tokens (500 questions; shaded: 4.0 to 5.0M)",
        tokens,
        token_axis,
        lambda e: 100 * e["success_rate"],
        "success (%)",
        window,
    )
    _figure(
        runs,
        out / FIGURES[1],
        "Test success against training episodes (500 questions)",
        episodes,
        "episodes (thousands)",
        lambda e: 100 * e["success_rate"],
        "success (%)",
    )
    _figure(
        runs,
        out / FIGURES[2],
        "Test success on levels 2 to 4 only (300 questions)",
        tokens,
        token_axis,
        lambda e: 100 * levels_2_to_4(e),
        "success (%)",
        window,
    )


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _print(runs_folder: str | Path) -> None:
    rows, differences = comparison(runs_folder)
    print("Test success of each run, in points\n")
    print(
        markdown_table(
            [
                "method",
                "seed",
                "steps",
                "per token",
                "per token, levels 2-4",
                "episode window",
                "per episode",
                "per episode, levels 2-4",
            ],
            [
                [
                    r["method"],
                    r["seed"],
                    r["last_step"],
                    100 * r["per_token"],
                    100 * r["per_token_2_4"],
                    f"100-{r['episode_window']}",
                    100 * r["per_episode"],
                    100 * r["per_episode_2_4"],
                ]
                for r in rows
            ],
        )
    )
    names = {
        "per_token": "per token",
        "per_token_2_4": "per token, levels 2-4",
        "per_episode": "per episode",
        "per_episode_2_4": "per episode, levels 2-4",
    }
    for measure, name in names.items():
        print(f"\nMethod minus uniform, {name}, in points\n")
        body = []
        for method, by_measure in differences.items():
            values = by_measure[measure]
            question = [values[s] for s in QUESTION_SEEDS if s in values]
            body.append(
                [
                    method,
                    *[100 * values[s] if s in values else None for s in SEEDS],
                    (
                        100 * _mean(question)
                        if len(question) == len(QUESTION_SEEDS)
                        else None
                    ),
                    100 * _mean(list(values.values())),
                ]
            )
        print(
            markdown_table(
                [
                    "method",
                    *[f"seed {s}" for s in SEEDS],
                    "mean, seeds 1 and 2",
                    "mean, every seed",
                ],
                body,
            )
        )


def main(argv: list[str] | None = None) -> int:
    """
    Print the tables and write the figures.

    Args:
        argv (`list[str]`, *optional*):
            The arguments; `sys.argv[1:]` when left out.

    Returns:
        `int`: 0.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("runs_folder")
    parser.add_argument("figures_folder")
    args = parser.parse_args(argv)
    _print(args.runs_folder)
    plot(args.runs_folder, args.figures_folder)
    return 0


__all__ = [
    "comparison",
    "episode_marks",
    "levels_2_to_4",
    "main",
    "per_episode",
    "per_token",
    "plot",
    "trial_runs",
]


if __name__ == "__main__":
    sys.exit(main())
