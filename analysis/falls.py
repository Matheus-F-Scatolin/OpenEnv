# SPDX-License-Identifier: BSD-3-Clause

"""Describe falls of test success between consecutive reference evaluations.

This reading was added after the reference results had been read. It is not part
of the pre-registered analysis and changes none of its numbers. `python -m
analysis.reference` writes it as `falls.json` and `falls.md`.
"""

from __future__ import annotations

import gzip
import json
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from statistics import fmean
from typing import Any

from analysis.calibration import SUCCESS_THRESHOLD
from analysis.figures import TOKEN_MARKS
from analysis.reference import (
    DEGENERATE_SHARE,
    DEGENERATE_WINDOW,
    measure_at_tokens,
    METHODS,
)
from analysis.tables import markdown_table
from trial.runlog import write_json_atomic

FALL_POINTS = 5.0
LARGE_FALL_POINTS = 10.0
Row = dict[str, Any]

# One line, no decimals: `a + b - c = d`, the result possibly written `= - d`.
_SUM = re.compile(
    r"(?<![\d,])(?<!\d\.)(-?\d[\d,]*(?:[ \t]*[+-][ \t]*-?\d[\d,]*)+)"
    r"[ \t]*=[ \t]*(-?)[ \t]*(\d[\d,]*)(?!\d|\.\d)"
)
_TERM = re.compile(r"([+-]?)[ \t]*(-?\d[\d,]*)")


def wrong_sum(text: str) -> bool:
    """
    Tell whether an answer writes a sum whose result is wrong.

    Args:
        text (`str`):
            One answer. Each `a + b - c = d` written on one line is checked, up to
            its first `=`; numbers with a decimal point are skipped, and commas
            inside numbers are ignored, as the model places them oddly.

    Returns:
        `bool`: Whether any written result differs from its terms.
    """
    for left, negative, right in _SUM.findall(text):
        value = sum(
            (-1 if sign == "-" else 1) * int(number.replace(",", ""))
            for sign, number in _TERM.findall(left)
        )
        if value != (-1 if negative else 1) * int(right.replace(",", "")):
            return True
    return False


def worst_cut_share(steps: list[Row]) -> tuple[float, int] | None:
    """
    Find the window of the degenerate-run rule with the most training answers cut.

    Args:
        steps (`list[dict]`):
            Ordered steps with each group's per-answer `truncated` flags, read
            over `DEGENERATE_WINDOW` steps as `analysis.reference.degenerate` does.

    Returns:
        `tuple[float, int]` or `None`: The share and the first step of the
        earliest worst window; None with fewer steps than one window.
    """
    counts = [
        (sum(flags), len(flags))
        for flags in (
            [flag for group in step["groups"] for flag in group["truncated"]]
            for step in steps
        )
    ]
    best: tuple[float, int] | None = None
    for start in range(len(counts) - DEGENERATE_WINDOW + 1):
        window = counts[start : start + DEGENERATE_WINDOW]
        answers = sum(n for _, n in window)
        share = sum(c for c, _ in window) / answers if answers else 0.0
        if best is None or share > best[0]:
            best = (share, steps[start]["step"])
    return best


def _events(folder: Path) -> list[Row]:
    return [
        json.loads(line) for line in (folder / "log.jsonl").read_text().splitlines()
    ]


def _records(folder: Path, levels: set[int]) -> dict[int, dict[str, Row]]:
    by_step: dict[int, dict[str, Row]] = {}
    for line in (folder / "evals.jsonl").read_text().splitlines():
        record = json.loads(line)
        if int(record["tier"]) in levels:
            by_step.setdefault(record["step"], {})[record["task"]] = record
    return by_step


def _texts(folder: Path, wanted: set[int]) -> dict[tuple[int, str], str] | None:
    path = folder / "eval_texts.jsonl.gz"
    if not path.exists():
        return None
    texts = {}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            entry = json.loads(line)
            if int(entry["step"]) in wanted:
                texts[(int(entry["step"]), entry["task"])] = entry["text"]
    return texts


def _count_wrong(
    texts: dict[tuple[int, str], str] | None, step: int, tasks: list[str]
) -> int | None:
    """Count answers with a wrong sum; None when any text is unavailable."""
    if texts is None or any((step, t) not in texts for t in tasks):
        return None
    return sum(wrong_sum(texts[(step, t)]) for t in tasks)


def _solved(records: dict[str, Row]) -> set[str]:
    return {t for t, r in records.items() if r["reward"] >= SUCCESS_THRESHOLD}


def _marks(run: Row, steps: list[Row], step: int, window: int | None) -> Row:
    """Name the pre-registered marks that read the evaluation at `step`."""
    evaluations = run["evaluations"]
    at = {e["step"] for e in evaluations}
    token_marks = []
    for mark in TOKEN_MARKS:
        crossing = next(
            (s["step"] for s in steps if s["gen_tokens_total"] >= mark), None
        )
        if crossing is None:
            continue
        if crossing == step:
            token_marks.append(dict(mark=mark, read="at this evaluation"))
        elif crossing not in at:
            for left, right in zip(evaluations, evaluations[1:]):
                if left["tokens"] <= mark <= right["tokens"] and (
                    right["tokens"] > left["tokens"]
                ):
                    if step in (left["step"], right["step"]):
                        token_marks.append(dict(mark=mark, read="interpolated"))
                    break
    episode_marks = [m for m in range(100, (window or 0) + 1, 100) if m == step]
    return dict(token_marks=token_marks, episode_marks=episode_marks)


def _measures(
    run: Row, steps: list[Row], window: int | None, values: dict[int, float | None]
) -> tuple[float | None, float | None]:
    """Recompute the run's two main measures from main-level success by step."""
    points = [
        dict(step=e["step"], tokens=e["tokens"], value=values[e["step"]])
        for e in run["evaluations"]
    ]
    marks = [measure_at_tokens(points, steps, mark) for mark in TOKEN_MARKS]
    per_token = None if any(v is None for v in marks) else fmean(marks)
    episode = [values.get(s) for s in range(100, (window or 0) + 1, 100)]
    per_episode = (
        None if not episode or any(v is None for v in episode) else fmean(episode)
    )
    return per_token, per_episode


def build_falls(report: Row) -> Row:
    """
    Describe every fall of 5 points or more and a what-if without the large ones.

    Args:
        report (`dict`):
            The result of `analysis.reference.build_report`, whose run paths are
            read again for evaluation records, answer texts and training steps.

    Returns:
        `dict`: The falls, the number of consecutive pairs compared per method,
        each run's worst window of cut training answers, the runs or texts that
        could not be read (left out, with the reason), and a what-if that
        replaces each fall of 10 points or more by the mean of its neighbouring
        evaluations, with no verdict unless a complete pair exists.

    Raises:
        `ValueError`: If the main measures recomputed here differ from the report.
    """
    levels = set(report["main_levels"])
    windows = {int(k): v for k, v in report["episode_windows"].items()}
    intervals = dict.fromkeys(METHODS, 0)
    falls, worst_windows, unavailable, measures = [], [], [], {}
    for run in report["runs"]:
        folder = Path(run["path"])
        who = dict(method=run["method"], seed=run["seed"])
        try:
            steps = [e for e in _events(folder) if e["event"] == "step"]
            worst = worst_cut_share(steps)
            cut_by_step = {
                s["step"]: sum(
                    flag for group in s["groups"] for flag in group["truncated"]
                )
                for s in steps
            }
            records = _records(folder, levels)
        except (KeyError, TypeError, ValueError, OSError) as error:
            reason = f"training steps or evaluation records not readable: {error!r}"
            unavailable.append(dict(who, reason=reason))
            continue
        window = windows.get(run["seed"])
        valid = [
            e for e in run["evaluations"] if e["scopes"]["main"]["success"] is not None
        ]
        intervals[run["method"]] = intervals.get(run["method"], 0) + max(
            0, len(valid) - 1
        )
        worst_windows.append(
            dict(
                who,
                share=worst[0] if worst else None,
                first_step=worst[1] if worst else None,
            )
        )
        values = {e["step"]: e["scopes"]["main"]["success"] for e in run["evaluations"]}
        recomputed = _measures(run, steps, window, values)
        for got, name in zip(recomputed, ("per_token", "per_episode")):
            expected = run[name]["main"]
            if expected is not None and (got is None or abs(got - expected) > 1e-9):
                raise ValueError(f"the what-if does not reproduce {name} of {folder}")
        measures[(run["seed"], run["method"])] = dict(
            run=run, steps=steps, window=window, values=values
        )
        pairs = [
            (a, b, valid[i + 2] if i + 2 < len(valid) else None)
            for i, (a, b) in enumerate(zip(valid, valid[1:]))
        ]
        found = [
            (a, b, c)
            for a, b, c in pairs
            if 100 * (a["scopes"]["main"]["success"] - b["scopes"]["main"]["success"])
            >= FALL_POINTS - 1e-9
        ]
        if not found:
            continue
        try:
            texts = _texts(folder, {e["step"] for a, b, _ in found for e in (a, b)})
        except (OSError, EOFError, ValueError) as error:
            texts = None
            reason = f"answer texts not readable: {error!r}"
            unavailable.append(dict(who, reason=reason))
        for a, b, c in found:
            before, at = records[a["step"]], records[b["step"]]
            lost = sorted(_solved(before) - _solved(at))
            gained = _solved(at) - _solved(before)
            unsolved = sorted(set(before) - _solved(before))
            later = [e for e in valid if e["step"] > b["step"]]
            recovered = next(
                (
                    e["step"]
                    for e in later
                    if e["scopes"]["main"]["success"]
                    >= a["scopes"]["main"]["success"] - 1e-12
                ),
                None,
            )
            cut = [
                n for step, n in cut_by_step.items() if a["step"] < step <= b["step"]
            ]
            falls.append(
                dict(
                    who,
                    step_before=a["step"],
                    step=b["step"],
                    tokens=b["tokens"],
                    questions=len(at),
                    solved_before=len(_solved(before)),
                    solved_at=len(_solved(at)),
                    success_before=a["scopes"]["main"]["success"],
                    success_at=b["scopes"]["main"]["success"],
                    drop_points=100
                    * (a["scopes"]["main"]["success"] - b["scopes"]["main"]["success"]),
                    lost=len(lost),
                    gained=len(gained),
                    lost_cut=sum(bool(at[t]["truncated"]) for t in lost),
                    lost_wrong_sum=_count_wrong(texts, b["step"], lost),
                    unsolved_before=len(unsolved),
                    unsolved_before_wrong_sum=_count_wrong(texts, a["step"], unsolved),
                    next_step=c["step"] if c else None,
                    solved_next=len(_solved(records[c["step"]])) if c else None,
                    success_next=c["scopes"]["main"]["success"] if c else None,
                    recovered_step=recovered,
                    training_cut=sum(cut),
                    training_steps_with_cut=sum(1 for n in cut if n),
                    **_marks(run, steps, b["step"], window),
                )
            )
    replaced, kept = [], []
    for fall in falls:
        if fall["drop_points"] < LARGE_FALL_POINTS - 1e-9:
            continue
        entry = dict(
            method=fall["method"],
            seed=fall["seed"],
            step=fall["step"],
            success=fall["success_at"],
        )
        if fall["success_next"] is None:
            kept.append(entry)
            continue
        replacement = (fall["success_before"] + fall["success_next"]) / 2
        measures[(fall["seed"], fall["method"])]["values"][fall["step"]] = replacement
        replaced.append(dict(entry, replacement=replacement))
    what_if: Row = dict(replaced=replaced, not_replaced=kept)
    seeds = sorted({seed for seed, _ in measures})
    for index, name in enumerate(("per_token", "per_episode")):
        differences = {}
        for seed in seeds:
            pair = [measures.get((seed, m)) for m in METHODS]
            if any(p is None for p in pair):
                continue
            values = [
                _measures(p["run"], p["steps"], p["window"], p["values"])[index]
                for p in pair
            ]
            if all(v is not None for v in values):
                differences[seed] = 100 * (values[1] - values[0])
        what_if[name] = dict(
            differences=differences,
            mean=fmean(differences.values()) if differences else None,
        )
    compared = [set(what_if[n]["differences"]) for n in ("per_token", "per_episode")]
    what_if["seeds_compared"] = sorted(compared[0] & compared[1])
    # No verdict unless both measures cover the same seeds and at least one pair.
    what_if["ahead_on_every_seed"] = (
        all(
            d > 0
            for name in ("per_token", "per_episode")
            for d in what_if[name]["differences"].values()
        )
        if compared[0] and compared[0] == compared[1]
        else None
    )
    return dict(
        fall_points=FALL_POINTS,
        large_fall_points=LARGE_FALL_POINTS,
        main_levels=sorted(levels),
        intervals=intervals,
        falls=falls,
        worst_windows=worst_windows,
        unavailable=unavailable,
        what_if=what_if,
    )


def _half_up(value: float, places: int = 1) -> str:
    """Round half up after removing binary noise, so 15.15 shows as 15.2."""
    exact = Decimal(repr(round(value, 10)))
    return str(exact.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def _percent(solved: int, questions: int) -> str:
    return _half_up(100 * solved / questions) + "%"


def _text(result: Row) -> str:
    """Render the falls, the worst windows and the what-if as Markdown."""
    lines = [
        "# Falls between consecutive evaluations",
        "",
        "Generated by `python -m analysis.reference`. This reading was added after the "
        "results had been read: it is not part of the pre-registered analysis, its "
        f"thresholds ({result['fall_points']:g} and {result['large_fall_points']:g} "
        "points) were chosen then, and it changes no number of the reference report.",
        "",
        f"Main levels: {result['main_levels']}. Consecutive evaluations compared: "
        + ", ".join(f"{m} {n}" for m, n in result["intervals"].items())
        + ". Success is solved questions over questions, rounded half up. "
        "A lost question was solved at the earlier evaluation and not at the fall; "
        "a wrong sum is a written `a + b = c` whose result is wrong. Nearly every "
        "unsolved answer writes one, so the baseline column gives the same count "
        "for the questions unsolved at the earlier evaluation.",
        "",
        "## Falls of 5 points or more",
        "",
    ]
    rows = []
    for f in result["falls"]:
        marks = [f"per episode, step {m}" for m in f["episode_marks"]] + [
            f"per token, {m['mark'] / 1e6:.2f} M ({m['read']})"
            for m in f["token_marks"]
        ]
        nxt = (
            f"step {f['next_step']}: {_percent(f['solved_next'], f['questions'])}"
            if f["next_step"] is not None
            else "none (last evaluation)"
        )
        rows.append(
            [
                f["method"],
                f["seed"],
                f"{f['step_before']} → {f['step']}",
                f"{f['tokens'] / 1e6:.2f} M",
                f"{_percent(f['solved_before'], f['questions'])} → "
                f"{_percent(f['solved_at'], f['questions'])}",
                f["lost"],
                f["gained"],
                f["lost_cut"],
                f["lost_wrong_sum"],
                None
                if f["unsolved_before_wrong_sum"] is None
                else f"{f['unsolved_before_wrong_sum']} of {f['unsolved_before']}",
                nxt,
                f["recovered_step"],
                "; ".join(marks) or "none",
                f"{f['training_cut']} in {f['training_steps_with_cut']} steps",
            ]
        )
    lines.append(
        markdown_table(
            [
                "Method",
                "Seed",
                "Steps",
                "Tokens at the fall",
                "Main-level success",
                "Lost",
                "Gained",
                "Lost and cut",
                "Lost with a wrong sum",
                "Baseline: unsolved before, with a wrong sum",
                "Next evaluation",
                "First step back at the earlier level",
                "Pre-registered marks reading it",
                "Cut training answers since the earlier evaluation",
            ],
            rows,
        )
    )
    lines += [
        "",
        f"## Worst {DEGENERATE_WINDOW}-step window of cut training answers",
        "",
        f"A run is degenerate when this share reaches {100 * DEGENERATE_SHARE:g}% "
        "(pre-registered rule).",
        "",
        markdown_table(
            ["Method", "Seed", "Share", "First step"],
            [
                [
                    w["method"],
                    w["seed"],
                    None if w["share"] is None else _half_up(100 * w["share"]) + "%",
                    w["first_step"],
                ]
                for w in result["worst_windows"]
            ],
        ),
        "",
        "## What-if, descriptive only",
        "",
        "Each fall of 10 points or more, in either method, is replaced by the mean of "
        "its two neighbouring evaluations; a fall at a run's last evaluation is not "
        "replaced. No test is computed. Curriculum minus uniform, in points:",
        "",
    ]
    what_if = result["what_if"]
    seeds = sorted(
        set(what_if["per_token"]["differences"])
        | set(what_if["per_episode"]["differences"])
    )
    lines.append(
        markdown_table(
            ["Measure", *[f"seed {s}" for s in seeds], "Mean"],
            [
                [
                    name,
                    *[
                        _half_up(what_if[name]["differences"][s])
                        if s in what_if[name]["differences"]
                        else None
                        for s in seeds
                    ],
                    None
                    if what_if[name]["mean"] is None
                    else _half_up(what_if[name]["mean"]),
                ]
                for name in ("per_token", "per_episode")
            ],
        )
    )
    lines += [
        "",
        "Replaced: "
        + (
            "; ".join(
                f"{r['method']} seed {r['seed']} step {r['step']} "
                f"({_half_up(100 * r['success'])}% → {_half_up(100 * r['replacement'])}%)"
                for r in what_if["replaced"]
            )
            or "none"
        )
        + ". Not replaced (last evaluation): "
        + (
            "; ".join(
                f"{r['method']} seed {r['seed']} step {r['step']}"
                for r in what_if["not_replaced"]
            )
            or "none"
        )
        + ". "
        + (
            "Curriculum ahead in both measures on every seed compared "
            f"({', '.join(map(str, what_if['seeds_compared']))}): "
            f"{what_if['ahead_on_every_seed']}."
            if what_if["ahead_on_every_seed"] is not None
            else "Not assessed: no seed has both measures for both methods."
        ),
    ]
    if result["unavailable"]:
        lines += [
            "",
            "## Left out",
            "",
            *[
                f"- {u['method']}, seed {u['seed']}: {u['reason']}"
                for u in result["unavailable"]
            ],
        ]
    return "\n".join(lines) + "\n"


def write_falls(result: Row, out: str | Path) -> list[Path]:
    """
    Atomically write the falls as full-precision JSON and a Markdown table.

    Args:
        result (`dict`):
            The result of `build_falls`.
        out (`str` or `Path`):
            Output folder; `analysis.reference.write_report` has checked it.

    Returns:
        `list[Path]`: `falls.json` and `falls.md`.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = [out / "falls.json", out / "falls.md"]
    write_json_atomic(files[0], result)
    temporary = files[1].with_suffix(".md.tmp")
    temporary.write_text(_text(result))
    temporary.replace(files[1])
    return files


__all__ = ["build_falls", "worst_cut_share", "write_falls", "wrong_sum"]
