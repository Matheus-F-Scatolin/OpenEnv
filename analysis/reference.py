# SPDX-License-Identifier: BSD-3-Clause

"""Read reference logs under the pre-registered rules, retaining failed checks.

Run `python -m analysis.reference --runs PATH --out PATH` to regenerate the
JSON, Markdown and two figures of the pre-registered reading, and `falls.json`
and `falls.md`, a reading added after the results (`analysis.falls`). No model,
network or new dependency is needed.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from analysis.calibration import signal_calibration, SUCCESS_THRESHOLD
from analysis.figures import LAST_EPISODE_MARK, TOKEN_MARKS
from analysis.report import BUDGET
from analysis.tables import markdown_table
from analysis.validate_run import check_run, find_run_dirs
from trial.runlog import write_json_atomic

SEEDS = (5, 6, 7, 8, 9)
METHODS = ("uniform", "curriculum")
T_CRITICAL = 2.776445
MAIN_LEVELS = {"rfc": tuple(range(2, 10)), "staircase": tuple(range(3, 10))}
DEGENERATE_WINDOW = 50
DEGENERATE_SHARE = 0.9
Row = dict[str, Any]


def paired_summary(differences: dict[int, float], *, inference: bool = True) -> Row:
    """
    Summarize curriculum minus uniform, already expressed in percentage points.

    Args:
        differences (`dict[int, float]`):
            One paired difference per available reference seed.
        inference (`bool`, *optional*, defaults to `True`):
            Allow inference with all five seeds; disabled for the sensitivity
            reading without degenerate seeds.

    Returns:
        `dict`: All values, mean, range and, only for five pairs, the two-sided
        df=4 t-test, interval, verdict and the separate -2-point margin reading.
        Constant differences use the limiting t and p (0 and 1 for all zeros).
    """
    if not set(differences) <= set(SEEDS):
        raise ValueError("only reference seeds 5 to 9 are allowed")
    values = list(differences.values())
    mean = fmean(values) if values else None
    result = dict(
        n=len(values),
        differences=differences,
        mean=mean,
        range=[min(values), max(values)] if values else None,
        t=None,
        p=None,
        interval=None,
        verdict="no test: fewer than five pairs",
        support="not assessed",
    )
    if not inference or len(values) != len(SEEDS):
        if not inference:
            result["verdict"] = "no test: descriptive reading"
        return result
    error = stdev(values) / math.sqrt(len(values))
    if error:
        t: float | str = mean / error
        x = abs(t) / math.hypot(2, t)
        p = max(0.0, min(1.0, 1 - 1.5 * x + 0.5 * x**3))
    else:
        t, p = (0, 1.0) if mean == 0 else ("inf" if mean > 0 else "-inf", 0.0)
    interval = [mean - T_CRITICAL * error, mean + T_CRITICAL * error]
    verdict = (
        ("better" if mean > 0 else "worse")
        if p <= 0.05
        else "no significant difference with five seeds"
    )
    support = "not supported"
    if mean >= -2 and verdict != "worse":
        support = "shown by the interval" if interval[0] >= -2 else "on the mean only"
    result.update(t=t, p=p, interval=interval, verdict=verdict, support=support)
    return result


def measure_at_tokens(
    evaluations: list[Row], steps: list[Row], mark: int
) -> float | None:
    """
    Read a token mark at its crossing step, otherwise interpolate evaluations.

    Args:
        evaluations (`list[dict]`):
            Ordered points with `step`, `tokens` and `value` (a fraction).
        steps (`list[dict]`):
            Training steps with `step` and `gen_tokens_total`.
        mark (`int`):
            Generated training tokens at which to read success.

    Returns:
        `float` or `None`: Success; None when no crossing or bracket exists.
        A missing value at a recorded crossing is not replaced by interpolation.
    """
    crossing = next((s["step"] for s in steps if s["gen_tokens_total"] >= mark), None)
    if crossing is None:
        return None
    for point in evaluations:
        if point["step"] == crossing:
            return point["value"]
    for left, right in zip(evaluations, evaluations[1:]):
        if (
            left["tokens"] <= mark <= right["tokens"]
            and right["tokens"] > left["tokens"]
        ):
            if left["value"] is None or right["value"] is None:
                return None
            fraction = (mark - left["tokens"]) / (right["tokens"] - left["tokens"])
            return left["value"] + fraction * (right["value"] - left["value"])
    return None


def degenerate(steps: list[Row]) -> bool:
    """
    Detect a 50-step window with at least 90% of all training answers cut.

    Args:
        steps (`list[dict]`):
            Ordered steps with each group's per-answer `truncated` flags.

    Returns:
        `bool`: Whether any such window exists, weighted by answers.
    """
    window: deque[tuple[int, int]] = deque()
    cut = answers = 0
    for step in steps:
        flags = [flag for group in step["groups"] for flag in group["truncated"]]
        window.append((sum(flags), len(flags)))
        cut += sum(flags)
        answers += len(flags)
        if len(window) > DEGENERATE_WINDOW:
            old_cut, old_answers = window.popleft()
            cut -= old_cut
            answers -= old_answers
        if (
            len(window) == DEGENERATE_WINDOW
            and answers
            and cut / answers >= DEGENERATE_SHARE
        ):
            return True
    return False


def _mean_or_none(values: list[float | None]) -> float | None:
    return fmean(values) if values and all(v is not None for v in values) else None


def _scopes(ladder: str) -> dict[str, tuple[int, ...]]:
    return {
        "main": MAIN_LEVELS[ladder],
        "all": tuple(range(10)),
        "excluded": tuple(t for t in range(10) if t not in MAIN_LEVELS[ladder]),
        **{f"level_{t}": (t,) for t in range(10)},
    }


def _diagnostics(config: Row, steps: list[Row], folder: Path) -> Row:
    """Count spend and signal; calibration uses estimates before each draw."""
    groups = [g for s in steps for g in s["groups"]]
    episodes = sum(len(g["rewards"]) for g in groups)
    tokens = sum(sum(g["tokens"]) for g in groups)
    signals = sum(s["groups_with_signal"] for s in steps)
    calibration = (
        signal_calibration([dict(config=config, steps=steps, dir=folder)])
        if steps
        else {}
    )
    levels = {}
    for tier in range(10):
        own = [g for g in groups if int(g["tier"]) == tier]
        count = sum(len(g["rewards"]) for g in own)
        cell = calibration.get((config["sampler"], "whole run", str(tier)), {})
        levels[str(tier)] = dict(
            episodes=count,
            episode_share=count / episodes if episodes else None,
            predicted_signal=cell.get("predicted"),
            observed_signal=cell.get("observed"),
        )
    return dict(
        episodes=episodes,
        generated_tokens=tokens,
        groups=len(groups),
        signal_groups=signals,
        signal_per_episode=signals / episodes if episodes else None,
        signal_per_million_tokens=signals * 1_000_000 / tokens if tokens else None,
        truncated_share=sum(sum(g["truncated"]) for g in groups) / episodes
        if episodes
        else None,
        tokens_per_answer=tokens / episodes if episodes else None,
        training_seconds=steps[-1]["train_elapsed_s"] if steps else 0,
        updates=sum(s["groups_with_signal"] > 0 for s in steps),
        levels=levels,
        calibration=[
            dict(method=m, window=w, level=t, **v)
            for (m, w, t), v in sorted(calibration.items())
        ],
    )


def _read_selected(folder: Path, events: list[Row], ladder: str) -> Row:
    """Retain validation problems alongside the first complete run's numbers."""
    config = events[0]
    issues: list[str] = []
    if config.get("token_budget") != BUDGET:
        issues.append(f"token budget differs from the pre-registered {BUDGET}")
    try:
        status, _ = check_run(folder)
    except (KeyError, TypeError, ValueError, OSError) as error:
        status = f"validator could not read the run: {error}"
    steps = [e for e in events if e["event"] == "step"]
    try:
        records = [
            json.loads(line)
            for line in (folder / "evals.jsonl").read_text().splitlines()
        ]
    except (OSError, ValueError) as error:
        records = []
        issues.append(f"evaluation records not available: {error}")
    by_step: dict[int, list[Row]] = {}
    for record in records:
        by_step.setdefault(record["step"], []).append(record)
    evaluations = []
    for event in (e for e in events if e["event"] == "eval"):
        own = by_step.get(event["step"], [])
        scopes = {}
        for scope, levels in _scopes(ladder).items():
            selected = [r for r in own if int(r["tier"]) in levels]
            expected = config["eval_per_tier"] * len(levels)
            # A missing answer is not a failure or a zero reward.
            valid = (
                len(selected) == expected
                and len({r["task"] for r in selected}) == expected
            )
            valid = valid and all(
                isinstance(r["reward"], (int, float)) and math.isfinite(r["reward"])
                for r in selected
            )
            if not valid:
                scopes[scope] = dict(n=len(selected), success=None, se=None)
            else:
                success = sum(r["reward"] >= SUCCESS_THRESHOLD for r in selected) / len(
                    selected
                )
                scopes[scope] = dict(
                    n=len(selected),
                    success=success,
                    se=math.sqrt(success * (1 - success) / len(selected)),
                )
        if any(v["success"] is None for v in scopes.values()):
            issues.append(
                f"eval at step {event['step']}: missing, repeated or unscored questions"
            )
        evaluations.append(
            dict(step=event["step"], tokens=event["gen_tokens_total"], scopes=scopes)
        )
    per_token = {}
    for scope in _scopes(ladder):
        points = [
            dict(
                step=e["step"], tokens=e["tokens"], value=e["scopes"][scope]["success"]
            )
            for e in evaluations
        ]
        per_token[scope] = _mean_or_none(
            [measure_at_tokens(points, steps, mark) for mark in TOKEN_MARKS]
        )
    try:
        diagnostics = _diagnostics(config, steps, folder)
        collapsed: bool | None = degenerate(steps)
    except (KeyError, TypeError, ValueError) as error:
        diagnostics, collapsed = {}, None
        issues.append(f"training diagnostics not available: {error}")
    return dict(
        path=str(folder),
        method=config["sampler"],
        seed=config["seed"],
        config=config,
        validation=status,
        issues=issues,
        last_step=steps[-1]["step"] if steps else 0,
        degenerate=collapsed,
        diagnostics=diagnostics,
        evaluations=evaluations,
        per_token=per_token,
        per_episode=dict.fromkeys(_scopes(ladder)),
        _initial=by_step.get(0),
    )


def _comparisons(
    runs: list[Row], scopes: dict[str, tuple[int, ...]], *, inference: bool = True
) -> tuple[Row, dict[int, int]]:
    """Use the same episode window for both methods of each complete pair."""
    indexed = {(r["seed"], r["method"]): r for r in runs}
    windows = {}
    differences: Row = {scope: {"per_token": {}, "per_episode": {}} for scope in scopes}
    for seed in SEEDS:
        pair = [indexed.get((seed, m)) for m in METHODS]
        if any(r is None for r in pair):
            continue
        uniform, curriculum = pair
        window = min(LAST_EPISODE_MARK, *(r["last_step"] // 100 * 100 for r in pair))
        windows[seed] = window
        for scope in scopes:
            token_values = [r["per_token"][scope] for r in pair]
            episode_values = []
            for run in pair:
                values = {
                    e["step"]: e["scopes"][scope]["success"] for e in run["evaluations"]
                }
                episode_values.append(
                    _mean_or_none([values.get(s) for s in range(100, window + 1, 100)])
                )
                run["per_episode"][scope] = episode_values[-1]
            for name, values in (
                ("per_token", token_values),
                ("per_episode", episode_values),
            ):
                if all(v is not None for v in values):
                    differences[scope][name][seed] = 100 * (values[1] - values[0])
    return {
        scope: {
            name: paired_summary(values, inference=inference and scope == "main")
            for name, values in measures.items()
        }
        for scope, measures in differences.items()
    }, windows


def build_report(folders: list[str | Path], *, ladder: str) -> Row:
    """
    Read one numbering's reference runs, never replacing a complete run.

    Args:
        folders (`list[str | Path]`):
            Input roots or run folders. Repeats may be in separate subfolders.
        ladder (`str`):
            `rfc` or `staircase`; results with different numbering never pool.

    Returns:
        `dict`: Run inventory, missing pairs, provenance warnings, per-question
        errors, primary and sensitivity comparisons, and training diagnostics.
        The first completed run is ordered by its logged start UTC (runs are
        sequential on one machine); ties are disclosed and ordered by path.
    """
    scopes = _scopes(ladder)
    warnings: list[str] = []
    inventory, candidates = [], []
    paths = sorted({p for folder in folders for p in find_run_dirs(folder)})
    for path in paths:
        try:
            events = [
                json.loads(line)
                for line in (path / "log.jsonl").read_text().splitlines()
            ]
            config = events[0]
            key = (config["seed"], config["sampler"])
        except (ValueError, IndexError, KeyError, TypeError) as error:
            inventory.append(dict(path=str(path), status=f"unreadable: {error}"))
            warnings.append(f"{path}: unreadable log; completion not verified")
            continue
        row = dict(path=str(path), seed=key[0], method=key[1])
        inventory.append(row)
        if config.get("ladder") != ladder:
            row["status"] = "other ladder"
            warnings.append(
                f"{path}: ladder {config.get('ladder')} not pooled with {ladder}"
            )
        elif key[0] not in SEEDS or key[1] not in METHODS:
            row["status"] = "outside reference methods/seeds"
        elif events[-1].get("event") != "done":
            row["status"] = "incomplete"
        else:
            candidates.append((config.get("utc_time", ""), str(path), events, row))
    chosen: dict[tuple[int, str], Row] = {}
    starts: set[tuple[int, str, str]] = set()
    for start, path, events, row in sorted(candidates, key=lambda c: (c[0], c[1])):
        key = (row["seed"], row["method"])
        if not start or (*key, start) in starts:
            warnings.append(
                f"{path}: first-complete order not verified (missing or tied start UTC); path breaks ties"
            )
        starts.add((*key, start))
        if key in chosen:
            row["status"] = "repeat"
        else:
            row["status"] = "selected"
            chosen[key] = _read_selected(Path(path), events, ladder)
    runs = [chosen[(s, m)] for s in SEEDS for m in METHODS if (s, m) in chosen]
    if runs:
        for field in ("model_snapshot", "versions", "prompt_date"):
            values = {json.dumps(r["config"].get(field), sort_keys=True) for r in runs}
            if len(values) > 1 or "null" in values:
                warnings.append(f"the runs differ in or lack {field}: {sorted(values)}")
        initial = runs[0]["_initial"]
        if initial is None or any(r["_initial"] != initial for r in runs):
            warnings.append(
                "step-0 question records differ or are missing; not corrected"
            )
    comparisons, windows = _comparisons(runs, scopes)
    collapsed = sorted({r["seed"] for r in runs if r["degenerate"]})
    sensitivity, _ = _comparisons(
        [r for r in runs if r["seed"] not in collapsed], scopes, inference=False
    )
    for run in runs:
        del run["_initial"]
    support = [
        comparisons["main"][measure]["support"]
        for measure in ("per_token", "per_episode")
    ]
    method_values: Row = {}
    for scope in scopes:
        method_values[scope] = {}
        for measure in ("per_token", "per_episode"):
            method_values[scope][measure] = {}
            for method in METHODS:
                values = {
                    r["seed"]: 100 * r[measure][scope]
                    for r in runs
                    if r["method"] == method and r[measure][scope] is not None
                }
                method_values[scope][measure][method] = dict(
                    values=values,
                    n=len(values),
                    mean=fmean(values.values()) if values else None,
                    range=[min(values.values()), max(values.values())]
                    if values
                    else None,
                )
    return dict(
        ladder=ladder,
        roots=[str(Path(p)) for p in folders],
        main_levels=list(MAIN_LEVELS[ladder]),
        inventory=inventory,
        runs=runs,
        warnings=warnings,
        missing=[
            dict(seed=s, method=m, status="not run")
            for s in SEEDS
            for m in METHODS
            if (s, m) not in chosen
        ],
        episode_windows=windows,
        degenerate_seeds=collapsed,
        comparisons=comparisons,
        method_values=method_values,
        without_degenerate=sensitivity,
        supports_proposal=all(
            s in ("shown by the interval", "on the mean only") for s in support
        )
        if all(comparisons["main"][m]["n"] == 5 for m in ("per_token", "per_episode"))
        else None,
    )


def _cell(value: Any) -> str:
    """Four significant digits keep small p-values and question errors readable."""
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_cell(v) for v in value) + "]"
    return str(value)


def _table(header: list[str], rows: list[list[Any]]) -> str:
    """Round the Markdown cells; the JSON keeps full precision."""
    return markdown_table(
        header, [[_cell(v) if v is not None else None for v in row] for row in rows]
    )


def _text(report: Row) -> str:
    """Render the same computed data that are exported as JSON."""
    parts = [
        "# Reference experiment",
        "",
        "Generated by `python -m analysis.reference`.",
        "",
        f"Numbering: {report['ladder']}; main levels: {report['main_levels']}.",
        "Differences and intervals are percentage points; success and SE in JSON are fractions.",
        "Table cells show four significant digits; reference.json keeps full precision.",
        "Tests are two-sided, paired, five seeds, df=4, without multiple-test correction.",
        "A t-test does not establish the -2-point margin: the interval reading is reported separately.",
        "Zero sample variance uses the limiting t/p; infinity is a JSON string.",
        "",
        "## Checks and run selection",
        "",
        *[f"- {w}" for w in report["warnings"]],
        _table(
            ["Run", "Status"], [[r["path"], r["status"]] for r in report["inventory"]]
        ),
        *[f"- Seed {r['seed']}, {r['method']}: not run." for r in report["missing"]],
        "",
        f"Common episode windows (last step): {report['episode_windows']}",
        f"Degenerate seeds, retained in the primary analysis: {report['degenerate_seeds']}",
        f"Pre-registered support for both main measures: {report['supports_proposal']}",
    ]
    for label, comparisons in (
        ("Primary reading", report["comparisons"]),
        ("Without degenerate seeds (no test)", report["without_degenerate"]),
    ):
        parts += ["", f"## {label}", ""]
        for scope, measures in comparisons.items():
            parts += [
                f"### {scope}",
                "",
                _table(
                    [
                        "Measure",
                        *[f"seed {s}" for s in SEEDS],
                        "n",
                        "Mean",
                        "Range",
                        "t",
                        "p",
                        "95% interval",
                        "Reading",
                        "-2 margin",
                    ],
                    [
                        [
                            name,
                            *[r["differences"].get(s) for s in SEEDS],
                            r["n"],
                            r["mean"],
                            r["range"],
                            r["t"],
                            r["p"],
                            r["interval"],
                            r["verdict"],
                            r["support"],
                        ]
                        for name, r in measures.items()
                    ],
                ),
                "",
            ]
    parts += [
        "",
        "## Each method across seeds (percent success)",
        "",
        _table(
            [
                "Scope",
                "Measure",
                "Method",
                *[f"seed {s}" for s in SEEDS],
                "n",
                "Mean",
                "Range",
            ],
            [
                [
                    scope,
                    measure,
                    method,
                    *[r["values"].get(s) for s in SEEDS],
                    r["n"],
                    r["mean"],
                    r["range"],
                ]
                for scope, measures in report["method_values"].items()
                for measure, methods in measures.items()
                for method, r in methods.items()
            ],
        ),
    ]
    parts += ["", "## Per-run checks and training spend", ""]
    for run in report["runs"]:
        parts += [
            f"### {run['method']}, seed {run['seed']}",
            "",
            f"Source: `{run['path']}`; validation: {run['validation']}; degenerate: {run['degenerate']}.",
            *[f"- {issue}" for issue in run["issues"]],
            "",
            "```json",
            json.dumps(
                dict(config=run["config"], diagnostics=run["diagnostics"]),
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            _table(
                [
                    "Step",
                    "Tokens",
                    "Scope",
                    "Questions",
                    "Success",
                    "SE over questions",
                ],
                [
                    [e["step"], e["tokens"], scope, v["n"], v["success"], v["se"]]
                    for e in run["evaluations"]
                    for scope, v in e["scopes"].items()
                ],
            ),
        ]
    return "\n".join(parts) + "\n"


def write_report(report: Row, out: str | Path) -> list[Path]:
    """
    Atomically regenerate tables, full-precision JSON, per-seed curves and falls.

    Args:
        report (`dict`):
            The result of `build_report`. Relative roots and run paths are
            resolved against the current folder, so call this from the folder
            `build_report` read them from.
        out (`str` or `Path`):
            Output folder, outside every input root and run folder.

    Returns:
        `list[Path]`: The six generated files; `falls.json` and `falls.md` are
        the post-results reading of `analysis.falls`.

    Raises:
        `ValueError`: If the output could write into a run's input tree, or if
        the falls reading cannot reproduce the report's main measures. Both are
        checked before any file is written.
    """
    from analysis.falls import build_falls, write_falls

    out = Path(out).resolve()
    for root in report["roots"]:
        root = Path(root).resolve()
        if out == root or root in out.parents:
            raise ValueError("report output must be outside run input roots")
    # Protect individual run folders even when several input roots were supplied.
    if any(
        Path(r["path"]).resolve() == out or Path(r["path"]).resolve() in out.parents
        for r in report["inventory"]
    ):
        raise ValueError("report output must be outside run folders")
    falls = build_falls(report)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    files = [out / "reference.json", out / "reference.md"]
    write_json_atomic(files[0], report)
    temporary = files[1].with_suffix(".md.tmp")
    temporary.write_text(_text(report))
    temporary.replace(files[1])
    for axis, name in (
        ("tokens", "success-per-token.png"),
        ("step", "success-per-episode.png"),
    ):
        fig, axes = plt.subplots(1, len(SEEDS), figsize=(16, 3), sharey=True)
        for seed, ax in zip(SEEDS, axes):
            for run in report["runs"]:
                if run["seed"] != seed:
                    continue
                evaluations = [
                    e
                    for e in run["evaluations"]
                    if e["scopes"]["main"]["success"] is not None
                ]
                multiplier = (
                    run["config"]["prompts_per_step"] * run["config"]["group_size"]
                    if axis == "step"
                    else 1
                )
                xs = [e[axis] * multiplier for e in evaluations]
                ys = [e["scopes"]["main"]["success"] for e in evaluations]
                errors = [e["scopes"]["main"]["se"] for e in evaluations]
                ax.plot(xs, ys, label=run["method"])
                ax.fill_between(
                    xs,
                    [y - e for y, e in zip(ys, errors)],
                    [y + e for y, e in zip(ys, errors)],
                    alpha=0.15,
                )
            ax.set(
                title=f"Seed {seed}",
                xlabel="Generated training tokens"
                if axis == "tokens"
                else "Training episodes",
                ylim=(0, 1),
            )
            if ax.lines:
                ax.legend()
        axes[0].set_ylabel("Main-level success ± question SE")
        fig.tight_layout()
        target = out / name
        temp = target.with_suffix(".png.tmp")
        fig.savefig(temp, format="png", metadata={"Software": "reference analysis"})
        plt.close(fig)
        temp.replace(target)
        files.append(target)
    files += write_falls(falls, out)
    return files


def main(argv: list[str] | None = None) -> int:
    """
    Generate the reference report without changing any run folder.

    Args:
        argv (`list[str]`, *optional*):
            Command-line arguments; omitted means `sys.argv[1:]`.

    Returns:
        `int`: Zero after generation; validation problems remain in the report.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--ladder", choices=tuple(MAIN_LEVELS), default="rfc")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = build_report(args.runs, ladder=args.ladder)
    for path in write_report(report, args.out):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEGENERATE_SHARE",
    "DEGENERATE_WINDOW",
    "build_report",
    "degenerate",
    "main",
    "measure_at_tokens",
    "paired_summary",
    "write_report",
]
