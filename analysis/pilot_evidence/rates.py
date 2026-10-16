# SPDX-License-Identifier: BSD-3-Clause

"""
Throughput, evaluation noise, time budgets and a matched-token comparison.

  python -m analysis.pilot_evidence.rates <pilot folder>
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

from analysis.pilot_evidence.pilot_logs import EPISODES_PER_STEP, read_pilot_log

# Steps of the curriculum arm before throttling slowed it down.
BEFORE_THROTTLING = 900


def text(pilot_dir: Path) -> str:
    """
    Return the `rates.py` section of the evidence.

    Args:
        pilot_dir (`Path`):
            The pilot's folder, `rfc-013/pilot`.

    Returns:
        `str`: The section's lines, each ending with a newline.
    """
    curriculum = read_pilot_log(pilot_dir, "curriculum")
    uniform = read_pilot_log(pilot_dir, "uniform")
    cs, ce = curriculum.steps, curriculum.evals
    us, ue = uniform.steps, uniform.evals
    seg = cs[:BEFORE_THROTTLING]
    el = seg[-1]["train_elapsed_s"]
    tok = sum(s["gen_tokens"] for s in seg)
    sg = sum(s["groups_with_signal"] for s in seg)
    lines = [
        f"curriculum, steps 1-{BEFORE_THROTTLING} (before throttling): "
        f"{BEFORE_THROTTLING * EPISODES_PER_STEP / el:.2f} eps/s, {tok / el:.0f} tok/s, "
        f"{el / BEFORE_THROTTLING:.2f} s/step, {60 * sg / el:.1f} signal groups/min, "
        f"{1e6 * sg / tok:.0f} per M tok"
    ]
    # Token rates as the evidence prints them: this line and the analyze.py section.
    curriculum_rate = round(tok / el, 0)
    uel = us[-1]["train_elapsed_s"]
    uniform_rate = round(sum(s["gen_tokens"] for s in us) / uel, 1)
    lines.append(
        f"uniform whole run: {len(us) * EPISODES_PER_STEP / uel:.2f} eps/s, "
        f"{uel / len(us):.2f} s/step"
    )
    for name, ev in (("uniform", ue), ("curriculum", ce)):
        acc = [100 * e["success_rate"] for e in ev]
        d = [b - a for a, b in zip(acc, acc[1:])]
        half = len(d) // 2
        lines.append(
            f"{name}: eval-to-eval diffs SD first half "
            f"{statistics.pstdev(d[:half]):.2f}, second half "
            f"{statistics.pstdev(d[half:]):.2f} points "
            "(divide by 1.41 for per-eval noise)"
        )
    for B in (2.0e6, 2.5e6, 3.0e6, 5.0e6):
        u, c = B / uniform_rate / 60, B / curriculum_rate / 60
        lines.append(
            f"budget {B / 1e6:.1f} M tokens: uniform {u:.0f} min, "
            f"curriculum(episodes) {c:.0f} min, tokens arm {c:.0f}-{u:.0f} min; "
            f"per seed (3 arms) {(u + 2 * c) / 60:.2f}-{(2 * u + c) / 60:.2f} h "
            "before evals"
        )
    lines.append(
        "eval seconds per 500-task eval: uniform median "
        f"{statistics.median(e['eval_seconds'] for e in ue)} curriculum median "
        f"{statistics.median(e['eval_seconds'] for e in ce)}"
    )
    # Each curriculum evaluation against the uniform one nearest in tokens.
    ut = [e["gen_tokens_total"] for e in ue]
    lines.append("matched-token comparison (curriculum eval -> nearest uniform eval):")
    for e in ce[1:]:
        i = min(range(len(ut)), key=lambda j: abs(ut[j] - e["gen_tokens_total"]))
        lines.append(
            f"  {e['gen_tokens_total'] / 1e6:.2f} M: "
            f"curriculum {100 * e['success_rate']:.1f}  "
            f"uniform {100 * ue[i]['success_rate']:.1f} (at {ut[i] / 1e6:.2f} M)  "
            f"diff {100 * (e['success_rate'] - ue[i]['success_rate']):+.1f}"
        )
    return "".join(line + "\n" for line in lines)


def main(argv: list[str] | None = None) -> None:
    """
    Print the section from the command line.

    Args:
        argv (`list[str]`, *optional*):
            The arguments; `sys.argv[1:]` when left out.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pilot_dir", type=Path)
    sys.stdout.write(text(parser.parse_args(argv).pilot_dir))


if __name__ == "__main__":
    main()
