# SPDX-License-Identifier: BSD-3-Clause

"""
Throughput, signal groups, memory and test evaluations of each method.

  python -m analysis.pilot_evidence.analyze <pilot folder>
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

from analysis.pilot_evidence.pilot_logs import (
    EPISODES_PER_STEP,
    MAX_TOKENS,
    read_pilot_log,
)


def text(pilot_dir: Path) -> str:
    """
    Return the `analyze.py` section of the evidence.

    Args:
        pilot_dir (`Path`):
            The pilot's folder, `rfc-013/pilot`.

    Returns:
        `str`: The section's lines, each ending with a newline.
    """
    lines = []
    for arm in ("uniform", "curriculum"):
        log = read_pilot_log(pilot_dir, arm)
        steps, evals = log.steps, log.evals
        n = len(steps)
        episodes = n * EPISODES_PER_STEP
        toks = sum(s["gen_tokens"] for s in steps)
        secs = steps[-1]["train_elapsed_s"]
        sig = sum(s["groups_with_signal"] for s in steps)
        gen_s = sum(s["gen_seconds"] for s in steps)
        tr_s = sum(s["train_seconds"] for s in steps)
        lines += [
            f"== {arm}: steps={n} episodes={episodes} tokens={toks} secs={secs}",
            f"   eps/s={episodes / secs:.2f} tok/s={toks / secs:.1f} "
            f"tok/completion={toks / episodes:.1f}",
            f"   signal groups={sig} per1k eps={1000 * sig / episodes:.1f} "
            f"per M tok={1e6 * sig / toks:.0f} per min={60 * sig / secs:.1f}",
            f"   gen_s={gen_s:.0f} ({100 * gen_s / secs:.0f}%) "
            f"train_s={tr_s:.0f} ({100 * tr_s / secs:.0f}%)",
            f"   gen tok/s (gen only)={toks / gen_s:.0f}; "
            f"train s per signal group={tr_s / max(sig, 1):.3f}",
        ]
        # Peak memory at a few steps, and the first step above each threshold.
        pm = [s["peak_mem_gb"] for s in steps]
        idx = [0, 9, 49, 99, 199, 499, 999, n - 1]
        lines.append(f"   peak_mem at steps {[(i + 1, pm[i]) for i in idx if i < n]}")
        for th in (8, 10, 12, 14, 15):
            first = next((s["step"] for s in steps if s["peak_mem_gb"] > th), None)
            lines.append(f"   first step peak_mem>{th}: {first}")
        gt = [s["gen_tokens"] for s in steps]
        lines.append(
            f"   gen_tokens/step: min={min(gt)} med={statistics.median(gt)} "
            f"p95={sorted(gt)[int(0.95 * n)]} max={max(gt)}  "
            f"(max possible {EPISODES_PER_STEP * MAX_TOKENS})"
        )
        lines.append(
            "   evals: step, Mtok, succ, by_tier, mean_completion_tokens, eval_s"
        )
        for e in evals:
            lines.append(
                f"     {e['step']:>5} {e['gen_tokens_total'] / 1e6:>5.2f} "
                f"{100 * e['success_rate']:>5.1f} {e['by_tier']} "
                f"{e['mean_completion_tokens']:>6.1f} {e['eval_seconds']}"
            )
        tot_eval = sum(e["eval_seconds"] for e in evals)
        lines.append(f"   total eval seconds={tot_eval:.0f} over {len(evals)} evals")
        # Tokens per completion and signal over time, by quarter of the run.
        q = n // 4
        for i in range(4):
            seg = steps[i * q : (i + 1) * q]
            t = sum(s["gen_tokens"] for s in seg)
            sg = sum(s["groups_with_signal"] for s in seg)
            start = steps[i * q - 1]["train_elapsed_s"] if i else 0
            el = seg[-1]["train_elapsed_s"] - start
            lines.append(
                f"   quarter {i + 1}: tok/compl={t / (len(seg) * EPISODES_PER_STEP):.1f} "
                f"signal/step={sg / len(seg):.2f} s/step={el / len(seg):.2f} "
                f"sig per Mtok={1e6 * sg / t:.0f}"
            )
        if arm == "curriculum":
            for i in (0, 99, 299, 499, 799, n - 1):
                lines.append(
                    f"   probs@ {steps[i]['step']} {steps[i]['sample_probs']} "
                    f"p_hat {steps[i]['p_hat']}"
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
