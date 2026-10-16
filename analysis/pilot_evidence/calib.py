# SPDX-License-Identifier: BSD-3-Clause

"""
Is `1 - p^G - (1-p)^G` calibrated against the signal-group rate the pilot saw?

  python -m analysis.pilot_evidence.calib <pilot folder>

For the curriculum arm it compares the signal groups the sampler's estimates
predict with those that happened. For the uniform arm it gives, per tier and
quarter, the success rate, the predicted and the measured signal rate, and the
tokens per completion.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from analysis.pilot_evidence.pilot_logs import (
    GROUP_SIZE,
    N_TIERS,
    ols,
    PROMPTS_PER_STEP,
    read_pilot_log,
    signal_chance,
    uniform_tier_counts,
)


@dataclass
class Quarter:
    """
    One quarter of the uniform arm.

    Attributes:
        p (`list[float]`):
            Per tier, the mean reward of its answers.
        signal (`list[float]`):
            Per tier, signal groups per group, from a regression on tier counts.
        tokens (`list[float]`):
            Per tier, tokens per completion, from a regression on tier counts.
        signal_groups (`int`):
            Groups with signal in the quarter.
        gen_tokens (`int`):
            Tokens generated in the quarter.
        steps (`int`):
            Steps in the quarter.
    """

    p: list[float]
    signal: list[float]
    tokens: list[float]
    signal_groups: int
    gen_tokens: int
    steps: int


def uniform_quarters(pilot_dir: Path) -> list[Quarter]:
    """
    Measure each tier of the uniform arm, quarter by quarter.

    Args:
        pilot_dir (`Path`):
            The pilot's folder, `rfc-013/pilot`.

    Returns:
        `list[Quarter]`: The four quarters of the run, in order.
    """
    us = read_pilot_log(pilot_dir, "uniform").steps
    comp = uniform_tier_counts(len(us))
    q = len(us) // 4
    quarters = []
    for i in range(4):
        seg = range(i * q, (i + 1) * q)
        p = []
        for k in range(N_TIERS):
            num = sum(
                comp[t][k] * us[t]["tier_mean_reward"].get(str(k), 0.0) for t in seg
            )
            den = sum(comp[t][k] for t in seg)
            p.append(num / den)
        signal = ols(
            [[float(c) for c in comp[t]] for t in seg],
            [us[t]["groups_with_signal"] for t in seg],
        )
        tokens = ols(
            [[float(GROUP_SIZE * c) for c in comp[t]] for t in seg],
            [us[t]["gen_tokens"] for t in seg],
        )
        quarters.append(
            Quarter(
                p=p,
                signal=signal,
                tokens=tokens,
                signal_groups=sum(us[t]["groups_with_signal"] for t in seg),
                gen_tokens=sum(us[t]["gen_tokens"] for t in seg),
                steps=len(seg),
            )
        )
    return quarters


def text(pilot_dir: Path) -> str:
    """
    Return the `calib.py` section of the evidence.

    Args:
        pilot_dir (`Path`):
            The pilot's folder, `rfc-013/pilot`.

    Returns:
        `str`: The section's lines, each ending with a newline.
    """
    lines = [
        "curriculum arm: predicted signal groups/step from p_hat (model) vs actual"
    ]
    cs = read_pilot_log(pilot_dir, "curriculum").steps
    n = len(cs)
    for a, b in [(1, 100), (100, 300), (300, 600), (600, 900), (900, n)]:
        pred = act = 0.0
        for t in range(a, b):
            prev = cs[t - 1]
            pred += PROMPTS_PER_STEP * sum(
                prev["sample_probs"][k] * signal_chance(prev["p_hat"][k])
                for k in prev["p_hat"]
            )
            act += cs[t]["groups_with_signal"]
        lines.append(
            f"  steps {a:>4}-{b:<4}: predicted {pred / (b - a):.2f}  "
            f"actual {act / (b - a):.2f}  ratio {act / pred:.2f}"
        )
    lines += [
        "",
        "uniform arm, per tier, by quarter: p (mean reward), model signal "
        "1-p^8-(1-p)^8, actual signal rate, tokens/completion,",
        "  actual signal groups per million tokens of that tier",
    ]
    for i, quarter in enumerate(uniform_quarters(pilot_dir)):
        lines.append(f"  quarter {i + 1}")
        for k in range(N_TIERS):
            p, s, c = quarter.p[k], quarter.signal[k], quarter.tokens[k]
            per_mtok = 1e6 * s / (GROUP_SIZE * c)
            lines.append(
                f"    tier {k}: p={p:.3f} model={signal_chance(p):.3f} "
                f"actual={s:.3f} tok={c:.1f} sig/Mtok={per_mtok:.0f}"
            )
        lines.append(
            f"    overall: {1e6 * quarter.signal_groups / quarter.gen_tokens:.0f} "
            "signal groups per M tokens; "
            f"{quarter.signal_groups / (PROMPTS_PER_STEP * quarter.steps):.3f} "
            "signal rate"
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
