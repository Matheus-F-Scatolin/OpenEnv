# SPDX-License-Identifier: BSD-3-Clause

"""
Tokens per completion by tier, and the tier mix of each method.

  python -m analysis.pilot_evidence.replay <pilot folder>

The uniform arm's batches come from replaying its sampler; regressing the
tokens of each step on its tier counts gives the tokens per completion of each
tier. The curriculum arm's expected mix comes from the probabilities it logged
one step earlier.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from analysis.pilot_evidence.pilot_logs import (
    EPISODES_PER_STEP,
    GROUP_SIZE,
    N_TIERS,
    ols,
    read_pilot_log,
    uniform_tier_counts,
)


def _mix(X: list[list[float]]) -> list[float]:
    return [sum(x[k] for x in X) / sum(map(sum, X)) for k in range(N_TIERS)]


def text(pilot_dir: Path) -> str:
    """
    Return the `replay.py` section of the evidence.

    Args:
        pilot_dir (`Path`):
            The pilot's folder, `rfc-013/pilot`.

    Returns:
        `str`: The section's lines, each ending with a newline.
    """
    steps = read_pilot_log(pilot_dir, "uniform").steps
    X, y, mismatch = [], [], 0
    for step, counts in zip(steps, uniform_tier_counts(len(steps))):
        present = {str(k) for k, c in enumerate(counts) if c}
        if present != set(step["tier_mean_reward"]):
            mismatch += 1
        X.append([GROUP_SIZE * c for c in counts])
        y.append(step["gen_tokens"])
    lines = [f"uniform replay: steps {len(steps)} tier-set mismatches {mismatch}"]
    cost = ols(X, y)
    lines.append(
        f"uniform per-tier tokens/completion (whole run): {[round(c, 1) for c in cost]}"
    )
    q = len(steps) // 4
    for i in range(4):
        part = ols(X[i * q : (i + 1) * q], y[i * q : (i + 1) * q])
        lines.append(f"  quarter {i + 1}: {[round(c, 1) for c in part]}")
    lines.append(f"  first 100 steps: {[round(c, 1) for c in ols(X[:100], y[:100])]}")
    mix_u = _mix(X)
    lines.append(f"uniform realised mix: {[round(m, 3) for m in mix_u]}")

    # The curriculum's expected batch: the probabilities logged one step earlier.
    cs = read_pilot_log(pilot_dir, "curriculum").steps
    Xc = [
        [EPISODES_PER_STEP * cs[t - 1]["sample_probs"][str(k)] for k in range(N_TIERS)]
        for t in range(1, len(cs))
    ]
    yc = [cs[t]["gen_tokens"] for t in range(1, len(cs))]
    lines.append(
        "curriculum per-tier tokens/completion (regression on expected mix, rough): "
        f"{[round(c, 1) for c in ols(Xc, yc)]}"
    )
    mix_c = _mix(Xc)
    lines.append(f"curriculum expected mix over run: {[round(m, 3) for m in mix_c]}")
    qc = len(Xc) // 4
    for i in range(4):
        part_mix = _mix(Xc[i * qc : (i + 1) * qc])
        lines.append(f"  quarter {i + 1} mix: {[round(m, 3) for m in part_mix]}")
    tok_u_mix_u = sum(m * c for m, c in zip(mix_u, cost))
    tok_c_mix_u_policy = sum(m * c for m, c in zip(mix_c, cost))
    lines.append(
        f"tokens/completion: uniform observed {sum(y) / (EPISODES_PER_STEP * len(y)):.1f}; "
        f"uniform-policy costs x uniform mix {tok_u_mix_u:.1f}; "
        f"uniform-policy costs x curriculum mix {tok_c_mix_u_policy:.1f}; "
        f"curriculum observed {sum(yc) / (EPISODES_PER_STEP * len(yc)):.1f}"
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
