# SPDX-License-Identifier: BSD-3-Clause

"""
Score sampler allocations against the per-tier profile of the pilot's uniform arm.

  python -m analysis.pilot_evidence.alloc

The profiles are the first and last quarters of the `calib.py` section: success
rate `p`, signal rate `s` and tokens per completion `c` by tier. A test checks
them against that section.
"""

from __future__ import annotations

import sys

from analysis.pilot_evidence.pilot_logs import GROUP_SIZE, N_TIERS, signal_chance

# The curriculum sampler's defaults; a test compares them with the pilot's sampler.
EPS = 0.05
DECAY = 0.9
# calib.py rounded; a negative signal estimate, or a tier always solved, is 0.
PROFILES = {
    "uniform arm, quarter 1": dict(
        p=[1.0, 0.993, 0.775, 0.103, 0.049],
        s=[0.0, 0.052, 0.388, 0.286, 0.227],
        c=[11.5, 25.7, 37.5, 62.3, 78.6],
    ),
    "uniform arm, quarter 4": dict(
        p=[1.0, 0.999, 0.948, 0.398, 0.267],
        s=[0.0, 0.0, 0.171, 0.456, 0.441],
        c=[11.0, 28.6, 40.8, 65.8, 79.3],
    ),
}


def _p_hat(p: float) -> float:
    # Steady state of the decayed Beta(1, 1) estimate: G / (1 - decay) = 80 episodes.
    n = GROUP_SIZE / (1 - DECAY)
    return (n * p + 1) / (n + 2)


def _norm(w: list[float]) -> list[float]:
    t = sum(w)
    return [x / t for x in w]


def _score(
    pi: list[float], s: list[float], c: list[float]
) -> tuple[float, float, float]:
    # Signal groups per 1,000 episodes, per million tokens, and tokens per episode.
    sig = sum(a * b for a, b in zip(pi, s))
    tok = sum(a * b for a, b in zip(pi, c))
    return 1000 * sig / GROUP_SIZE, 1e6 * sig / (GROUP_SIZE * tok), tok


def text() -> str:
    """
    Return the `alloc.py` section of the evidence.

    Returns:
        `str`: The section's lines, each ending with a newline.
    """
    lines = []
    for name, pr in PROFILES.items():
        p, s, c = pr["p"], pr["s"], pr["c"]
        ph = [_p_hat(x) for x in p]
        best_per_token = max(range(N_TIERS), key=lambda j: s[j] / c[j])
        best_per_episode = max(range(N_TIERS), key=lambda j: s[j])
        allocs = {
            "uniform": [0.2] * N_TIERS,
            "RFC cost=episodes  (model+eps)": _norm(
                [signal_chance(x) + EPS for x in ph]
            ),
            "RFC cost=tokens    (model+eps)/c": _norm(
                [(signal_chance(x) + EPS) / cc for x, cc in zip(ph, c)]
            ),
            "empirical signal   (s+eps)/c": _norm(
                [(x + EPS) / cc for x, cc in zip(s, c)]
            ),
            "oracle per token   argmax s/c": [
                1.0 if i == best_per_token else 0.0 for i in range(N_TIERS)
            ],
            "oracle per episode argmax s": [
                1.0 if i == best_per_episode else 0.0 for i in range(N_TIERS)
            ],
        }
        lines.append(
            f"== {name}:  p_hat={[round(x, 3) for x in ph]}  "
            f"model signal={[round(signal_chance(x), 3) for x in ph]}  "
            f"measured signal={s}"
        )
        base = _score(allocs["uniform"], s, c)
        for k, pi in allocs.items():
            e, t, tok = _score(pi, s, c)
            lines.append(
                f"  {k:<34} probs={[round(x, 2) for x in pi]}  "
                f"sig/1k ep={e:5.1f} ({e / base[0]:.2f}x)  "
                f"sig/Mtok={t:5.0f} ({t / base[1]:.2f}x)  tok/ep={tok:5.1f}"
            )
    return "".join(line + "\n" for line in lines)


def main() -> None:
    """
    Print the section from the command line. It reads no file.
    """
    sys.stdout.write(text())


if __name__ == "__main__":
    main()
