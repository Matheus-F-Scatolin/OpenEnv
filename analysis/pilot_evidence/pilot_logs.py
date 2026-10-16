# SPDX-License-Identifier: BSD-3-Clause

"""
Read the pilot's logs, replay its uniform sampler, fit regressions: for every section.

The pilot ran seed 0 of two methods, `uniform` and `curriculum`, twice: a
20-step `smoke` run, then the 150-minute `pilot` run. Each log is
`<pilot folder>/runs/<kind>/llama3.2-1b-<method>-s0/log.jsonl`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from samplers import UniformSampler
from trial.runlog import LOG_NAME, read_log
from trial.tasks import make_tasks, PILOT_LADDER

# The pilot's settings; a test compares them with the config event of each log.
SEED = 0
PROMPTS_PER_STEP = 8
GROUP_SIZE = 8
EPISODES_PER_STEP = PROMPTS_PER_STEP * GROUP_SIZE
MAX_TOKENS = 384
TRAIN_PER_TIER = 2000
TEST_PER_TIER = 100
N_TIERS = len(PILOT_LADDER)


@dataclass
class PilotLog:
    """
    The events of one pilot log, split by event type.

    Attributes:
        config (`dict[str, Any]`):
            The first event: the run's settings.
        steps (`list[dict[str, Any]]`):
            The `step` events, in order.
        evals (`list[dict[str, Any]]`):
            The `eval` events, in order.
    """

    config: dict[str, Any]
    steps: list[dict[str, Any]]
    evals: list[dict[str, Any]]


def read_pilot_log(pilot_dir: Path, method: str, kind: str = "pilot") -> PilotLog:
    """
    Read one log of the pilot.

    Args:
        pilot_dir (`Path`):
            The pilot's folder, `rfc-013/pilot`.
        method (`str`):
            `"uniform"` or `"curriculum"`.
        kind (`str`, *optional*, defaults to `"pilot"`):
            `"pilot"` for the 150-minute run, `"smoke"` for the 20-step run.

    Returns:
        `PilotLog`: The config, step and eval events of that log.
    """
    run_dir = pilot_dir / "runs" / kind / f"llama3.2-1b-{method}-s{SEED}"
    events = read_log(run_dir / LOG_NAME)
    return PilotLog(
        config=events[0],
        steps=[e for e in events if e["event"] == "step"],
        evals=[e for e in events if e["event"] == "eval"],
    )


def uniform_tier_counts(n_steps: int) -> list[list[int]]:
    """
    Replay the pilot's uniform sampler to recover the tiers of every step.

    The uniform sampler never reads rewards, so drawing one batch per step from
    a sampler with the pilot's tasks and seed gives the batches the pilot ran.

    Args:
        n_steps (`int`):
            Number of steps to replay.

    Returns:
        `list[list[int]]`: Per step, the number of prompts of each tier.
    """
    tasks = make_tasks("train", TRAIN_PER_TIER, PILOT_LADDER)
    sampler = UniformSampler(tasks, rng_seed=SEED)
    counts = []
    for _ in range(n_steps):
        step = [0] * N_TIERS
        for task in sampler.sample(PROMPTS_PER_STEP):
            step[task["difficulty"]] += 1
        counts.append(step)
    return counts


def signal_chance(p: float) -> float:
    """
    Return the chance that a group has signal when each answer succeeds with `p`.

    Args:
        p (`float`):
            Success rate of one answer.

    Returns:
        `float`: `1 - p**G - (1 - p)**G`, with `G` the group size.
    """
    return 1 - p**GROUP_SIZE - (1 - p) ** GROUP_SIZE


def ols(X: list[list[float]], y: list[float]) -> list[float]:
    """
    Fit `y` as a linear function of the rows of `X`, without an intercept.

    Args:
        X (`list[list[float]]`):
            One row of regressors per observation.
        y (`list[float]`):
            One value per observation.

    Returns:
        `list[float]`: The least-squares coefficient of each regressor.
    """
    k = len(X[0])
    A = [[sum(x[i] * x[j] for x in X) for j in range(k)] for i in range(k)]
    b = [sum(x[i] * yy for x, yy in zip(X, y)) for i in range(k)]
    # Solve the normal equations by Gauss-Jordan elimination with partial pivoting.
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(k):
        p = max(range(i, k), key=lambda r: abs(M[r][i]))
        M[i], M[p] = M[p], M[i]
        for r in range(k):
            if r != i:
                f = M[r][i] / M[i][i]
                M[r] = [a - f * c for a, c in zip(M[r], M[i])]
    return [M[i][k] / M[i][i] for i in range(k)]


__all__ = [
    "EPISODES_PER_STEP",
    "GROUP_SIZE",
    "MAX_TOKENS",
    "N_TIERS",
    "PROMPTS_PER_STEP",
    "PilotLog",
    "SEED",
    "TEST_PER_TIER",
    "TRAIN_PER_TIER",
    "ols",
    "read_pilot_log",
    "signal_chance",
    "uniform_tier_counts",
]
