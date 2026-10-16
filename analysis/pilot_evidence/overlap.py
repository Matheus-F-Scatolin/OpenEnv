# SPDX-License-Identifier: BSD-3-Clause

"""
Question collisions between the pilot-style train and test seeds.

  python -m analysis.pilot_evidence.overlap

For the pilot's 5 tiers and for the 10-level staircase, it counts the distinct
questions of each split and the test questions that are also train questions.
Each question is built as the environment's `reset()` builds it: a
`reasoning_gym` dataset of size 1 with the task's seed.
"""

from __future__ import annotations

import sys
from typing import Any

import reasoning_gym
from analysis.pilot_evidence.pilot_logs import TEST_PER_TIER, TRAIN_PER_TIER
from trial.tasks import DATASET, make_tasks, PILOT_LADDER, STAIRCASE_LADDER

LADDERS = (
    ("pilot tiers", PILOT_LADDER),
    ("staircase d=0..9 (library levels, no negation)", STAIRCASE_LADDER),
)


def _questions(
    split: str, per_tier: int, ladder: list[dict[str, Any]]
) -> list[list[str]]:
    questions: list[list[str]] = [[] for _ in ladder]
    for task in make_tasks(split, per_tier, ladder):
        cfg = ladder[task["difficulty"]]
        dataset = reasoning_gym.create_dataset(
            DATASET, size=1, seed=task["seed"], **cfg
        )
        questions[task["difficulty"]].append(dataset[0]["question"])
    return questions


def text() -> str:
    """
    Return the `overlap.py` section of the evidence.

    Returns:
        `str`: The section's lines, each ending with a newline.
    """
    lines = []
    for name, ladder in LADDERS:
        lines += [
            f"== {name}",
            "  tier terms x digits | distinct train /2000 | distinct test /100 "
            "| test questions also in train",
        ]
        train = _questions("train", TRAIN_PER_TIER, ladder)
        test = _questions("test", TEST_PER_TIER, ladder)
        for k, cfg in enumerate(ladder):
            tr, te = train[k], test[k]
            s = set(tr)
            neg = " neg" if cfg.get("allow_negation") else "    "
            lines.append(
                f"  {k:>3}   {cfg['min_terms']} x {cfg['min_digits']}{neg}     "
                f"| {len(s):>5}               | {len(set(te)):>3}                "
                f"| {sum(q in s for q in te):>3}/100"
            )
    return "".join(line + "\n" for line in lines)


def main() -> None:
    """
    Print the section from the command line. It reads no file.
    """
    sys.stdout.write(text())


if __name__ == "__main__":
    main()
