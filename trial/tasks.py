# SPDX-License-Identifier: BSD-3-Clause

"""
The tasks of the experiment: the difficulty ladders, the prompt, the answer parser.

Copied from the pilot's script with the ladder as a parameter. A test compares
every piece with its source in `pilot/smoke.py`.
"""

from __future__ import annotations

import re
from typing import Any

# The day of the pilot. The model's chat template writes today's date otherwise.
PROMPT_DATE = "26 Sep 2026"

# chain_sum difficulty tiers: more terms, more digits, then negative numbers.
DATASET = "chain_sum"
PILOT_LADDER = [
    dict(min_terms=2, max_terms=2, min_digits=1, max_digits=1),
    dict(min_terms=3, max_terms=3, min_digits=2, max_digits=2),
    dict(min_terms=4, max_terms=4, min_digits=3, max_digits=3),
    dict(min_terms=5, max_terms=5, min_digits=4, max_digits=4, allow_negation=True),
    dict(min_terms=6, max_terms=6, min_digits=5, max_digits=5, allow_negation=True),
]
# Ten finer levels, as (numbers, digits), without negative numbers.
STAIRCASE_LADDER = [
    dict(min_terms=terms, max_terms=terms, min_digits=digits, max_digits=digits)
    for terms, digits in (
        (2, 1),
        (3, 1),
        (3, 2),
        (4, 2),
        (4, 3),
        (5, 3),
        (5, 4),
        (6, 4),
        (6, 5),
        (7, 5),
    )
]
LADDERS: dict[str, list[dict[str, Any]]] = {
    "pilot": PILOT_LADDER,
    "staircase": STAIRCASE_LADDER,
}
SPLIT_SEED_OFFSET = {"train": 0, "test": 500_000_000}

SYSTEM_PROMPT = (
    "Solve the arithmetic problem. Work step by step, briefly, then give only the "
    "final number inside <answer></answer> tags, for example <answer>42</answer>."
)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\{\s*([^{}]*?)\s*\}")
NUMBER_RE = re.compile(r"-?\d[\d,]*")


def extract_answer(completion: str) -> str:
    """Turn a completion into the env action: <answer> tags, then \\boxed{}, then the last number.

    Lenient on purpose, so the experiment measures arithmetic rather than how fast
    the model learns one output format. Strict tag usage is still reported as
    ``format_rate``. The reward itself always comes from the environment.
    """
    for pattern in (ANSWER_RE, BOXED_RE, NUMBER_RE):
        found = pattern.findall(completion)
        if found:
            return found[-1].replace(",", "").strip()
    return ""


def make_tasks(
    split: str, per_tier: int, ladder: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Build the task specs of one split.

    Args:
        split (`str`):
            `"train"` or `"test"`. The two use disjoint seeds.
        per_tier (`int`):
            Number of tasks per level.
        ladder (`list[dict[str, Any]]`):
            The level settings, easiest first.

    Returns:
        `list[dict[str, Any]]`: `per_tier` tasks for each level, level by
        level, each with `id`, `split`, `group`, `difficulty`, `index`, `seed`.
    """
    return [
        {
            "id": f"{DATASET}-{split}-t{tier}-{i}",
            "split": split,
            "group": DATASET,
            "difficulty": tier,
            "index": i,
            "seed": SPLIT_SEED_OFFSET[split] + tier * 1_000_003 + i,
        }
        for tier in range(len(ladder))
        for i in range(per_tier)
    ]


__all__ = [
    "ANSWER_RE",
    "DATASET",
    "LADDERS",
    "PILOT_LADDER",
    "PROMPT_DATE",
    "STAIRCASE_LADDER",
    "SYSTEM_PROMPT",
    "extract_answer",
    "make_tasks",
]
