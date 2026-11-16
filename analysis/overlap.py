# SPDX-License-Identifier: BSD-3-Clause

"""Count train/test question overlap on the reference experiment's task splits."""

from __future__ import annotations

import argparse

from analysis.pilot_evidence.pilot_logs import TEST_PER_TIER, TRAIN_PER_TIER
from trial.env import build_tasks


def overlap(ladder: str) -> list[dict[str, int]]:
    """
    Count overlapping and distinct test questions for each difficulty level.

    Uses the trainer's task builder and environment with the reference
    experiment's split sizes, shared with the pilot's settings.

    Args:
        ladder (`str`):
            Task numbering: `rfc` or `staircase`.

    Returns:
        `list[dict[str, int]]`: Level, overlapping test-task count and distinct
        test-question count, in difficulty order. Repeated test questions count
        once per task in `overlap`, and once per text in `distinct_test`.
    """
    env, train, test = build_tasks(ladder, TRAIN_PER_TIER, TEST_PER_TIER)
    train_questions = {level: set() for level in range(len(env.ladder))}
    test_questions: dict[int, list[str]] = {level: [] for level in train_questions}
    for task in train:
        train_questions[task["difficulty"]].add(env.question(task))
    for task in test:
        test_questions[task["difficulty"]].append(env.question(task))
    return [
        {
            "level": level,
            "overlap": sum(q in train_questions[level] for q in questions),
            "distinct_test": len(set(questions)),
        }
        for level, questions in test_questions.items()
    ]


def main(argv: list[str] | None = None) -> None:
    """
    Print one line per difficulty level without writing any files.

    Args:
        argv (`list[str]`, *optional*):
            Command-line arguments; `sys.argv[1:]` when omitted.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ladder", required=True, choices=("rfc", "staircase"))
    args = parser.parse_args(argv)
    for row in overlap(args.ladder):
        print(
            f"level {row['level']}: overlap={row['overlap']}, "
            f"distinct_test={row['distinct_test']}"
        )


if __name__ == "__main__":
    main()


__all__ = ["main", "overlap"]
