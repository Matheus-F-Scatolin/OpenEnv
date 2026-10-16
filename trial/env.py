# SPDX-License-Identifier: BSD-3-Clause

"""
Run one task per episode on the in-process `reasoning_gym_env`.

The environment is the pilot's vendored copy, imported from `pilot/vendor/`.
The reward is whatever the environment's `step()` returns.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from trial.tasks import DATASET, extract_answer

_VENDOR = str(Path(__file__).resolve().parent.parent / "pilot" / "vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from reasoning_gym_env.models import ReasoningGymAction
from reasoning_gym_env.server.reasoning_gym_environment import ReasoningGymEnvironment


class TaskEnv:
    """
    Asks the environment for a task's question and for the reward of an answer.

    Args:
        ladder (`list[dict[str, Any]]`):
            The level settings; a task's `difficulty` indexes into it.
    """

    def __init__(self, ladder: list[dict[str, Any]]) -> None:
        self.ladder = ladder
        self.env = ReasoningGymEnvironment()
        self._questions: dict[str, str] = {}

    def _reset(self, task: dict[str, Any]) -> Any:
        return self.env.reset(
            dataset_name=DATASET,
            dataset_config=self.ladder[task["difficulty"]],
            seed=task["seed"],
            size=1,
        )

    def question(self, task: dict[str, Any]) -> str:
        """
        Return the question of a task.

        Args:
            task (`dict[str, Any]`):
                A task spec from `make_tasks`.

        Returns:
            `str`: The question text, as the environment's `reset()` gives it.
        """
        if task["id"] not in self._questions:
            self._questions[task["id"]] = self._reset(task).question
        return self._questions[task["id"]]

    def reward(self, task: dict[str, Any], completion: str) -> float:
        """
        Return the environment's reward for a completion.

        Args:
            task (`dict[str, Any]`):
                A task spec from `make_tasks`.
            completion (`str`):
                The text the model wrote.

        Returns:
            `float`: The reward of the environment's `step()`.

        Raises:
            `ValueError`: If the environment returns no reward.
        """
        self._reset(task)
        answer = extract_answer(completion)
        reward = self.env.step(ReasoningGymAction(answer=answer)).reward
        if reward is None:
            raise ValueError("environment returned no reward")
        return float(reward)

    def correct_answer(self, task: dict[str, Any]) -> str | None:
        """
        Return the answer the environment expects for a task.

        Args:
            task (`dict[str, Any]`):
                A task spec from `make_tasks`.

        Returns:
            `str` or `None`: The correct answer, which the environment reveals
            after a step.
        """
        self._reset(task)
        return self.env.step(ReasoningGymAction(answer="")).correct_answer


__all__ = ["TaskEnv"]
