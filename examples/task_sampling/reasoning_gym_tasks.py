# SPDX-License-Identifier: BSD-3-Clause

"""
Ask and score `reasoning_gym_env` tasks for the training loop, through a server
that serves the Task API.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reasoning_gym_env import ReasoningGymAction, ReasoningGymEnv


def answer_from_reply(reply: str) -> str:
    """
    Return the answer a reply gives.

    It is the content of the reply's last complete `<answer>...</answer>` block, or
    the whole reply when there is none, without surrounding whitespace: the rule of
    `reasoning_gym_env.harness.text_answer_model_step`, so that training and
    evaluation read a reply the same way.

    Args:
        reply (`str`):
            The model's text.

    Returns:
        `str`: The answer to submit.
    """
    before, _, _ = reply.rpartition("</answer>")
    _, opening, inside = before.rpartition("<answer>")
    return (inside if opening else reply).strip()


class ReasoningGymTasks:
    """
    The training loop's environment for `reasoning_gym_env` tasks.

    A question and each answer get an episode of their own: `reset(split, index,
    seed)`, then, for an answer, one `step`. The reward is the environment's.

    Args:
        base_url (`str`):
            URL of the environment server.
        seed (`int`, *optional*, defaults to `0`):
            The seed of every reset.
    """

    def __init__(self, base_url: str, *, seed: int = 0) -> None:
        self._client = ReasoningGymEnv(base_url=base_url).sync()
        self._client.connect()
        self._seed = seed

    def _reset(self, task: Mapping[str, Any]) -> Any:
        return self._client.reset(
            split=task["split"], index=task["index"], seed=self._seed
        )

    def question(self, task: Mapping[str, Any]) -> str:
        """Return the question of a task."""
        return self._reset(task).observation.question

    def reward(self, task: Mapping[str, Any], reply: str) -> float | None:
        """Return the environment's reward for a reply to a task."""
        self._reset(task)
        action = ReasoningGymAction(answer=answer_from_reply(reply))
        return self._client.step(action).reward

    def close(self) -> None:
        """Close the connection to the server."""
        self._client.close()


__all__ = ["answer_from_reply", "ReasoningGymTasks"]
