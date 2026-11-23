# SPDX-License-Identifier: BSD-3-Clause

"""
Ask and score `reasoning_gym_env` tasks for the training loop, through a server
that serves the Task API.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from typing import Any

from openenv.core.harness import ModelStep, ModelStepResult
from openenv.core.llm_client import ToolCall
from reasoning_gym_env import ReasoningGymAction, ReasoningGymEnv

_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
_BOXED = re.compile(r"\\boxed\{\s*([^{}]*?)\s*\}")
_NUMBER = re.compile(r"-?\d[\d,]*")


def answer_from_reply(reply: str) -> str:
    """
    Return the answer a reply gives to a chain-sum question.

    It is the content of the reply's last `<answer>...</answer>` block, else of its
    last `\\boxed{...}`, else its last number, without thousands separators, or an
    empty string when there is none. The rule is lenient on purpose: a small model
    that has not learned to write the tags is still scored on its sum. Training and
    evaluation both read replies with it.

    Args:
        reply (`str`):
            The model's text.

    Returns:
        `str`: The answer to submit.
    """
    for pattern in (_ANSWER, _BOXED, _NUMBER):
        found = pattern.findall(reply)
        if found:
            return found[-1].replace(",", "").strip()
    return ""


def chain_sum_model_step(model_step: ModelStep) -> ModelStep:
    """
    Wrap a model step so that a reply in plain text is submitted as its answer.

    A reply that calls a tool is left unchanged. To any other reply the step adds
    one call to the environment's `answer` tool, with what `answer_from_reply` reads
    in it, so that evaluation reads a reply as training does. The environment scores
    the answer; this step scores nothing.

    Args:
        model_step (`ModelStep`):
            The model step to wrap.

    Returns:
        `ModelStep`: The wrapped step.
    """

    def step(
        messages: list[dict[str, Any]], tools: list[Any], sampling: dict[str, Any]
    ) -> ModelStepResult:
        result = model_step(messages, tools, sampling)
        if result.response.tool_calls:
            return result
        answer = answer_from_reply(result.response.content)
        call = ToolCall(id="text-answer", name="answer", args={"answer": answer})
        return dataclasses.replace(
            result, response=dataclasses.replace(result.response, tool_calls=[call])
        )

    return step


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


__all__ = ["answer_from_reply", "chain_sum_model_step", "ReasoningGymTasks"]
