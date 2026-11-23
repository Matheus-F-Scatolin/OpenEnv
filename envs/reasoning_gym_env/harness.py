# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Harness-oriented Reasoning Gym session adapters.

Follows the pattern introduced by ``openspiel_env.harness``: exposes a
ReasoningGymEnv client as a ``ResourceSession`` driven through MCP-style
tools, so it can be consumed by ``openenv.core.harness`` adapters (e.g.
the collect pipeline).
"""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Mapping
from typing import Any, Callable

from openenv.core.env_server.mcp_types import Tool
from openenv.core.harness import (
    ModelStep,
    ModelStepResult,
    ResourceSessionFactory,
    StepEnvSessionAdapter,
    ToolResult,
)
from openenv.core.llm_client import ToolCall

from .client import ReasoningGymEnv
from .models import ReasoningGymAction

_REASONING_GYM_TOOLS: list[Tool] = [
    Tool(
        name="answer",
        description="Submit the final answer for the current question.",
        input_schema={
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The answer to submit.",
                },
            },
            "required": ["answer"],
        },
    ),
]


def _build_tool_result() -> Callable[..., ToolResult]:
    def builder(
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        state: Any,
    ) -> ToolResult:
        observation = result.observation
        return ToolResult(
            data={
                "answer": arguments.get("answer"),
                "score": observation.score,
                "correct_answer": observation.correct_answer,
                "reward": result.reward,
                "done": result.done,
            },
            done=bool(result.done),
            metadata={
                "reward": result.reward,
                "state": state.model_dump() if hasattr(state, "model_dump") else state,
            },
        )

    return builder


def text_answer_model_step(model_step: ModelStep) -> ModelStep:
    """
    Wrap a model step so that a reply in plain text is submitted as the answer.

    The environment scores only a call to the `answer` tool, so a reply without a
    tool call would end the episode without a score. The returned step leaves a reply
    that calls a tool unchanged. To any other reply it adds one `answer` call whose
    argument is the content of the last complete `<answer>...</answer>` block, or the
    whole reply when there is none, without surrounding whitespace. The environment
    then scores that text, an empty one included; this step never scores anything.

    Use it for evaluation, not to collect training data: the transcript then holds a
    tool call that the model did not write.

    Args:
        model_step (`ModelStep`):
            The model step to wrap, for example `build_model_step(OpenAIClient(...))`.

    Returns:
        `ModelStep`: A model step that returns the wrapped step's result, with the
        `answer` call added to a reply that has no tool call. Every other field of the
        result and of its response is kept.

    Examples:

    ```python
    from openenv.core.harness.collect import build_model_step
    from openenv.core.llm_client import OpenAIClient

    model_step = text_answer_model_step(
        build_model_step(OpenAIClient("http://localhost:8000/v1", None, model="m"))
    )
    ```
    """

    def step(
        messages: list[dict[str, Any]],
        tools: list[Tool],
        sampling: dict[str, Any],
    ) -> ModelStepResult:
        result = model_step(messages, tools, sampling)
        if result.response.tool_calls:
            return result
        content = result.response.content
        before, _, _ = content.rpartition("</answer>")
        _, opening, inside = before.rpartition("<answer>")
        answer = (inside if opening else content).strip()
        call = ToolCall(id="text-answer", name="answer", args={"answer": answer})
        return dataclasses.replace(
            result, response=dataclasses.replace(result.response, tool_calls=[call])
        )

    return step


class ReasoningGymSessionFactory(ResourceSessionFactory):
    """Create ReasoningGym-backed resource sessions for harness rollouts."""

    def __init__(
        self,
        client_factory: Callable[[], ReasoningGymEnv],
        *,
        dataset_name: str,
        dataset_config: dict[str, Any] | None = None,
    ):
        self._client_factory = client_factory
        self._dataset_name = dataset_name
        self._dataset_config = dataset_config or {}

    def create(
        self,
        task: Any = None,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> StepEnvSessionAdapter:
        """
        Start a session for one episode.

        A task spec with `split` and `index`, as the environment's Task API serves
        it, is started with `reset(split=..., index=..., seed=...)`. Any other task
        starts a one-question dataset of the factory's dataset, with a random seed
        when none is given.

        Args:
            task (`Any`, *optional*):
                The task spec.
            seed (`int`, *optional*):
                Seed of the reset.
            episode_id (`str`, *optional*):
                Id of the episode.

        Returns:
            `StepEnvSessionAdapter`: The session.
        """
        client = self._client_factory()
        if isinstance(task, Mapping) and "split" in task and "index" in task:
            reset_kwargs = {"split": task["split"], "index": task["index"]}
            effective_seed = seed
        else:
            reset_kwargs = {
                "dataset_name": self._dataset_name,
                "dataset_config": self._dataset_config,
                "size": 1,
            }
            effective_seed = seed if seed is not None else random.randint(0, 2**31 - 1)

        return StepEnvSessionAdapter(
            client=client,
            task=task,
            seed=effective_seed,
            episode_id=episode_id,
            tool_specs=list(_REASONING_GYM_TOOLS),
            action_builder=lambda name, arguments: ReasoningGymAction(
                answer=str(arguments["answer"]),
            ),
            initial_messages_builder=lambda result, current_task: [
                {
                    "role": "user",
                    "content": result.observation.question,
                }
            ],
            tool_result_builder=_build_tool_result(),
            reset_kwargs=reset_kwargs,
        )


__all__ = [
    "ReasoningGymSessionFactory",
    "text_answer_model_step",
]
