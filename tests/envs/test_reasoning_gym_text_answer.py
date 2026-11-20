# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the model step that submits a text reply as reasoning_gym's answer."""

from __future__ import annotations

from typing import Any

import pytest
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core.harness import HarnessRunLimits, MCPHarnessAdapter, ModelStepResult
from openenv.core.llm_client import LLMResponse, ToolCall
from reasoning_gym_env.harness import ReasoningGymSessionFactory, text_answer_model_step
from reasoning_gym_env.models import ReasoningGymAction, ReasoningGymObservation


def _replying(response: LLMResponse, calls: list[tuple] | None = None):
    """A model step that always returns `response` and records its arguments."""

    def model_step(messages, tools, sampling):
        if calls is not None:
            calls.append((messages, tools, sampling))
        return ModelStepResult(
            response=response,
            prompt_ids=[1, 2],
            completion_ids=[3, 4, 5],
            logprobs=[-0.1, -0.2, -0.3],
        )

    return model_step


class _FakeReasoningGymClient:
    """Stands in for `ReasoningGymEnv`: one question, then the environment's score."""

    def __init__(self, reward: float):
        self.reward = reward
        self.actions: list[ReasoningGymAction] = []

    def reset(self, **kwargs: Any) -> StepResult[ReasoningGymObservation]:
        return StepResult(
            observation=ReasoningGymObservation(question="What is 40 + 2?"),
            reward=None,
            done=False,
        )

    def step(self, action: ReasoningGymAction) -> StepResult[ReasoningGymObservation]:
        self.actions.append(action)
        return StepResult(
            observation=ReasoningGymObservation(
                score=self.reward, correct_answer="42", done=True, reward=self.reward
            ),
            reward=self.reward,
            done=True,
        )

    def state(self) -> State:
        return State(episode_id="fake", step_count=len(self.actions))

    def close(self) -> None:
        pass


class TestTextAnswerModelStep:
    @pytest.mark.parametrize(
        ("content", "answer"),
        [
            ("I think <answer> 42 </answer>", "42"),
            ("<answer>1</answer> no, <answer>2</answer>", "2"),
            ("<answer>1</answer> then <answer>2", "1"),
            ("<answer>a <answer>b</answer>", "b"),
            ("  42\n", "42"),
            ("<answer>42", "<answer>42"),
            ("", ""),
        ],
    )
    def test_text_reply_becomes_one_answer_call(self, content: str, answer: str):
        step = text_answer_model_step(_replying(LLMResponse(content=content)))

        result = step([{"role": "user", "content": "q"}], [], {})

        assert result.response.tool_calls == [
            ToolCall(id="text-answer", name="answer", args={"answer": answer})
        ]
        assert result.response.content == content

    def test_reply_with_a_tool_call_is_unchanged(self):
        response = LLMResponse(
            content="calling",
            tool_calls=[ToolCall(id="c1", name="answer", args={"answer": "7"})],
        )
        inner = _replying(response)
        expected = inner([], [], {})

        result = text_answer_model_step(inner)([], [], {})

        assert result == expected

    def test_keeps_the_wrapped_step_arguments_and_tokens(self):
        calls: list[tuple] = []
        step = text_answer_model_step(_replying(LLMResponse(content="42"), calls))
        messages = [{"role": "user", "content": "q"}]

        result = step(messages, [], {"temperature": 0.0, "max_tokens": 8})

        assert calls == [(messages, [], {"temperature": 0.0, "max_tokens": 8})]
        assert result.prompt_ids == [1, 2]
        assert result.completion_ids == [3, 4, 5]
        assert result.logprobs == [-0.1, -0.2, -0.3]


class TestTextAnswerInARollout:
    def test_environment_scores_the_text_answer(self):
        client = _FakeReasoningGymClient(reward=0.5)
        factory = ReasoningGymSessionFactory(lambda: client, dataset_name="chain_sum")
        step = text_answer_model_step(
            _replying(LLMResponse(content="The sum is <answer>42</answer>."))
        )

        rollout = MCPHarnessAdapter().run_white_box(
            model_step=step,
            session=factory.create(task=None, seed=0),
            limits=HarnessRunLimits(max_turns=2),
        )

        assert client.actions == [ReasoningGymAction(answer="42")]
        assert rollout.tool_trace[-1].result.done is True
        assert rollout.tool_trace[-1].result.metadata["reward"] == 0.5
        assert rollout.messages[1]["tool_calls"][0]["function"]["name"] == "answer"
