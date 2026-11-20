# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the training loop of `examples/task_sampling`, without MLX."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from openenv.core.evals import EvalConfig, EvalHarness, EvalSchedule
from openenv.core.harness import ModelStepResult
from openenv.core.llm_client import LLMResponse
from openenv.core.task_sampler import SequentialSampler, UniformSampler
from reasoning_gym_env.harness import text_answer_model_step

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "task_sampling"


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


training_loop = _load("training_loop")
reasoning_gym_tasks = _load("reasoning_gym_tasks")
curriculum_sampler = _load("curriculum_sampler")


class FakePolicy:
    """
    Answers each question with the next reply of `replies[question]`, in turn; a
    prompt's ids are the characters of its question, so the question can be read
    back from them.
    """

    def __init__(self, replies: dict[str, list[str]]) -> None:
        self.replies = replies
        self.turn: dict[str, int] = {}
        self.generations: list[tuple[int, int, float]] = []
        self.updates: list[list[tuple[list[int], list[int], float]]] = []

    def encode_prompt(self, question: str) -> list[int]:
        return [ord(char) for char in question]

    def generate(
        self, prompts: list[list[int]], max_tokens: int, temperature: float
    ) -> list[Any]:
        self.generations.append((len(prompts), max_tokens, temperature))
        completions = []
        for prompt in prompts:
            question = "".join(chr(token) for token in prompt)
            turn = self.turn.get(question, 0)
            self.turn[question] = turn + 1
            reply = self.replies[question][turn % len(self.replies[question])]
            completions.append(
                training_loop.Completion(tokens=list(range(len(reply) + 1)), text=reply)
            )
        return completions

    def update(self, samples: list[tuple[list[int], list[int], float]]) -> float:
        self.updates.append(samples)
        return 0.25


class FakeEnv:
    """Rewards computed here, as an environment would; the reply "?" is unscored."""

    def question(self, task: dict[str, Any]) -> str:
        return f"question {task['id']}"

    def reward(self, task: dict[str, Any], reply: str) -> float | None:
        if reply == "?":
            return None
        return float(reply == task["answer"])


class CountingHarness(EvalHarness):
    def __init__(self) -> None:
        self.runs = 0

    def run(
        self,
        harness_version: str,
        library_versions: dict[str, str],
        dataset: str,
        eval_parameters: dict[str, Any],
    ) -> dict[str, Any]:
        self.runs += 1
        return {"run": self.runs}


def tasks(*names: str) -> list[dict[str, Any]]:
    return [
        {"id": name, "answer": "4", "difficulty": i} for i, name in enumerate(names)
    ]


def train(policy: FakePolicy, sampler: Any, **options: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    settings = {
        "steps": 2,
        "prompts_per_step": 3,
        "group_size": 2,
        "max_tokens": 16,
        "temperature": 1.0,
        "log": records.append,
        **options,
    }
    training_loop.train(policy, FakeEnv(), sampler, **settings)
    return records


def advantage(reward: float, rewards: list[float]) -> float:
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    return (reward - mean) / (std + 1e-4)


class TestTrain:
    def test_every_group_updates_the_sampler_and_only_signal_trains(self) -> None:
        policy = FakePolicy(
            {
                "question mixed": ["4", "50"],
                "question solved": ["4"],
                "question failed": ["0"],
            }
        )
        sampler = UniformSampler(tasks("mixed", "solved", "failed"), rng_seed=0)

        records = train(policy, sampler)

        metrics = sampler.metrics()
        assert metrics["groups"] == 6
        assert metrics["episodes"] == 12
        assert metrics["signal_groups"] == 2
        assert metrics["cost"] == 2 * (2 + 3 + 2 + 2 + 2 + 2)
        prompt = policy.encode_prompt("question mixed")
        assert (
            policy.updates
            == [
                [
                    (prompt, [0, 1], advantage(1.0, [1.0, 0.0])),
                    (prompt, [0, 1, 2], advantage(0.0, [1.0, 0.0])),
                ]
            ]
            * 2
        )
        assert policy.generations == [(6, 16, 1.0)] * 2
        assert [record["signal_groups"] for record in records] == [1, 1]
        assert [record["loss"] for record in records] == [0.25, 0.25]

    def test_unscored_rollouts_are_reported_and_not_trained(self) -> None:
        policy = FakePolicy({"question a": ["4", "?", "1"]})
        sampler = UniformSampler(tasks("a"), rng_seed=0)

        records = train(policy, sampler, steps=1, prompts_per_step=1, group_size=3)

        assert records[0]["groups"] == [
            {"task": "a", "rewards": [1.0, None, 0.0], "tokens": [2, 2, 2]}
        ]
        assert sampler.metrics()["episodes"] == 3
        prompt = policy.encode_prompt("question a")
        assert policy.updates == [
            [
                (prompt, [0, 1], advantage(1.0, [1.0, 0.0])),
                (prompt, [0, 1], advantage(0.0, [1.0, 0.0])),
            ]
        ]

    def test_no_step_without_signal(self) -> None:
        policy = FakePolicy({"question a": ["4"], "question b": ["4", "?"]})
        sampler = UniformSampler(tasks("a", "b"), rng_seed=0)

        records = train(policy, sampler, prompts_per_step=2)

        assert policy.updates == []
        assert [record["loss"] for record in records] == [None, None]
        assert sampler.metrics()["groups"] == 4

    def test_log_records(self) -> None:
        policy = FakePolicy({"question a": ["4", "40"]})
        sampler = UniformSampler(tasks("a"), rng_seed=0)

        records = train(policy, sampler, prompts_per_step=1)

        assert [record["step"] for record in records] == [1, 2]
        assert records[0]["generated_tokens"] == 2 + 3
        assert records[0]["mean_reward"] == 0.5
        assert records[1]["sampler"] == sampler.metrics()

    def test_finite_sampler_ends_training(self) -> None:
        policy = FakePolicy({"question a": ["4"], "question b": ["4"]})
        sampler = SequentialSampler(tasks("a", "b"))

        records = train(policy, sampler, steps=5, prompts_per_step=1)

        assert [record["step"] for record in records] == [1, 2]

    def test_curriculum_sampler(self) -> None:
        policy = FakePolicy({"question a": ["4", "1"], "question b": ["1"]})
        sampler = curriculum_sampler.CurriculumSampler(
            tasks("a", "b"), group_size=2, rng_seed=0
        )

        train(policy, sampler, steps=4, prompts_per_step=2)

        assert sampler.metrics()["groups"] == 8


class TestSchedule:
    @pytest.mark.parametrize(
        ("steps", "evaluated"), [(5, [0, 2, 4, 5]), (4, [0, 2, 4]), (1, [0, 1])]
    )
    def test_start_interval_and_last_step(
        self, steps: int, evaluated: list[int]
    ) -> None:
        policy = FakePolicy({"question a": ["4", "1"]})
        sampler = UniformSampler(tasks("a"), rng_seed=0)
        seen: list[tuple[int, dict[str, Any]]] = []
        schedule = EvalSchedule(
            CountingHarness(),
            EvalConfig(
                harness_name="CountingHarness",
                harness_version="0",
                library_versions={},
                dataset="fake",
                eval_parameters={},
            ),
            every_n_steps=2,
            on_result=lambda step, result: seen.append((step, sampler.metrics())),
        )

        train(policy, sampler, steps=steps, prompts_per_step=1, schedule=schedule)

        assert [step for step, _ in seen] == evaluated
        assert seen[0][1]["groups"] == 0
        assert seen[-1][1]["groups"] == steps


class TestReasoningGymTasks:
    @pytest.mark.parametrize(
        "reply",
        [
            "<answer>42</answer>",
            "first <answer> 7 </answer> then <answer>8</answer> done",
            "no tags at all: 13 ",
            "<answer>open but not closed",
            "</answer> closed first <answer>5",
            "",
        ],
    )
    def test_reads_a_reply_as_the_evaluation_does(self, reply: str) -> None:
        step = text_answer_model_step(
            lambda messages, tools, sampling: ModelStepResult(
                response=LLMResponse(content=reply)
            )
        )

        call = step([], [], {}).response.tool_calls[0]

        assert reasoning_gym_tasks.answer_from_reply(reply) == call.args["answer"]

    def test_resets_each_task_and_scores_each_reply(self, monkeypatch) -> None:
        calls: list[tuple[str, Any]] = []

        class FakeClient:
            def __init__(self, base_url: str) -> None:
                calls.append(("init", base_url))

            def sync(self) -> FakeClient:
                return self

            def connect(self) -> FakeClient:
                calls.append(("connect", None))
                return self

            def reset(self, **kwargs: Any) -> SimpleNamespace:
                calls.append(("reset", kwargs))
                return SimpleNamespace(observation=SimpleNamespace(question="1 + 3 ="))

            def step(self, action: Any) -> SimpleNamespace:
                calls.append(("step", action.answer))
                return SimpleNamespace(reward=1.0)

            def close(self) -> None:
                calls.append(("close", None))

        monkeypatch.setattr(reasoning_gym_tasks, "ReasoningGymEnv", FakeClient)
        task = {"id": "t", "split": "train", "index": 7}

        env = reasoning_gym_tasks.ReasoningGymTasks("http://env", seed=3)
        question = env.question(task)
        reward = env.reward(task, "so <answer>4</answer>")
        env.close()

        assert question == "1 + 3 ="
        assert reward == 1.0
        reset = ("reset", {"split": "train", "index": 7, "seed": 3})
        assert calls == [
            ("init", "http://env"),
            ("connect", None),
            reset,
            reset,
            ("step", "4"),
            ("close", None),
        ]
