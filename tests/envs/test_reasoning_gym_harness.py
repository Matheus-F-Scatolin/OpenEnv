# SPDX-License-Identifier: BSD-3-Clause

"""Task selection through the Reasoning Gym session factory."""

from __future__ import annotations

from typing import Any

import pytest
from openenv.core.client_types import StepResult
from reasoning_gym_env.harness import ReasoningGymSessionFactory
from reasoning_gym_env.models import ReasoningGymObservation


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def reset(self, **kwargs: Any) -> StepResult[ReasoningGymObservation]:
        self.calls.append(kwargs)
        return StepResult(
            observation=ReasoningGymObservation(question="Question"),
            reward=0.0,
            done=False,
        )

    def close(self) -> None:
        pass


class TestTaskSession:
    @pytest.mark.parametrize("seed", [None, 0, 7])
    @pytest.mark.parametrize("episode_id", [None, "e"])
    def test_task_controls_reset(
        self, seed: int | None, episode_id: str | None
    ) -> None:
        client = FakeClient()
        factory = ReasoningGymSessionFactory(
            lambda: client, dataset_name="leg_counting"
        )
        session = factory.create(
            task={"split": "test", "index": 3}, seed=seed, episode_id=episode_id
        )
        try:
            expected = {"split": "test", "index": 3}
            if seed is not None:
                expected["seed"] = seed
            if episode_id is not None:
                expected["episode_id"] = episode_id
            assert client.calls == [expected]
        finally:
            session.close()

    def test_without_task_keeps_dataset_options(self) -> None:
        client = FakeClient()
        factory = ReasoningGymSessionFactory(
            lambda: client,
            dataset_name="leg_counting",
            dataset_config={"min_animals": 5},
        )
        session = factory.create(seed=7)
        session.close()
        assert client.calls == [
            {
                "dataset_name": "leg_counting",
                "dataset_config": {"min_animals": 5},
                "size": 1,
                "seed": 7,
            }
        ]

    def test_without_task_still_chooses_seed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "reasoning_gym_env.harness.random.randint", lambda a, b: 123
        )
        client = FakeClient()
        session = ReasoningGymSessionFactory(
            lambda: client, dataset_name="chain_sum"
        ).create()
        session.close()
        assert client.calls[0]["seed"] == 123
