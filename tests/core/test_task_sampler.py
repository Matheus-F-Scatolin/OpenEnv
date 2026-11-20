# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the task samplers of `openenv.core.task_sampler`."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any, get_type_hints

import pytest
from openenv.core.task_sampler import (
    _has_signal,
    SequentialSampler,
    TaskSampler,
    UniformSampler,
)

TASKS = [{"id": i, "difficulty": i % 2} for i in range(4)]
TEN = [{"id": i} for i in range(10)]

EMPTY_METRICS = {
    "groups": 0,
    "signal_groups": 0,
    "episodes": 0,
    "signal_groups_per_episode": 0.0,
    "zero_signal_fraction": 0.0,
}


def _ids(tasks: list[dict]) -> list[int]:
    return [task["id"] for task in tasks]


class TestHasSignal:
    @pytest.mark.parametrize(
        "rewards", [[], [1.0], [0.0] * 8, [1.0] * 8, [0.3] * 8, [0.5, 0.5 + 1e-9]]
    )
    def test_equal_rewards_have_no_signal(self, rewards: list[float]) -> None:
        assert not _has_signal(rewards)

    @pytest.mark.parametrize(
        "rewards", [[0.0, 1.0], [1.0] * 7 + [0.0], [0.0] * 7 + [0.01], [0.5, 1.0]]
    )
    def test_mixed_rewards_have_signal(self, rewards: list[float]) -> None:
        assert _has_signal(rewards)


class TestCounters:
    def test_a_new_sampler_counts_nothing(self) -> None:
        assert UniformSampler(TASKS).metrics() == EMPTY_METRICS

    def test_counts_groups_episodes_and_reported_cost(self) -> None:
        sampler = UniformSampler(TASKS)
        sampler.update(TASKS[0], [1.0, 0.0], costs=[10, 30])
        sampler.update(TASKS[1], [1.0, 1.0], costs=[5, 5])
        sampler.update(TASKS[2], [0.0, 0.5], costs=[5, 5])
        metrics = sampler.metrics()
        assert (metrics["groups"], metrics["signal_groups"], metrics["episodes"]) == (
            3,
            2,
            6,
        )
        assert metrics["cost"] == 60.0
        assert metrics["signal_groups_per_episode"] == 2 / 6
        assert metrics["signal_groups_per_cost"] == 2 / 60
        assert metrics["zero_signal_fraction"] == pytest.approx(1 / 3)

    @pytest.mark.parametrize(
        "rewards,costs", [([], None), ([1.0, 0.0], [1.0]), ([1.0, 0.0], [1.0, -1.0])]
    )
    def test_a_rejected_update_changes_nothing(
        self, rewards: list[float], costs: list[float] | None
    ) -> None:
        sampler = UniformSampler(TASKS)
        before = sampler.state_dict()
        with pytest.raises(ValueError):
            sampler.update(TASKS[0], rewards, costs=costs)
        assert sampler.state_dict() == before


class _FullSampler:
    def sample(self, n: int) -> list[Any]:
        return []

    def update(
        self,
        task: Any,
        rewards: Sequence[float | None],
        costs: Sequence[float] | None = None,
    ) -> None:
        pass

    def metrics(self) -> dict[str, float]:
        return {}

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        pass


class _NoMetricsSampler:
    def sample(self, n: int) -> list[Any]:
        return []

    def update(
        self,
        task: Any,
        rewards: Sequence[float | None],
        costs: Sequence[float] | None = None,
    ) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        pass


class TestProtocol:
    def test_a_class_with_the_five_methods_is_a_task_sampler(self) -> None:
        assert isinstance(_FullSampler(), TaskSampler)

    def test_a_class_without_metrics_is_not(self) -> None:
        assert not isinstance(_NoMetricsSampler(), TaskSampler)

    @pytest.mark.parametrize("sampler_type", [UniformSampler, SequentialSampler])
    def test_the_samplers_satisfy_the_protocol(self, sampler_type: type) -> None:
        assert isinstance(sampler_type(TASKS), TaskSampler)


class TestUniformSampler:
    def test_every_task_once_per_pass_and_a_new_shuffle_after(self) -> None:
        sampler = UniformSampler(TEN, rng_seed=3)
        first, second = _ids(sampler.sample(10)), _ids(sampler.sample(10))
        assert sorted(first) == sorted(second) == list(range(10))
        assert first != second

    def test_the_seed_decides_the_order(self) -> None:
        a = _ids(UniformSampler(TEN, rng_seed=1).sample(30))
        assert a == _ids(UniformSampler(TEN, rng_seed=1).sample(30))
        assert a != _ids(UniformSampler(TEN, rng_seed=2).sample(30))

    def test_the_draws_of_a_seed_do_not_change(self) -> None:
        # A run repeated with the same seed must see the same tasks.
        draws = _ids(UniformSampler(TEN, rng_seed=0).sample(12))
        assert draws == [6, 9, 0, 2, 4, 3, 5, 1, 8, 7, 5, 3]

    def test_save_in_the_middle_of_a_pass_and_restore(self) -> None:
        sampler = UniformSampler(TEN, rng_seed=5)
        sampler.sample(7)
        sampler.update(TEN[0], [1.0, 0.0], costs=[3, 4])
        state = json.loads(json.dumps(sampler.state_dict()))
        restored = UniformSampler(TEN, rng_seed=99)
        restored.load_state_dict(state)
        assert _ids(restored.sample(40)) == _ids(sampler.sample(40))
        assert restored.state_dict() == sampler.state_dict()

    def test_bad_arguments_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            UniformSampler([])
        sampler = UniformSampler(TEN)
        with pytest.raises(ValueError):
            sampler.sample(-1)
        assert sampler.sample(0) == []
        with pytest.raises(ValueError):
            UniformSampler(TEN[:3]).load_state_dict(sampler.state_dict())


class TestContract:
    @pytest.mark.parametrize("sampler_type", [UniformSampler, SequentialSampler])
    def test_cost_annotation_matches_protocol(self, sampler_type: type) -> None:
        assert (
            get_type_hints(sampler_type.update)["costs"]
            == get_type_hints(TaskSampler.update)["costs"]
        )

    @pytest.mark.parametrize("factory", [UniformSampler, SequentialSampler])
    def test_metrics_and_unscored_accounting(
        self, factory: Callable[[list[dict]], TaskSampler]
    ) -> None:
        sampler = factory(TASKS)
        assert sampler.metrics() == EMPTY_METRICS
        sampler.update(TASKS[0], [1.0, None, 0.0], [1, 2, 3])
        sampler.update(TASKS[0], [None, None], [4, 5])
        assert sampler.metrics() == {
            "groups": 2,
            "signal_groups": 1,
            "episodes": 5,
            "cost": 15.0,
            "signal_groups_per_episode": 0.2,
            "zero_signal_fraction": 0.5,
            "signal_groups_per_cost": 1 / 15,
        }

    @pytest.mark.parametrize("factory", [UniformSampler, SequentialSampler])
    def test_a_group_with_one_scored_reward_has_no_signal(
        self, factory: Callable[[list[dict]], TaskSampler]
    ) -> None:
        sampler = factory(TASKS)
        sampler.update(TASKS[0], [1.0, None])
        sampler.update(TASKS[0], [None, 0.0, None])
        assert sampler.metrics()["signal_groups"] == 0

    @pytest.mark.parametrize("first", [None, [0.0, 0.0]])
    def test_cost_rule_survives_json_roundtrip(self, first: list[float] | None) -> None:
        sampler = UniformSampler(TASKS)
        sampler.update(TASKS[0], [None, None], first)
        restored = UniformSampler(TASKS)
        restored.load_state_dict(json.loads(json.dumps(sampler.state_dict())))
        before = restored.state_dict()
        with pytest.raises(ValueError):
            restored.update(TASKS[0], [None, None], [1, 2] if first is None else None)
        assert restored.state_dict() == before
        if first is not None:
            assert restored.metrics()["cost"] == 0
            assert "signal_groups_per_cost" not in restored.metrics()


class TestSequential:
    @pytest.mark.parametrize("seeds", [None, [7, 3]])
    def test_order_and_resume(self, seeds: list[int] | None) -> None:
        sampler = SequentialSampler(TASKS, seeds=seeds)
        expected = TASKS if seeds is None else [(t, s) for t in TASKS for s in seeds]
        assert sampler.sample(1) == expected[:1]
        restored = SequentialSampler(TASKS, seeds=seeds)
        restored.load_state_dict(json.loads(json.dumps(sampler.state_dict())))
        assert restored.sample(100) == expected[1:]
        assert restored.sample(2) == []
        assert restored.sample(0) == []
        with pytest.raises(ValueError):
            restored.sample(-1)

    def test_empty_seeds_are_refused(self) -> None:
        with pytest.raises(ValueError):
            SequentialSampler(TASKS, seeds=[])


class TestSavedState:
    def test_another_version_is_refused(self) -> None:
        state = UniformSampler(TASKS).state_dict()
        state["version"] = 2
        sampler = UniformSampler(TASKS, rng_seed=4)
        before = sampler.state_dict()
        with pytest.raises(ValueError, match="version 1"):
            sampler.load_state_dict(state)
        assert sampler.state_dict() == before

    def test_another_kind_is_refused(self) -> None:
        state = SequentialSampler(TASKS).state_dict()
        sampler = UniformSampler(TASKS)
        before = sampler.state_dict()
        with pytest.raises(ValueError):
            sampler.load_state_dict(state)
        assert sampler.state_dict() == before

    def test_other_seeds_are_refused(self) -> None:
        state = SequentialSampler(TASKS, seeds=[7, 3]).state_dict()
        sampler = SequentialSampler(TASKS)
        before = sampler.state_dict()
        with pytest.raises(ValueError):
            sampler.load_state_dict(state)
        assert sampler.state_dict() == before
