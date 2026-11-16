# SPDX-License-Identifier: BSD-3-Clause

"""The sampler contract for scored and unscored groups."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import get_type_hints

import pytest
from samplers import CurriculumSampler, TaskSampler, UniformSampler
from samplers.base import CountingSampler
from samplers.simple import SequentialSampler

TASKS = [{"id": i, "difficulty": i % 2} for i in range(4)]


class TestReferenceContract:
    @pytest.mark.parametrize(
        "sampler_type", [UniformSampler, SequentialSampler, CurriculumSampler]
    )
    def test_cost_annotation_matches_protocol(
        self, sampler_type: type[TaskSampler]
    ) -> None:
        assert (
            get_type_hints(sampler_type.update)["costs"]
            == get_type_hints(TaskSampler.update)["costs"]
        )

    @pytest.mark.parametrize(
        "factory",
        [
            UniformSampler,
            SequentialSampler,
            lambda t: CurriculumSampler(t, group_size=4),
        ],
    )
    def test_metrics_and_unscored_accounting(
        self, factory: Callable[[list[dict]], CountingSampler]
    ) -> None:
        sampler = factory(TASKS)
        assert isinstance(sampler, TaskSampler)
        assert sampler.metrics() == dict(
            groups=0,
            signal_groups=0,
            episodes=0,
            signal_groups_per_episode=0.0,
            zero_signal_fraction=0.0,
        )
        sampler.update(TASKS[0], [1.0, None, 0.0], [1, 2, 3])
        sampler.update(TASKS[0], [None, None], [4, 5])
        assert sampler.metrics() == dict(
            groups=2,
            signal_groups=1,
            episodes=5,
            cost=15.0,
            signal_groups_per_episode=0.2,
            zero_signal_fraction=0.5,
            signal_groups_per_cost=1 / 15,
        )

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

    @pytest.mark.parametrize("cost", ["episodes", "tokens"])
    def test_missing_rewards_do_not_decay_successes(self, cost: str) -> None:
        a = CurriculumSampler(TASKS, group_size=4, cost=cost)
        b = CurriculumSampler(TASKS, group_size=4, cost=cost)
        a.update(TASKS[0], [1.0, None, 0.0], [2, 4, 6])
        b.update(TASKS[0], [1.0, 0.0], [4, 4])
        assert a.p_hat(0) == b.p_hat(0)
        before = a.p_hat(0)
        a.update(TASKS[0], [None, None], [8, 8])
        assert a.p_hat(0) == before
        if cost == "tokens":
            assert a.cost_hat(0) == pytest.approx(4.4)

    def test_id_key_and_version(self) -> None:
        a = CurriculumSampler([{"difficulty": 2}], group_size=4)
        assert a.keys == [2]
        b = CurriculumSampler([{"task_id": "a"}], group_size=4, id_key="task_id")
        assert b.keys == ["a"]
        state = b.state_dict()
        assert state["settings"]["id_key"] == "task_id"
        state["version"] = 1
        with pytest.raises(ValueError, match="version 2"):
            b.load_state_dict(state)

    def test_token_cost_zero_is_rejected_without_mutation(self) -> None:
        a = CurriculumSampler(TASKS, group_size=4, cost="tokens")
        before = a.state_dict()
        with pytest.raises(ValueError):
            a.update(TASKS[0], [None, None], [0, 1])
        assert a.state_dict() == before


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
