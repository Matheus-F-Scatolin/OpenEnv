# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the curriculum task sampler of `examples/task_sampling/`."""

from __future__ import annotations

import importlib.util
import json
import math
import random
from pathlib import Path
from typing import get_type_hints

import pytest
from openenv.core.task_sampler import TaskSampler, UniformSampler

EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "task_sampling"
    / "curriculum_sampler.py"
)
SPEC = importlib.util.spec_from_file_location("curriculum_sampler", EXAMPLE)
curriculum_sampler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(curriculum_sampler)
CurriculumSampler = curriculum_sampler.CurriculumSampler


def make_tasks(levels: int = 3, per_level: int = 5) -> list[dict]:
    return [
        {"id": f"t{level}-{i}", "difficulty": level}
        for level in range(levels)
        for i in range(per_level)
    ]


TASKS = make_tasks()


def _ids(tasks: list[dict]) -> list[str]:
    return [task["id"] for task in tasks]


class TestSuccessEstimate:
    def test_starts_at_one_half(self) -> None:
        sampler = CurriculumSampler(TASKS, group_size=8)
        assert [sampler.p_hat(k) for k in sampler.keys] == [0.5, 0.5, 0.5]

    def test_counts_successes_with_a_prior_of_one_each(self) -> None:
        sampler = CurriculumSampler(TASKS, group_size=4)
        sampler.update(TASKS[0], [1.0, 1.0, 1.0, 0.0])
        assert sampler.p_hat(0) == (3 + 1) / (4 + 2)

    def test_old_groups_decay(self) -> None:
        sampler = CurriculumSampler(TASKS, group_size=2, decay=0.5)
        sampler.update(TASKS[0], [1.0, 1.0])
        sampler.update(TASKS[0], [0.0, 0.0])
        assert sampler.p_hat(0) == (0.5 * 2 + 1) / (0.5 * 2 + 2 + 2)

    def test_partial_credit_is_not_a_success(self) -> None:
        sampler = CurriculumSampler(TASKS, group_size=2)
        sampler.update(TASKS[0], [0.99, 0.5])
        assert sampler.p_hat(0) == 1 / 4
        lenient = CurriculumSampler(TASKS, group_size=2, success_threshold=0.5)
        lenient.update(TASKS[0], [0.99, 0.5])
        assert lenient.p_hat(0) == 3 / 4


class TestWeights:
    def test_the_formula(self) -> None:
        sampler = CurriculumSampler(TASKS, group_size=2, eps=0.05)
        sampler.update(TASKS[0], [1.0, 1.0])
        p = sampler.p_hat(0)
        weights = [1 - p**2 - (1 - p) ** 2 + 0.05, 0.5 + 0.05, 0.5 + 0.05]
        expected = [w / sum(weights) for w in weights]
        assert list(sampler.probabilities().values()) == pytest.approx(expected)

    def test_costs_do_not_change_the_draws(self) -> None:
        plain = CurriculumSampler(TASKS, group_size=2, rng_seed=4)
        with_costs = CurriculumSampler(TASKS, group_size=2, rng_seed=4)
        for _ in range(20):
            batch = plain.sample(4)
            assert _ids(batch) == _ids(with_costs.sample(4))
            for i, task in enumerate(batch):
                rewards = [float(i % 2), 1.0]
                plain.update(task, rewards)
                with_costs.update(task, rewards, costs=[7.0 * (i + 1), 90.0])
        assert plain.probabilities() == with_costs.probabilities()


class TestLevels:
    def test_levels_keep_the_order_in_which_they_first_appear(self) -> None:
        tasks = [{"id": f"t{level}", "difficulty": level} for level in range(12)]
        assert CurriculumSampler(tasks, group_size=2).keys == list(range(12))
        assert sorted(range(12), key=str) != list(range(12))
        reverse = CurriculumSampler(tasks[::-1], group_size=2)
        assert reverse.keys == list(range(11, -1, -1))

    def test_a_task_without_the_key_is_its_own_level(self) -> None:
        sampler = CurriculumSampler([{"id": "a"}, {"id": "b"}], group_size=2)
        assert sampler.keys == ["a", "b"]

    def test_tasks_of_a_level_are_drawn_without_replacement(self) -> None:
        tasks = [{"id": f"t{i}", "difficulty": 0} for i in range(5)]
        sampler = CurriculumSampler(tasks, group_size=2)
        everything = sorted(task["id"] for task in tasks)
        assert sorted(_ids(sampler.sample(5))) == everything
        assert sorted(_ids(sampler.sample(5))) == everything

    def test_the_draws_of_a_seed_do_not_change(self) -> None:
        # A run repeated with the same seed must see the same tasks.
        draws = _ids(CurriculumSampler(TASKS, group_size=2, rng_seed=0).sample(10))
        assert draws == [
            "t2-1",
            "t2-4",
            "t1-0",
            "t0-2",
            "t1-2",
            "t1-3",
            "t2-3",
            "t0-0",
            "t1-4",
            "t1-1",
        ]


class TestRejected:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"group_size": 1},
            {"group_size": 8, "decay": 0.0},
            {"group_size": 8, "decay": 1.5},
            {"group_size": 8, "eps": -0.1},
        ],
    )
    def test_constructor(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            CurriculumSampler(TASKS, **kwargs)

    @pytest.mark.parametrize(
        "task,rewards,costs",
        [
            (TASKS[0], [1.0, 0.0], None),
            (TASKS[0], [1.0, 0.0], [5.0]),
            (TASKS[0], [1.0, 0.0], [5.0, -1.0]),
            ({"id": "x", "difficulty": 9}, [1.0, 0.0], [5.0, 5.0]),
        ],
    )
    def test_update_changes_nothing(
        self, task: dict, rewards: list[float], costs: list[float] | None
    ) -> None:
        sampler = CurriculumSampler(TASKS, group_size=2)
        sampler.update(TASKS[0], [1.0, 0.0], costs=[3.0, 3.0])
        before = sampler.state_dict()
        with pytest.raises(ValueError):
            sampler.update(task, rewards, costs=costs)
        assert sampler.state_dict() == before

    def test_negative_n(self) -> None:
        with pytest.raises(ValueError):
            CurriculumSampler(TASKS, group_size=2).sample(-1)


class TestState:
    def _used(self, **kwargs) -> CurriculumSampler:
        sampler = CurriculumSampler(TASKS, group_size=2, rng_seed=7, **kwargs)
        for step in range(15):
            for i, task in enumerate(sampler.sample(4)):
                sampler.update(task, [float((step + i) % 2), 1.0], costs=[9.0, 4.0 + i])
        return sampler

    def test_round_trip_through_json(self) -> None:
        sampler = self._used()
        state = json.loads(json.dumps(sampler.state_dict()))
        restored = CurriculumSampler(TASKS, group_size=2, rng_seed=0)
        restored.load_state_dict(state)
        assert restored.probabilities() == sampler.probabilities()
        assert _ids(restored.sample(50)) == _ids(sampler.sample(50))
        assert restored.state_dict() == sampler.state_dict()

    def test_a_state_saved_with_other_settings_is_refused(self) -> None:
        state = self._used().state_dict()
        for other in (
            CurriculumSampler(TASKS, group_size=4),
            CurriculumSampler(TASKS, group_size=2, decay=0.5),
            CurriculumSampler(make_tasks(levels=4), group_size=2),
        ):
            before = other.state_dict()
            with pytest.raises(ValueError):
                other.load_state_dict(state)
            assert other.state_dict() == before

    def test_a_state_saved_with_other_levels_is_refused(self) -> None:
        state = self._used().state_dict()
        moved = [{**task, "difficulty": task["difficulty"] + 5} for task in TASKS]
        other = CurriculumSampler(moved, group_size=2)
        before = other.state_dict()
        with pytest.raises(ValueError):
            other.load_state_dict(state)
        assert other.state_dict() == before


class TestContract:
    CONTRACT_TASKS = [{"id": i, "difficulty": i % 2} for i in range(4)]

    def test_cost_annotation_matches_protocol(self) -> None:
        assert (
            get_type_hints(CurriculumSampler.update)["costs"]
            == get_type_hints(TaskSampler.update)["costs"]
        )

    def test_metrics_and_unscored_accounting(self) -> None:
        tasks = self.CONTRACT_TASKS
        sampler = CurriculumSampler(tasks, group_size=4)
        assert isinstance(sampler, TaskSampler)
        assert sampler.metrics() == {
            "groups": 0,
            "signal_groups": 0,
            "episodes": 0,
            "signal_groups_per_episode": 0.0,
            "zero_signal_fraction": 0.0,
        }
        sampler.update(tasks[0], [1.0, None, 0.0], [1, 2, 3])
        sampler.update(tasks[0], [None, None], [4, 5])
        assert sampler.metrics() == {
            "groups": 2,
            "signal_groups": 1,
            "episodes": 5,
            "cost": 15.0,
            "signal_groups_per_episode": 0.2,
            "zero_signal_fraction": 0.5,
            "signal_groups_per_cost": 1 / 15,
        }

    def test_missing_rewards_do_not_decay_successes(self) -> None:
        tasks = self.CONTRACT_TASKS
        a = CurriculumSampler(tasks, group_size=4)
        b = CurriculumSampler(tasks, group_size=4)
        a.update(tasks[0], [1.0, None, 0.0], [2, 4, 6])
        b.update(tasks[0], [1.0, 0.0], [4, 4])
        assert a.p_hat(0) == b.p_hat(0)
        before = a.p_hat(0)
        a.update(tasks[0], [None, None], [8, 8])
        assert a.p_hat(0) == before

    def test_an_unscored_group_does_not_decay_the_estimate(self) -> None:
        tasks = self.CONTRACT_TASKS
        sampler = CurriculumSampler(tasks, group_size=4)
        sampler.update(tasks[0], [1.0, 1.0, 0.0])
        before = sampler.p_hat(0)
        sampler.update(tasks[0], [None, None, None])
        assert sampler.p_hat(0) == before

    def test_id_key_and_version(self) -> None:
        a = CurriculumSampler([{"difficulty": 2}], group_size=4)
        assert a.keys == [2]
        b = CurriculumSampler([{"task_id": "a"}], group_size=4, id_key="task_id")
        assert b.keys == ["a"]
        state = b.state_dict()
        assert state["settings"]["id_key"] == "task_id"
        state["version"] = 2
        with pytest.raises(ValueError, match="version 1"):
            b.load_state_dict(state)


# A closed loop with a simulated policy. The policy has one skill number. A task
# of level k succeeds with a chance that falls as k rises above the skill; a
# group with mixed rewards raises the skill a little, more when the level is
# close to it; answers get longer with the level.

LEVELS, GROUP, PER_LEVEL = 8, 8, 200
SIM_TASKS = [
    {"id": f"t{level}-{i}", "difficulty": level}
    for level in range(LEVELS)
    for i in range(PER_LEVEL)
]


class SimulatedPolicy:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.skill = 1.5

    def rollout(self, task: dict) -> tuple[list[float], list[float]]:
        chance = 1 / (1 + math.exp(-2.5 * (self.skill - task["difficulty"])))
        rewards = [float(self.rng.random() < chance) for _ in range(GROUP)]
        length = 10.0 + 12.0 * task["difficulty"]
        costs = [max(1.0, self.rng.gauss(length, 0.2 * length)) for _ in range(GROUP)]
        return rewards, costs

    def learn(self, task: dict, rewards: list[float]) -> None:
        if len(set(rewards)) > 1:
            self.skill += 0.004 * math.exp(-abs(task["difficulty"] - self.skill))


def run(sampler: TaskSampler, seed: int, episodes: int) -> float:
    policy = SimulatedPolicy(seed)
    while sampler.metrics()["episodes"] < episodes:
        for task in sampler.sample(8):
            rewards, costs = policy.rollout(task)
            sampler.update(task, rewards, costs=costs)
            policy.learn(task, rewards)
    return policy.skill


class TestSimulatedPolicy:
    @pytest.mark.parametrize("seed", range(5))
    def test_the_curriculum_finds_more_groups_with_signal(self, seed: int) -> None:
        uniform = UniformSampler(SIM_TASKS, rng_seed=seed)
        curriculum = CurriculumSampler(SIM_TASKS, group_size=GROUP, rng_seed=seed)
        skill_uniform = run(uniform, seed, 20_000)
        skill_curriculum = run(curriculum, seed, 20_000)
        u, c = uniform.metrics(), curriculum.metrics()
        assert c["signal_groups_per_episode"] > u["signal_groups_per_episode"]
        assert c["signal_groups_per_cost"] > u["signal_groups_per_cost"]
        assert skill_curriculum > skill_uniform

    @pytest.mark.parametrize("seed", range(5))
    def test_the_curriculum_follows_the_skill(self, seed: int) -> None:
        sampler = CurriculumSampler(SIM_TASKS, group_size=GROUP, rng_seed=seed)
        skill = run(sampler, seed, 20_000)
        probabilities = sampler.probabilities()
        favourite = max(probabilities, key=probabilities.get)
        assert abs(favourite - skill) <= 1.0
