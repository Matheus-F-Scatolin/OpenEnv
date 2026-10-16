# SPDX-License-Identifier: BSD-3-Clause

"""The curriculum sampler: estimates, weights, costs, state."""

from __future__ import annotations

import json

import pytest
from samplers import CurriculumSampler


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

    def test_cost_in_tokens_divides_the_weight(self) -> None:
        sampler = CurriculumSampler(TASKS, group_size=2, cost="tokens")
        for level, tokens in ((0, 10.0), (1, 20.0), (2, 40.0)):
            sampler.update(TASKS[level * 5], [1.0, 0.0], costs=[tokens, tokens])
        probs = list(sampler.probabilities().values())
        assert probs == pytest.approx([4 / 7, 2 / 7, 1 / 7])

    def test_costs_do_not_change_the_draws_when_cost_is_episodes(self) -> None:
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


class TestCostEstimate:
    def test_moving_average(self) -> None:
        sampler = CurriculumSampler(TASKS, group_size=2, cost="tokens")
        sampler.update(TASKS[0], [1.0, 0.0], costs=[10, 10])
        sampler.update(TASKS[0], [1.0, 0.0], costs=[20, 20])
        assert sampler.cost_hat(0) == pytest.approx(11.0)

    def test_an_unseen_level_gets_the_mean_of_the_seen_ones(self) -> None:
        sampler = CurriculumSampler(TASKS, group_size=2, cost="tokens")
        assert sampler.cost_hat(2) == 1.0
        sampler.update(TASKS[0], [1.0, 0.0], costs=[10, 10])
        sampler.update(TASKS[5], [1.0, 0.0], costs=[30, 30])
        assert sampler.cost_hat(2) == pytest.approx(20.0)

    def test_episodes_cost_is_always_one(self) -> None:
        sampler = CurriculumSampler(TASKS, group_size=2)
        sampler.update(TASKS[0], [1.0, 0.0], costs=[10, 10])
        assert sampler.cost_hat(0) == 1.0


def test_levels_keep_the_order_in_which_they_first_appear() -> None:
    tasks = [{"id": f"t{level}", "difficulty": level} for level in range(12)]
    assert CurriculumSampler(tasks, group_size=2).keys == list(range(12))
    assert sorted(range(12), key=str) != list(range(12))
    reverse = CurriculumSampler(tasks[::-1], group_size=2)
    assert reverse.keys == list(range(11, -1, -1))


def test_a_task_without_the_key_is_its_own_level() -> None:
    sampler = CurriculumSampler([{"id": "a"}, {"id": "b"}], group_size=2)
    assert sampler.keys == ["a", "b"]


class TestRejected:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"group_size": 1},
            {"group_size": 8, "decay": 0.0},
            {"group_size": 8, "decay": 1.5},
            {"group_size": 8, "eps": -0.1},
            {"group_size": 8, "cost": "seconds"},
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
        sampler = CurriculumSampler(TASKS, group_size=2, cost="tokens")
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

    @pytest.mark.parametrize("cost", ["episodes", "tokens"])
    def test_round_trip_through_json(self, cost: str) -> None:
        sampler = self._used(cost=cost)
        state = json.loads(json.dumps(sampler.state_dict()))
        restored = CurriculumSampler(TASKS, group_size=2, rng_seed=0, cost=cost)
        restored.load_state_dict(state)
        assert restored.probabilities() == sampler.probabilities()
        assert _ids(restored.sample(50)) == _ids(sampler.sample(50))
        assert restored.state_dict() == sampler.state_dict()

    def test_a_state_saved_with_other_settings_is_refused(self) -> None:
        state = self._used().state_dict()
        for other in (
            CurriculumSampler(TASKS, group_size=4),
            CurriculumSampler(TASKS, group_size=2, cost="tokens"),
            CurriculumSampler(make_tasks(levels=4), group_size=2),
        ):
            before = other.state_dict()
            with pytest.raises(ValueError):
                other.load_state_dict(state)
            assert other.state_dict() == before
