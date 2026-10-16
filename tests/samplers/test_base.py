# SPDX-License-Identifier: BSD-3-Clause

"""The shared pieces: the signal test and the counters."""

from __future__ import annotations

import pytest
from samplers import has_signal, TaskSampler, UniformSampler

TASKS = [{"id": i} for i in range(4)]


class TestHasSignal:
    @pytest.mark.parametrize(
        "rewards", [[], [1.0], [0.0] * 8, [1.0] * 8, [0.3] * 8, [0.5, 0.5 + 1e-9]]
    )
    def test_equal_rewards_have_no_signal(self, rewards: list[float]) -> None:
        assert not has_signal(rewards)

    @pytest.mark.parametrize(
        "rewards", [[0.0, 1.0], [1.0] * 7 + [0.0], [0.0] * 7 + [0.01], [0.5, 1.0]]
    )
    def test_mixed_rewards_have_signal(self, rewards: list[float]) -> None:
        assert has_signal(rewards)


class TestCounters:
    def test_a_new_sampler_counts_nothing(self) -> None:
        sampler = UniformSampler(TASKS)
        assert (sampler.groups, sampler.signal_groups, sampler.episodes) == (0, 0, 0)
        assert sampler.cost == 0.0
        assert sampler.signal_groups_per_episode == 0.0
        assert sampler.signal_groups_per_cost == 0.0
        assert sampler.zero_signal_fraction == 0.0

    def test_counts_groups_episodes_and_reported_cost(self) -> None:
        sampler = UniformSampler(TASKS)
        sampler.update(TASKS[0], [1.0, 0.0], costs=[10, 30])
        sampler.update(TASKS[1], [1.0, 1.0], costs=[5, 5])
        sampler.update(TASKS[2], [0.0, 0.5])
        assert (sampler.groups, sampler.signal_groups, sampler.episodes) == (3, 2, 6)
        assert sampler.cost == 50.0
        assert sampler.signal_groups_per_episode == 2 / 6
        assert sampler.signal_groups_per_cost == 2 / 50
        assert sampler.zero_signal_fraction == pytest.approx(1 / 3)

    @pytest.mark.parametrize(
        "rewards,costs", [([], None), ([1.0, 0.0], [1.0]), ([1.0, 0.0], [1.0, 0.0])]
    )
    def test_a_rejected_update_changes_nothing(
        self, rewards: list[float], costs: list[float] | None
    ) -> None:
        sampler = UniformSampler(TASKS)
        before = sampler.state_dict()
        with pytest.raises(ValueError):
            sampler.update(TASKS[0], rewards, costs=costs)
        assert sampler.state_dict() == before


def test_the_samplers_satisfy_the_protocol() -> None:
    assert isinstance(UniformSampler(TASKS), TaskSampler)
