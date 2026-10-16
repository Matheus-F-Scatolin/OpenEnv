# SPDX-License-Identifier: BSD-3-Clause

"""
A closed loop with a simulated policy, as the RFC's rollout plan asks.

The policy has one skill number. A task of level k succeeds with a chance that
falls as k rises above the skill; a group with mixed rewards raises the skill a
little, more when the level is close to it; answers get longer with the level.
"""

from __future__ import annotations

import math
import random

import pytest
from samplers import CurriculumSampler, TaskSampler, UniformSampler

LEVELS, GROUP, PER_LEVEL = 8, 8, 200
TASKS = [
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
    while sampler.episodes < episodes:
        for task in sampler.sample(8):
            rewards, costs = policy.rollout(task)
            sampler.update(task, rewards, costs=costs)
            policy.learn(task, rewards)
    return policy.skill


@pytest.mark.parametrize("seed", range(5))
def test_the_curriculum_finds_more_groups_with_signal(seed: int) -> None:
    uniform = UniformSampler(TASKS, rng_seed=seed)
    curriculum = CurriculumSampler(TASKS, group_size=GROUP, rng_seed=seed)
    skill_uniform = run(uniform, seed, 20_000)
    skill_curriculum = run(curriculum, seed, 20_000)
    assert curriculum.signal_groups_per_episode > uniform.signal_groups_per_episode
    assert skill_curriculum > skill_uniform


@pytest.mark.parametrize("seed", range(5))
def test_the_curriculum_follows_the_skill(seed: int) -> None:
    sampler = CurriculumSampler(TASKS, group_size=GROUP, rng_seed=seed)
    skill = run(sampler, seed, 20_000)
    probabilities = sampler.probabilities()
    favourite = max(probabilities, key=probabilities.get)
    assert abs(favourite - skill) <= 1.0
