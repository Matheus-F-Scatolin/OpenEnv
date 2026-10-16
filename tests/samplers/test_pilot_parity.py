# SPDX-License-Identifier: BSD-3-Clause

"""The library reproduces the pilot's samplers draw for draw."""

from __future__ import annotations

import json
import random
from pathlib import Path

import curriculum as pilot
import pytest
from samplers import CurriculumSampler, UniformSampler

PILOT_STATE = (
    Path(pilot.__file__).parent
    / "runs"
    / "pilot"
    / "llama3.2-1b-uniform-s0"
    / "sampler_state.json"
)
GROUP = 8


def make_tasks(per_level: int) -> list[dict]:
    return [
        {"id": f"t{level}-{i}", "difficulty": level, "index": i}
        for level in range(5)
        for i in range(per_level)
    ]


def made_up_rewards(rng: random.Random, level: int) -> list[float]:
    """Rewards with all-equal groups, 0/1 groups and partial credit."""
    kind = rng.random()
    if kind < 0.25:
        return [float(level < 2)] * GROUP
    if kind < 0.35:
        return [0.01] * GROUP
    chance = max(0.05, 0.95 - 0.22 * level)
    return [
        1.0 if rng.random() < chance else rng.choice([0.0, 0.0, 0.01, 0.5])
        for _ in range(GROUP)
    ]


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("per_level", [2000, 7])
def test_curriculum_matches_the_pilot(seed: int, per_level: int) -> None:
    tasks = make_tasks(per_level)
    old = pilot.CurriculumSampler(tasks, group_size=GROUP, seed=seed)
    new = CurriculumSampler(tasks, group_size=GROUP, rng_seed=seed)
    rewards_rng = random.Random(1000 + seed)
    for _ in range(125):
        batch = old.sample(8)
        assert [t["id"] for t in new.sample(8)] == [t["id"] for t in batch]
        for task in batch:
            rewards = made_up_rewards(rewards_rng, task["difficulty"])
            old.update(task, rewards)
            new.update(task, rewards)
        assert new.keys == old.keys
        assert new.probabilities() == old.probabilities()
        assert [new.p_hat(k) for k in new.keys] == [old.p_hat(k) for k in old.keys]
    assert new.rng.getstate() == old.rng.getstate()


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("per_level", [2000, 7])
def test_uniform_matches_the_pilot(seed: int, per_level: int) -> None:
    tasks = make_tasks(per_level)
    old = pilot.UniformSampler(tasks, seed=seed)
    new = UniformSampler(tasks, rng_seed=seed)
    for _ in range(125):
        assert [t["id"] for t in new.sample(8)] == [t["id"] for t in old.sample(8)]
    assert new.rng.getstate() == old.rng.getstate()


def test_uniform_reaches_the_state_the_pilot_stored() -> None:
    stored = json.loads(PILOT_STATE.read_text())
    sampler = UniformSampler([{"id": i} for i in range(10_000)], rng_seed=0)
    for _ in range(2196):
        sampler.sample(8)
    state = sampler.state_dict()
    assert state["order"] == stored["order"]
    assert state["rng"] == stored["rng"]
