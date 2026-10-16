# SPDX-License-Identifier: BSD-3-Clause

"""The uniform sampler."""

from __future__ import annotations

import json

import pytest
from samplers import UniformSampler

TASKS = [{"id": i} for i in range(10)]


def _ids(tasks: list[dict]) -> list[int]:
    return [task["id"] for task in tasks]


def test_every_task_once_per_pass_and_a_new_shuffle_after() -> None:
    sampler = UniformSampler(TASKS, rng_seed=3)
    first, second = _ids(sampler.sample(10)), _ids(sampler.sample(10))
    assert sorted(first) == sorted(second) == list(range(10))
    assert first != second


def test_the_seed_decides_the_order() -> None:
    a = _ids(UniformSampler(TASKS, rng_seed=1).sample(30))
    assert a == _ids(UniformSampler(TASKS, rng_seed=1).sample(30))
    assert a != _ids(UniformSampler(TASKS, rng_seed=2).sample(30))


def test_save_in_the_middle_of_a_pass_and_restore() -> None:
    sampler = UniformSampler(TASKS, rng_seed=5)
    sampler.sample(7)
    sampler.update(TASKS[0], [1.0, 0.0], costs=[3, 4])
    state = json.loads(json.dumps(sampler.state_dict()))
    restored = UniformSampler(TASKS, rng_seed=99)
    restored.load_state_dict(state)
    assert _ids(restored.sample(40)) == _ids(sampler.sample(40))
    assert restored.state_dict() == sampler.state_dict()


def test_bad_arguments_are_rejected() -> None:
    with pytest.raises(ValueError):
        UniformSampler([])
    sampler = UniformSampler(TASKS)
    with pytest.raises(ValueError):
        sampler.sample(-1)
    assert sampler.sample(0) == []
    with pytest.raises(ValueError):
        UniformSampler(TASKS[:3]).load_state_dict(sampler.state_dict())
