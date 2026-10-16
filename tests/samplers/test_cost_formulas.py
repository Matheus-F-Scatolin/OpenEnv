# SPDX-License-Identifier: BSD-3-Clause

"""The three cost-aware forms of the curriculum, compared in the experiment."""

from __future__ import annotations

import pytest
from samplers import CurriculumSampler

G, EPS, K = 8, 0.05, 5
TASKS = [{"id": f"t{k}-{i}", "difficulty": k} for k in range(K) for i in range(3)]
# The pilot's last-quarter success rate and tokens per answer, per level.
SUCCESS = [1.0, 0.999, 0.948, 0.398, 0.267]
COST = [11.0, 28.6, 40.8, 65.8, 79.3]
# Hand-computed shares, to catch a wrong formula in the test itself.
EXPECTED = {
    ("additive", "fixed"): [0.227, 0.092, 0.194, 0.273, 0.213],
    ("mixture", "fixed"): [0.181, 0.104, 0.204, 0.281, 0.230],
    ("mixture", "decayed"): [0.050, 0.056, 0.231, 0.367, 0.296],
}


def at_steady_state(floor: str, prior: str) -> CurriculumSampler:
    """A sampler whose estimates sit at the pilot's last-quarter numbers."""
    sampler = CurriculumSampler(
        TASKS, group_size=G, cost="tokens", floor=floor, prior=prior
    )
    state = sampler.state_dict()
    state["successes"] = [80 * p for p in SUCCESS]
    state["failures"] = [80 * (1 - p) for p in SUCCESS]
    state["costs"] = list(COST)
    sampler.load_state_dict(state)
    return sampler


def by_formula(floor: str, prior: str) -> list[float]:
    """The same shares, computed from the formulas as written in the notes."""
    if prior == "fixed":
        p = [(80 * s + 1) / 82 for s in SUCCESS]
    else:
        p = list(SUCCESS)
    signal = [1 - x**G - (1 - x) ** G for x in p]
    if floor == "additive":
        weights = [(s + EPS) / c for s, c in zip(signal, COST)]
        return [w / sum(weights) for w in weights]
    weights = [s / c for s, c in zip(signal, COST)]
    return [(1 - K * EPS) * w / sum(weights) + EPS for w in weights]


@pytest.mark.parametrize("floor,prior", list(EXPECTED))
def test_each_form_follows_its_formula(floor: str, prior: str) -> None:
    probabilities = list(at_steady_state(floor, prior).probabilities().values())
    assert probabilities == pytest.approx(by_formula(floor, prior))
    assert probabilities == pytest.approx(EXPECTED[(floor, prior)], abs=0.0006)
    assert sum(probabilities) == pytest.approx(1.0)


def test_the_defaults_are_the_rfc() -> None:
    plain = CurriculumSampler(TASKS, group_size=G, cost="tokens")
    explicit = CurriculumSampler(
        TASKS, group_size=G, cost="tokens", floor="additive", prior="fixed"
    )
    assert plain.state_dict() == explicit.state_dict()


@pytest.mark.parametrize("costs", [[1, 1, 1, 1, 1], [1, 10, 100, 1000, 10000]])
def test_with_the_mixture_no_level_falls_below_eps(costs: list[float]) -> None:
    sampler = CurriculumSampler(
        TASKS, group_size=2, cost="tokens", floor="mixture", prior="decayed"
    )
    for level, cost in enumerate(costs):
        for _ in range(30):
            rewards = [1.0, 1.0] if level % 2 else [1.0, 0.0]
            sampler.update(TASKS[level * 3], rewards, costs=[cost, cost])
    probabilities = sampler.probabilities()
    assert min(probabilities.values()) >= EPS - 1e-12
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_mixture_with_no_signal_anywhere_is_uniform() -> None:
    sampler = CurriculumSampler(
        TASKS, group_size=2, floor="mixture", prior="decayed", decay=0.5
    )
    for level in range(K):
        for _ in range(2000):
            sampler.update(TASKS[level * 3], [1.0, 1.0])
    assert list(sampler.probabilities().values()) == pytest.approx([1 / K] * K)


def test_a_solved_level_fades_only_with_the_decayed_prior() -> None:
    def after_50(prior: str) -> float:
        sampler = CurriculumSampler(TASKS, group_size=G, floor="mixture", prior=prior)
        for _ in range(50):
            sampler.update(TASKS[0], [1.0] * G)
        return sampler.probabilities()[0]

    assert after_50("decayed") == pytest.approx(EPS, abs=0.001)
    assert after_50("fixed") == pytest.approx(0.067, abs=0.003)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"floor": "other"},
        {"prior": "other"},
        {"floor": "mixture", "eps": 0.21},
    ],
)
def test_bad_options_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        CurriculumSampler(TASKS, group_size=G, **kwargs)


def test_a_state_saved_with_another_form_is_refused() -> None:
    state = at_steady_state("mixture", "decayed").state_dict()
    with pytest.raises(ValueError):
        CurriculumSampler(TASKS, group_size=G, cost="tokens").load_state_dict(state)
