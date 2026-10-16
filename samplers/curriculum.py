# SPDX-License-Identifier: BSD-3-Clause

"""A curriculum that samples the levels where a group is likely to carry signal."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

from .base import CountingSampler

_COST_UNITS = ("episodes", "tokens")
_FLOORS = ("additive", "mixture")
_PRIORS = ("fixed", "decayed")


class CurriculumSampler(CountingSampler):
    """
    Sample levels in proportion to the chance that a group there has signal.

    Tasks are grouped into levels by the value of `key`. For each level `k` the
    sampler keeps decayed counts of successes and failures and estimates the
    success rate `p_k` with a prior of one success and one failure. A group of
    `G` answers has at least one success and one failure with probability
    `1 - p_k**G - (1 - p_k)**G`; the level's weight is that signal plus `eps`,
    divided by the level's cost. Within a level, tasks are shuffled and handed
    out without replacement.

    Levels keep the order in which they first appear in `tasks`.

    Args:
        tasks (`Sequence[Any]`):
            The task specs to sample from. Each is a mapping with an `id`.
        group_size (`int`):
            Number of answers per group, `G` above.
        key (`str`, *optional*, defaults to `"difficulty"`):
            The field of a task spec that names its level. A task without it is
            a level of its own, named by its `id`.
        cost (`str`, *optional*, defaults to `"episodes"`):
            `"episodes"`: every level costs the same. `"tokens"`: a level costs
            the moving average of the costs reported for it, and `update` then
            requires `costs`.
        decay (`float`, *optional*, defaults to `0.9`):
            Factor applied to a level's counts and cost average at each report
            for that level. `1.0` never forgets.
        eps (`float`, *optional*, defaults to `0.05`):
            Added to every level's signal, so that no level stops being sampled.
        success_threshold (`float`, *optional*, defaults to `1.0`):
            A reward at or above it counts as a success.
        rng_seed (`int`, *optional*, defaults to `0`):
            Seed of the sampler's own random generator.
        floor (`str`, *optional*, defaults to `"additive"`):
            For the experiment that compares formulas; the default is the
            RFC's. `"additive"`: the weight is `(signal + eps) / cost`.
            `"mixture"`: the weight is `signal / cost`, and every one of the
            `K` levels then keeps at least `eps` of the probability:
            `(1 - K * eps) * weight / total + eps`.
        prior (`str`, *optional*, defaults to `"fixed"`):
            For the experiment that compares formulas; the default is the
            RFC's. `"fixed"`: the prior of one success and one failure never
            fades. `"decayed"`: the counts start at one success and one
            failure and decay like the data, so the estimate of a solved level
            approaches 1 and its signal approaches 0.

    Attributes:
        keys (`list[Hashable]`):
            The levels, in the order in which they first appear in `tasks`.

    Raises:
        `ValueError`: If `tasks` is empty, `group_size` is below 2, `decay` is
            not in `(0, 1]`, `eps` is negative, `cost`, `floor` or `prior` is not a
            known name, or `floor="mixture"` with `eps` times the number of levels
            above 1.

    Examples:

    ```python
    sampler = CurriculumSampler(tasks, group_size=8, cost="tokens")
    for task in sampler.sample(8):
        rewards, tokens = run_group(task)
        sampler.update(task, rewards, costs=tokens)
    ```
    """

    def __init__(
        self,
        tasks: Sequence[Any],
        *,
        group_size: int,
        key: str = "difficulty",
        cost: str = "episodes",
        decay: float = 0.9,
        eps: float = 0.05,
        success_threshold: float = 1.0,
        rng_seed: int = 0,
        floor: str = "additive",
        prior: str = "fixed",
    ) -> None:
        if group_size < 2:
            raise ValueError(f"group_size must be at least 2, got {group_size}")
        if not 0 < decay <= 1:
            raise ValueError(f"decay must be in (0, 1], got {decay}")
        if eps < 0:
            raise ValueError(f"eps must be zero or more, got {eps}")
        if cost not in _COST_UNITS:
            raise ValueError(f"cost must be one of {_COST_UNITS}, got {cost!r}")
        if floor not in _FLOORS:
            raise ValueError(f"floor must be one of {_FLOORS}, got {floor!r}")
        if prior not in _PRIORS:
            raise ValueError(f"prior must be one of {_PRIORS}, got {prior!r}")
        super().__init__(tasks, rng_seed=rng_seed)
        self.floor = floor
        self.prior = prior
        self.group_size = group_size
        self.key = key
        self.cost_unit = cost
        self.decay = decay
        self.eps = eps
        self.success_threshold = success_threshold
        self._buckets: dict[Hashable, list[int]] = {}
        for index, task in enumerate(self.tasks):
            self._buckets.setdefault(self._level(task), []).append(index)
        self.keys: list[Hashable] = list(self._buckets)
        self._orders: dict[Hashable, list[int]] = {k: [] for k in self.keys}
        if floor == "mixture" and eps * len(self.keys) > 1:
            raise ValueError(
                f"eps={eps} times {len(self.keys)} levels is above 1: "
                "the mixture floor cannot give every level that much"
            )
        start = 1.0 if prior == "decayed" else 0.0
        self._successes: dict[Hashable, float] = {k: start for k in self.keys}
        self._failures: dict[Hashable, float] = {k: start for k in self.keys}
        self._costs: dict[Hashable, float | None] = {k: None for k in self.keys}

    def _level(self, task: Any) -> Hashable:
        return task.get(self.key, task["id"])

    def p_hat(self, k: Hashable) -> float:
        """
        Estimated success rate of a level.

        Args:
            k (`Hashable`):
                A level, one of `keys`.

        Returns:
            `float`: The decayed successes plus one, over the decayed answers
            plus two. `0.5` for a level never reported. With
            `prior="decayed"` the one success and one failure are part of the
            decayed counts.
        """
        successes, failures = self._successes[k], self._failures[k]
        if self.prior == "decayed":
            return successes / (successes + failures)
        return (successes + 1.0) / (successes + failures + 2.0)

    def cost_hat(self, k: Hashable) -> float:
        """
        Estimated cost of one answer at a level.

        Args:
            k (`Hashable`):
                A level, one of `keys`.

        Returns:
            `float`: `1.0` when `cost="episodes"`. Otherwise the moving average
            of the costs reported for the level; for a level with none yet, the
            mean over the levels that have one, or `1.0` when none has.
        """
        if self.cost_unit == "episodes":
            return 1.0
        own = self._costs[k]
        if own is not None:
            return own
        seen = [value for value in self._costs.values() if value is not None]
        return sum(seen) / len(seen) if seen else 1.0

    def _signal(self, k: Hashable) -> float:
        p, g = self.p_hat(k), self.group_size
        return 1.0 - p**g - (1.0 - p) ** g

    def probabilities(self) -> dict[Hashable, float]:
        """
        The probability with which the next task comes from each level.

        Returns:
            `dict[Hashable, float]`: One entry per level, in the order of `keys`.
        """
        if self.floor == "additive":
            weights = {
                k: (self._signal(k) + self.eps) / self.cost_hat(k) for k in self.keys
            }
            total = sum(weights.values())
            return {k: weight / total for k, weight in weights.items()}
        weights = {k: self._signal(k) / self.cost_hat(k) for k in self.keys}
        total = sum(weights.values())
        if total <= 0:
            return {k: 1.0 / len(self.keys) for k in self.keys}
        share = 1.0 - self.eps * len(self.keys)
        return {k: share * weight / total + self.eps for k, weight in weights.items()}

    def sample(self, n: int) -> list[Any]:
        """
        Draw `n` levels with the current probabilities, and a task from each.

        Args:
            n (`int`):
                Number of task specs to return.

        Returns:
            `list[Any]`: `n` task specs, in the order they must be run.

        Raises:
            `ValueError`: If `n` is negative.
        """
        self._check_n(n)
        probabilities = self.probabilities()
        picks = self.rng.choices(
            self.keys, weights=[probabilities[k] for k in self.keys], k=n
        )
        out = []
        for k in picks:
            order = self._orders[k]
            if not order:
                order.extend(self._buckets[k])
                self.rng.shuffle(order)
            out.append(self.tasks[order.pop()])
        return out

    def update(
        self,
        task: Any,
        rewards: Sequence[float],
        costs: Sequence[float] | None = None,
    ) -> None:
        """
        Report one group, and move its level's estimates.

        Args:
            task (`Any`):
                The task spec the group ran.
            rewards (`Sequence[float]`):
                One reward per answer of the group.
            costs (`Sequence[float]`, *optional*):
                One cost per answer. Required when `cost="tokens"`.

        Raises:
            `ValueError`: If `rewards` is empty, `costs` has another length or
                a value that is not above zero, `costs` is missing although
                `cost="tokens"`, or the task's level is not one of `keys`.
                Nothing changes then.
        """
        self._check_group(rewards, costs)
        if costs is None and self.cost_unit == "tokens":
            raise ValueError('cost="tokens" needs the costs of every group')
        k = self._level(task)
        if k not in self._buckets:
            raise ValueError(f"unknown level {k!r}: not among the sampler's tasks")
        self._count(rewards, costs)
        successes = sum(reward >= self.success_threshold for reward in rewards)
        self._successes[k] = self.decay * self._successes[k] + successes
        failures = len(rewards) - successes
        self._failures[k] = self.decay * self._failures[k] + failures
        if costs is not None:
            mean = sum(costs) / len(costs)
            previous = self._costs[k]
            self._costs[k] = (
                mean
                if previous is None
                else self.decay * previous + (1 - self.decay) * mean
            )

    def _settings(self) -> dict[str, Any]:
        return {
            "group_size": self.group_size,
            "key": self.key,
            "cost": self.cost_unit,
            "decay": self.decay,
            "eps": self.eps,
            "success_threshold": self.success_threshold,
            "floor": self.floor,
            "prior": self.prior,
        }

    def state_dict(self) -> dict[str, Any]:
        """
        Return the sampler's state.

        Returns:
            `dict[str, Any]`: A JSON-safe snapshot: the settings, the levels,
            their estimates and remaining shuffles, the generator and the
            counters.
        """
        return {
            **self._base_state("curriculum"),
            "settings": self._settings(),
            "keys": [str(k) for k in self.keys],
            "successes": [self._successes[k] for k in self.keys],
            "failures": [self._failures[k] for k in self.keys],
            "costs": [self._costs[k] for k in self.keys],
            "orders": [list(self._orders[k]) for k in self.keys],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """
        Restore a state saved by `state_dict`.

        Args:
            state (`dict[str, Any]`):
                A snapshot from a curriculum sampler with the same tasks and
                settings.

        Raises:
            `ValueError`: If the state is of another kind of sampler or
                version, or was saved with other tasks, levels or settings.
                Nothing changes then.
        """
        self._check_state(state, "curriculum")
        if state["settings"] != self._settings():
            raise ValueError(
                f"the state was saved with settings {state['settings']}, "
                f"this sampler has {self._settings()}"
            )
        if state["keys"] != [str(k) for k in self.keys]:
            raise ValueError("the state was saved with other levels")
        self._load_base_state(state)
        for index, k in enumerate(self.keys):
            self._successes[k] = state["successes"][index]
            self._failures[k] = state["failures"][index]
            self._costs[k] = state["costs"][index]
            self._orders[k] = list(state["orders"][index])


__all__ = ["CurriculumSampler"]
