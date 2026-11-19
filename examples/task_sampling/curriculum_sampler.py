# SPDX-License-Identifier: BSD-3-Clause

"""
A curriculum task sampler that favours the levels where a group of answers is
likely to carry signal.

`CurriculumSampler` follows [`~openenv.core.task_sampler.TaskSampler`] and
shares the counters of the samplers in `openenv.core.task_sampler`, so its
`metrics()` report the same numbers.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

# The counters and metrics() of the samplers in openenv.core.task_sampler.
from openenv.core.task_sampler import _CountingSampler


class CurriculumSampler(_CountingSampler):
    """
    Sample levels in proportion to the chance that a group there has signal.

    Tasks are grouped into levels by the value of `key`. For each level `k` the
    sampler keeps decayed counts of successes and failures and estimates the
    success rate `p_k` with a prior of one success and one failure. A group of
    `G` answers has at least one success and one failure with probability
    `1 - p_k**G - (1 - p_k)**G`; the level's weight is that signal plus `eps`.
    Within a level, tasks are shuffled and handed out without replacement. The
    costs given to `update` are counted and reported by `metrics()`; they do
    not change the weights.

    Levels keep the order in which they first appear in `tasks`.

    Args:
        tasks (`Sequence[Any]`):
            The task specs, mappings with `key` or `id_key`.
        group_size (`int`):
            Number of answers per group, `G` above.
        key (`str`, *optional*, defaults to `"difficulty"`):
            The field of a task spec that names its level. A task without it is
            a level of its own, named by its `id_key`.
        id_key (`str`, *optional*, defaults to `"id"`):
            Field identifying tasks without a level.
        decay (`float`, *optional*, defaults to `0.9`):
            Factor applied to success and failure counts on reports with scored
            rewards.
        eps (`float`, *optional*, defaults to `0.05`):
            Added to every level's signal, so that no level stops being sampled.
        success_threshold (`float`, *optional*, defaults to `1.0`):
            A reward at or above it counts as a success.
        rng_seed (`int`, *optional*, defaults to `0`):
            Seed of the sampler's own random generator.

    Attributes:
        keys (`list[Hashable]`):
            The levels, in the order in which they first appear in `tasks`.

    Raises:
        `ValueError`: If `tasks` is empty, `group_size` is below 2, `decay` is
            not in `(0, 1]`, or `eps` is negative.
        `KeyError`: If a task has neither `key` nor `id_key`.

    Examples:

    ```python
    sampler = CurriculumSampler(tasks, group_size=8)
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
        id_key: str = "id",
        decay: float = 0.9,
        eps: float = 0.05,
        success_threshold: float = 1.0,
        rng_seed: int = 0,
    ) -> None:
        if group_size < 2:
            raise ValueError(f"group_size must be at least 2, got {group_size}")
        if not 0 < decay <= 1:
            raise ValueError(f"decay must be in (0, 1], got {decay}")
        if eps < 0:
            raise ValueError(f"eps must be zero or more, got {eps}")
        super().__init__(tasks, rng_seed=rng_seed)
        self.group_size = group_size
        self.key = key
        self.id_key = id_key
        self.decay = decay
        self.eps = eps
        self.success_threshold = success_threshold
        self._buckets: dict[Hashable, list[int]] = {}
        for index, task in enumerate(self._tasks):
            self._buckets.setdefault(self._level(task), []).append(index)
        self.keys: list[Hashable] = list(self._buckets)
        self._orders: dict[Hashable, list[int]] = {k: [] for k in self.keys}
        self._successes: dict[Hashable, float] = {k: 0.0 for k in self.keys}
        self._failures: dict[Hashable, float] = {k: 0.0 for k in self.keys}

    def _level(self, task: Any) -> Hashable:
        return task[self.key] if self.key in task else task[self.id_key]

    def p_hat(self, k: Hashable) -> float:
        """
        Estimated success rate of a level.

        Args:
            k (`Hashable`):
                A level, one of `keys`.

        Returns:
            `float`: The decayed successes plus one, over the decayed answers
            plus two. `0.5` for a level never reported.
        """
        successes, failures = self._successes[k], self._failures[k]
        return (successes + 1.0) / (successes + failures + 2.0)

    def _signal(self, k: Hashable) -> float:
        p, g = self.p_hat(k), self.group_size
        return 1.0 - p**g - (1.0 - p) ** g

    def probabilities(self) -> dict[Hashable, float]:
        """
        The probability with which the next task comes from each level.

        Returns:
            `dict[Hashable, float]`: One entry per level, in the order of `keys`.
        """
        weights = {k: self._signal(k) + self.eps for k in self.keys}
        total = sum(weights.values())
        return {k: weight / total for k, weight in weights.items()}

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
        picks = self._rng.choices(
            self.keys, weights=[probabilities[k] for k in self.keys], k=n
        )
        out = []
        for k in picks:
            order = self._orders[k]
            if not order:
                order.extend(self._buckets[k])
                self._rng.shuffle(order)
            out.append(self._tasks[order.pop()])
        return out

    def update(
        self,
        task: Any,
        rewards: Sequence[float | None],
        costs: Sequence[float] | None = None,
    ) -> None:
        """
        Report one group, and move its level's estimates.

        Args:
            task (`Any`):
                The task spec the group ran.
            rewards (`Sequence[float | None]`):
                One reward per answer. `None` is unscored: counted as spend,
                never in reward estimates.
            costs (`Sequence[float]`, *optional*):
                One nonnegative cost per answer, for every group or none.

        Raises:
            `ValueError`: If `rewards` is empty, `costs` has another length or
                a negative value, costs are not supplied consistently, or the
                task's level is not one of `keys`. Nothing changes then.
        """
        self._check_group(rewards, costs)
        k = self._level(task)
        if k not in self._buckets:
            raise ValueError(f"unknown level {k!r}: not among the sampler's tasks")
        self._count(rewards, costs)
        scored = [reward for reward in rewards if reward is not None]
        if scored:
            successes = sum(reward >= self.success_threshold for reward in scored)
            self._successes[k] = self.decay * self._successes[k] + successes
            failures = len(scored) - successes
            self._failures[k] = self.decay * self._failures[k] + failures

    def _settings(self) -> dict[str, Any]:
        return {
            "group_size": self.group_size,
            "key": self.key,
            "id_key": self.id_key,
            "decay": self.decay,
            "eps": self.eps,
            "success_threshold": self.success_threshold,
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
            self._orders[k] = list(state["orders"][index])


__all__ = ["CurriculumSampler"]
