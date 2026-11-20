# SPDX-License-Identifier: BSD-3-Clause

"""
Task samplers: which task specs a trainer or an evaluation runs next.

A sampler chooses tasks and is told the rewards that came back. It only reads
rewards: the environment computes them.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

_STATE_VERSION = 1
_SIGNAL_STD = 1e-6


def _has_signal(rewards: Sequence[float]) -> bool:
    """
    Say whether the rewards of one group differ from each other.

    A group whose rewards are all equal gives a group-relative method such as
    GRPO no gradient: every advantage is zero.

    Args:
        rewards (`Sequence[float]`):
            The rewards of one group, one per answer.

    Returns:
        `bool`: `True` when the population standard deviation of the rewards is
        at least `1e-6`.
    """
    if not rewards:
        return False
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    return std >= _SIGNAL_STD


@runtime_checkable
class TaskSampler(Protocol):
    """
    Chooses the tasks a trainer runs, and learns from their rewards.

    A task spec is whatever the environment's task listing returns; a sampler
    passes it through unchanged.
    """

    def sample(self, n: int) -> list[Any]:
        """
        Return the task specs for the next `n` groups.

        Args:
            n (`int`):
                Number of task specs to return.

        Returns:
            `list[Any]`: The task specs, in the order they must be run: `n` of
            them, or, from a finite sampler at its end, the ones left, then an
            empty list.
        """
        ...

    def update(
        self,
        task: Any,
        rewards: Sequence[float | None],
        costs: Sequence[float] | None = None,
    ) -> None:
        """
        Report the outcome of one group.

        Args:
            task (`Any`):
                The task spec the group ran, as `sample` returned it.
            rewards (`Sequence[float | None]`):
                One reward per answer, as the environment gave it. `None` marks
                an answer the environment could not score: it counts as spend,
                never in reward estimates.
            costs (`Sequence[float]`, *optional*):
                One cost per answer, scored or not, in the unit the trainer
                counts (for example generated tokens).
        """
        ...

    def metrics(self) -> dict[str, float]:
        """
        Return numbers for the trainer's log.

        Returns:
            `dict[str, float]`: What the sampler tracks, by name.
        """
        ...

    def state_dict(self) -> dict[str, Any]:
        """
        Return the sampler's state.

        Returns:
            `dict[str, Any]`: A JSON-safe snapshot that `load_state_dict` accepts.
        """
        ...

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """
        Restore a state saved by `state_dict`.

        Args:
            state (`dict[str, Any]`):
                A snapshot from a sampler built with the same tasks and settings.
        """
        ...


class _CountingSampler:
    """
    The tasks, the random generator and the counters the samplers here share.

    Args:
        tasks (`Sequence[Any]`):
            The task specs to sample from.
        rng_seed (`int`, *optional*, defaults to `0`):
            Seed of the sampler's own random generator.

    Raises:
        `ValueError`: If `tasks` is empty.
    """

    def __init__(self, tasks: Sequence[Any], *, rng_seed: int = 0) -> None:
        self._tasks = list(tasks)
        if not self._tasks:
            raise ValueError("no tasks to sample from")
        self._rng = random.Random(rng_seed)
        self._groups = 0
        self._signal_groups = 0
        self._episodes = 0
        self._cost = 0.0
        self._costs_given: bool | None = None

    @staticmethod
    def _check_n(n: int) -> None:
        if n < 0:
            raise ValueError(f"n must be zero or more, got {n}")

    def _check_group(
        self, rewards: Sequence[float | None], costs: Sequence[float] | None
    ) -> None:
        if not rewards:
            raise ValueError("a group needs at least one reward")
        if self._costs_given is not None and self._costs_given != (costs is not None):
            raise ValueError("costs must be supplied for every group or for none")
        if costs is None:
            return
        if len(costs) != len(rewards):
            raise ValueError(
                f"got {len(costs)} costs for {len(rewards)} rewards: one each"
            )
        if any(cost < 0 for cost in costs):
            raise ValueError("costs must not be negative")

    def _count(
        self, rewards: Sequence[float | None], costs: Sequence[float] | None
    ) -> None:
        self._costs_given = costs is not None
        self._groups += 1
        self._signal_groups += _has_signal([r for r in rewards if r is not None])
        self._episodes += len(rewards)
        if costs is not None:
            self._cost += float(sum(costs))

    def metrics(self) -> dict[str, float]:
        """
        Return the counters and signal rates of every reported group.

        Returns:
            `dict[str, float]`: `groups`, `signal_groups` and `episodes`
            (counts; a group has signal when the standard deviation of its
            scored rewards is at least `1e-6`), `signal_groups_per_episode` and
            `zero_signal_fraction`; `cost` once costs were reported, and
            `signal_groups_per_cost` once they sum to more than zero.
        """
        result = {
            "groups": self._groups,
            "signal_groups": self._signal_groups,
            "episodes": self._episodes,
            "signal_groups_per_episode": (
                self._signal_groups / self._episodes if self._episodes else 0.0
            ),
            "zero_signal_fraction": (
                1 - self._signal_groups / self._groups if self._groups else 0.0
            ),
        }
        if self._costs_given:
            result["cost"] = self._cost
            if self._cost > 0:
                result["signal_groups_per_cost"] = self._signal_groups / self._cost
        return result

    def _base_state(self, kind: str) -> dict[str, Any]:
        version, internal, gauss = self._rng.getstate()
        return {
            "version": _STATE_VERSION,
            "kind": kind,
            "n_tasks": len(self._tasks),
            "costs_given": self._costs_given,
            "rng": [version, list(internal), gauss],
            "groups": self._groups,
            "signal_groups": self._signal_groups,
            "episodes": self._episodes,
            "cost": self._cost,
        }

    def _check_state(self, state: dict[str, Any], kind: str) -> None:
        if state.get("version") != _STATE_VERSION or state.get("kind") != kind:
            raise ValueError(
                f"not a version {_STATE_VERSION} state of a {kind} sampler"
            )
        if state["n_tasks"] != len(self._tasks):
            raise ValueError(
                f"the state was saved with {state['n_tasks']} tasks, "
                f"this sampler has {len(self._tasks)}"
            )

    def _load_base_state(self, state: dict[str, Any]) -> None:
        self._costs_given = state["costs_given"]
        version, internal, gauss = state["rng"]
        self._rng.setstate((version, tuple(internal), gauss))
        self._groups = state["groups"]
        self._signal_groups = state["signal_groups"]
        self._episodes = state["episodes"]
        self._cost = state["cost"]


class UniformSampler(_CountingSampler):
    """
    Every task equally often: shuffled, without replacement.

    The tasks are shuffled and handed out one by one; when all were handed out
    they are shuffled again. Rewards are counted and do not change the order.

    Args:
        tasks (`Sequence[Any]`):
            The task specs to sample from.
        rng_seed (`int`, *optional*, defaults to `0`):
            Seed of the shuffles.

    Raises:
        `ValueError`: If `tasks` is empty.

    Examples:

    ```python
    sampler = UniformSampler(tasks, rng_seed=0)
    for task in sampler.sample(8):
        rewards = run_group(task)
        sampler.update(task, rewards)
    ```
    """

    def __init__(self, tasks: Sequence[Any], *, rng_seed: int = 0) -> None:
        super().__init__(tasks, rng_seed=rng_seed)
        self._order: list[int] = []

    def sample(self, n: int) -> list[Any]:
        """
        Return the next `n` tasks of the current shuffle.

        Args:
            n (`int`):
                Number of task specs to return.

        Returns:
            `list[Any]`: `n` task specs, in the order they must be run.

        Raises:
            `ValueError`: If `n` is negative.
        """
        self._check_n(n)
        out = []
        for _ in range(n):
            if not self._order:
                self._order = list(range(len(self._tasks)))
                self._rng.shuffle(self._order)
            out.append(self._tasks[self._order.pop()])
        return out

    def update(
        self,
        task: Any,
        rewards: Sequence[float | None],
        costs: Sequence[float] | None = None,
    ) -> None:
        """
        Count one group. The sampling order does not depend on it.

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
                a negative value, or costs are not supplied consistently.
                Nothing is counted then.
        """
        self._check_group(rewards, costs)
        self._count(rewards, costs)

    def state_dict(self) -> dict[str, Any]:
        """
        Return the sampler's state.

        Returns:
            `dict[str, Any]`: A JSON-safe snapshot: the generator, what is left
            of the current shuffle, and the counters.
        """
        return {**self._base_state("uniform"), "order": list(self._order)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """
        Restore a state saved by `state_dict`.

        Args:
            state (`dict[str, Any]`):
                A snapshot from a uniform sampler with the same number of tasks.

        Raises:
            `ValueError`: If the state is of another kind of sampler, another
                version, or another number of tasks. Nothing changes then.
        """
        self._check_state(state, "uniform")
        self._load_base_state(state)
        self._order = list(state["order"])


class SequentialSampler(_CountingSampler):
    """
    Return a finite split in order, optionally with each requested seed.

    At the end of the split, `sample(n)` returns the items left, possibly fewer
    than `n`, and then an empty list. Rewards are counted and do not change the
    order.

    Args:
        tasks (`Sequence[Any]`):
            Task specs in evaluation order.
        seeds (`Sequence[int]`, *optional*):
            Seeds to pair with each task. With them, `sample` returns
            `(task, seed)` pairs, each task with every seed before the next
            task.

    Raises:
        `ValueError`: If `tasks` is empty, or `seeds` is given and empty.

    Examples:

    ```python
    sampler = SequentialSampler(test_tasks, seeds=[0, 1])
    while batch := sampler.sample(64):
        for task, seed in batch:
            run_episode(task, seed)
    ```
    """

    def __init__(
        self, tasks: Sequence[Any], *, seeds: Sequence[int] | None = None
    ) -> None:
        super().__init__(tasks)
        self._seeds = None if seeds is None else list(seeds)
        if self._seeds == []:
            raise ValueError("seeds is empty: leave it out to run each task once")
        self._position = 0

    def sample(self, n: int) -> list[Any]:
        """
        Return up to `n` remaining specs, or task and seed pairs.

        Args:
            n (`int`):
                Maximum number of items.

        Returns:
            `list[Any]`: The next items in order: fewer than `n` at the end of
            the split, and an empty list once it is exhausted.

        Raises:
            `ValueError`: If `n` is negative.
        """
        self._check_n(n)
        width = 1 if self._seeds is None else len(self._seeds)
        end = min(self._position + n, len(self._tasks) * width)
        out = []
        for index in range(self._position, end):
            task = self._tasks[index // width]
            out.append(
                task if self._seeds is None else (task, self._seeds[index % width])
            )
        self._position = end
        return out

    def update(
        self,
        task: Any,
        rewards: Sequence[float | None],
        costs: Sequence[float] | None = None,
    ) -> None:
        """
        Count a group without changing the iteration order.

        Args:
            task (`Any`):
                The reported task spec.
            rewards (`Sequence[float | None]`):
                One reward per answer. `None` is unscored: counted as spend,
                never in reward estimates.
            costs (`Sequence[float]`, *optional*):
                One nonnegative cost per answer, for every group or none.

        Raises:
            `ValueError`: If `rewards` is empty, `costs` has another length or
                a negative value, or costs are not supplied consistently.
                Nothing is counted then.
        """
        self._check_group(rewards, costs)
        self._count(rewards, costs)

    def state_dict(self) -> dict[str, Any]:
        """
        Return the sampler's state.

        Returns:
            `dict[str, Any]`: A JSON-safe snapshot: the counters, the seeds and
            the position in the split.
        """
        return {
            **self._base_state("sequential"),
            "seeds": self._seeds,
            "position": self._position,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """
        Restore the position in the split and the counters.

        Args:
            state (`dict[str, Any]`):
                A snapshot from a sequential sampler with the same number of
                tasks and the same seeds.

        Raises:
            `ValueError`: If the state is of another kind of sampler, another
                version, another number of tasks or other seeds. Nothing changes
                then.
        """
        self._check_state(state, "sequential")
        if state["seeds"] != self._seeds:
            raise ValueError("the state was saved with other seeds")
        self._load_base_state(state)
        self._position = state["position"]


__all__ = ["SequentialSampler", "TaskSampler", "UniformSampler"]
