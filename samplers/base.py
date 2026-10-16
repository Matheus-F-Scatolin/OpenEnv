# SPDX-License-Identifier: BSD-3-Clause

"""
The task sampler contract, and what every sampler shares.

A sampler decides which tasks a trainer runs next and is told the rewards that
came back. It only reads rewards: the environment computes them.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

STATE_VERSION = 1
_SIGNAL_STD = 1e-6


def has_signal(rewards: Sequence[float]) -> bool:
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
            `list[Any]`: `n` task specs, in the order they must be run.
        """
        ...

    def update(
        self,
        task: Any,
        rewards: Sequence[float],
        costs: Sequence[float] | None = None,
    ) -> None:
        """
        Report the outcome of one group.

        Args:
            task (`Any`):
                The task spec the group ran, as `sample` returned it.
            rewards (`Sequence[float]`):
                One reward per answer of the group, as the environment gave it.
            costs (`Sequence[float]`, *optional*):
                One cost per answer, in the unit the trainer counts (for
                example generated tokens).
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


class CountingSampler:
    """
    Base for samplers: holds the tasks, the random generator and the counters.

    Args:
        tasks (`Sequence[Any]`):
            The task specs to sample from.
        rng_seed (`int`, *optional*, defaults to `0`):
            Seed of the sampler's own random generator.

    Attributes:
        groups (`int`):
            Number of groups reported.
        signal_groups (`int`):
            Number of reported groups for which `has_signal` holds.
        episodes (`int`):
            Number of answers reported.
        cost (`float`):
            Sum of the costs reported. It does not grow when a group is
            reported without costs.

    Raises:
        `ValueError`: If `tasks` is empty.
    """

    _COUNTERS = ("groups", "signal_groups", "episodes", "cost")

    def __init__(self, tasks: Sequence[Any], *, rng_seed: int = 0) -> None:
        if not tasks:
            raise ValueError("no tasks to sample from")
        self.tasks = list(tasks)
        self.rng = random.Random(rng_seed)
        self.groups = 0
        self.signal_groups = 0
        self.episodes = 0
        self.cost = 0.0

    @staticmethod
    def _check_n(n: int) -> None:
        if n < 0:
            raise ValueError(f"n must be zero or more, got {n}")

    @staticmethod
    def _check_group(rewards: Sequence[float], costs: Sequence[float] | None) -> None:
        if not rewards:
            raise ValueError("a group needs at least one reward")
        if costs is None:
            return
        if len(costs) != len(rewards):
            raise ValueError(
                f"got {len(costs)} costs for {len(rewards)} rewards: one each"
            )
        if any(cost <= 0 for cost in costs):
            raise ValueError("every cost must be above zero")

    def _count(self, rewards: Sequence[float], costs: Sequence[float] | None) -> None:
        self.groups += 1
        self.signal_groups += has_signal(rewards)
        self.episodes += len(rewards)
        if costs is not None:
            self.cost += float(sum(costs))

    @property
    def signal_groups_per_episode(self) -> float:
        """Groups with signal per reported answer; 0.0 before any report."""
        return self.signal_groups / self.episodes if self.episodes else 0.0

    @property
    def signal_groups_per_cost(self) -> float:
        """Groups with signal per unit of reported cost; 0.0 when no cost was reported."""
        return self.signal_groups / self.cost if self.cost else 0.0

    @property
    def zero_signal_fraction(self) -> float:
        """Share of the reported groups that had no signal; 0.0 before any report."""
        return 1 - self.signal_groups / self.groups if self.groups else 0.0

    def _base_state(self, kind: str) -> dict[str, Any]:
        version, internal, gauss = self.rng.getstate()
        return {
            "version": STATE_VERSION,
            "kind": kind,
            "n_tasks": len(self.tasks),
            "rng": [version, list(internal), gauss],
            **{name: getattr(self, name) for name in self._COUNTERS},
        }

    def _check_state(self, state: dict[str, Any], kind: str) -> None:
        if state.get("version") != STATE_VERSION or state.get("kind") != kind:
            raise ValueError(f"not a version {STATE_VERSION} state of a {kind} sampler")
        if state["n_tasks"] != len(self.tasks):
            raise ValueError(
                f"the state was saved with {state['n_tasks']} tasks, "
                f"this sampler has {len(self.tasks)}"
            )

    def _load_base_state(self, state: dict[str, Any]) -> None:
        version, internal, gauss = state["rng"]
        self.rng.setstate((version, tuple(internal), gauss))
        for name in self._COUNTERS:
            setattr(self, name, state[name])


__all__ = ["CountingSampler", "TaskSampler", "has_signal"]
