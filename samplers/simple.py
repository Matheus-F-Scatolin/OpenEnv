# SPDX-License-Identifier: BSD-3-Clause

"""Samplers that do not look at rewards."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .base import CountingSampler


class UniformSampler(CountingSampler):
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
                self._order = list(range(len(self.tasks)))
                self.rng.shuffle(self._order)
            out.append(self.tasks[self._order.pop()])
        return out

    def update(
        self,
        task: Any,
        rewards: Sequence[float],
        costs: Sequence[float] | None = None,
    ) -> None:
        """
        Count one group. The sampling order does not depend on it.

        Args:
            task (`Any`):
                The task spec the group ran.
            rewards (`Sequence[float]`):
                One reward per answer of the group.
            costs (`Sequence[float]`, *optional*):
                One cost per answer.

        Raises:
            `ValueError`: If `rewards` is empty, or `costs` has another length
                or a value that is not above zero. Nothing is counted then.
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
                version, or another number of tasks.
        """
        self._check_state(state, "uniform")
        self._load_base_state(state)
        self._order = list(state["order"])


__all__ = ["UniformSampler"]
