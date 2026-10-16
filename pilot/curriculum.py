"""Prototype of the RFC 013 task samplers.

A task spec is a plain dict with at least ``id``; the curriculum also reads an
optional key (``difficulty`` by default). Samplers only pick tasks and read the
rewards the environment already computed; they never compute rewards.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Sequence


class _Sampler:
    def __init__(self, tasks: Sequence[dict], success_threshold: float = 1.0,
                 seed: int = 0, window: int = 200):
        if not tasks:
            raise ValueError("no tasks to sample from")
        self.tasks = list(tasks)
        self.success_threshold = success_threshold
        self.rng = random.Random(seed)
        self._recent_zero_signal: deque[bool] = deque(maxlen=window)
        self.groups_seen = 0
        self.zero_signal_groups = 0

    def sample(self, n: int) -> list[dict]:
        raise NotImplementedError

    def update(self, task: dict, rewards: Sequence[float]) -> None:
        """Report every reward of one GRPO group for ``task``."""
        zero = len(set(rewards)) <= 1
        self.groups_seen += 1
        self.zero_signal_groups += zero
        self._recent_zero_signal.append(zero)

    @property
    def zero_signal_fraction(self) -> float:
        """Share of recent groups whose rewards were all equal (no GRPO gradient)."""
        if not self._recent_zero_signal:
            return 0.0
        return sum(self._recent_zero_signal) / len(self._recent_zero_signal)

    def state_dict(self) -> dict[str, Any]:
        return {"rng": self.rng.getstate(), "groups_seen": self.groups_seen,
                "zero_signal_groups": self.zero_signal_groups,
                "recent": list(self._recent_zero_signal)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rng.setstate(_as_rng_state(state["rng"]))
        self.groups_seen = state["groups_seen"]
        self.zero_signal_groups = state["zero_signal_groups"]
        self._recent_zero_signal.clear()
        self._recent_zero_signal.extend(state["recent"])


class UniformSampler(_Sampler):
    """Shuffled sampling without replacement; reshuffles after each pass."""

    def __init__(self, tasks: Sequence[dict], **kwargs):
        super().__init__(tasks, **kwargs)
        self._order: list[int] = []

    def sample(self, n: int) -> list[dict]:
        out = []
        for _ in range(n):
            if not self._order:
                self._order = list(range(len(self.tasks)))
                self.rng.shuffle(self._order)
            out.append(self.tasks[self._order.pop()])
        return out

    def state_dict(self) -> dict[str, Any]:
        return {**super().state_dict(), "order": list(self._order)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self._order = list(state["order"])


class CurriculumSampler(_Sampler):
    """Sample keys where a GRPO group of size ``group_size`` is likely to carry signal.

    For each key k (e.g. a difficulty tier) it keeps decayed success/failure
    counts, estimates p_k with a Beta(1, 1) prior, and samples k with weight
    ``1 - p_k**G - (1 - p_k)**G + eps``: the probability that a group of G
    rollouts has at least one success and one failure, plus an exploration floor.
    """

    def __init__(self, tasks: Sequence[dict], group_size: int, key: str = "difficulty",
                 decay: float = 0.9, eps: float = 0.05, **kwargs):
        super().__init__(tasks, **kwargs)
        self.group_size = group_size
        self.key = key
        self.decay = decay
        self.eps = eps
        self._buckets: dict[Any, list[dict]] = {}
        for t in self.tasks:
            self._buckets.setdefault(t.get(key, t["id"]), []).append(t)
        self.keys = sorted(self._buckets, key=str)
        self._orders: dict[Any, list[int]] = {k: [] for k in self.keys}
        self.stats: dict[Any, list[float]] = {k: [0.0, 0.0] for k in self.keys}

    def p_hat(self, k: Any) -> float:
        s, f = self.stats[k]
        return (s + 1.0) / (s + f + 2.0)

    def weight(self, k: Any) -> float:
        p, g = self.p_hat(k), self.group_size
        return 1.0 - p**g - (1.0 - p) ** g + self.eps

    def probabilities(self) -> dict[Any, float]:
        w = {k: self.weight(k) for k in self.keys}
        total = sum(w.values())
        return {k: v / total for k, v in w.items()}

    def _pop(self, k: Any) -> dict:
        order = self._orders[k]
        if not order:
            order.extend(range(len(self._buckets[k])))
            self.rng.shuffle(order)
        return self._buckets[k][order.pop()]

    def sample(self, n: int) -> list[dict]:
        probs = self.probabilities()
        keys = self.rng.choices(self.keys, weights=[probs[k] for k in self.keys], k=n)
        return [self._pop(k) for k in keys]

    def update(self, task: dict, rewards: Sequence[float]) -> None:
        super().update(task, rewards)
        k = task.get(self.key, task["id"])
        succ = sum(r >= self.success_threshold for r in rewards)
        s, f = self.stats[k]
        self.stats[k] = [self.decay * s + succ, self.decay * f + (len(rewards) - succ)]

    def state_dict(self) -> dict[str, Any]:
        return {**super().state_dict(),
                "stats": {str(k): v for k, v in self.stats.items()},
                "orders": {str(k): v for k, v in self._orders.items()}}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        by_str = {str(k): k for k in self.keys}
        self.stats = {by_str[k]: list(v) for k, v in state["stats"].items()}
        self._orders = {by_str[k]: list(v) for k, v in state["orders"].items()}


def _as_rng_state(state):
    # JSON turns the tuple returned by Random.getstate() into nested lists.
    version, internal, gauss = state
    return (version, tuple(internal), gauss)
