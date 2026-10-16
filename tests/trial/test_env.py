# SPDX-License-Identifier: BSD-3-Clause

"""The task adapter preserves scored rewards and refuses an unscored answer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("reasoning_gym")

from trial.env import TaskEnv
from trial.tasks import make_tasks, PILOT_LADDER


def _env_with_reward(reward: float | None, monkeypatch: pytest.MonkeyPatch) -> TaskEnv:
    env = TaskEnv(PILOT_LADDER)

    def step(action: object) -> SimpleNamespace:
        return SimpleNamespace(reward=reward)

    monkeypatch.setattr(env.env, "step", step)
    return env


class TestReward:
    def test_missing_reward_is_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        env = _env_with_reward(None, monkeypatch)
        task = make_tasks("train", 1, PILOT_LADDER)[0]

        with pytest.raises(ValueError, match="environment returned no reward"):
            env.reward(task, "<answer>42</answer>")

    @pytest.mark.parametrize("reward", [0.0, 0.12345678901234567, 1.0])
    def test_scored_rewards_are_preserved(
        self, reward: float, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = _env_with_reward(reward, monkeypatch)
        task = make_tasks("train", 1, PILOT_LADDER)[0]

        assert env.reward(task, "<answer>42</answer>") == reward
