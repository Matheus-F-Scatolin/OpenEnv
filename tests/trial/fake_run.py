"""Make a small real run folder with the fake policy, for tests."""

from __future__ import annotations

from pathlib import Path

from tests.trial.fake_policy import FakePolicy
from trial.env import TaskEnv
from trial.loop import build_sampler, RunConfig, train_run
from trial.runlog import RunLog
from trial.tasks import make_tasks, PILOT_LADDER


def run(tmp_path: Path, name: str = "run", **overrides) -> Path:
    """Train with the fake policy and return the run folder."""
    settings = {
        "model": "fake",
        "sampler": "curriculum",
        "out": str(tmp_path),
        "steps": 12,
        "train_per_tier": 30,
        "eval_per_tier": 4,
        "eval_every": 6,
        "eval_at_start": True,
        **overrides,
    }
    config = RunConfig(**settings)
    env = TaskEnv(PILOT_LADDER)
    train_tasks = make_tasks("train", config.train_per_tier, PILOT_LADDER)
    test_tasks = make_tasks("test", config.eval_per_tier, PILOT_LADDER)
    sampler = build_sampler(
        config.sampler, train_tasks, group_size=config.group_size, seed=config.seed
    )
    run_dir = tmp_path / name
    log = RunLog(run_dir, echo=False)
    train_run(config, FakePolicy(config.seed), env, sampler, log, test_tasks)
    log.close()
    run.sampler = sampler
    return run_dir
