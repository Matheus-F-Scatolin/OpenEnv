"""Every method name trains, and the ladders and the baseline work."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("reasoning_gym")

from analysis.validate_run import check_run
from tests.trial.fake_policy import FakePolicy
from tests.trial.fake_run import run
from trial.env import TaskEnv
from trial.loop import baseline_run, METHODS, RunConfig
from trial.runlog import is_complete, read_log, RunLog
from trial.tasks import LADDERS, make_tasks, STAIRCASE_LADDER


def test_the_method_names() -> None:
    assert METHODS == (
        "uniform",
        "curriculum",
        "curriculum-tokens",
        "curriculum-tokens-mix",
        "curriculum-tokens-mix-decay",
    )


@pytest.mark.parametrize("method", METHODS[2:])
def test_cost_aware_methods_train_and_log_their_estimates(
    method: str, tmp_path: Path
) -> None:
    run_dir = run(tmp_path, sampler=method, steps=100, eval_every=100)
    log = read_log(run_dir / "log.jsonl")
    steps = [e for e in log if e["event"] == "step"]
    assert len(steps) == 100 and is_complete(run_dir)
    for step in steps:
        assert set(step["sampler"]) == {"p_hat", "cost_hat", "probs"}
        assert sum(step["sampler"]["probs"].values()) == pytest.approx(1.0)
    assert steps[-1]["sampler"]["cost_hat"]["0"] < steps[-1]["sampler"]["cost_hat"]["4"]
    settings = log[0]["sampler_settings"]
    assert settings["cost"] == "tokens"
    assert settings["floor"] == ("mixture" if "mix" in method else "additive")
    assert settings["prior"] == ("decayed" if "decay" in method else "fixed")
    assert check_run(run_dir)[0] == "OK"


class TestStaircase:
    def test_ten_levels_without_negative_numbers(self) -> None:
        assert LADDERS["staircase"] is STAIRCASE_LADDER
        shape = [(t["min_terms"], t["min_digits"]) for t in STAIRCASE_LADDER]
        assert shape == [
            (2, 1),
            (3, 1),
            (3, 2),
            (4, 2),
            (4, 3),
            (5, 3),
            (5, 4),
            (6, 4),
            (6, 5),
            (7, 5),
        ]
        for tier in STAIRCASE_LADDER:
            assert tier["min_terms"] == tier["max_terms"]
            assert tier["min_digits"] == tier["max_digits"]
            assert "allow_negation" not in tier

    def test_train_and_test_seeds_are_disjoint(self) -> None:
        train = {t["seed"] for t in make_tasks("train", 2000, STAIRCASE_LADDER)}
        test = {t["seed"] for t in make_tasks("test", 100, STAIRCASE_LADDER)}
        assert len(train) == 20_000 and len(test) == 1000 and not train & test

    def test_a_baseline_writes_a_complete_run_folder(self, tmp_path: Path) -> None:
        config = RunConfig(
            model="fake",
            sampler="baseline",
            out=str(tmp_path),
            cmd="baseline",
            ladder="staircase",
            eval_per_tier=100,
        )
        tasks = make_tasks("test", 100, STAIRCASE_LADDER)
        log = RunLog(tmp_path / "fake-baseline-s0", echo=False)
        baseline_run(config, FakePolicy(), TaskEnv(STAIRCASE_LADDER), log, tasks)
        log.close()
        events = read_log(log.run_dir / "log.jsonl")
        assert [e["event"] for e in events] == ["config", "eval", "done"]
        assert events[-1]["stop_reason"] == "baseline"
        assert len(events[1]["by_tier"]) == 10
        records = (log.run_dir / "evals.jsonl").read_text().splitlines()
        assert len(records) == 1000 and json.loads(records[0])["step"] == 0
        assert check_run(log.run_dir)[0] == "OK"
