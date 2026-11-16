# SPDX-License-Identifier: BSD-3-Clause

"""Scheduled evaluation preserves marks and reconciles sampler accounting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("reasoning_gym")

from analysis.validate_run import check_run
from tests.trial.fake_run import run
from trial.loop import eval_due
from trial.runlog import read_log


class TestEvalMetrics:
    @pytest.mark.parametrize("every,tokens", [(3, 0), (0, 1000), (3, 1000)])
    def test_marks_and_counters(self, tmp_path: Path, every: int, tokens: int) -> None:
        folder = run(tmp_path, steps=7, eval_every=every, eval_every_tokens=tokens)
        events = read_log(folder / "log.jsonl")
        steps = [e for e in events if e["event"] == "step"]
        expected = [0]
        before = signals = 0
        metrics = {}
        for step in steps:
            number, total = step["step"], step["gen_tokens_total"]
            signals += step["groups_with_signal"]
            own = step["sampler_metrics"]
            assert own["groups"] == number * 8
            assert own["episodes"] == number * 64
            assert own["signal_groups"] == signals
            assert own["cost"] == total
            metrics[number] = own
            if number == 7 or eval_due(number, before, total, every, tokens):
                expected.append(number)
            before = total
        evaluations = [e for e in events if e["event"] == "eval"]
        assert [e["step"] for e in evaluations] == expected
        assert evaluations[0]["sampler_metrics"]["groups"] == 0
        assert "cost" not in evaluations[0]["sampler_metrics"]
        for evaluation in evaluations[1:]:
            assert evaluation["sampler_metrics"] == metrics[evaluation["step"]]
        assert check_run(folder)[0] == "OK"

    @pytest.mark.parametrize("kind", ["step", "eval"])
    @pytest.mark.parametrize("field", ["cost", "groups", "episodes", "signal_groups"])
    def test_corrupted_metrics_are_named(
        self, tmp_path: Path, kind: str, field: str
    ) -> None:
        folder = run(tmp_path, steps=2, eval_every=2)
        path = folder / "log.jsonl"
        events = read_log(path)
        event = next(e for e in events if e["event"] == kind and e["step"] > 0)
        event["sampler_metrics"][field] += 1
        path.write_text("".join(json.dumps(e) + "\n" for e in events))
        assert f"sampler_metrics.{field}" in check_run(folder)[0]

    def test_baseline_has_no_sampler_metrics(self, tmp_path: Path) -> None:
        from tests.trial.fake_policy import FakePolicy
        from trial.env import build_tasks
        from trial.loop import baseline_run, RunConfig
        from trial.runlog import RunLog

        env, _, tasks = build_tasks("pilot", 1, 2)
        config = RunConfig(
            model="fake", sampler="baseline", out=str(tmp_path), cmd="baseline"
        )
        folder = tmp_path / "baseline"
        log = RunLog(folder, echo=False)
        baseline_run(config, FakePolicy(), env, log, tasks)
        log.close()
        assert all(
            "sampler_metrics" not in event for event in read_log(folder / "log.jsonl")
        )
        assert check_run(folder)[0] == "OK"

    def test_harness_preserves_task_order_and_greedy_batches(self) -> None:
        from openenv.core.evals import EvalConfig
        from tests.trial.fake_policy import FakePolicy
        from trial.env import build_tasks
        from trial.loop import BatchedEvalHarness

        class RecordingPolicy(FakePolicy):
            def __init__(self) -> None:
                super().__init__()
                self.calls: list[tuple[int, int, float]] = []

            def generate(
                self, prompts: list[list[int]], max_tokens: int, temperature: float
            ) -> list:
                self.calls.append((len(prompts), max_tokens, temperature))
                return super().generate(prompts, max_tokens, temperature)

        env, _, tasks = build_tasks("pilot", 1, 27)
        policy = RecordingPolicy()
        harness = BatchedEvalHarness(policy, env=env, tasks=tasks)
        config = EvalConfig(
            harness_name=harness.name,
            harness_version="1",
            library_versions={},
            dataset="reasoning_gym:chain_sum",
            eval_parameters={"max_tokens": 96, "temperature": 0.0},
        )
        result = harness.run_from_config(config)
        assert result.config == config
        assert result.scores["n_tasks"] == len(tasks)
        assert [row["task"] for row in harness.records] == [
            task["id"] for task in tasks
        ]
        assert [row["task"] for row in harness.texts] == [task["id"] for task in tasks]
        assert policy.calls == [(64, 96, 0.0), (64, 96, 0.0), (7, 96, 0.0)]
