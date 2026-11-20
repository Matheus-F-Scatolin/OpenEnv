# SPDX-License-Identifier: BSD-3-Clause

"""The schedule executes a real EvalHarness contract and forwards its result."""

from __future__ import annotations

from typing import Any

import pytest
from openenv.core.evals import EvalConfig, EvalHarness, EvalResult, EvalSchedule


class FakeHarness(EvalHarness):
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def run(
        self,
        harness_version: str,
        library_versions: dict[str, str],
        dataset: str,
        eval_parameters: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append((harness_version, library_versions, dataset, eval_parameters))
        return {"calls": len(self.calls)}


class TestSchedule:
    @pytest.mark.parametrize("every", [0, 3])
    def test_marks_forced_runs_and_callback(self, every: int) -> None:
        harness = FakeHarness()
        config = EvalConfig(
            harness_name=harness.name,
            harness_version="1",
            library_versions={"test": "1"},
            dataset="test",
            eval_parameters={"temperature": 0.0},
        )
        seen: list[tuple[int, EvalResult]] = []
        schedule = EvalSchedule(
            harness,
            config,
            every_n_steps=every,
            on_result=lambda step, result: seen.append((step, result)),
        )
        for step in range(7):
            result = schedule.maybe_run(step)
            assert (result is not None) == (every != 0 and step % every == 0)
        forced = schedule.run(1)
        assert seen[-1] == (1, forced)
        assert forced.config == config
        assert forced.scores == {"calls": len(seen)}
        assert [step for step, _ in seen] == ([0, 3, 6, 1] if every else [1])
        assert harness.calls == [
            ("1", {"test": "1"}, "test", {"temperature": 0.0})
        ] * len(seen)
        before = len(seen)
        for method in (schedule.run, schedule.maybe_run):
            with pytest.raises(ValueError, match="step"):
                method(-1)
        assert len(seen) == before

    def test_negative_interval(self) -> None:
        config = EvalConfig(
            harness_name="fake",
            harness_version="1",
            library_versions={},
            dataset="test",
            eval_parameters={},
        )
        with pytest.raises(ValueError, match="every_n_steps"):
            EvalSchedule(
                FakeHarness(),
                config,
                every_n_steps=-1,
                on_result=lambda step, result: None,
            )
