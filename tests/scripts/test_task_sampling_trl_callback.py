# SPDX-License-Identifier: BSD-3-Clause

"""Run the TRL evaluation callback example with a fake or the real Transformers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from openenv.core.evals import EvalConfig, EvalHarness, EvalSchedule

EXAMPLE = (
    Path(__file__).resolve().parents[2] / "examples/task_sampling/trl_eval_callback.py"
)


class FakeHarness(EvalHarness):
    def __init__(self) -> None:
        self.runs = 0

    def run(
        self,
        harness_version: str,
        library_versions: dict[str, str],
        dataset: str,
        eval_parameters: dict[str, Any],
    ) -> dict[str, Any]:
        self.runs += 1
        return {"success_rate": 0.5}


def make_schedule(every: int, events: list) -> EvalSchedule:
    config = EvalConfig(
        harness_name="FakeHarness",
        harness_version="1",
        library_versions={},
        dataset="test",
        eval_parameters={},
    )
    return EvalSchedule(
        FakeHarness(),
        config,
        every_n_steps=every,
        on_result=lambda step, result: events.append(("eval", step)),
    )


def load_example() -> ModuleType:
    spec = importlib.util.spec_from_file_location("trl_eval_callback", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train(callback: Any, state: Any, first: int, last: int) -> None:
    callback.on_train_begin(None, state, None)
    for step in range(first, last + 1):
        state.global_step = step
        callback.on_step_end(None, state, None)


@pytest.fixture
def example(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    transformers = ModuleType("transformers")
    transformers.TrainerCallback = type("TrainerCallback", (), {})
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return load_example()


class TestEvalScheduleCallback:
    @pytest.mark.parametrize("every", [0, 3])
    def test_evaluates_the_schedules_steps_after_sync(
        self, example: ModuleType, every: int
    ) -> None:
        twin: list = []
        twin_schedule = make_schedule(every, twin)
        for step in range(8):
            twin_schedule.maybe_run(step)
        assert twin == ([("eval", 0), ("eval", 3), ("eval", 6)] if every else [])

        events: list = []
        callback = example.EvalScheduleCallback(
            make_schedule(every, events), sync=lambda: events.append("sync")
        )
        state = SimpleNamespace(global_step=0, is_world_process_zero=True)
        train(callback, state, 1, 7)
        assert events == [event for pair in twin for event in ("sync", pair)]

    def test_other_processes_sync_but_do_not_evaluate(
        self, example: ModuleType
    ) -> None:
        events: list = []
        schedule = make_schedule(2, events)
        callback = example.EvalScheduleCallback(
            schedule, sync=lambda: events.append("sync")
        )
        state = SimpleNamespace(global_step=0, is_world_process_zero=False)
        train(callback, state, 1, 3)
        assert events == ["sync", "sync"]
        assert schedule.harness.runs == 0

    def test_a_resumed_run_does_not_repeat_its_last_evaluation(
        self, example: ModuleType
    ) -> None:
        events: list = []
        callback = example.EvalScheduleCallback(
            make_schedule(5, events), sync=lambda: events.append("sync")
        )
        state = SimpleNamespace(global_step=5, is_world_process_zero=True)
        train(callback, state, 6, 10)
        assert events == ["sync", ("eval", 10)]


class TestWithTransformers:
    def test_runs_inside_the_trainers_callback_handler(self, tmp_path: Path) -> None:
        transformers = pytest.importorskip("transformers")
        from transformers.trainer_callback import CallbackHandler

        events: list = []
        callback = load_example().EvalScheduleCallback(
            make_schedule(2, events), sync=lambda: events.append("sync")
        )
        handler = CallbackHandler([callback], None, None, None, None)
        args = transformers.TrainingArguments(output_dir=str(tmp_path), report_to=[])
        state = transformers.TrainerState()
        control = transformers.TrainerControl()
        handler.on_train_begin(args, state, control)
        for _ in range(4):
            state.global_step += 1
            handler.on_step_end(args, state, control)
        assert events == [
            "sync",
            ("eval", 0),
            "sync",
            ("eval", 2),
            "sync",
            ("eval", 4),
        ]
