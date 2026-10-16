"""Whether two logs describe the same training."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from trial.compare_logs import compare_logs, main

PILOT_RUNS = Path(__file__).resolve().parent.parent.parent / "pilot" / "runs"


def _log(experiment: str, method: str) -> Path:
    return PILOT_RUNS / experiment / f"llama3.2-1b-{method}-s0" / "log.jsonl"


def _copy(source: Path, target: Path, edit) -> Path:
    events = [json.loads(line) for line in source.read_text().splitlines()]
    target.write_text("".join(json.dumps(e) + "\n" for e in edit(events)))
    return target


@pytest.mark.parametrize("method", ["uniform", "curriculum"])
def test_the_two_stored_runs_agree_on_their_first_20_steps(
    method: str, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [str(_log("smoke", method)), str(_log("pilot", method)), "--steps", "20"]
    )
    assert code == 0
    assert capsys.readouterr().out.strip() == (
        "identical in 20/20 steps, 0 evaluations compared"
    )


def test_a_log_agrees_with_itself_on_its_evaluations() -> None:
    log = _log("smoke", "curriculum")
    assert compare_logs(log, log, 20) == (20, 3, [])


def test_one_changed_field_is_one_named_difference(tmp_path: Path) -> None:
    def edit(events: list[dict]) -> list[dict]:
        step = next(e for e in events if e["event"] == "step" and e["step"] == 7)
        step["gen_tokens"] += 1
        return events

    changed = _copy(_log("smoke", "uniform"), tmp_path / "log.jsonl", edit)
    steps, _, differences = compare_logs(_log("smoke", "uniform"), changed, 20)
    assert steps == 19
    assert len(differences) == 1
    assert "step 7" in differences[0] and "gen_tokens" in differences[0]


def test_times_and_memory_are_ignored(tmp_path: Path) -> None:
    def edit(events: list[dict]) -> list[dict]:
        for event in events:
            if event["event"] == "step":
                event["seconds"] += 1.0
                event["peak_mem_gb"] += 1.0
                event["only_here"] = 1
        return events

    changed = _copy(_log("smoke", "uniform"), tmp_path / "log.jsonl", edit)
    assert compare_logs(_log("smoke", "uniform"), changed, 20)[2] == []


class TestRefused:
    def test_a_second_config_event(self, tmp_path: Path) -> None:
        changed = _copy(
            _log("smoke", "uniform"), tmp_path / "log.jsonl", lambda e: e + [e[0]]
        )
        with pytest.raises(ValueError, match="config"):
            compare_logs(_log("smoke", "uniform"), changed, 20)
        assert main([str(_log("smoke", "uniform")), str(changed), "--steps", "20"]) == 2

    def test_a_step_number_twice(self, tmp_path: Path) -> None:
        def edit(events: list[dict]) -> list[dict]:
            return events + [e for e in events if e.get("step") == 3][:1]

        changed = _copy(_log("smoke", "uniform"), tmp_path / "log.jsonl", edit)
        with pytest.raises(ValueError, match="twice"):
            compare_logs(_log("smoke", "uniform"), changed, 20)

    def test_fewer_steps_than_asked(self) -> None:
        with pytest.raises(ValueError, match="20 steps"):
            compare_logs(_log("smoke", "uniform"), _log("smoke", "uniform"), 30)
