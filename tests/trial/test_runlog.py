"""Reading a run log and deciding whether a run finished."""

from __future__ import annotations

import json
from pathlib import Path

from trial.runlog import is_complete, read_log, run_dir_for

EVENTS = [
    {"event": "config", "seed": 0},
    {"event": "step", "step": 1},
    {"event": "done", "stop_reason": "steps", "step": 1},
]


def _write(run_dir: Path, text: str) -> Path:
    run_dir.mkdir(parents=True)
    (run_dir / "log.jsonl").write_text(text)
    return run_dir


def _lines(events: list[dict]) -> str:
    return "".join(json.dumps(event) + "\n" for event in events)


class TestReadLog:
    def test_returns_every_event_in_order(self, tmp_path: Path) -> None:
        run_dir = _write(tmp_path / "run", _lines(EVENTS))
        assert read_log(run_dir / "log.jsonl") == EVENTS

    def test_skips_a_line_cut_in_the_middle(self, tmp_path: Path) -> None:
        run_dir = _write(tmp_path / "run", _lines(EVENTS[:2]) + '{"event": "st')
        assert read_log(run_dir / "log.jsonl") == EVENTS[:2]


class TestIsComplete:
    def test_complete(self, tmp_path: Path) -> None:
        assert is_complete(_write(tmp_path / "run", _lines(EVENTS)))

    def test_cut_in_the_middle_of_a_line(self, tmp_path: Path) -> None:
        text = _lines(EVENTS[:2]) + '{"event": "do'
        assert not is_complete(_write(tmp_path / "run", text))

    def test_empty_log(self, tmp_path: Path) -> None:
        assert not is_complete(_write(tmp_path / "run", ""))

    def test_missing_folder(self, tmp_path: Path) -> None:
        assert not is_complete(tmp_path / "missing")


def test_run_dir_for_matches_the_pilot_layout() -> None:
    expected = Path("trial/runs/trial/llama3.2-1b-curriculum-s2")
    assert run_dir_for("trial/runs/trial", "llama3.2-1b", "curriculum", 2) == expected
