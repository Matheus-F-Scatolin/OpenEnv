"""The run queue: order, skipping finished runs, surviving failures."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from tests.trial.pinned_parser import build_parser
from trial.run_queue import run_queue
from trial.runlog import is_complete, run_dir_for
from trial.train import build_parser as train_parser, config_from_args

ROOT = Path(__file__).resolve().parent.parent.parent

# A stand-in for a trainer: writes a small complete log into its run folder.
STAND_IN = """
import json, os, sys
run_dir, mode = sys.argv[1], sys.argv[2]
os.makedirs(run_dir)
with open(os.path.join(run_dir, "log.jsonl"), "x") as log:
    log.write(json.dumps({"event": "config", "offline": os.environ["HF_HUB_OFFLINE"]}) + "\\n")
    if mode == "fail":
        sys.exit("boom: the stand-in was told to fail")
    log.write(json.dumps({"event": "done"}) + "\\n")
with open("order.txt", "a") as order:
    order.write(run_dir + "\\n")
"""


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "trial" / "lists").mkdir(parents=True)
    (tmp_path / "stand_in.py").write_text(STAND_IN)
    return tmp_path


def _list(root: Path, name: str, runs: list[tuple[str, str]]) -> Path:
    path = root / "trial" / "lists" / name
    entries = [
        {"run_dir": run_dir, "cmd": ["stand_in.py", run_dir, mode]}
        for run_dir, mode in runs
    ]
    path.write_text(json.dumps(entries))
    return path


def _statuses(root: Path) -> list[str]:
    log = root / "trial" / "lists" / "queue-log.jsonl"
    return [json.loads(line)["status"] for line in log.read_text().splitlines()]


def test_runs_in_order_then_skips_what_finished(root: Path) -> None:
    first = _list(root, "a.json", [("runs/one", "ok"), ("runs/two", "ok")])
    second = _list(root, "b.json", [("runs/three", "ok")])
    summary = run_queue([first, second], root=root, min_free_gb=0)
    assert summary == {"finished": 3, "failed": 0, "skipped": 0}
    assert (root / "order.txt").read_text().split() == [
        "runs/one",
        "runs/two",
        "runs/three",
    ]
    again = run_queue([first, second], root=root, min_free_gb=0)
    assert again == {"finished": 0, "failed": 0, "skipped": 3}
    assert not (root / "trial" / "lists" / "queue.pid").exists()


def test_an_incomplete_run_is_set_aside_and_run_again(root: Path) -> None:
    runs = _list(root, "a.json", [("runs/one", "ok"), ("runs/two", "ok")])
    run_queue([runs], root=root, min_free_gb=0)
    for expected in ("runs/two.incomplete-1", "runs/two.incomplete-2"):
        (root / "runs" / "two" / "log.jsonl").write_text('{"event": "config"}\n')
        summary = run_queue([runs], root=root, min_free_gb=0)
        assert summary == {"finished": 1, "failed": 0, "skipped": 1}
        assert (root / expected / "log.jsonl").exists()
        assert is_complete(root / "runs" / "two")


def test_a_failed_run_does_not_stop_the_queue(root: Path) -> None:
    runs = _list(root, "a.json", [("runs/bad", "fail"), ("runs/good", "ok")])
    summary = run_queue([runs], root=root, min_free_gb=0)
    assert summary == {"finished": 1, "failed": 1, "skipped": 0}
    assert _statuses(root) == ["failed", "finished"]
    output = (root / "trial" / "lists" / "output" / "01-bad.txt").read_text()
    assert "boom" in output


def test_the_run_gets_the_offline_flag(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    runs = _list(root, "a.json", [("runs/one", "ok")])
    run_queue([runs], root=root, min_free_gb=0)
    config = json.loads((root / "runs/one/log.jsonl").read_text().splitlines()[0])
    assert config["offline"] == "1"


def test_a_second_queue_refuses_to_start(root: Path) -> None:
    runs = _list(root, "a.json", [("runs/one", "ok")])
    (root / "trial" / "lists" / "queue.pid").write_text(str(os.getpid()))
    with pytest.raises(RuntimeError, match="already running"):
        run_queue([runs], root=root, min_free_gb=0)
    assert not (root / "runs").exists()


def test_a_stale_pid_file_is_ignored(root: Path) -> None:
    runs = _list(root, "a.json", [("runs/one", "ok")])
    (root / "trial" / "lists" / "queue.pid").write_text("999999999")
    assert run_queue([runs], root=root, min_free_gb=0)["finished"] == 1


def test_dry_run_creates_nothing(root: Path, capsys: pytest.CaptureFixture) -> None:
    runs = _list(root, "a.json", [("runs/one", "ok")])
    before = sorted(p.name for p in root.rglob("*"))
    run_queue([runs], root=root, min_free_gb=0, dry_run=True)
    assert sorted(p.name for p in root.rglob("*")) == before
    assert capsys.readouterr().out.strip() == "run  runs/one"


def test_refuses_outside_the_project_and_on_a_full_disk(root: Path) -> None:
    runs = _list(root, "a.json", [("runs/one", "ok")])
    with pytest.raises(RuntimeError, match="trial/lists"):
        run_queue([runs], root=root / "trial", min_free_gb=0)
    with pytest.raises(RuntimeError, match="free"):
        run_queue([runs], root=root, min_free_gb=10**9)


def test_no_file_shadows_a_standard_library_module() -> None:
    shadows = [
        str(path.relative_to(ROOT))
        for folder in ("samplers", "trial", "analysis")
        for path in (ROOT / folder).rglob("*.py")
        if path.stem in sys.stdlib_module_names
    ]
    assert shadows == []


class TestRunLists:
    """Every run list in `trial/lists/` can be started as written."""

    entries = [
        (path.name, entry)
        for path in sorted((ROOT / "trial" / "lists").glob("*.json"))
        for entry in json.loads(path.read_text())
    ]

    def test_there_are_run_lists(self) -> None:
        assert len(self.entries) == 56

    def test_run_folders_are_unique(self) -> None:
        folders = [entry["run_dir"] for _, entry in self.entries]
        assert len(folders) == len(set(folders))

    @pytest.mark.parametrize("name,entry", entries)
    def test_entry_is_valid(self, name: str, entry: dict) -> None:
        cmd = entry["cmd"]
        prefix = "/tmp/" if name == "rehearsal-mac.json" else "trial/runs/"
        assert entry["run_dir"].startswith(prefix)
        if cmd[0] == "trial/smoke_pinned.py":
            args = build_parser().parse_args(cmd[1:])
        else:
            assert cmd[:2] == ["-m", "trial.train"]
            args = config_from_args(train_parser().parse_args(cmd[2:]))
        expected = run_dir_for(args.out, args.model, args.sampler, args.seed)
        assert Path(entry["run_dir"]) == expected
        assert args.cmd == "baseline" or "--steps" in cmd or "--token-budget" in cmd
        assert args.model == "llama3.2-1b"
        if name != "rehearsal-mac.json" and not name.startswith("gate-queue"):
            assert args.eval_per_tier == 100
