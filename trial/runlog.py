# SPDX-License-Identifier: BSD-3-Clause

"""
Read the log of a run and decide whether the run finished.

A run writes one JSON event per line to `log.jsonl` in its own folder. The run
is complete only when the last event of that file is the `done` event.
"""

from __future__ import annotations

import gzip
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, IO

LOG_NAME = "log.jsonl"
LIBRARIES = (
    "mlx",
    "mlx-lm",
    "transformers",
    "tokenizers",
    "reasoning-gym",
    "openenv",
    "numpy",
)


def read_log(path: str | Path) -> list[dict[str, Any]]:
    """
    Read the events of a log file.

    Args:
        path (`str` or `Path`):
            The `log.jsonl` file of a run.

    Returns:
        `list[dict[str, Any]]`: The events, in file order. A line that is not
        valid JSON (a write cut short) is left out.
    """
    events = []
    for line in Path(path).read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def is_complete(run_dir: str | Path) -> bool:
    """
    Say whether the run that wrote a folder finished.

    Args:
        run_dir (`str` or `Path`):
            The folder of one run.

    Returns:
        `bool`: `True` when the folder has a log whose last event is `done`.
    """
    log = Path(run_dir) / LOG_NAME
    if not log.is_file():
        return False
    events = read_log(log)
    return bool(events) and events[-1].get("event") == "done"


def run_dir_for(out: str | Path, model: str, method: str, seed: int) -> Path:
    """
    Build the folder of one run, as the pilot's script names it.

    Args:
        out (`str` or `Path`):
            The folder of the experiment, the value of `--out`.
        model (`str`):
            The model name, the value of `--model`.
        method (`str`):
            The sampling method, the value of `--sampler`.
        seed (`int`):
            The seed of the run.

    Returns:
        `Path`: `<out>/<model>-<method>-s<seed>`.
    """
    return Path(out) / f"{model}-{method}-s{seed}"


def write_json_atomic(path: str | Path, value: Any) -> None:
    """
    Write a JSON file under a temporary name, then rename it.

    Args:
        path (`str` or `Path`):
            The file to write.
        value (`Any`):
            A JSON-safe value.
    """
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value))
    os.replace(temporary, path)


def _lines(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(record) + "\n" for record in records)


def _sync(handle: IO[Any]) -> None:
    handle.flush()
    os.fsync(handle.fileno())


class RunLog:
    """
    The files one run writes into its folder.

    Args:
        run_dir (`str` or `Path`):
            The folder of the run. It is created when missing.
        echo (`bool`, *optional*, defaults to `True`):
            Also print each event, so the queue's output file shows progress.

    Raises:
        `FileExistsError`: If the folder already has a log. A run never
            continues or overwrites another.
    """

    def __init__(self, run_dir: str | Path, *, echo: bool = True) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._log = (self.run_dir / LOG_NAME).open("x")
        self._echo = echo

    def emit(self, record: dict[str, Any]) -> None:
        """
        Append one event to `log.jsonl` and sync it to disk.

        Args:
            record (`dict[str, Any]`):
                The event, with its `event` field.
        """
        line = json.dumps(record)
        if self._echo:
            print(line, flush=True)
        self._log.write(line + "\n")
        _sync(self._log)

    def _append(self, name: str, records: list[dict[str, Any]]) -> None:
        opener = gzip.open if name.endswith(".gz") else open
        with opener(self.run_dir / name, "at") as handle:
            handle.write(_lines(records))
            handle.flush()
            os.fsync(handle.fileno())

    def write_eval(
        self, records: list[dict[str, Any]], texts: list[dict[str, Any]]
    ) -> None:
        """
        Append the records of one evaluation.

        Args:
            records (`list[dict[str, Any]]`):
                One line per test question, for `evals.jsonl`.
            texts (`list[dict[str, Any]]`):
                The answer texts, for `eval_texts.jsonl.gz`.
        """
        self._append("evals.jsonl", records)
        self._append("eval_texts.jsonl.gz", texts)

    def write_train_texts(self, texts: list[dict[str, Any]]) -> None:
        """
        Append the answer texts of one training step.

        Args:
            texts (`list[dict[str, Any]]`):
                One line per answer, for `train_texts.jsonl.gz`.
        """
        self._append("train_texts.jsonl.gz", texts)

    def write_questions(self, questions: list[dict[str, Any]]) -> None:
        """
        Write the test questions, once.

        Args:
            questions (`list[dict[str, Any]]`):
                One line per test question, for `questions.jsonl`.
        """
        self._append("questions.jsonl", questions)

    def close(self) -> None:
        """Close the log file."""
        self._log.close()


def _git(root: Path, *args: str) -> str | None:
    """Output of a read-only git command in `root`, or `None` when it fails."""
    try:
        done = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    return done.stdout if done.returncode == 0 else None


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


# Written by the experiment itself under the project folder: not a change of code.
_WRITTEN_BY_RUNS = ("trial/runs", "trial/lists/queue-log.jsonl")


def environment_info(root: str | Path, prompt_date: str) -> dict[str, Any]:
    """
    Describe the environment of a run, for its `config` event.

    Args:
        root (`str` or `Path`):
            The project folder; git is asked about this folder only.
        prompt_date (`str`):
            The date pinned in the prompt.

    Returns:
        `dict[str, Any]`: `prompt_date`, `utc_time`, `python`, `platform`,
        `machine`, `versions`, `hf_hub_offline`, `git_commit` and `git_dirty`.
        `git_dirty` is true when a tracked file under `root` differs from the
        commit. Untracked files do not count, and neither do the run folders
        and the queue's log, which the experiment itself writes. No
        environment variable other than `HF_HUB_OFFLINE` is read.
    """
    root = Path(root)
    commit = _git(root, "rev-parse", "HEAD")
    status = _git(
        root,
        "status",
        "--porcelain",
        "--untracked-files=no",
        "--",
        ".",
        *(f":(exclude){path}" for path in _WRITTEN_BY_RUNS),
    )
    return {
        "prompt_date": prompt_date,
        "utc_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "versions": {name: _version(name) for name in LIBRARIES},
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE") == "1",
        "git_commit": commit.strip() if commit else None,
        "git_dirty": bool(status.strip()) if status is not None else None,
    }


__all__ = [
    "LIBRARIES",
    "LOG_NAME",
    "RunLog",
    "environment_info",
    "is_complete",
    "read_log",
    "run_dir_for",
    "write_json_atomic",
]
