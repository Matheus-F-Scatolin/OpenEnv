# SPDX-License-Identifier: BSD-3-Clause

"""
Run the runs of one or more run lists, one after another.

A run list is a JSON list of `{"run_dir": ..., "cmd": [...]}` entries, where
`cmd` is what follows `python`. A run whose folder is complete is skipped, so
the same command can be started again after a crash or an interruption. A
folder left incomplete is renamed aside and its run starts again from zero.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trial.preflight import free_gb, MIN_FREE_GB
from trial.runlog import is_complete

LISTS_DIR = Path("trial") / "lists"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _other_queue_is_alive(pid_file: Path) -> bool:
    """Whether the process named by a pid file still exists."""
    try:
        os.kill(int(pid_file.read_text()), 0)
    except (FileNotFoundError, ValueError, OverflowError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def _set_aside(run_dir: Path) -> Path:
    """Rename an incomplete run folder to the first free `.incomplete-N` name."""
    number = 1
    while (target := run_dir.with_name(f"{run_dir.name}.incomplete-{number}")).exists():
        number += 1
    run_dir.rename(target)
    return target


def run_queue(
    list_paths: list[str | Path],
    *,
    root: str | Path | None = None,
    dry_run: bool = False,
    min_free_gb: float = MIN_FREE_GB,
) -> dict[str, int]:
    """
    Run every entry of the given run lists, in order.

    Args:
        list_paths (`list[str | Path]`):
            The run lists. Their entries run in the order given.
        root (`str` or `Path`, *optional*):
            The project folder (`rfc-013/`), where every run starts. Defaults
            to the current folder.
        dry_run (`bool`, *optional*, defaults to `False`):
            Print what would happen to each entry, and run and write nothing.
        min_free_gb (`float`, *optional*, defaults to `10.0`):
            Refuse to start with less free disk space than this.

    Returns:
        `dict[str, int]`: How many runs finished, failed and were skipped.

    Raises:
        `RuntimeError`: If `root` is not the project folder, the disk is almost
            full, or another queue is running.
    """
    root = Path.cwd() if root is None else Path(root)
    lists_dir = root / LISTS_DIR
    if not lists_dir.is_dir():
        raise RuntimeError(f"no {LISTS_DIR}/ in {root}: start the queue from rfc-013/")
    entries: list[dict[str, Any]] = []
    for path in list_paths:
        entries += json.loads(Path(path).read_text())

    if dry_run:
        for entry in entries:
            run_dir = root / entry["run_dir"]
            if is_complete(run_dir):
                print(f"skip: complete  {entry['run_dir']}")
            elif run_dir.exists():
                print(f"run again: incomplete  {entry['run_dir']}")
            else:
                print(f"run  {entry['run_dir']}")
        return {"finished": 0, "failed": 0, "skipped": 0}

    if free_gb(root) < min_free_gb:
        raise RuntimeError(f"less than {min_free_gb:.0f} GB of disk are free")
    pid_file = lists_dir / "queue.pid"
    if _other_queue_is_alive(pid_file):
        raise RuntimeError(
            f"a queue is already running (process {pid_file.read_text()})"
        )
    output_dir = lists_dir / "output"
    output_dir.mkdir(exist_ok=True)
    pid_file.write_text(str(os.getpid()))
    summary = {"finished": 0, "failed": 0, "skipped": 0}
    try:
        for position, entry in enumerate(entries, start=1):
            run_dir = root / entry["run_dir"]
            label = f"[{position}/{len(entries)}] {entry['run_dir']}"
            if is_complete(run_dir):
                print(f"skip: complete  {label}", flush=True)
                summary["skipped"] += 1
                continue
            if run_dir.exists():
                aside = _set_aside(run_dir)
                print(f"set aside: incomplete  {aside.name}", flush=True)
            output = output_dir / f"{position:02d}-{run_dir.name}.txt"
            print(f"start  {label}", flush=True)
            started = _now()
            with output.open("a") as handle:
                handle.write(f"== {started} {' '.join(entry['cmd'])}\n")
                handle.flush()
                process = subprocess.Popen(
                    [sys.executable, *entry["cmd"]],
                    cwd=root,
                    env={**os.environ, "HF_HUB_OFFLINE": "1"},
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                try:
                    code: int | None = process.wait()
                    status = (
                        "finished" if code == 0 and is_complete(run_dir) else "failed"
                    )
                except KeyboardInterrupt:
                    process.terminate()
                    code = process.wait()
                    status = "interrupted"
            record = {
                "run_dir": entry["run_dir"],
                "cmd": entry["cmd"],
                "start": started,
                "end": _now(),
                "exit_code": code,
                "status": status,
            }
            with (lists_dir / "queue-log.jsonl").open("a") as log:
                log.write(json.dumps(record) + "\n")
            print(
                f"{status}  {label}  (output: {output.relative_to(root)})", flush=True
            )
            if status == "interrupted":
                break
            summary[status] += 1
    finally:
        pid_file.unlink(missing_ok=True)
    print(
        f"queue: {summary['finished']} finished, {summary['failed']} failed, "
        f"{summary['skipped']} skipped, of {len(entries)}",
        flush=True,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    """
    Run the queue from the command line.

    Args:
        argv (`list[str]`, *optional*):
            The arguments; `sys.argv[1:]` when left out.

    Returns:
        `int`: 0 when no run failed, 1 otherwise, 2 when the queue did not start.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lists", nargs="+", help="run lists, in the order to run")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = run_queue(args.lists, dry_run=args.dry_run)
    except RuntimeError as error:
        print(f"queue not started: {error}")
        return 2
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
