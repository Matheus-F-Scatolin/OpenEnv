# SPDX-License-Identifier: BSD-3-Clause

"""
Check a machine before it runs the experiments.

Prints the installed library versions against the lock file, the model
snapshot, the free disk space, the power source, and the date the prompt will
carry with and without the pin. Exits with 1 when a run should not start.
"""

from __future__ import annotations

import importlib.metadata
import os
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path

from trial.runlog import LIBRARIES
from trial.tasks import PROMPT_DATE

MODEL_REPO = "mlx-community/Llama-3.2-1B-Instruct-bf16"
MIN_FREE_GB = 10.0


def locked_versions(lock_path: str | Path) -> dict[str, set[str]]:
    """
    Read the versions a lock file allows for each package.

    Args:
        lock_path (`str` or `Path`):
            A `uv.lock` file.

    Returns:
        `dict[str, set[str]]`: Package name to its locked versions. A package
        has two when the lock differs between Python versions.
    """
    with Path(lock_path).open("rb") as handle:
        lock = tomllib.load(handle)
    versions: dict[str, set[str]] = {}
    for package in lock["package"]:
        versions.setdefault(package["name"], set()).add(package["version"])
    return versions


def version_report(
    lock_path: str | Path,
    installed: Callable[[str], str] = importlib.metadata.version,
) -> tuple[list[str], list[str]]:
    """
    Compare the installed pinned libraries with the lock file.

    Args:
        lock_path (`str` or `Path`):
            A `uv.lock` file.
        installed (`Callable[[str], str]`, *optional*):
            Returns the installed version of a package. Defaults to the
            metadata of the running environment.

    Returns:
        `tuple[list[str], list[str]]`: One printable line per library, and the
        names of the libraries whose installed version is not in the lock.
    """
    locked = locked_versions(lock_path)
    lines, wrong = [], []
    for name in LIBRARIES:
        try:
            have = installed(name)
        except importlib.metadata.PackageNotFoundError:
            have = "not installed"
        allowed = sorted(locked.get(name, set()))
        ok = have in allowed
        if not ok:
            wrong.append(name)
        mark = "ok" if ok else "DIFFERENT"
        lines.append(f"{name:14} installed {have:14} lock {', '.join(allowed)}  {mark}")
    return lines, wrong


def free_gb(path: str | Path) -> float:
    """
    Free disk space where a path lives.

    Args:
        path (`str` or `Path`):
            Any existing path on the disk to measure.

    Returns:
        `float`: Free space in gigabytes (10^9 bytes).
    """
    return shutil.disk_usage(path).free / 1e9


def today_date_line(snapshot: str, date_string: str | None) -> str:
    """
    Render a prompt with the model's chat template and return its date line.

    Args:
        snapshot (`str`):
            Local folder of the model snapshot.
        date_string (`str`, *optional*):
            The date to pin; the template uses today's date when left out.

    Returns:
        `str`: The line of the prompt that starts with `Today Date:`.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    extra = {} if date_string is None else {"date_string": date_string}
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "1 + 1 ="}],
        add_generation_prompt=True,
        tokenize=False,
        **extra,
    )
    return next(line for line in prompt.splitlines() if line.startswith("Today Date:"))


def _shell(command: list[str]) -> str:
    """Output of a command, or a short note when it cannot run."""
    try:
        done = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as error:
        return f"({error})"
    return (done.stdout + done.stderr).strip()


def main() -> int:
    """
    Print the checks and say whether a run may start.

    Returns:
        `int`: 0 when every check passed, 1 otherwise.
    """
    root = Path(__file__).resolve().parent.parent
    problems: list[str] = []

    print("== versions ==")
    lines, wrong = version_report(root / "uv.lock")
    print("\n".join(lines))
    problems += [f"{name}: installed version is not the locked one" for name in wrong]
    print(f"python         {sys.version.split()[0]}")
    pilot_python = root / "pilot" / ".venv" / "bin" / "python"
    if pilot_python.exists():
        print(f"pilot python   {_shell([str(pilot_python), '--version'])}")

    print("== model ==")
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    print(f"HF_HUB_OFFLINE {'1' if offline else 'not 1'}")
    if not offline:
        problems.append("HF_HUB_OFFLINE is not 1: run `export HF_HUB_OFFLINE=1`")
    snapshot = None
    try:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(MODEL_REPO, local_files_only=True)
        print(f"snapshot       {Path(snapshot).name}")
    except Exception as error:  # the hub raises several unrelated types here
        problems.append(f"the model is not in the local cache: {error}")

    print("== machine ==")
    free = free_gb(root)
    print(f"free disk      {free:.1f} GB")
    if free < MIN_FREE_GB:
        problems.append(f"less than {MIN_FREE_GB:.0f} GB of disk are free")
    print(f"power          {_shell(['pmset', '-g', 'ac']) or 'no charger reported'}")
    low_power = [
        line.strip()
        for line in _shell(["pmset", "-g"]).splitlines()
        if "lowpowermode" in line
    ]
    print(f"low power mode {low_power[0] if low_power else 'not reported'}")

    print("== prompt date ==")
    if snapshot is not None:
        pinned = today_date_line(snapshot, PROMPT_DATE)
        print(f"pinned         {pinned}")
        print(f"not pinned     {today_date_line(snapshot, None)}")
        if pinned != f"Today Date: {PROMPT_DATE}":
            problems.append(f"the pinned prompt does not say {PROMPT_DATE}")

    for problem in problems:
        print(f"FAIL: {problem}")
    print("preflight: " + ("FAILED" if problems else "OK"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
