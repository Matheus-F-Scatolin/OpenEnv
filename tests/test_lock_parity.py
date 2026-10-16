"""The project lock keeps every library at the version the pilot used."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _versions(lock_path: Path) -> dict[str, set[str]]:
    """Map each locked package to its set of versions, without the project itself."""
    with lock_path.open("rb") as handle:
        lock = tomllib.load(handle)
    versions: dict[str, set[str]] = {}
    for package in lock["package"]:
        if package.get("source") == {"virtual": "."}:
            continue
        versions.setdefault(package["name"], set()).add(package["version"])
    return versions


def changed_packages(pilot_lock: Path, project_lock: Path) -> list[str]:
    """Names of the pilot's packages that the project lock lacks or pins differently."""
    pilot = _versions(pilot_lock)
    project = _versions(project_lock)
    return sorted(name for name in pilot if project.get(name) != pilot[name])


def test_project_lock_keeps_the_pilot_versions() -> None:
    assert changed_packages(ROOT / "pilot" / "uv.lock", ROOT / "uv.lock") == []


def test_a_changed_version_is_reported(tmp_path: Path) -> None:
    text = (ROOT / "uv.lock").read_text()
    marker = 'name = "mlx"\nversion = "'
    assert marker in text
    start = text.index(marker) + len(marker)
    end = text.index('"', start)
    changed = tmp_path / "uv.lock"
    changed.write_text(text[:start] + "0.0.0" + text[end:])
    assert changed_packages(ROOT / "pilot" / "uv.lock", changed) == ["mlx"]
