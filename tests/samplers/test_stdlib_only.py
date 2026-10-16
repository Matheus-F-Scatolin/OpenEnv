# SPDX-License-Identifier: BSD-3-Clause

"""The library imports nothing outside the standard library."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import samplers

PACKAGE = Path(samplers.__file__).parent


def test_every_absolute_import_is_standard_library() -> None:
    foreign = []
    files = sorted(PACKAGE.glob("*.py"))
    assert len(files) >= 4
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            else:
                continue
            foreign += [
                f"{path.name}: {name}"
                for name in names
                if name.split(".")[0] not in sys.stdlib_module_names
            ]
    assert foreign == []


def test_every_file_starts_with_the_license_line() -> None:
    files = sorted(PACKAGE.glob("*.py")) + sorted(Path(__file__).parent.glob("*.py"))
    assert len(files) >= 11
    for path in files:
        first = path.read_text().splitlines()[0]
        assert first == "# SPDX-License-Identifier: BSD-3-Clause", path.name
