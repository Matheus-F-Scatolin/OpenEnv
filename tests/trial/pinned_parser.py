"""Build the command-line parser of `trial/smoke_pinned.py` without importing MLX."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
PINNED = ROOT / "trial" / "smoke_pinned.py"


def pinned_constant(name: str, path: Path = PINNED) -> Any:
    """Return the value of a module-level literal assignment of the script."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and node.targets[0].id == name:
            return ast.literal_eval(node.value)
    raise KeyError(name)


def build_parser(path: Path = PINNED) -> argparse.ArgumentParser:
    """Run the statements of the script's `main()` that build its parser."""
    tree = ast.parse(path.read_text())
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    body = [
        statement
        for statement in main.body
        if "parse_args" not in ast.unparse(statement)
        and "args.fn" not in ast.unparse(statement)
    ]
    namespace: dict[str, Any] = {
        "argparse": argparse,
        "__doc__": "",
        "MODELS": pinned_constant("MODELS", path),
        "cmd_baseline": None,
        "cmd_train": None,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["ap"]
