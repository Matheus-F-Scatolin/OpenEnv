"""The numerical code of the MLX policy is the pilot's, unchanged."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
PILOT = ast.parse((ROOT / "pilot" / "smoke.py").read_text())
OURS = ast.parse((ROOT / "trial" / "mlx_policy.py").read_text())


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _assignment(tree: ast.Module, name: str) -> str:
    return next(
        ast.dump(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and node.targets[0].id == name
    )


@pytest.mark.parametrize(
    "name", ["load_policy", "generate", "sequence_logp_loss", "train_step"]
)
def test_copied_functions_have_the_pilots_syntax_tree(name: str) -> None:
    assert ast.dump(_function(OURS, name)) == ast.dump(_function(PILOT, name))


def test_models_table_is_the_pilots() -> None:
    assert _assignment(OURS, "MODELS") == _assignment(PILOT, "MODELS")


def test_prompt_tokens_differs_only_by_the_date() -> None:
    ours = _function(OURS, "prompt_tokens")
    assert ours.args.args.pop().arg == "date_string"
    calls = [
        node
        for node in ast.walk(ours)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "apply_chat_template"
    ]
    assert len(calls) == 1
    dates = [k for k in calls[0].keywords if k.arg == "date_string"]
    assert len(dates) == 1 and ast.unparse(dates[0].value) == "date_string"
    calls[0].keywords.remove(dates[0])
    assert ast.dump(ours) == ast.dump(_function(PILOT, "prompt_tokens"))


def test_only_this_file_imports_mlx() -> None:
    offenders = []
    for folder in ("samplers", "trial", "analysis"):
        for path in (ROOT / folder).rglob("*.py"):
            if path.name in ("mlx_policy.py", "smoke_pinned.py"):
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                if any(n.split(".")[0] in ("mlx", "mlx_lm") for n in names):
                    offenders.append(path.name)
    assert offenders == []
