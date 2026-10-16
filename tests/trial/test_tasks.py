"""The task definitions are the pilot's, copied unchanged."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from tests.trial.pinned_parser import pinned_constant
from trial import tasks

PILOT = Path(__file__).resolve().parent.parent.parent / "pilot" / "smoke.py"
NAMES = {
    "DATASET",
    "TIERS",
    "SPLIT_SEED_OFFSET",
    "SYSTEM_PROMPT",
    "ANSWER_RE",
    "BOXED_RE",
    "NUMBER_RE",
    "make_tasks",
    "extract_answer",
}


def _pilot_namespace() -> dict[str, Any]:
    """Execute only the task definitions of the pilot's script, without MLX."""
    kept = []
    for node in ast.parse(PILOT.read_text()).body:
        if isinstance(node, ast.Assign) and node.targets[0].id in NAMES:
            kept.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in NAMES:
            kept.append(node)
    namespace: dict[str, Any] = {"re": __import__("re")}
    exec(compile(ast.Module(body=kept, type_ignores=[]), str(PILOT), "exec"), namespace)
    return namespace


def _function(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text())
    return next(
        ast.dump(node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_constants_equal_the_pilots() -> None:
    pilot = _pilot_namespace()
    assert tasks.DATASET == pilot["DATASET"]
    assert tasks.PILOT_LADDER == pilot["TIERS"]
    assert tasks.LADDERS["pilot"] is tasks.PILOT_LADDER
    assert tasks.SPLIT_SEED_OFFSET == pilot["SPLIT_SEED_OFFSET"]
    assert tasks.SYSTEM_PROMPT == pilot["SYSTEM_PROMPT"]
    for name in ("ANSWER_RE", "BOXED_RE", "NUMBER_RE"):
        assert getattr(tasks, name) == pilot[name]


def test_tasks_equal_the_pilots() -> None:
    pilot = _pilot_namespace()
    for split, per_tier in (("train", 2000), ("test", 100)):
        ours = tasks.make_tasks(split, per_tier, tasks.PILOT_LADDER)
        assert ours == pilot["make_tasks"](split, per_tier)


def test_extract_answer_is_the_pilots_function() -> None:
    assert _function(Path(tasks.__file__), "extract_answer") == _function(
        PILOT, "extract_answer"
    )
    assert tasks.extract_answer("so <answer> 1,234 </answer>") == "1234"


def test_the_prompt_date_is_the_pinned_scripts() -> None:
    assert tasks.PROMPT_DATE == pinned_constant("PROMPT_DATE")
