"""The pinned script is the pilot's script with six named changes."""

from __future__ import annotations

import difflib

from tests.trial.pinned_parser import build_parser, PINNED, pinned_constant
from trial.tasks import PROMPT_DATE

PILOT = PINNED.parent.parent / "pilot" / "smoke.py"
ALLOWED = ("prompt_date", "date_string", "pilot", "done", "token_budget", '"x"')


def _diff() -> tuple[list[str], list[str]]:
    lines = difflib.ndiff(
        PILOT.read_text().splitlines(), PINNED.read_text().splitlines()
    )
    added, removed = [], []
    for line in lines:
        if line.startswith("+ "):
            added.append(line[2:])
        elif line.startswith("- "):
            removed.append(line[2:])
    return added, removed


def test_every_added_line_belongs_to_a_named_change() -> None:
    added, _ = _diff()
    unexplained = [
        line
        for line in added
        if not any(word in line.lower() for word in ALLOWED)
        and "--token-budget" not in line
    ]
    assert unexplained == []


def test_only_the_replaced_lines_of_the_pilot_are_missing() -> None:
    _, removed = _diff()
    assert [line.strip() for line in removed] == [
        'sys.path.insert(0, str(Path(__file__).parent / "vendor"))',
        "enable_thinking=False)",
        'log = (run_dir / "log.jsonl").open("a")',
        'emit({"event": "config", **{k: v for k, v in vars(args).items() '
        'if k != "fn"}, "trainable_params": n_train})',
        "if (args.eval_every and step % args.eval_every == 0) or out_of_time "
        "or step == args.steps:",
        "if out_of_time:",
    ]


def test_the_date_is_the_day_of_the_pilot() -> None:
    assert pinned_constant("PROMPT_DATE") == "26 Sep 2026" == PROMPT_DATE


def test_the_parser_has_the_pilot_defaults_plus_the_token_budget() -> None:
    args = ["train"]
    pinned = vars(build_parser(PINNED).parse_args(args))
    pilot = vars(build_parser(PILOT).parse_args(args))
    assert pinned.pop("token_budget") == 0
    assert pinned == pilot
    assert pilot["steps"] == 20 and pilot["model"] == "qwen3-0.6b"
