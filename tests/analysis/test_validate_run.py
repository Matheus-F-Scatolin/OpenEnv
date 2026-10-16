"""The run checker: complete, consistent run folders."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("reasoning_gym")

from analysis.validate_run import check_run, main, validate
from tests.trial.fake_run import run

PILOT_SMOKE = (
    Path(__file__).resolve().parent.parent.parent
    / "pilot"
    / "runs"
    / "smoke"
    / "llama3.2-1b-uniform-s0"
)


@pytest.fixture(scope="module")
def good(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("runs") / "trial"
    return run(base, "fake-curriculum-s0", steps=0, token_budget=6000)


def damaged(good: Path, tmp_path: Path, edit, name: str = "log.jsonl") -> Path:
    """Copy a good run folder and change one of its files, line by line."""
    copy = tmp_path / "trial" / good.name
    shutil.copytree(good, copy)
    lines = [json.loads(line) for line in (copy / name).read_text().splitlines()]
    (copy / name).write_text("".join(json.dumps(x) + "\n" for x in edit(lines)))
    return copy


def version_1(tmp_path: Path, date: str = "26 Sep 2026") -> Path:
    """The stored pilot log, as the pinned script would have written it."""
    copy = tmp_path / "ten-run-pinned" / PILOT_SMOKE.name
    copy.mkdir(parents=True)
    lines = (PILOT_SMOKE / "log.jsonl").read_text().splitlines()
    config = {**json.loads(lines[0]), "prompt_date": date}
    done = {"event": "done", "stop_reason": "steps", "step": 20}
    (copy / "log.jsonl").write_text(
        "\n".join([json.dumps(config), *lines[1:], json.dumps(done)]) + "\n"
    )
    return copy


def test_a_good_run_is_ok(good: Path) -> None:
    assert check_run(good)[0] == "OK"


def _drop_done(lines):
    return lines[:-1]


def _change_tokens(lines):
    step = next(x for x in lines if x["event"] == "step")
    step["groups"][0]["tokens"][0] += 1
    return lines


def _change_total(lines):
    next(x for x in lines if x["event"] == "step")["gen_tokens_total"] += 1
    return lines


def _drop_step(lines):
    second = next(x for x in lines if x["event"] == "step" and x["step"] == 2)
    return [x for x in lines if x is not second]


def _drop_group(lines):
    next(x for x in lines if x["event"] == "step")["groups"].pop()
    return lines


def _no_date(lines):
    lines[0]["prompt_example"] = lines[0]["prompt_example"].replace("26 Sep", "12 Oct")
    return lines


def _no_commit(lines):
    del lines[0]["git_commit"]
    return lines


def _change_success(lines):
    next(x for x in lines if x["event"] == "eval")["success_rate"] += 0.01
    return lines


def _stop_early(lines):
    lines[0]["token_budget"] *= 10
    return lines


def _drop_final_eval(lines):
    last = max(x["step"] for x in lines if x["event"] == "eval")
    return [x for x in lines if not (x["event"] == "eval" and x["step"] == last)]


def _change_eval_total(lines):
    [x for x in lines if x["event"] == "eval"][-1]["gen_tokens_total"] += 1
    return lines


@pytest.mark.parametrize(
    "edit,word",
    [
        (_drop_done, "done"),
        (_change_tokens, "tokens"),
        (_change_total, "gen_tokens_total"),
        (_drop_step, "step"),
        (_drop_group, "groups"),
        (_no_date, "Today Date"),
        (_no_commit, "git_commit"),
        (_change_success, "success_rate"),
        (_stop_early, "budget"),
        (_drop_final_eval, "no evaluation at the last step"),
        (_change_eval_total, "eval at step"),
    ],
)
def test_a_damaged_log_is_reported(good: Path, tmp_path: Path, edit, word: str) -> None:
    status = check_run(damaged(good, tmp_path, edit))[0]
    assert status != "OK" and word in status


def test_a_missing_evaluation_record_is_reported(good: Path, tmp_path: Path) -> None:
    copy = damaged(good, tmp_path, lambda lines: lines[:-1], name="evals.jsonl")
    assert "records" in check_run(copy)[0]


def test_evaluation_records_must_carry_the_token_count_of_their_evaluation(
    good: Path, tmp_path: Path
) -> None:
    def edit(records):
        records[-1]["gen_tokens_total"] += 1
        return records

    status = check_run(damaged(good, tmp_path, edit, name="evals.jsonl"))[0]
    assert status != "OK" and "evals.jsonl" in status


def test_version_1(tmp_path: Path) -> None:
    assert check_run(version_1(tmp_path))[0] == "OK (version 1)"


class TestVersion2Done:
    @pytest.mark.parametrize("field", ["step", "gen_tokens_total", "episodes"])
    @pytest.mark.parametrize("missing", [False, True])
    def test_final_counters_must_match_recorded_steps(
        self, good: Path, tmp_path: Path, field: str, missing: bool
    ) -> None:
        def edit(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if missing:
                del events[-1][field]
            else:
                events[-1][field] += 1
            return events

        status = check_run(damaged(good, tmp_path, edit))[0]
        assert status != "OK" and field in status

    def test_a_false_stop_reason_cannot_hide_an_unfinished_budget(
        self, good: Path, tmp_path: Path
    ) -> None:
        def edit(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
            events[0]["token_budget"] *= 10
            events[-1]["stop_reason"] = "steps"
            return events

        status = check_run(damaged(good, tmp_path, edit))[0]
        assert status != "OK" and "budget" in status

    @pytest.mark.parametrize("reason", ["steps", "baseline", "unknown", None])
    def test_the_stop_reason_must_agree_with_the_reached_limit(
        self, good: Path, tmp_path: Path, reason: str | None
    ) -> None:
        def edit(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
            events[-1]["stop_reason"] = reason
            return events

        status = check_run(damaged(good, tmp_path, edit))[0]
        assert status != "OK" and "stop_reason" in status

    def test_training_cannot_finish_before_its_step_limit(
        self, good: Path, tmp_path: Path
    ) -> None:
        def edit(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
            events[0]["token_budget"] = 0
            events[0]["steps"] = events[-1]["step"] + 1
            events[-1]["stop_reason"] = "steps"
            return events

        status = check_run(damaged(good, tmp_path, edit))[0]
        assert status != "OK" and "limit" in status

    @pytest.mark.parametrize("limit", ["steps", "token_budget"])
    def test_training_cannot_continue_after_a_limit(
        self, good: Path, tmp_path: Path, limit: str
    ) -> None:
        def edit(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
            first_step = next(e for e in events if e["event"] == "step")
            events[0][limit] = 1 if limit == "steps" else first_step["gen_tokens_total"]
            events[-1]["stop_reason"] = limit
            return events

        status = check_run(damaged(good, tmp_path, edit))[0]
        assert status != "OK" and "continued" in status

    @pytest.mark.parametrize(
        "steps,budget,reason",
        [(2, 100_000, "steps"), (100, 1000, "token_budget"), (2, 1, "token_budget")],
    )
    def test_the_first_limit_reached_ends_training(
        self, tmp_path: Path, steps: int, budget: int, reason: str
    ) -> None:
        run_dir = run(tmp_path, steps=steps, token_budget=budget)
        events = [
            json.loads(line)
            for line in (run_dir / "log.jsonl").read_text().splitlines()
        ]
        assert events[-1]["stop_reason"] == reason
        assert check_run(run_dir)[0] == "OK"

    def test_token_budget_has_priority_when_both_limits_are_reached(
        self, good: Path, tmp_path: Path
    ) -> None:
        def edit(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
            events[0]["steps"] = events[-1]["step"]
            return events

        run_dir = damaged(good, tmp_path, edit)
        assert check_run(run_dir)[0] == "OK"
        events = [
            json.loads(line)
            for line in (run_dir / "log.jsonl").read_text().splitlines()
        ]
        events[-1]["stop_reason"] = "steps"
        (run_dir / "log.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events)
        )
        assert "stop_reason" in check_run(run_dir)[0]

    def test_a_baseline_reason_cannot_hide_training(
        self, good: Path, tmp_path: Path
    ) -> None:
        def edit(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
            events[0]["cmd"] = "baseline"
            events[-1]["stop_reason"] = "baseline"
            return events

        status = check_run(damaged(good, tmp_path, edit))[0]
        assert status != "OK" and "baseline" in status


def test_folder_of_runs_and_exit_code(
    good: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shutil.copytree(good, tmp_path / "trial" / good.name)
    shutil.copytree(good, tmp_path / "trial" / (good.name + ".incomplete-1"))
    version_1(tmp_path)
    assert main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert out.count(": OK\n") == 1 and "OK (version 1)" in out
    assert "skipped (incomplete, moved aside by the queue)" in out
    damaged(good, tmp_path / "other", _drop_done)
    assert main([str(tmp_path)]) == 1


class TestAcrossRuns:
    def test_different_prompt_dates_are_reported(
        self, good: Path, tmp_path: Path
    ) -> None:
        shutil.copytree(good, tmp_path / "trial" / good.name)
        version_1(tmp_path, date="12 Oct 2026")
        lines = validate([tmp_path])
        assert any("prompt_date" in line for line in lines[-1:])

    def test_version_1_and_2_with_the_same_date_agree(
        self, good: Path, tmp_path: Path
    ) -> None:
        shutil.copytree(good, tmp_path / "trial" / good.name)
        version_1(tmp_path)
        assert all("differ" not in line for line in validate([tmp_path]))

    def test_different_library_versions_are_reported(
        self, good: Path, tmp_path: Path
    ) -> None:
        def edit(lines):
            lines[0]["versions"]["mlx"] = "0.0.0"
            return lines

        shutil.copytree(good, tmp_path / "trial" / good.name)
        damaged(good, tmp_path / "cost-full", edit)
        assert any("versions" in line for line in validate([tmp_path]))

    def test_gate_folders_are_left_out(self, good: Path, tmp_path: Path) -> None:
        def edit(lines):
            lines[0]["versions"]["mlx"] = "0.0.0"
            return lines

        shutil.copytree(good, tmp_path / "trial" / good.name)
        copy = damaged(good, tmp_path / "x", edit)
        (tmp_path / "gate-queue").mkdir()
        shutil.move(str(copy), tmp_path / "gate-queue" / copy.name)
        assert all("differ" not in line for line in validate([tmp_path]))
