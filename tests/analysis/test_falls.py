# SPDX-License-Identifier: BSD-3-Clause

"""Describe falls between consecutive evaluations on synthetic logs, without a model."""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

import pytest
from analysis.falls import build_falls, worst_cut_share, write_falls, wrong_sum
from analysis.reference import degenerate, DEGENERATE_SHARE

STEPS = 125
EVALS = (0, 50, 100, 125)
QUESTIONS = 10


def _tokens(step: int) -> int:
    return step * 40_000


def _write(root: Path, method: str, solved: dict[int, int], texts: bool) -> Path:
    """One run on level 2 only: `solved[step]` questions are right at each evaluation."""
    folder = root / f"fake-{method}-s5"
    folder.mkdir(parents=True)
    events = [dict(event="config", sampler=method, seed=5)]
    for step in range(1, STEPS + 1):
        cut = {90: [True, False], 95: [True, True]}.get(step, [False, False])
        if method == "curriculum":
            cut = [False, False]
        events.append(
            dict(
                event="step",
                step=step,
                gen_tokens_total=_tokens(step),
                groups=[dict(tier=2, rewards=[1.0, 0.0], truncated=cut)],
            )
        )
    records, lines = [], []
    for step in EVALS:
        for i in range(QUESTIONS):
            right = i < solved[step]
            lost_here = method == "uniform" and step == 100 and 4 <= i < 8
            records.append(
                dict(
                    step=step,
                    gen_tokens_total=_tokens(step),
                    task=f"q{i}",
                    tier=2,
                    reward=1.0 if right else 0.0,
                    tokens=8,
                    truncated=lost_here and i < 6,
                    formatted=False,
                )
            )
            text = (
                "1. 500 + 406 = 906\n2. 906 + 695 = 1,601\nThe final answer is 1,601."
            )
            if lost_here and i >= 6:
                text = "1. 500 + 406 = 906\n2. 906 + 695 = 1,701\nThe final answer is 1,701."
            lines.append(json.dumps(dict(step=str(step), task=f"q{i}", text=text)))
    (folder / "log.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (folder / "evals.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    if texts:
        with gzip.open(folder / "eval_texts.jsonl.gz", "wt") as handle:
            handle.write("\n".join(lines) + "\n")
    return folder


def _report(root: Path, *, texts: bool = True) -> dict:
    """A report as `build_report` shapes it, for one pair of seed 5."""
    shapes = {
        "uniform": {0: 8, 50: 8, 100: 4, 125: 9},
        "curriculum": {0: 10, 50: 10, 100: 10, 125: 10},
    }
    runs = []
    for method, solved in shapes.items():
        folder = _write(root, method, solved, texts)
        success = {s: solved[s] / QUESTIONS for s in EVALS}
        per_token = 0.65 if method == "uniform" else 1.0
        runs.append(
            dict(
                path=str(folder),
                method=method,
                seed=5,
                evaluations=[
                    dict(
                        step=s,
                        tokens=_tokens(s),
                        scopes={"main": dict(n=QUESTIONS, success=success[s], se=0.0)},
                    )
                    for s in EVALS
                ],
                per_token={"main": per_token},
                per_episode={"main": success[100]},
            )
        )
    return dict(main_levels=[2], episode_windows={5: 100}, runs=runs)


class TestWrongSum:
    @pytest.mark.parametrize(
        ("text", "wrong"),
        [
            ("1. 500 + 406 = 906\n2. 906 + 695 = 1,601", False),
            ("3. 1,601 + 411 = 2,212", True),
            ("1. 389 - 971 = -582\n2. -582 + 489 = -93", False),
            ("1. 5220 - -3913 = 9133", False),
            ("1336 + 2697 + 6093 = 10126", False),
            ("1336 + 2697 + 6093 = 1,0126", False),
            ("-3913 - 5849 = -9772", True),
            ("The final answer is 2,012.", False),
            ("3. 555 - 842 = -287\n\n-287\n\n-287 = 287", False),
            ("Total: 12\n-3 + 5 = 2", False),
            ("81617 - 83340 = - 1723", False),
            ("81617 - 83340 = - 9663", True),
            ("59551.82 + 79225 = 1,38776.82", False),
            ("906 + 695 = 1,601.", False),
            ("906 + 695 = 1,701.", True),
        ],
    )
    def test_checks_every_written_sum(self, text: str, wrong: bool) -> None:
        assert wrong_sum(text) is wrong


class TestWorstCutShare:
    def test_reads_the_worst_full_window_like_the_degenerate_rule(self) -> None:
        steps = [
            dict(step=s, groups=[dict(truncated=[s in (60, 61), False])])
            for s in range(1, 101)
        ]
        assert worst_cut_share(steps) == (2 / 100, 12)

    def test_fewer_steps_than_one_window_have_no_share(self) -> None:
        steps = [dict(step=s, groups=[dict(truncated=[True])]) for s in range(1, 50)]
        assert worst_cut_share(steps) is None

    @pytest.mark.parametrize("seed", range(20))
    def test_agrees_with_the_degenerate_rule(self, seed: int) -> None:
        rng = random.Random(seed)
        bias = rng.random()
        steps = [
            dict(
                step=s,
                groups=[
                    dict(
                        truncated=[
                            rng.random() < bias for _ in range(rng.randint(0, 4))
                        ]
                    )
                ],
            )
            for s in range(1, rng.randint(30, 120))
        ]
        worst = worst_cut_share(steps)
        assert degenerate(steps) is (worst is not None and worst[0] >= DEGENERATE_SHARE)


class TestFalls:
    def test_a_fall_is_described_from_the_records_and_the_texts(
        self, tmp_path: Path
    ) -> None:
        result = build_falls(_report(tmp_path))
        assert result["intervals"] == {"uniform": 3, "curriculum": 3}
        (fall,) = result["falls"]
        assert (fall["method"], fall["seed"], fall["step_before"], fall["step"]) == (
            "uniform",
            5,
            50,
            100,
        )
        assert (fall["solved_before"], fall["solved_at"], fall["questions"]) == (
            8,
            4,
            10,
        )
        assert fall["drop_points"] == pytest.approx(40.0)
        assert (fall["lost"], fall["gained"]) == (4, 0)
        assert (fall["lost_cut"], fall["lost_wrong_sum"]) == (2, 2)
        assert (fall["unsolved_before"], fall["unsolved_before_wrong_sum"]) == (2, 0)
        assert (fall["next_step"], fall["solved_next"]) == (125, 9)
        assert fall["recovered_step"] == 125
        assert fall["episode_marks"] == [100]
        assert fall["token_marks"] == [
            dict(mark=4_000_000, read="at this evaluation"),
            *(
                dict(mark=m, read="interpolated")
                for m in (4_250_000, 4_500_000, 4_750_000)
            ),
        ]
        assert (fall["training_cut"], fall["training_steps_with_cut"]) == (3, 2)
        worst = {(w["method"], w["seed"]): w for w in result["worst_windows"]}
        assert worst[("uniform", 5)]["share"] == pytest.approx(3 / 100)

    def test_without_texts_the_wrong_sums_are_unavailable_not_zero(
        self, tmp_path: Path
    ) -> None:
        (fall,) = build_falls(_report(tmp_path, texts=False))["falls"]
        assert fall["lost_wrong_sum"] is None
        assert fall["lost_cut"] == 2

    def test_what_if_reproduces_the_report_then_replaces_large_falls(
        self, tmp_path: Path
    ) -> None:
        what_if = build_falls(_report(tmp_path))["what_if"]
        assert what_if["replaced"] == [
            dict(
                method="uniform",
                seed=5,
                step=100,
                success=0.4,
                replacement=pytest.approx(0.85),
            )
        ]
        assert what_if["per_episode"]["differences"] == {5: pytest.approx(15.0)}
        assert what_if["per_token"]["differences"] == {5: pytest.approx(12.5)}
        assert what_if["ahead_on_every_seed"] is True
        assert what_if["seeds_compared"] == [5]

    def test_a_report_the_what_if_cannot_reproduce_is_refused(
        self, tmp_path: Path
    ) -> None:
        report = _report(tmp_path)
        report["runs"][0]["per_token"]["main"] = 0.5
        with pytest.raises(ValueError, match="reproduce"):
            build_falls(report)

    def test_without_a_complete_pair_nothing_is_claimed(self, tmp_path: Path) -> None:
        report = _report(tmp_path)
        report["runs"] = report["runs"][:1]
        what_if = build_falls(report)["what_if"]
        assert what_if["seeds_compared"] == []
        assert what_if["ahead_on_every_seed"] is None

    def test_malformed_steps_leave_the_run_out_and_say_why(
        self, tmp_path: Path
    ) -> None:
        report = _report(tmp_path)
        log = Path(report["runs"][0]["path"]) / "log.jsonl"
        events = [json.loads(line) for line in log.read_text().splitlines()]
        del events[5]["groups"][0]["truncated"]
        log.write_text("".join(json.dumps(e) + "\n" for e in events))
        result = build_falls(report)
        assert result["falls"] == []
        assert [(u["method"], u["seed"]) for u in result["unavailable"]] == [
            ("uniform", 5)
        ]
        assert result["what_if"]["ahead_on_every_seed"] is None

    def test_unreadable_texts_leave_wrong_sums_unavailable(
        self, tmp_path: Path
    ) -> None:
        report = _report(tmp_path)
        texts = Path(report["runs"][0]["path"]) / "eval_texts.jsonl.gz"
        texts.write_bytes(texts.read_bytes()[:40])
        result = build_falls(report)
        (fall,) = result["falls"]
        assert fall["lost_wrong_sum"] is None
        assert fall["unsolved_before_wrong_sum"] is None
        assert "texts" in result["unavailable"][0]["reason"]

    def test_written_atomically_with_rounded_tables(self, tmp_path: Path) -> None:
        files = write_falls(build_falls(_report(tmp_path / "in")), tmp_path / "out")
        assert [f.name for f in files] == ["falls.json", "falls.md"]
        text = files[1].read_text()
        assert "after the results" in text
        assert "| uniform | 5 | 50 → 100 |" in text
        assert "| 80.0% → 40.0% |" in text
        saved = json.loads(files[0].read_text())
        assert saved["falls"][0]["drop_points"] == pytest.approx(40.0)
        assert not list((tmp_path / "out").glob("*.tmp"))
