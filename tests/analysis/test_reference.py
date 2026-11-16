# SPDX-License-Identifier: BSD-3-Clause

"""Apply the reference pre-registration to synthetic logs, without a model."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from analysis.reference import (
    build_report,
    degenerate,
    main,
    measure_at_tokens,
    paired_summary,
    SEEDS,
    T_CRITICAL,
    write_report,
)
from trial.loop import RunConfig, summarize_eval
from trial.tasks import PROMPT_DATE


def _write_run(
    root: Path,
    method: str = "uniform",
    seed: int = 5,
    *,
    steps: int = 200,
    ladder: str = "rfc",
    date: str = "2026-11-15T00:00:00+00:00",
    complete: bool = True,
) -> Path:
    """Write internally consistent small records on the full token/step axes."""
    from dataclasses import asdict

    folder = root / f"fake-{method}-s{seed}"
    folder.mkdir(parents=True)
    config = asdict(
        RunConfig(model="fake", out=str(root), sampler=method, seed=seed, ladder=ladder)
    )
    config.update(
        event="config",
        schema_version=2,
        prompt_date=PROMPT_DATE,
        prompt_example=f"Today Date: {PROMPT_DATE}",
        model_snapshot="synthetic",
        versions={"fake": "1"},
        git_commit="synthetic",
        utc_time=date,
        sampler_settings={},
        token_budget=5_000_000,
        eval_per_tier=2,
        group_size=2,
        prompts_per_step=1,
    )
    marks = {0, steps, *range(100, steps + 1, 100)}
    marks.update(
        math.ceil(mark * steps / 5_000_000)
        for mark in range(250_000, 5_000_001, 250_000)
    )
    events, records = [config], []
    for step in range(steps + 1):
        total = step * 5_000_000 // steps
        if step:
            tokens = total - (step - 1) * 5_000_000 // steps
            events.append(
                dict(
                    event="step",
                    step=step,
                    gen_tokens=tokens,
                    gen_tokens_total=total,
                    groups_with_signal=1,
                    train_elapsed_s=step * 0.5,
                    groups=[
                        dict(
                            tier=step % 10,
                            rewards=[1.0, 0.0],
                            tokens=[tokens // 2, tokens - tokens // 2],
                            truncated=[False, False],
                        )
                    ],
                    sampler={"p_hat": {str(t): 0.8 for t in range(10)}},
                )
            )
        if step in marks:
            own = [
                dict(
                    step=step,
                    gen_tokens_total=total,
                    task=f"t{tier}-{i}",
                    tier=tier,
                    reward=float(i == 0 or (method == "curriculum" and step > 0)),
                    tokens=8,
                    formatted=True,
                    truncated=False,
                )
                for tier in range(10)
                for i in range(2)
            ]
            records.extend(own)
            events.append(
                dict(
                    event="eval",
                    step=step,
                    gen_tokens_total=total,
                    **summarize_eval(own),
                )
            )
    if complete:
        events.append(
            dict(
                event="done",
                step=steps,
                gen_tokens_total=5_000_000,
                episodes=2 * steps,
                stop_reason="token_budget",
            )
        )
    (folder / "log.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (folder / "evals.jsonl").write_text("".join(json.dumps(e) + "\n" for e in records))
    return folder


class TestStatistics:
    def test_five_paired_differences_and_known_df4_probability(self) -> None:
        # Mean 2, sample sd sqrt(5), so t=2 exactly.
        result = paired_summary(dict(zip(SEEDS, [-1, 1, 2, 3, 5])))
        assert result["t"] == pytest.approx(2)
        assert result["p"] == pytest.approx(0.11611652351681556)
        assert result["interval"] == pytest.approx([2 - T_CRITICAL, 2 + T_CRITICAL])
        assert result["verdict"] == "no significant difference with five seeds"
        assert result["support"] == "shown by the interval"

    def test_mean_only_and_significant_worse_are_distinct(self) -> None:
        assert (
            paired_summary(dict(zip(SEEDS, [-10, -5, 0, 5, 10])))["support"]
            == "on the mean only"
        )
        result = paired_summary(dict(zip(SEEDS, [-7, -6, -5, -4, -3])))
        assert result["verdict"] == "worse"
        assert result["support"] == "not supported"

    @pytest.mark.parametrize("n", range(5))
    def test_no_inference_with_fewer_than_five_pairs(self, n: int) -> None:
        result = paired_summary({s: 5.0 for s in SEEDS[:n]})
        assert result["n"] == n
        assert result["p"] is None
        assert result["interval"] is None

    @pytest.mark.parametrize("value,t,p", [(0, 0, 1), (5, "inf", 0), (-5, "-inf", 0)])
    def test_constant_differences_have_an_explicit_limit(
        self, value: float, t: float | str, p: float
    ) -> None:
        result = paired_summary(dict.fromkeys(SEEDS, value))
        assert (result["t"], result["p"]) == (t, p)
        json.dumps(result, allow_nan=False)

    def test_constants_match_the_pre_registration_and_lists(self) -> None:
        root = Path(__file__).resolve().parents[2]
        entries = json.loads((root / "trial/lists/reference.json").read_text())
        assert sorted(
            {int(e["cmd"][e["cmd"].index("--seed") + 1]) for e in entries}
        ) == list(SEEDS)
        assert str(T_CRITICAL) in (root / "notes/expectations.md").read_text()

    @pytest.mark.parametrize("ladder", ["rfc", "staircase"])
    def test_main_levels_match_the_actual_question_overlap(self, ladder: str) -> None:
        pytest.importorskip("reasoning_gym")
        from analysis.overlap import overlap
        from analysis.reference import MAIN_LEVELS

        assert (
            tuple(r["level"] for r in overlap(ladder) if r["overlap"] == 0)
            == MAIN_LEVELS[ladder]
        )


class TestWindows:
    def test_token_crossing_uses_the_crossing_step_not_interpolation(self) -> None:
        evaluations = [
            dict(step=0, tokens=0, value=0.0),
            dict(step=2, tokens=120, value=1.0),
        ]
        steps = [dict(step=1, gen_tokens_total=60), dict(step=2, gen_tokens_total=120)]
        assert measure_at_tokens(evaluations, steps, 100) == 1.0
        assert measure_at_tokens(evaluations, steps, 50) == pytest.approx(50 / 120)
        assert measure_at_tokens(evaluations, steps, 200) is None

    def test_degenerate_means_a_50_step_aggregate_at_90_percent(self) -> None:
        cut = dict(groups=[dict(truncated=[True] * 9 + [False])])
        assert not degenerate([cut] * 49)
        assert degenerate([cut] * 50)
        assert not degenerate(
            [dict(groups=[dict(truncated=[True] * 8 + [False] * 2)])] * 60
        )
        assert degenerate(
            [dict(groups=[dict(truncated=[False] * 10)])] * 5 + [cut] * 50
        )


class TestReferenceReport:
    def test_missing_reward_is_unavailable_and_an_incomplete_repeat_cannot_win(
        self, tmp_path: Path
    ) -> None:
        p = _write_run(tmp_path) / "evals.jsonl"
        records = [json.loads(line) for line in p.read_text().splitlines()]
        for record in records:
            if record["step"] > 0 and record["tier"] == 2:
                record["reward"] = None
        p.write_text("".join(json.dumps(r) + "\n" for r in records))
        _write_run(tmp_path, "curriculum")
        result = build_report([tmp_path], ladder="rfc")
        assert result["comparisons"]["main"]["per_token"]["n"] == 0
        assert result["runs"][0]["per_token"]["main"] is None
        assert result["runs"][0]["validation"] != "OK"

    def test_shorter_pair_never_uses_the_longer_arms_later_evaluations(
        self, tmp_path: Path
    ) -> None:
        p = _write_run(tmp_path, steps=300) / "evals.jsonl"
        records = [json.loads(line) for line in p.read_text().splitlines()]
        for record in records:
            if record["step"] > 200:
                record["reward"] = 0.0
        p.write_text("".join(json.dumps(r) + "\n" for r in records))
        _write_run(tmp_path, "curriculum", steps=200)
        result = build_report([tmp_path], ladder="rfc")
        assert result["comparisons"]["main"]["per_episode"]["differences"] == {5: 50.0}

    def test_overlapping_levels_cannot_change_the_main_measure(
        self, tmp_path: Path
    ) -> None:
        _write_run(tmp_path)
        p = _write_run(tmp_path, "curriculum") / "evals.jsonl"
        records = [json.loads(line) for line in p.read_text().splitlines()]
        for record in records:
            if record["step"] and record["tier"] < 2:
                record["reward"] = 0.0
        p.write_text("".join(json.dumps(r) + "\n" for r in records))
        result = build_report([tmp_path], ladder="rfc")
        assert result["comparisons"]["main"]["per_token"]["differences"][5] == 50.0
        assert result["comparisons"]["all"]["per_token"]["differences"][
            5
        ] == pytest.approx(30.0)

    def test_five_pairs_enable_only_the_two_main_tests(self, tmp_path: Path) -> None:
        for seed in SEEDS:
            for method in ("uniform", "curriculum"):
                _write_run(tmp_path, method, seed)
        result = build_report([tmp_path], ladder="rfc")
        assert result["comparisons"]["main"]["per_token"]["n"] == 5
        assert result["comparisons"]["main"]["per_token"]["verdict"] == "better"
        assert result["comparisons"]["all"]["per_token"]["p"] is None
        assert result["supports_proposal"] is True
        assert result["method_values"]["main"]["per_token"]["uniform"][
            "values"
        ] == dict.fromkeys(SEEDS, 50.0)
        assert (
            result["method_values"]["main"]["per_episode"]["curriculum"]["mean"]
            == 100.0
        )

    def test_paired_windows_question_error_and_all_level_views(
        self, tmp_path: Path
    ) -> None:
        _write_run(tmp_path, steps=300)
        _write_run(tmp_path, "curriculum", steps=200)
        result = build_report([tmp_path], ladder="rfc")
        assert result["comparisons"]["main"]["per_token"]["differences"] == {5: 50.0}
        assert result["comparisons"]["main"]["per_episode"]["differences"] == {5: 50.0}
        assert result["episode_windows"] == {5: 200}
        assert len(result["missing"]) == 8
        first = result["runs"][0]
        assert first["validation"] == "OK"
        main_eval = first["evaluations"][0]["scopes"]["main"]
        assert main_eval["n"] == 16
        assert main_eval["se"] == pytest.approx(math.sqrt(0.25 / 16))
        assert set(first["evaluations"][0]["scopes"]) == {
            "main",
            "all",
            "excluded",
            *(f"level_{t}" for t in range(10)),
        }
        assert first["diagnostics"]["updates"] == 300
        assert sum(
            t["episode_share"] for t in first["diagnostics"]["levels"].values()
        ) == pytest.approx(1)
        assert not result["warnings"]

    def test_staircase_excludes_three_levels_and_reports_mixed_input(
        self, tmp_path: Path
    ) -> None:
        _write_run(tmp_path / "fallback", ladder="staircase")
        _write_run(tmp_path / "other", ladder="rfc")
        result = build_report([tmp_path], ladder="staircase")
        assert result["runs"][0]["evaluations"][0]["scopes"]["main"]["n"] == 14
        assert any("ladder" in w for w in result["warnings"])

    def test_first_complete_invalid_run_is_never_replaced(self, tmp_path: Path) -> None:
        first = _write_run(tmp_path / "z-first")
        config = (
            (first / "log.jsonl")
            .read_text()
            .replace(
                '"prompt_example": "Today Date: 26 Sep 2026"',
                '"prompt_example": "wrong"',
            )
        )
        (first / "log.jsonl").write_text(config)
        _write_run(tmp_path / "a-repeat", date="2026-11-16T00:00:00+00:00")
        _write_run(
            tmp_path / "unfinished", complete=False, date="2026-11-14T00:00:00+00:00"
        )
        result = build_report([tmp_path], ladder="rfc")
        assert result["runs"][0]["path"] == str(first)
        assert result["runs"][0]["validation"] != "OK"
        assert sorted(r["status"] for r in result["inventory"]) == [
            "incomplete",
            "repeat",
            "selected",
        ]

    def test_initial_results_and_provenance_differences_are_reported(
        self, tmp_path: Path
    ) -> None:
        _write_run(tmp_path)
        p = _write_run(tmp_path, "curriculum") / "evals.jsonl"
        records = p.read_text().splitlines()
        r = json.loads(records[0])
        r["reward"] = 0.0
        p.write_text(json.dumps(r) + "\n" + "\n".join(records[1:]) + "\n")
        log = p.with_name("log.jsonl")
        log.write_text(log.read_text().replace('"synthetic"', '"different"'))
        result = build_report([tmp_path], ladder="rfc")
        assert any("step-0" in w for w in result["warnings"])
        assert any("model_snapshot" in w for w in result["warnings"])

    def test_degenerate_seed_remains_in_primary_and_leaves_secondary(
        self, tmp_path: Path
    ) -> None:
        p = _write_run(tmp_path) / "log.jsonl"
        p.write_text(
            p.read_text().replace(
                '"truncated": [false, false]', '"truncated": [true, true]'
            )
        )
        _write_run(tmp_path, "curriculum")
        result = build_report([tmp_path], ladder="rfc")
        assert result["degenerate_seeds"] == [5]
        assert result["comparisons"]["main"]["per_token"]["n"] == 1
        assert result["without_degenerate"]["main"]["per_token"]["n"] == 0

    def test_empty_and_missing_measurements_are_explicit(self, tmp_path: Path) -> None:
        assert len(build_report([tmp_path], ladder="rfc")["missing"]) == 10
        p = _write_run(tmp_path, steps=50)
        (p / "evals.jsonl").unlink()
        _write_run(tmp_path, "curriculum", steps=50)
        result = build_report([tmp_path], ladder="rfc")
        assert result["comparisons"]["main"]["per_token"]["n"] == 0
        assert result["comparisons"]["main"]["per_episode"]["n"] == 0
        assert result["runs"][0]["issues"]

    def test_calibration_uses_previous_step_estimates(self, tmp_path: Path) -> None:
        _write_run(tmp_path, "curriculum", steps=10)
        result = build_report([tmp_path], ladder="rfc")
        levels = result["runs"][0]["diagnostics"]["levels"]
        assert levels["1"]["predicted_signal"] == pytest.approx(0.5)
        assert levels["2"]["predicted_signal"] == pytest.approx(0.32)
        assert levels["2"]["observed_signal"] == 1

    def test_markdown_rounds_numbers_and_json_keeps_full_precision(
        self, tmp_path: Path
    ) -> None:
        pytest.importorskip("matplotlib")
        runs, out = tmp_path / "input", tmp_path / "output"
        _write_run(runs)
        result = build_report([runs], ladder="rfc")
        row = result["comparisons"]["main"]["per_token"]
        row["mean"] = 8.224999999999996
        row["p"] = 1.23456e-05
        row["range"] = [2.675000000000005, 15.149999999999997]
        write_report(result, out)
        text = (out / "reference.md").read_text()
        assert "| 8.225 |" in text
        assert "| 1.235e-05 |" in text
        assert "| [2.675, 15.15] |" in text
        assert "8.224999999999996" not in text
        saved = json.loads((out / "reference.json").read_text())
        assert saved["comparisons"]["main"]["per_token"]["mean"] == 8.224999999999996
        assert saved["comparisons"]["main"]["per_token"]["p"] == 1.23456e-05

    def test_one_command_is_deterministic_and_cannot_write_inside_runs(
        self, tmp_path: Path
    ) -> None:
        pytest.importorskip("matplotlib")
        runs, out = tmp_path / "input", tmp_path / "output"
        _write_run(runs)
        result = build_report([runs], ladder="rfc")
        write_report(result, out)
        first = {p.name: p.read_bytes() for p in out.iterdir()}
        assert {
            "reference.json",
            "reference.md",
            "success-per-token.png",
            "success-per-episode.png",
            "falls.json",
            "falls.md",
        } == set(first)
        assert main(["--runs", str(runs), "--out", str(out)]) == 0
        assert first == {p.name: p.read_bytes() for p in out.iterdir()}
        with pytest.raises(ValueError, match="run"):
            write_report(result, runs / "reports")

    def test_falls_are_built_before_any_file_is_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import analysis.falls

        runs, out = tmp_path / "input", tmp_path / "output"
        _write_run(runs)
        result = build_report([runs], ladder="rfc")

        def refuse(report: dict) -> dict:
            raise ValueError("refused")

        monkeypatch.setattr(analysis.falls, "build_falls", refuse)
        with pytest.raises(ValueError, match="refused"):
            write_report(result, out)
        assert not out.exists() or not list(out.iterdir())

    def test_relative_roots_stay_relative_and_still_guard_the_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_run(tmp_path / "input")
        monkeypatch.chdir(tmp_path)
        result = build_report(["input"], ladder="rfc")
        assert result["roots"] == ["input"]
        with pytest.raises(ValueError, match="input roots"):
            write_report(result, "input/reports")
