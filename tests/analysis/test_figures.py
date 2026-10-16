"""The trial's comparison, checked on small logs with known numbers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("reasoning_gym")
pytest.importorskip("matplotlib")

from analysis.figures import (
    comparison,
    FIGURES,
    levels_2_to_4,
    main,
    per_episode,
    per_token,
    plot,
)


def _eval(step: int, tokens: int, success: float) -> dict:
    hard = (5 * success - 2) / 3
    return {
        "event": "eval",
        "step": step,
        "gen_tokens_total": tokens,
        "success_rate": success,
        "by_tier": {"0": 1.0, "1": 1.0, "2": hard, "3": hard, "4": hard},
    }


def _write_run(
    folder: Path, method: str, seed: int, last_step: int, evals: list[dict]
) -> None:
    run_dir = folder / f"fake-{method}-s{seed}"
    run_dir.mkdir(parents=True)
    events = [
        {
            "event": "config",
            "sampler": method,
            "seed": seed,
            "prompts_per_step": 8,
            "group_size": 8,
        },
        *({"event": "step", "step": s} for s in range(1, last_step + 1)),
        *evals,
        {"event": "done", "step": last_step},
    ]
    (run_dir / "log.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))


def _flat(success: float, last_step: int) -> list[dict]:
    # `success` from 4M tokens on, so the per-token measure is `success`.
    return [
        _eval(0, 0, 0.5),
        _eval(last_step - 1, 4_000_000, success),
        _eval(last_step, 5_000_000, success),
    ]


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    for seed in (1, 2):
        _write_run(tmp_path / "trial", "uniform", seed, 800, _flat(0.5, 800))
        _write_run(tmp_path / "trial", "curriculum", seed, 800, _flat(0.7, 800))
    # A short run cuts the episode window of its seed.
    _write_run(tmp_path / "cost-full", "curriculum-tokens", 2, 250, _flat(0.5, 250))
    _write_run(tmp_path / "trial", "uniform", 9, 5, _flat(0.5, 5)[:1])
    (tmp_path / "trial" / "fake-uniform-s9" / "log.jsonl").write_text(
        json.dumps({"event": "config", "sampler": "uniform", "seed": 9}) + "\n"
    )
    return tmp_path


class TestMeasures:
    def test_per_token_reads_the_marks_between_evaluations(self) -> None:
        evals = [_eval(0, 0, 0.5), _eval(10, 4_000_000, 0.6), _eval(20, 5_000_000, 0.8)]
        # Marks 4.0, 4.25, 4.5, 4.75, 5.0M: 0.6, 0.65, 0.7, 0.75, 0.8.
        assert per_token(evals) == pytest.approx(0.7)

    def test_per_episode_reads_steps_100_to_the_window(self) -> None:
        evals = [_eval(0, 0, 0.5), _eval(400, 1, 0.9)]
        # Steps 100 and 200: 0.6 and 0.7.
        assert per_episode(evals, 200) == pytest.approx(0.65)

    def test_levels_2_to_4(self) -> None:
        assert levels_2_to_4(_eval(0, 0, 0.8)) == pytest.approx(2 / 3)

    def test_a_mark_outside_the_evaluations_is_refused(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            per_token([_eval(0, 0, 0.5), _eval(5, 3_000_000, 0.5)])


class TestComparison:
    def test_rows_and_differences(self, runs: Path) -> None:
        rows, differences = comparison(runs)
        assert [(r["method"], r["seed"]) for r in rows] == [
            ("uniform", 1),
            ("curriculum", 1),
            ("uniform", 2),
            ("curriculum", 2),
            ("curriculum-tokens", 2),
        ]
        assert differences["curriculum"]["per_token"] == {
            1: pytest.approx(0.2),
            2: pytest.approx(0.2),
        }

    def test_short_run_cuts_the_window_of_its_seed_only(self, runs: Path) -> None:
        rows, _ = comparison(runs)
        window = {(r["method"], r["seed"]): r["episode_window"] for r in rows}
        assert window[("uniform", 1)] == 700
        assert window[("uniform", 2)] == 200
        assert window[("curriculum-tokens", 2)] == 200

    def test_incomplete_runs_are_left_out(self, runs: Path) -> None:
        rows, _ = comparison(runs)
        assert 9 not in {r["seed"] for r in rows}


class TestOutput:
    def test_figures_are_written(self, runs: Path, tmp_path: Path) -> None:
        out = tmp_path / "figures"
        plot(runs, out)
        assert sorted(p.name for p in out.iterdir()) == sorted(FIGURES)

    def test_main_prints_both_tables(
        self, runs: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main([str(runs), str(tmp_path / "figures")]) == 0
        printed = capsys.readouterr().out
        assert "| uniform | 1 |" in printed
        assert "| curriculum | 20.0 | 20.0 | – |" in printed
