# SPDX-License-Identifier: BSD-3-Clause

"""The report: the verdict rules, the window of each pair, the pilot's numbers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("reasoning_gym")
pytest.importorskip("matplotlib")

from analysis.figures import comparison
from analysis.report import (
    budget_summary,
    early_steps,
    pair_differences,
    REPORT_FILES,
    verdict,
    write_tables,
)
from analysis.tables import load_pilot_runs

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "pilot"
RUNS = ROOT / "trial" / "runs"
EVIDENCE = ROOT / "notes" / "results" / "pilot-evidence.txt"


def _eval(step: int, tokens: int, success: float) -> dict:
    return {
        "event": "eval",
        "step": step,
        "gen_tokens_total": tokens,
        "success_rate": success,
        "by_tier": {str(t): success for t in range(5)},
    }


def _write_run(
    folder: Path, method: str, seed: int, last_step: int, late_success: float
) -> None:
    # Success 0.5 up to step 200, then `late_success`; 5M tokens at the last step.
    per_step = 5_000_000 // last_step
    steps = [*range(0, last_step, 100), last_step]
    evals = [_eval(s, s * per_step, 0.5 if s <= 200 else late_success) for s in steps]
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


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    for seed in (1, 2):
        _write_run(tmp_path / "trial", "uniform", seed, 800, 0.5)
        _write_run(tmp_path / "trial", "curriculum", seed, 800, 0.9)
    # Seed 2 also has a run that stops at step 250, as the collapsed run did.
    _write_run(tmp_path / "cost-full", "curriculum-tokens", 2, 250, 0.5)
    return tmp_path


class TestVerdict:
    @pytest.mark.parametrize(
        "values, expected",
        [
            ({1: 0.05, 2: 0.05}, "better"),
            ({1: 0.10, 2: -0.01}, "at least as good"),
            ({1: -0.05, 2: -0.05}, "worse"),
            ({1: -0.03, 2: 0.005}, "at least as good"),
            ({1: -0.06, 2: 0.01}, "no clear difference"),
            ({1: 0.05, 2: 0.05, 3: -0.5}, "better"),
            ({1: 0.05}, None),
        ],
    )
    def test_the_rules_of_the_expectations(
        self, values: dict[int, float], expected: str | None
    ) -> None:
        assert verdict(values) == expected


class TestEarlySteps:
    def test_length_and_cut_answers_per_block(self) -> None:
        run = {
            "config": {"prompts_per_step": 1, "group_size": 2},
            "steps": [
                {"step": 1, "gen_tokens": 10, "groups": [{"truncated": [False, True]}]},
                {"step": 2, "gen_tokens": 30, "groups": [{"truncated": [True, True]}]},
                {"step": 3, "gen_tokens": 4, "groups": [{"truncated": [False, False]}]},
            ],
        }
        assert early_steps(run, block=2, blocks=3) == [(10.0, 0.75), (2.0, 0.0)]

    def test_a_version_1_log_has_no_cut_share(self) -> None:
        run = {
            "config": {"prompts_per_step": 1, "group_size": 2},
            "steps": [{"step": 1, "gen_tokens": 10}],
        }
        assert early_steps(run, block=2, blocks=1) == [(5.0, None)]


class TestPairWindow:
    def test_the_pre_registered_window_is_cut_for_the_whole_seed(
        self, runs: Path
    ) -> None:
        _, differences = comparison(runs)
        # Seed 2 is read at steps 100 and 200 only, where both are at 0.5.
        assert differences["curriculum"]["per_episode"][2] == pytest.approx(0.0)
        assert differences["curriculum"]["per_episode"][1] == pytest.approx(5 * 0.4 / 7)

    def test_each_pair_is_read_up_to_the_steps_both_runs_reached(
        self, runs: Path
    ) -> None:
        rows, _ = comparison(runs)
        curriculum = pair_differences(rows, "uniform", "per_episode")["curriculum"]
        # Steps 100 to 700: 0, 0, then 0.4 five times.
        assert curriculum[2] == pytest.approx(5 * 0.4 / 7)
        assert curriculum[1] == pytest.approx(5 * 0.4 / 7)
        short = pair_differences(rows, "uniform", "per_episode")["curriculum-tokens"]
        assert short[2] == pytest.approx(0.0)

    def test_a_reference_other_than_uniform(self, runs: Path) -> None:
        rows, _ = comparison(runs)
        tokens = pair_differences(rows, "curriculum", "per_token")["curriculum-tokens"]
        assert set(tokens) == {2}
        assert "uniform" in pair_differences(rows, "curriculum", "per_token")


class TestPilot:
    def test_the_pilot_is_read_as_seed_0(self, tmp_path: Path) -> None:
        rows, differences = comparison(tmp_path, pilot=PILOT)
        measured = {
            r["method"]: (
                round(100 * r["per_token"], 1),
                round(100 * r["per_episode"], 1),
            )
            for r in rows
        }
        # The numbers written in notes/expectations.md before the runs.
        assert measured == {"uniform": (72.2, 61.4), "curriculum": (64.5, 63.6)}
        assert round(100 * differences["curriculum"]["per_token"][0], 1) == -7.7

    def test_what_5_million_tokens_bought_in_the_pilot(self) -> None:
        summary = {
            run["config"]["sampler"]: budget_summary(run)
            for run in load_pilot_runs(PILOT)
        }
        # The numbers quoted from the pilot's logs before the runs.
        uniform, curriculum = summary["uniform"], summary["curriculum"]
        assert (uniform["steps"], uniform["episodes"], uniform["updates"]) == (
            1766,
            113_024,
            1467,
        )
        assert (curriculum["steps"], curriculum["episodes"], curriculum["updates"]) == (
            755,
            48_320,
            672,
        )
        assert round(uniform["training_minutes"], 1) == 120.7
        assert round(curriculum["training_minutes"], 1) == 88.8


@pytest.mark.skipif(not (RUNS / "trial").is_dir(), reason="the runs are not here")
class TestReport:
    def test_it_writes_every_file_and_repeats_byte_for_byte(
        self, tmp_path: Path
    ) -> None:
        first, second = tmp_path / "first", tmp_path / "second"
        write_tables(RUNS, PILOT, first)
        write_tables(RUNS, PILOT, second)
        assert sorted(p.name for p in first.iterdir()) == sorted(REPORT_FILES)
        for name in REPORT_FILES:
            assert (first / name).read_bytes() == (second / name).read_bytes()
        # The second reading of the pilot is the stored file, unchanged.
        assert (first / "pilot-evidence.txt").read_bytes() == EVIDENCE.read_bytes()
