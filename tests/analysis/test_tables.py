"""The first tables, checked against numbers recomputed from the raw logs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

pytest.importorskip("reasoning_gym")

from analysis.tables import cost_probe, ladder, main, shadow, ten_run
from tests.trial.fake_run import run
from trial.runlog import read_log


def _steps(run_dir: Path) -> list[dict]:
    return [e for e in read_log(run_dir / "log.jsonl") if e["event"] == "step"]


def _evals(run_dir: Path) -> dict[int, dict]:
    log = read_log(run_dir / "log.jsonl")
    return {e["step"]: e for e in log if e["event"] == "eval"}


@pytest.fixture(scope="module")
def ten(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("ten") / "ten-run"
    for seed in (0, 1, 2):
        for method in ("uniform", "curriculum"):
            run(
                base,
                f"fake-{method}-s{seed}",
                sampler=method,
                seed=seed,
                steps=30,
                eval_every=10,
            )
    (base / "fake-uniform-s9.incomplete-1").mkdir()
    return base


class TestTenRun:
    def test_one_row_per_complete_run(self, ten: Path) -> None:
        rows, _ = ten_run(ten)
        assert [(r["method"], r["seed"]) for r in rows] == [
            (m, s) for s in (0, 1, 2) for m in ("uniform", "curriculum")
        ]

    def test_lengths_and_signal_counts(self, ten: Path) -> None:
        rows, _ = ten_run(ten)
        row = next(r for r in rows if (r["method"], r["seed"]) == ("curriculum", 1))
        run_dir = ten / "fake-curriculum-s1"
        evals, steps = _evals(run_dir), _steps(run_dir)
        assert row["length"] == {
            s: evals[s]["mean_completion_tokens"] for s in (0, 10, 20, 30)
        }
        assert row["length_by_tier"] == evals[30]["mean_completion_tokens_by_tier"]
        assert row["signal_groups"] == sum(s["groups_with_signal"] for s in steps)
        tokens = [
            t
            for step in steps[10:20]
            for group in step["groups"]
            if group["tier"] == 2
            for t in group["tokens"]
        ]
        assert row["train_tokens"]["11-20"]["2"] == pytest.approx(
            sum(tokens) / len(tokens)
        )

    def test_summary_counts_only_new_seeds(self, ten: Path) -> None:
        rows, summary = ten_run(ten)
        length = {(r["method"], r["seed"]): r["length"][30] for r in rows}
        expected = {
            seed: length[("curriculum", seed)] - length[("uniform", seed)]
            for seed in (0, 1, 2)
        }
        assert summary["difference_at_30"] == pytest.approx(expected)
        new = [expected[1], expected[2]]
        assert summary["new_seeds"] == [1, 2]
        assert summary["new_seeds_above_20"] == sum(d >= 20 for d in new)
        assert summary["mean"] == pytest.approx(sum(new) / 2)
        assert summary["range"] == pytest.approx((min(new), max(new)))

    def test_command_line_prints_tables(
        self, ten: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["ten-run", str(ten)]) == 0
        out = capsys.readouterr().out
        assert out.count("| curriculum |") >= 3 and "seed 0" in out


def test_ladder(ten: Path) -> None:
    run_dir = ten / "fake-uniform-s0"
    rows = ladder(run_dir)[run_dir.name]
    records = [
        json.loads(line)
        for line in (run_dir / "evals.jsonl").read_text().splitlines()
        if json.loads(line)["step"] == 0
    ]
    for row in rows:
        own = [r for r in records if r["tier"] == row["tier"]]
        p = sum(r["reward"] >= 1.0 for r in own) / len(own)
        assert (row["n"], row["success"]) == (len(own), p)
        assert row["standard_error"] == pytest.approx(math.sqrt(p * (1 - p) / len(own)))
        assert row["mean_tokens"] == sum(r["tokens"] for r in own) / len(own)
    assert [row["tier"] for row in rows] == [0, 1, 2, 3, 4]


@pytest.fixture(scope="module")
def probes(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("probes") / "cost-probe"
    for method in ("curriculum-tokens", "curriculum-tokens-mix-decay"):
        run(base, f"fake-{method}-s0", sampler=method, steps=100, eval_every=100)
    return base


def test_cost_probe(probes: Path) -> None:
    rows = cost_probe(probes)
    assert [r["method"] for r in rows] == [
        "curriculum-tokens",
        "curriculum-tokens-mix-decay",
    ]
    row = rows[0]
    steps = _steps(probes / "fake-curriculum-tokens-s0")[50:]
    groups = [g for s in steps for g in s["groups"]]
    assert len(groups) == 400
    easy = [g for g in groups if g["tier"] == 0]
    tokens = sum(sum(g["tokens"]) for g in groups)
    assert row["episode_share"]["0"] == pytest.approx(len(easy) / 400)
    assert row["token_share"]["0"] == pytest.approx(
        sum(sum(g["tokens"]) for g in easy) / tokens
    )
    assert sum(row["episode_share"].values()) == pytest.approx(1.0)
    assert row["mean_probs"]["0"] == pytest.approx(
        sum(s["sampler"]["probs"]["0"] for s in steps) / 50
    )
    signal = sum(s["groups_with_signal"] for s in steps)
    assert row["signal_per_1000_episodes"] == pytest.approx(1000 * signal / 3200)
    assert row["signal_per_million_tokens"] == pytest.approx(1e6 * signal / tokens)
    assert (
        row["eval_length_at_100"]
        == (_evals(probes / "fake-curriculum-tokens-s0")[100]["mean_completion_tokens"])
    )


class TestShadow:
    def test_reproduces_the_runs_own_probabilities(self, probes: Path) -> None:
        run_dir = probes / "fake-curriculum-tokens-mix-decay-s0"
        traces, _ = shadow(run_dir)
        logged = [s["sampler"]["probs"] for s in _steps(run_dir)]
        assert traces["curriculum-tokens-mix-decay"] == logged

    def test_every_formula_per_quarter(self, probes: Path) -> None:
        traces, quarters = shadow(probes / "fake-curriculum-tokens-s0")
        assert list(quarters) == [
            "curriculum",
            "curriculum-tokens",
            "curriculum-tokens-mix",
            "curriculum-tokens-mix-decay",
        ]
        first = quarters["curriculum"][0]
        expected = sum(p["0"] for p in traces["curriculum"][:25]) / 25
        assert first["0"] == pytest.approx(expected)
        assert all(len(q) == 4 for q in quarters.values())
