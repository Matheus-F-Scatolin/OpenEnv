# SPDX-License-Identifier: BSD-3-Clause

"""The pilot's evidence: recomputed byte for byte, its hand-entered numbers checked."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

pytest.importorskip("reasoning_gym")

import curriculum as pilot
from analysis.pilot_evidence import alloc, calib, evidence
from analysis.pilot_evidence.__main__ import main
from analysis.pilot_evidence.pilot_logs import (
    GROUP_SIZE,
    MAX_TOKENS,
    PROMPTS_PER_STEP,
    read_pilot_log,
    SEED,
    TEST_PER_TIER,
    TRAIN_PER_TIER,
)

RFC = Path(__file__).resolve().parents[2]
PILOT = RFC / "pilot"
STORED = RFC / "notes" / "results" / "pilot-evidence.txt"
SECTIONS = (
    "replay.py",
    "analyze.py",
    "calib.py",
    "alloc.py",
    "rates.py",
    "repro.py",
    "overlap.py",
)


@pytest.fixture(scope="module")
def text() -> str:
    return evidence(PILOT)


def test_evidence_is_the_stored_text(text: str) -> None:
    assert text.encode() == STORED.read_bytes()


def test_each_section_starts_with_an_empty_line_and_its_header(text: str) -> None:
    headers = [line for line in text.split("\n") if line.startswith("=====")]
    assert headers == [f"===== {name} =====" for name in SECTIONS]
    assert text.startswith("\n=====")
    assert text.count("\n\n=====") == len(SECTIONS) - 1


def test_out_writes_the_text_and_leaves_no_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "analysis.pilot_evidence.__main__.evidence", lambda pilot_dir: "\n== a\n"
    )
    out = tmp_path / "evidence.txt"
    main([str(PILOT), "--out", str(out)])
    assert out.read_text() == "\n== a\n"
    assert [p.name for p in tmp_path.iterdir()] == ["evidence.txt"]


@pytest.mark.parametrize("kind", ["pilot", "smoke"])
@pytest.mark.parametrize("method", ["uniform", "curriculum"])
def test_constants_are_the_pilot_config(kind: str, method: str) -> None:
    config = read_pilot_log(PILOT, method, kind).config
    assert config["prompts_per_step"] == PROMPTS_PER_STEP
    assert config["group_size"] == GROUP_SIZE
    assert config["max_tokens"] == MAX_TOKENS
    assert config["train_per_tier"] == TRAIN_PER_TIER
    assert config["seed"] == SEED
    if kind == "pilot":
        assert config["eval_per_tier"] == TEST_PER_TIER


def test_alloc_settings_are_the_pilot_sampler_defaults() -> None:
    defaults = inspect.signature(pilot.CurriculumSampler).parameters
    assert alloc.EPS == defaults["eps"].default
    assert alloc.DECAY == defaults["decay"].default


@pytest.mark.parametrize("quarter", [1, 4])
def test_alloc_profiles_are_the_calib_measurements(quarter: int) -> None:
    measured = calib.uniform_quarters(PILOT)[quarter - 1]
    profile = alloc.PROFILES[f"uniform arm, quarter {quarter}"]
    assert profile["p"] == [round(p, 3) for p in measured.p]
    assert profile["c"] == [round(c, 1) for c in measured.tokens]
    # A negative estimate, or a tier where every answer was right, counts as no signal.
    assert profile["s"] == [
        0.0 if s < 0 or p == 1.0 else round(s, 3)
        for p, s in zip(measured.p, measured.signal)
    ]
