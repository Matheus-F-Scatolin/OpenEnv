"""The command line of the new trainer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from trial.train import build_parser, config_from_args

SMOKE_LOG = (
    Path(__file__).resolve().parent.parent.parent
    / "pilot"
    / "runs"
    / "smoke"
    / "llama3.2-1b-curriculum-s0"
    / "log.jsonl"
)
GOLDEN = "train --model llama3.2-1b --sampler curriculum --steps 20 --eval-every 10 --eval-at-start --out runs/smoke"
TRIAL = "train --model llama3.2-1b --sampler uniform --seed 1 --eval-at-start --eval-per-tier 100 --token-budget 5000000 --eval-every-tokens 250000 --eval-every 100 --out trial/runs/trial"


def _config(line: str):
    return config_from_args(build_parser().parse_args(line.split()))


def test_golden_settings_equal_the_stored_config() -> None:
    stored = json.loads(SMOKE_LOG.read_text().splitlines()[0])
    config = vars(_config(GOLDEN))
    for field in (
        "model",
        "sampler",
        "steps",
        "prompts_per_step",
        "group_size",
        "max_tokens",
        "temperature",
        "lr",
        "lora_rank",
        "micro_batch",
        "train_per_tier",
        "eval_per_tier",
        "eval_every",
        "eval_at_start",
        "seed",
    ):
        assert config[field] == stored[field], field


def test_trial_flags() -> None:
    config = _config(TRIAL)
    assert (config.steps, config.eval_every, config.token_budget) == (0, 100, 5_000_000)
    assert config.eval_every_tokens == 250_000 and config.seed == 1
    assert config.snapshot_every_tokens == 1_000_000 and config.ladder == "pilot"


@pytest.mark.parametrize(
    "line",
    [
        "train --sampler uniform --steps 5 --out x",
        "train --model m --steps 5 --out x",
        "train --model m --sampler uniform --steps 5",
        "train --model m --sampler random --steps 5 --out x",
        "train --model m --sampler uniform --out x",
        "baseline --out x",
    ],
)
def test_a_forgotten_flag_is_an_error(line: str) -> None:
    with pytest.raises(SystemExit):
        _config(line)


def test_baseline() -> None:
    config = _config("baseline --model llama3.2-1b --eval-per-tier 8 --out x")
    assert (config.cmd, config.sampler, config.seed) == ("baseline", "baseline", 0)
