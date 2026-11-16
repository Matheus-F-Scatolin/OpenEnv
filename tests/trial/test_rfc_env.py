# SPDX-License-Identifier: BSD-3-Clause

"""The published task split is the trainer's source of questions."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("reasoning_gym")

import reasoning_gym
from analysis.validate_run import check_run
from tests.trial.fake_run import run
from trial.env import build_tasks
from trial.tasks import STAIRCASE_LADDER
from trial.train import build_parser, main


class TestPublishedTasks:
    def test_every_question_matches_the_staircase_settings(self) -> None:
        env, train, test = build_tasks("rfc", 2000, 100)
        assert len(train) == 20000 and len(test) == 1000
        assert not {t["id"] for t in train} & {t["id"] for t in test}
        for tasks, offset in [(train, 0), (test, 500000000)]:
            for task in tasks:
                entry = reasoning_gym.create_dataset(
                    "chain_sum",
                    seed=offset + task["index"],
                    size=1,
                    **STAIRCASE_LADDER[task["difficulty"]],
                )[0]
                assert env.question(task) == entry["question"]
                assert env.correct_answer(task) == entry["answer"]
                assert env.reward(task, f"<answer>{entry['answer']}</answer>") == 1.0

    def test_missing_reward_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        env, train, _ = build_tasks("rfc", 1, 1)
        monkeypatch.setattr(
            env.env, "step", lambda action: SimpleNamespace(reward=None)
        )
        with pytest.raises(ValueError, match="environment returned no reward"):
            env.reward(train[0], "42")

    @pytest.mark.parametrize("train,test", [(2001, 1), (1, 101), (-1, 1), (1, -1)])
    def test_excessive_counts_are_refused(self, train: int, test: int) -> None:
        with pytest.raises(ValueError):
            build_tasks("rfc", train, test)

    def test_invalid_selection_creates_no_run_folder(self, tmp_path: Path) -> None:
        out = tmp_path / "absent"
        with pytest.raises(ValueError):
            main(
                [
                    "baseline",
                    "--model",
                    "fake",
                    "--ladder",
                    "rfc",
                    "--eval-per-tier",
                    "101",
                    "--out",
                    str(out),
                ]
            )
        assert not out.exists()

    @pytest.mark.parametrize("cmd", ["baseline", "train"])
    def test_parser_accepts_ladder(self, cmd: str) -> None:
        args = [cmd, "--model", "fake", "--out", "/tmp/unused", "--ladder", "rfc"]
        if cmd == "train":
            args += ["--sampler", "uniform", "--steps", "1"]
        assert build_parser().parse_args(args).ladder == "rfc"

    def test_fake_run_uses_published_tasks(self, tmp_path: Path) -> None:
        folder = run(tmp_path / "trial", "fake-curriculum-s0", ladder="rfc", steps=2)
        assert check_run(folder)[0] == "OK"
        config = json.loads((folder / "log.jsonl").read_text().splitlines()[0])
        assert config["question_seed"] == 0
        assert config["tiers"] == STAIRCASE_LADDER
        for line in (folder / "questions.jsonl").read_text().splitlines():
            assert json.loads(line)["task"].startswith("chain_sum-test-")
