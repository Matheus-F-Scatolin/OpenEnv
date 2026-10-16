"""The training loop, with the fake policy and the real environment."""

from __future__ import annotations

import gzip
import json
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("reasoning_gym")

from tests.trial.fake_run import run
from trial.loop import build_sampler, eval_due, stop_reason, summarize_eval
from trial.runlog import environment_info, is_complete, read_log
from trial.tasks import PILOT_LADDER

PILOT_RUNS = Path(__file__).resolve().parent.parent.parent / "pilot" / "runs" / "pilot"
TIMES = {
    "train_elapsed_s",
    "seconds",
    "gen_seconds",
    "train_seconds",
    "peak_mem_gb",
    "eval_seconds",
    "gen_tokens_per_sec",
    "utc_time",
    "out",
}


def git(repo: Path, *args: str) -> None:
    """Run a git command in a temporary repository, with a throwaway identity."""
    identity = ["-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run(["git", *identity, *args], cwd=repo, check=True, capture_output=True)


def events(run_dir: Path, kind: str) -> list[dict]:
    return [e for e in read_log(run_dir / "log.jsonl") if e["event"] == kind]


def without_times(run_dir: Path) -> list[dict]:
    return [
        {k: v for k, v in event.items() if k not in TIMES}
        for event in read_log(run_dir / "log.jsonl")
    ]


def pilot_tokens(method: str) -> list[int]:
    log = read_log(PILOT_RUNS / f"llama3.2-1b-{method}-s0" / "log.jsonl")
    return [e["gen_tokens"] for e in log if e["event"] == "step"]


class TestTrainRun:
    def test_writes_a_complete_version_2_log(self, tmp_path: Path) -> None:
        run_dir = run(tmp_path)
        log = read_log(run_dir / "log.jsonl")
        assert [e["event"] for e in log] == (
            ["config", "eval"]
            + ["step"] * 6
            + ["eval"]
            + ["step"] * 6
            + ["eval", "done"]
        )
        config, done = log[0], log[-1]
        assert config["schema_version"] == 2
        assert config["prompt_date"] == "26 Sep 2026"
        assert "Today Date: 26 Sep 2026" in config["prompt_example"]
        assert config["tiers"] == PILOT_LADDER
        assert done["stop_reason"] == "steps" and done["step"] == 12
        assert done["episodes"] == 64 * 12
        assert is_complete(run_dir)
        sampler = run.sampler
        assert sampler.episodes == 64 * 12
        assert sampler.cost == done["gen_tokens_total"]

    def test_the_same_run_twice_gives_the_same_log(self, tmp_path: Path) -> None:
        first, second = run(tmp_path, "a"), run(tmp_path, "b")
        assert without_times(first) == without_times(second)

    def test_groups_add_up_to_the_step(self, tmp_path: Path) -> None:
        steps = events(run(tmp_path), "step")
        total = 0
        for step in steps:
            assert len(step["groups"]) == 8
            assert all(len(group["rewards"]) == 8 for group in step["groups"])
            tokens = sum(sum(group["tokens"]) for group in step["groups"])
            assert tokens == step["gen_tokens"]
            total += tokens
            assert step["gen_tokens_total"] == total
            assert set(step["sampler"]) == {"p_hat", "cost_hat", "probs"}
        assert any(step["groups_with_signal"] for step in steps)

    def test_evaluation_records_add_up_to_the_totals(self, tmp_path: Path) -> None:
        run_dir = run(tmp_path)
        records = [
            json.loads(line)
            for line in (run_dir / "evals.jsonl").read_text().splitlines()
        ]
        for evaluation in events(run_dir, "eval"):
            own = [r for r in records if r["step"] == evaluation["step"]]
            assert len(own) == evaluation["n_tasks"] == 20
            summary = summarize_eval(own)
            for field in ("success_rate", "by_tier", "mean_completion_tokens"):
                assert summary[field] == evaluation[field]
        with gzip.open(run_dir / "eval_texts.jsonl.gz", "rt") as handle:
            assert len(handle.read().splitlines()) == len(records)
        with gzip.open(run_dir / "train_texts.jsonl.gz", "rt") as handle:
            assert len(handle.read().splitlines()) == 64 * 12
        questions = (run_dir / "questions.jsonl").read_text().splitlines()
        assert len(questions) == 20
        assert set(json.loads(questions[0])) == {
            "task",
            "tier",
            "question",
            "correct_answer",
        }

    def test_a_token_budget_alone_goes_past_step_20(self, tmp_path: Path) -> None:
        run_dir = run(tmp_path, steps=0, token_budget=20_000, eval_every=0)
        done = events(run_dir, "done")[0]
        steps = events(run_dir, "step")
        assert done["stop_reason"] == "token_budget" and done["step"] > 20
        assert steps[-1]["gen_tokens_total"] >= 20_000 > steps[-2]["gen_tokens_total"]
        assert events(run_dir, "eval")[-1]["step"] == done["step"]

    def test_neither_steps_nor_budget_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="--steps"):
            run(tmp_path, steps=0)
        assert not (tmp_path / "run" / "log.jsonl").exists() or not is_complete(
            tmp_path / "run"
        )

    def test_uniform_logs_no_curriculum_fields(self, tmp_path: Path) -> None:
        step = events(run(tmp_path, sampler="uniform"), "step")[0]
        assert "p_hat" not in step and "sampler" not in step

    def test_one_snapshot_per_crossing(self, tmp_path: Path) -> None:
        run_dir = run(tmp_path, eval_every=1, snapshot_every_tokens=2000, steps=8)
        total = events(run_dir, "done")[0]["gen_tokens_total"]
        snapshots = sorted(p.name for p in run_dir.glob("adapters-*.safetensors"))
        assert len(snapshots) == total // 2000 >= 2
        assert (run_dir / "adapters.safetensors").exists()
        assert (
            json.loads((run_dir / "sampler_state.json").read_text())["episodes"] == 512
        )
        assert not list(run_dir.glob("*tmp*"))

    def test_a_folder_with_a_log_is_refused(self, tmp_path: Path) -> None:
        run(tmp_path)
        with pytest.raises(FileExistsError):
            run(tmp_path)


class TestSchedule:
    @pytest.mark.parametrize("method,last", [("uniform", 1766), ("curriculum", 755)])
    def test_the_pilot_would_stop_at_5_million_tokens(
        self, method: str, last: int
    ) -> None:
        total = 0
        for step, tokens in enumerate(pilot_tokens(method), start=1):
            total += tokens
            if stop_reason(step, total, 0, 5_000_000):
                break
        assert step == last
        assert stop_reason(step, total, 0, 5_000_000) == "token_budget"

    @pytest.mark.parametrize(
        "method,every,expected",
        [
            ("uniform", 0, 21),
            ("curriculum", 0, 21),
            ("uniform", 100, 38),
            ("curriculum", 100, 28),
        ],
    )
    def test_number_of_evaluations_on_the_pilots_tokens(
        self, method: str, every: int, expected: int
    ) -> None:
        total, evaluations = 0, 1
        for step, tokens in enumerate(pilot_tokens(method), start=1):
            before, total = total, total + tokens
            stop = stop_reason(step, total, 0, 5_000_000)
            if stop or eval_due(step, before, total, every, 250_000):
                evaluations += 1
            if stop:
                break
        assert evaluations == expected

    def test_stop_reason(self) -> None:
        assert stop_reason(5, 100, 10, 0) is None
        assert stop_reason(10, 100, 10, 0) == "steps"
        assert stop_reason(3, 100, 10, 100) == "token_budget"
        assert stop_reason(3, 99, 0, 100) is None

    def test_one_evaluation_when_a_step_crosses_two_marks(self) -> None:
        assert eval_due(7, 90, 310, 0, 100)
        assert not eval_due(7, 110, 190, 0, 100)
        assert eval_due(10, 0, 0, 10, 0) and not eval_due(11, 0, 0, 10, 0)


def test_unknown_method_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown"):
        build_sampler("random", [{"id": 1}], group_size=8, seed=0)


class TestEnvironmentInfo:
    def test_keys_and_no_secrets(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HF_TOKEN", "hf_MARKER_VALUE")
        info = environment_info(tmp_path, "26 Sep 2026")
        assert set(info) == {
            "prompt_date",
            "utc_time",
            "python",
            "platform",
            "machine",
            "versions",
            "hf_hub_offline",
            "git_commit",
            "git_dirty",
        }
        assert "MARKER" not in json.dumps(info)
        assert set(info["versions"]) == {
            "mlx",
            "mlx-lm",
            "transformers",
            "tokenizers",
            "reasoning-gym",
            "openenv",
            "numpy",
        }

    def test_git_dirty_ignores_untracked_files(self, tmp_path: Path) -> None:
        git(tmp_path, "init")
        (tmp_path / "tracked.txt").write_text("a")
        git(tmp_path, "add", "tracked.txt")
        git(tmp_path, "commit", "-m", "first")
        (tmp_path / "runs").mkdir()
        (tmp_path / "runs" / "log.jsonl").write_text("{}")
        info = environment_info(tmp_path, "26 Sep 2026")
        assert len(info["git_commit"]) == 40 and info["git_dirty"] is False
        (tmp_path / "tracked.txt").write_text("b")
        assert environment_info(tmp_path, "26 Sep 2026")["git_dirty"] is True

    def test_git_dirty_ignores_what_the_experiment_writes(self, tmp_path: Path) -> None:
        queue_log = tmp_path / "trial" / "lists" / "queue-log.jsonl"
        run_log = tmp_path / "trial" / "runs" / "trial" / "m-uniform-s1" / "log.jsonl"
        code = tmp_path / "trial" / "loop.py"
        for path in (queue_log, run_log, code):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("first\n")
        git(tmp_path, "init")
        git(tmp_path, "add", "-A")
        git(tmp_path, "commit", "-m", "first")
        # The queue appends to its log and sets an incomplete run folder aside.
        queue_log.write_text("first\nsecond\n")
        run_log.parent.rename(run_log.parent.with_name("m-uniform-s1.incomplete-1"))
        assert environment_info(tmp_path, "26 Sep 2026")["git_dirty"] is False
        code.write_text("changed\n")
        assert environment_info(tmp_path, "26 Sep 2026")["git_dirty"] is True
