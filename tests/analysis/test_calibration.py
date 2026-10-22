# SPDX-License-Identifier: BSD-3-Clause

"""The signal model against the logs, the graded rewards and the machine time."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("reasoning_gym")
pytest.importorskip("matplotlib")

from analysis.calibration import (
    calibration_text,
    graded_rewards,
    group_correlation,
    PRIOR_P_HAT,
    signal_calibration,
    signal_chance,
    WINDOWS,
)
from analysis.figures import EXPERIMENTS, trial_runs
from analysis.report import (
    cost_text,
    GENERATED,
    machine_hours,
    REPORT_FILES,
    write_tables,
)
from analysis.tables import load_runs
from samplers import has_signal
from samplers.base import _SIGNAL_STD
from tests.trial.fake_run import run as fake_run
from trial.loop import build_sampler
from trial.tasks import LADDERS, make_tasks

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "pilot"
RUNS = ROOT / "trial" / "runs"
REAL_RUN = RUNS / "trial" / "llama3.2-1b-curriculum-s1"

Groups = list[tuple[int, list[float]]]


def _write_run(
    folder: Path,
    method: str,
    seed: int,
    steps: list[Groups],
    *,
    p_hats: list[float] | None = None,
    threshold: float = 1.0,
    train_seconds: float = 3600.0,
    eval_seconds: tuple[float, float] = (60.0, 60.0),
    eval_rewards: tuple[float, ...] = (1.0, 0.0),
) -> None:
    # `steps` holds the level and rewards of each group; `p_hats` the estimate
    # a curriculum logs after each step, the same for every level. The last
    # step reaches 5 million tokens, so every table of the report can read it.
    n = len(steps)
    per_step = -(-5_000_000 // n)
    settings = {} if method == "uniform" else {"success_threshold": threshold}
    events: list[dict] = [
        {
            "event": "config",
            "sampler": method,
            "seed": seed,
            "prompts_per_step": len(steps[0]),
            "group_size": len(steps[0][0][1]),
            "train_per_tier": 2,
            "ladder": "pilot",
            "sampler_settings": settings,
        }
    ]
    for number, groups in enumerate(steps, start=1):
        record = {
            "event": "step",
            "step": number,
            "train_elapsed_s": train_seconds * number / n,
            "gen_tokens": per_step,
            "groups_with_signal": sum(has_signal(rewards) for _, rewards in groups),
            "gen_tokens_total": per_step * number,
            "groups": [
                {
                    "task": f"chain_sum-train-t{tier}-0",
                    "tier": tier,
                    "rewards": rewards,
                    "tokens": [10] * len(rewards),
                    "truncated": [False] * len(rewards),
                }
                for tier, rewards in groups
            ],
        }
        if method != "uniform":
            p = 0.5 if p_hats is None else p_hats[number - 1]
            record["sampler"] = {
                "p_hat": {str(t): p for t in range(5)},
                "cost_hat": {str(t): 10.0 for t in range(5)},
                "probs": {str(t): 0.2 for t in range(5)},
            }
        events.append(record)
    for number, seconds in zip((0, n), eval_seconds):
        events.append(
            {
                "event": "eval",
                "step": number,
                "train_elapsed_s": train_seconds * number / n,
                "gen_tokens_total": per_step * number,
                "success_rate": 0.5,
                "by_tier": {str(t): 0.5 for t in range(5)},
                "mean_completion_tokens": 10.0,
                "eval_seconds": seconds,
            }
        )
    events.append({"event": "done", "step": n})
    run_dir = folder / f"fake-{method}-s{seed}"
    run_dir.mkdir(parents=True)
    (run_dir / "log.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (run_dir / "evals.jsonl").write_text(
        "".join(json.dumps({"step": 0, "reward": r}) + "\n" for r in eval_rewards)
    )


def _one_group(*rewards: list[float], tier: int = 0) -> list[Groups]:
    return [[(tier, list(r))] for r in rewards]


class TestSignalChance:
    def test_the_binomial_chance_of_a_group_with_signal(self) -> None:
        assert signal_chance(0.5, 4) == 0.875
        assert signal_chance(0.0, 8) == 0.0
        assert signal_chance(1.0, 8) == 0.0


class TestCalibration:
    @pytest.fixture
    def folder(self, tmp_path: Path) -> Path:
        steps = _one_group([1, 0, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [1, 0, 0, 0])
        # Partial credit and no success, at level 1 of the first step.
        steps[0].append((1, [0.5, 0, 0, 0]))
        # Logged after each step; the steps were drawn with 0.5, 0.9, 0.2, 0.7.
        p_hats = [0.9, 0.2, 0.7, 0.4]
        _write_run(tmp_path / "trial", "curriculum", 1, steps, p_hats=p_hats)
        uniform = _one_group([1, 1, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [1, 0, 0, 0])
        _write_run(tmp_path / "trial", "uniform", 1, uniform)
        return tmp_path

    @pytest.fixture
    def runs(self, folder: Path) -> list[dict]:
        return trial_runs(folder)

    def test_a_curriculum_step_is_predicted_by_the_previous_estimate(
        self, runs: list[dict]
    ) -> None:
        cells = signal_calibration(runs)
        # 1 - p^4 - (1 - p)^4 for p = 0.5, 0.9, 0.2 and 0.7.
        assert cells[("curriculum", "whole run", "0")] == {
            "groups": 4,
            "predicted": pytest.approx((0.875 + 0.3438 + 0.5888 + 0.7518) / 4),
            "observed": 0.5,
        }
        assert cells[("curriculum", "first quarter", "0")] == {
            "groups": 1,
            "predicted": 0.875,
            "observed": 1.0,
        }
        assert cells[("curriculum", "last quarter", "0")] == {
            "groups": 1,
            "predicted": pytest.approx(0.7518),
            "observed": 1.0,
        }

    def test_rewards_that_differ_below_the_threshold_have_signal(
        self, runs: list[dict]
    ) -> None:
        # No answer succeeds, but the rewards are not all equal.
        assert signal_calibration(runs)[("curriculum", "whole run", "1")] == {
            "groups": 1,
            "predicted": 0.875,
            "observed": 1.0,
        }

    def test_the_text_gives_predicted_then_observed(self, folder: Path) -> None:
        text = calibration_text(folder)
        last = text[text.index("### Last quarter") : text.index("\n## Correlation")]
        # Drawn with 0.7: 1 - 0.7^4 - 0.3^4 = 0.7518; the group had signal.
        assert (
            "| curriculum | `p_hat` before the step | 0 | 1 | 0.752 | 1.000 | 1.33 |"
            in last.splitlines()
        )

    def test_the_text_gives_the_spread_that_has_signal_needs(
        self, folder: Path
    ) -> None:
        # The text writes the number by hand.
        assert _SIGNAL_STD == 1e-6
        text = calibration_text(folder)
        assert "of its rewards is at least 1e-6 (`samplers.has_signal`)" in text

    def test_uniform_is_predicted_by_the_success_share_of_its_window(
        self, runs: list[dict]
    ) -> None:
        cells = signal_calibration(runs)
        whole = cells[("uniform", "whole run", "0")]
        assert whole["predicted"] == pytest.approx(signal_chance(7 / 16, 4))
        assert (whole["observed"], whole["groups"]) == (0.5, 4)
        first = cells[("uniform", "first quarter", "0")]
        assert (first["predicted"], first["observed"]) == (0.875, 1.0)
        last = cells[("uniform", "last quarter", "0")]
        assert last["predicted"] == pytest.approx(1 - 0.25**4 - 0.75**4)

    def test_uniform_takes_the_success_share_of_each_run(self, tmp_path: Path) -> None:
        # Pooled, the two runs would succeed half the time and predict 0.875.
        _write_run(tmp_path / "trial", "uniform", 1, _one_group(*[[1] * 4] * 4))
        _write_run(tmp_path / "trial", "uniform", 2, _one_group(*[[0] * 4] * 4))
        cells = signal_calibration(trial_runs(tmp_path))
        assert cells[("uniform", "whole run", "0")]["predicted"] == 0.0

    def test_the_first_quarter_rounds_up_and_the_last_down(
        self, tmp_path: Path
    ) -> None:
        # Of 5 steps, the first quarter holds steps 1 and 2, the last step 5.
        _write_run(tmp_path / "trial", "uniform", 1, _one_group(*[[1, 0, 0, 0]] * 5))
        cells = signal_calibration(trial_runs(tmp_path))
        groups = [cells[("uniform", window, "0")]["groups"] for window in WINDOWS]
        assert groups == [5, 2, 1]

    @pytest.mark.parametrize(
        "runs, start",
        [
            ([("uniform", 1)], "The 1 run of seed 1 in {folders}. The pilot"),
            (
                [("uniform", 1), ("curriculum", 1)],
                "The 2 runs of seed 1 in {folders}. The pilot",
            ),
            (
                [("uniform", 1), ("uniform", 2)],
                "The 2 runs of seeds 1 and 2 in {folders}, pooled over the seeds. ",
            ),
            (
                [("uniform", 2), ("curriculum", 1), ("uniform", 3)],
                "The 3 runs of seeds 1, 2 and 3 in {folders}, pooled over the seeds. ",
            ),
        ],
    )
    def test_the_text_names_the_runs_it_read(
        self, tmp_path: Path, runs: list[tuple[str, int]], start: str
    ) -> None:
        for method, seed in runs:
            steps = _one_group(*[[1, 0, 0, 0]] * 4)
            _write_run(tmp_path / "trial", method, seed, steps)
        folders = "`trial/`, `cost-full/` and `seed3/`"
        assert calibration_text(tmp_path).startswith(start.format(folders=folders))

    def test_a_fresh_sampler_has_the_prior_estimate(self) -> None:
        tasks = make_tasks("train", 2, LADDERS["pilot"])
        for method in ("curriculum", "curriculum-tokens-mix-decay"):
            sampler = build_sampler(method, tasks, group_size=8, seed=0)
            assert {sampler.p_hat(k) for k in sampler.keys} == {PRIOR_P_HAT}


def _probabilities(
    p_hat: dict[str, float], cost_hat: dict[str, float], settings: dict
) -> dict[str, float]:
    # The curriculum's sampling probabilities, rebuilt from logged estimates.
    eps = settings["eps"]
    signal = {k: signal_chance(p, settings["group_size"]) for k, p in p_hat.items()}
    if settings["floor"] == "additive":
        weights = {k: (signal[k] + eps) / cost_hat[k] for k in signal}
        return {k: w / sum(weights.values()) for k, w in weights.items()}
    weights = {k: signal[k] / cost_hat[k] for k in signal}
    share = 1 - eps * len(weights)
    return {k: share * w / sum(weights.values()) + eps for k, w in weights.items()}


def _check_draws(run_dir: Path) -> None:
    # Replays the sampler on the logged groups: before each step it must hold
    # the estimates and probabilities logged at the previous step, and draw
    # the tasks the step logged.
    (run,) = load_runs(run_dir)
    config, steps = run["config"], run["steps"]
    settings = config["sampler_settings"]
    tasks = make_tasks("train", config["train_per_tier"], LADDERS[config["ladder"]])
    sampler = build_sampler(
        config["sampler"], tasks, group_size=config["group_size"], seed=config["seed"]
    )
    p_hat = dict.fromkeys(steps[0]["sampler"]["p_hat"], PRIOR_P_HAT)
    probs = None
    for step in steps:
        assert {str(k): sampler.p_hat(k) for k in sampler.keys} == p_hat
        drawn_with = {str(k): v for k, v in sampler.probabilities().items()}
        assert probs is None or drawn_with == probs
        cost_hat = {str(k): sampler.cost_hat(k) for k in sampler.keys}
        assert _probabilities(p_hat, cost_hat, settings) == pytest.approx(drawn_with)
        drawn = sampler.sample(config["prompts_per_step"])
        assert [task["id"] for task in drawn] == [g["task"] for g in step["groups"]]
        for task, group in zip(drawn, step["groups"]):
            sampler.update(task, group["rewards"], costs=group["tokens"])
        p_hat, probs = step["sampler"]["p_hat"], step["sampler"]["probs"]


class TestWhichEstimateDrewAStep:
    @pytest.mark.parametrize("method", ["curriculum", "curriculum-tokens-mix-decay"])
    def test_each_step_is_drawn_with_the_estimate_logged_before_it(
        self, tmp_path: Path, method: str
    ) -> None:
        _check_draws(fake_run(tmp_path, sampler=method, steps=30, eval_every=10))

    @pytest.mark.skipif(not REAL_RUN.is_dir(), reason="the runs are not here")
    def test_on_a_real_run(self) -> None:
        _check_draws(REAL_RUN)


class TestCorrelation:
    @pytest.fixture
    def folder(self, tmp_path: Path) -> Path:
        last = [
            *[(0, [1, 1, 1, 1])] * 5,
            *[(0, [0, 0, 0, 0])] * 5,
            *[(1, [1, 0, 0, 0])] * 5,
            *[(1, [1, 1, 1, 0])] * 5,
            *[(2, [1, 1, 0, 0])] * 10,
            *[(3, [1, 0, 0, 0])] * 9,
            *[(4, [0, 0, 0, 0])] * 10,
        ]
        early = [(0, [1, 0, 0, 0])]
        _write_run(tmp_path / "trial", "uniform", 1, [early, early, early, last])
        seed_2 = [*[(0, [1, 1, 1, 1])] * 5, *[(0, [0, 0, 0, 0])] * 5]
        _write_run(tmp_path / "seed3", "uniform", 2, [early, early, early, seed_2])
        return tmp_path

    @pytest.fixture
    def cells(self, folder: Path) -> dict:
        return group_correlation(trial_runs(folder))

    def test_all_or_nothing_groups_have_rho_1(self, cells: dict) -> None:
        # Only the last quarter is read, pooled over the seeds.
        assert cells[("uniform", "0")] == {
            "groups": 20,
            "p": 0.5,
            "rho": 1.0,
            "all_fail": 0.5,
            "binomial_all_fail": 0.5**4,
        }

    def test_binomial_spread_has_rho_0(self, cells: dict) -> None:
        assert cells[("uniform", "1")]["rho"] == 0.0

    def test_equal_counts_have_the_lowest_rho(self, cells: dict) -> None:
        assert cells[("uniform", "2")]["rho"] == pytest.approx(-1 / 3)

    def test_rho_is_empty_below_10_groups_or_without_variation(
        self, cells: dict
    ) -> None:
        assert cells[("uniform", "3")]["rho"] is None
        assert cells[("uniform", "3")]["p"] == 0.25
        assert cells[("uniform", "4")]["rho"] is None
        assert cells[("uniform", "4")]["all_fail"] == 1.0

    def test_the_text_gives_p_then_rho(self, folder: Path) -> None:
        text = calibration_text(folder)
        table = text[text.index("\n## Correlation") : text.index("\n## How graded")]
        rows = table.splitlines()
        assert "| uniform | 0 | 20 | 0.500 | 1.00 | 0.500 | 0.062 |" in rows
        assert "| uniform | 2 | 10 | 0.500 | -0.33 | 0.000 | 0.062 |" in rows

    def test_only_a_p_of_exactly_1_leaves_rho_out(self, tmp_path: Path) -> None:
        # Level 0: one failure in 2000 answers, p = 0.9995; level 1: no failure.
        last = [
            *[(0, [1, 1, 1, 1])] * 499,
            (0, [1, 1, 1, 0]),
            *[(1, [1, 1, 1, 1])] * 10,
        ]
        early = [(0, [1, 0, 0, 0])]
        _write_run(tmp_path / "trial", "uniform", 1, [early, early, early, last])
        rho = group_correlation(trial_runs(tmp_path))[("uniform", "0")]["rho"]
        assert -0.005 < rho < 0
        text = calibration_text(tmp_path)
        assert "It is a dash when p is exactly 0 or 1, or below 10 groups" in text
        rows = text[text.index("\n## Correlation") : text.index("\n## How graded")]
        # p is rounded to 1.000, and rho to 0.00, without a sign.
        assert "| uniform | 0 | 500 | 1.000 | 0.00 | 0.000 | 0.000 |" in rows
        assert "| uniform | 1 | 10 | 1.000 | – | 0.000 | 0.000 |" in rows
        assert "-0.00" not in text


class TestGradedRewards:
    @pytest.fixture
    def counts(self, tmp_path: Path) -> dict:
        folder = tmp_path / "trial"
        _write_run(
            folder,
            "curriculum",
            1,
            _one_group([0.5, 0, 0, 0], [1, 1, 0.25, 0], [1, 1, 1, 1], [0, 0, 0, 0]),
            eval_rewards=(1.0, 0.5, 0.0, 0.75),
        )
        _write_run(
            folder,
            "uniform",
            1,
            _one_group([1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]),
        )
        # Two successes and two failures whose rewards differ by less than
        # `has_signal` sees.
        _write_run(
            folder,
            "curriculum-tokens",
            1,
            _one_group([1.0, 0.999999, 1.0, 0.999999], *[[0, 0, 0, 0]] * 3),
        )
        return graded_rewards(trial_runs(tmp_path))

    def test_a_run_is_scored_with_its_own_threshold(self, tmp_path: Path) -> None:
        # 0.5 succeeds at this run's threshold and fails at the default, 1.0.
        steps = _one_group(*[[0.5, 0, 0, 0]] * 4)
        _write_run(tmp_path / "trial", "curriculum", 1, steps, threshold=0.5)
        runs = trial_runs(tmp_path)
        counts = graded_rewards(runs)["curriculum"]
        assert (counts["signal"], counts["signal_without_success"]) == (4, 0)
        assert group_correlation(runs)[("curriculum", "0")]["p"] == 0.25

    def test_partial_credit_and_signal_without_success(self, counts: dict) -> None:
        assert counts["curriculum"] == {
            "runs": 1,
            "rollouts": 16,
            "partial": 2,
            "groups": 4,
            "signal": 2,
            "signal_without_success": 1,
            "mixed_without_signal": 0,
            "answers": 4,
            "partial_answers": 2,
        }

    def test_successes_and_failures_without_signal(self, counts: dict) -> None:
        tokens = counts["curriculum-tokens"]
        assert (tokens["signal"], tokens["mixed_without_signal"]) == (0, 1)
        assert tokens["partial"] == 2

    def test_every_method_together(self, counts: dict) -> None:
        assert counts["every method"] == {
            "runs": 3,
            "rollouts": 48,
            "partial": 4,
            "groups": 12,
            "signal": 3,
            "signal_without_success": 1,
            "mixed_without_signal": 1,
            "answers": 8,
            "partial_answers": 2,
        }

    def test_the_text_gives_each_count_in_its_column(self, tmp_path: Path) -> None:
        steps = _one_group(
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0.5, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [0.25, 0.25, 0, 0],
        )
        rewards = (1.0, 0.5, 0.0, 0.75, 0.0)
        _write_run(tmp_path / "trial", "curriculum", 1, steps, eval_rewards=rewards)
        text = calibration_text(tmp_path)
        rows = text[text.index("\n## How graded") :].splitlines()
        # 24 answers, 3 partial; 6 groups, 4 with signal, 2 of them without a
        # success: 2 / 6 of the groups and 2 / 4 of those with signal.
        assert (
            "| curriculum | 1 | 24 | 3 | 12.500% | 6 | 4 | 2 | 33.333% | 50.000% | 0 |"
            in rows
        )
        # 5 evaluation answers, 2 of them partial.
        assert "| curriculum | 5 | 2 | 40.000% |" in rows


@pytest.fixture(scope="module")
def full(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # Two runs of 100 steps that every table of the report can read.
    base = tmp_path_factory.mktemp("full")
    patterns = [[1, 0, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1], [0.5, 0, 0, 0]]
    steps = [
        [(n % 5, patterns[n % 4]), ((n + 1) % 5, patterns[(n + 1) % 4])]
        for n in range(100)
    ]
    _write_run(
        base / "trial",
        "uniform",
        1,
        steps,
        train_seconds=7200.0,
        eval_seconds=(450.0, 450.0),
    )
    _write_run(
        base / "trial",
        "curriculum",
        1,
        steps,
        train_seconds=3600.0,
        eval_seconds=(450.0, 450.0),
    )
    return base


class TestMachineTime:
    def test_training_clock_plus_every_evaluation(self, tmp_path: Path) -> None:
        _write_run(
            tmp_path / "trial",
            "uniform",
            1,
            _one_group(*[[1, 0]] * 4),
            train_seconds=7200.0,
            eval_seconds=(360.0, 540.0),
        )
        (run,) = trial_runs(tmp_path)
        assert machine_hours(run) == {
            "training": 2.0,
            "evaluation": 0.25,
            "total": 2.25,
        }

    def test_cost_md_ends_with_the_machine_time(self, full: Path) -> None:
        text = cost_text(full, PILOT)
        last = text[text.rindex("\n## ") :]
        assert last.startswith("\n## Machine time per run\n")
        assert (
            "everything outside the timed part of each step and each evaluation, "
            "such as loading the model, writing logs and saving adapters, is in "
            "neither"
        ) in " ".join(last.split())
        assert "| uniform | 1 | 2.00 | 0.25 | 2.25 |" in last
        assert "| curriculum | 1 | 1.00 | 0.25 | 1.25 |" in last
        # The pilot logs both clocks: it is shown as seed 0, outside the means.
        assert "| uniform | 0 | 2.50 | 0.14 | 2.64 |" in last
        assert "| uniform | 1 | 2.25 | 2.25 | 2.25 |" in last
        assert "| every method | 2 | 1.75 | 1.25 | 2.25 |" in last


class TestReportFile:
    def test_it_comes_just_before_the_pilot_evidence(self) -> None:
        assert REPORT_FILES == (
            "trial.md",
            "cost.md",
            "ten-run.md",
            "ladder.md",
            "cost-probe.md",
            "shadow.md",
            "calibration.md",
            "pilot-evidence.txt",
        )

    def test_write_tables_writes_it_and_repeats_byte_for_byte(
        self, full: Path, tmp_path: Path
    ) -> None:
        first, second = tmp_path / "first", tmp_path / "second"
        write_tables(full, PILOT, first)
        write_tables(full, PILOT, second)
        for name in REPORT_FILES:
            assert (first / name).read_bytes() == (second / name).read_bytes()
        text = (first / "calibration.md").read_text()
        assert text.startswith("# ") and GENERATED in text
        for title in (
            "## Predicted and observed groups with signal",
            "## Correlation of successes inside a group",
            "## How graded the rewards are",
        ):
            assert title in text


@pytest.fixture(scope="module")
def real_runs() -> list[dict]:
    return [r for r in trial_runs(RUNS) if r["config"]["seed"] in (1, 2, 3)]


@pytest.mark.skipif(
    not all((RUNS / name).is_dir() for name in EXPERIMENTS),
    reason="the runs are not here",
)
class TestRealRuns:
    # The numbers of an independent reading of the same logs.
    def test_curriculum_last_quarter_levels_3_and_4(
        self, real_runs: list[dict]
    ) -> None:
        cells = signal_calibration(real_runs)
        found = {
            tier: (
                round(cells[("curriculum", "last quarter", tier)]["predicted"], 3),
                round(cells[("curriculum", "last quarter", tier)]["observed"], 3),
            )
            for tier in ("3", "4")
        }
        assert found == {"3": (0.963, 0.519), "4": (0.889, 0.465)}

    def test_graded_rewards_over_the_twelve_runs(self, real_runs: list[dict]) -> None:
        assert graded_rewards(real_runs)["every method"] == {
            "runs": 12,
            "rollouts": 950_976,
            "partial": 1088,
            "groups": 118_872,
            "signal": 32_008,
            "signal_without_success": 176,
            "mixed_without_signal": 0,
            "answers": 195_000,
            "partial_answers": 120,
        }

    def test_machine_hours_per_run(self, real_runs: list[dict]) -> None:
        totals = [machine_hours(run)["total"] for run in real_runs]
        assert (round(min(totals), 2), round(max(totals), 2)) == (1.66, 2.55)
        assert round(sum(totals) / len(totals), 2) == 2.06
