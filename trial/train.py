# SPDX-License-Identifier: BSD-3-Clause

"""
The command line of the trial trainer.

  train     A GRPO run (LoRA) with one sampling method, until a step limit or
            a budget of generated tokens.
  baseline  One greedy evaluation of the untrained model on the test tasks.

Every run writes its own folder, `<out>/<model>-<method>-s<seed>/`.
"""

from __future__ import annotations

import argparse
import sys

from trial.loop import baseline_run, build_sampler, METHODS, RunConfig, train_run
from trial.runlog import run_dir_for, RunLog
from trial.tasks import LADDERS, make_tasks


def build_parser() -> argparse.ArgumentParser:
    """
    Build the parser. It needs no MLX, so tests can use it anywhere.

    Returns:
        `argparse.ArgumentParser`: The parser of `train` and `baseline`.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("baseline")
    b.add_argument("--model", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--eval-per-tier", type=int, default=40)
    b.add_argument("--max-tokens", type=int, default=384)
    b.add_argument("--ladder", choices=sorted(LADDERS), default="pilot")

    t = sub.add_parser("train")
    t.add_argument("--model", required=True)
    t.add_argument("--sampler", choices=METHODS, required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=0, help="0 means no step limit")
    t.add_argument(
        "--token-budget",
        type=int,
        default=0,
        help="stop once this many tokens were generated in training; 0 means off",
    )
    t.add_argument("--prompts-per-step", type=int, default=8)
    t.add_argument("--group-size", type=int, default=8)
    t.add_argument("--max-tokens", type=int, default=384)
    t.add_argument("--temperature", type=float, default=1.0)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--lora-rank", type=int, default=16)
    t.add_argument("--micro-batch", type=int, default=4)
    t.add_argument("--train-per-tier", type=int, default=2000)
    t.add_argument("--eval-per-tier", type=int, default=40)
    t.add_argument("--eval-every", type=int, default=0, help="in steps; 0 means off")
    t.add_argument(
        "--eval-every-tokens", type=int, default=0, help="in tokens; 0 means off"
    )
    t.add_argument("--eval-at-start", action="store_true")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--ladder", choices=sorted(LADDERS), default="pilot")
    t.add_argument("--snapshot-every-tokens", type=int, default=1_000_000)
    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    """
    Turn parsed arguments into the settings of a run.

    Args:
        args (`argparse.Namespace`):
            The result of `build_parser().parse_args(...)`.

    Returns:
        `RunConfig`: The settings. A baseline has the method `baseline`.

    Raises:
        `SystemExit`: If `train` has neither `--steps` nor `--token-budget`.
    """
    values = vars(args).copy()
    if args.cmd == "baseline":
        values["sampler"] = "baseline"
    elif not args.steps and not args.token_budget:
        sys.exit("give --steps or --token-budget: a run needs a stop rule")
    return RunConfig(**values)


def main(argv: list[str] | None = None) -> None:
    """
    Run one training or one baseline.

    Args:
        argv (`list[str]`, *optional*):
            The arguments; `sys.argv[1:]` when left out.
    """
    config = config_from_args(build_parser().parse_args(argv))
    # The run folder is claimed before the model loads: a second run stops at once.
    log = RunLog(run_dir_for(config.out, config.model, config.sampler, config.seed))

    from trial.env import TaskEnv
    from trial.mlx_policy import MlxPolicy

    ladder = LADDERS[config.ladder]
    training = config.cmd == "train"
    policy = MlxPolicy(
        model=config.model,
        lora_rank=config.lora_rank if training else None,
        lr=config.lr,
        micro_batch=config.micro_batch,
        seed=config.seed,
    )
    env = TaskEnv(ladder)
    test_tasks = make_tasks("test", config.eval_per_tier, ladder)
    if training:
        train_tasks = make_tasks("train", config.train_per_tier, ladder)
        sampler = build_sampler(
            config.sampler, train_tasks, group_size=config.group_size, seed=config.seed
        )
        train_run(config, policy, env, sampler, log, test_tasks)
    else:
        baseline_run(config, policy, env, log, test_tasks)
    log.close()


if __name__ == "__main__":
    main()
