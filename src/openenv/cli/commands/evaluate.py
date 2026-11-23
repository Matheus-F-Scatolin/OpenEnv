# SPDX-License-Identifier: BSD-3-Clause

"""
Evaluate a model on a split of a deployed environment's tasks.
"""

from __future__ import annotations

from itertools import islice
from pathlib import Path
from typing import Annotated

import typer
from openenv import __version__
from openenv.core.evals import (
    compare_eval_results,
    EnvEvalHarness,
    EvalConfig,
    EvalResult,
)
from openenv.core.harness import MCPHarnessAdapter
from openenv.core.task_api_client import TaskAPIClient

from .._cli_utils import console
from .collect import _build_llm_model_step, _build_session_factory, _SYSTEM_PROMPT


def evaluate(
    env: Annotated[
        str,
        typer.Argument(
            help="Env spec in 'family:variant' form (e.g. reasoning_gym:chain_sum).",
        ),
    ],
    base_url: Annotated[
        str,
        typer.Option(
            "--base-url",
            help="Env server URL. The environment must implement the Task API.",
        ),
    ],
    split: Annotated[
        str,
        typer.Option("--split", help="Task split to evaluate, as the server lists it."),
    ],
    llm_endpoint: Annotated[
        str,
        typer.Option(
            "--llm-endpoint",
            help="Base URL of an OpenAI-compatible server, e.g. http://localhost:8080/v1.",
        ),
    ],
    model: Annotated[
        str, typer.Option("--model", help="Model id sent to the endpoint.")
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            file_okay=False,
            help="New or empty directory for the episodes and result.json.",
        ),
    ],
    limit: Annotated[
        int | None,
        typer.Option("--limit", min=1, help="Evaluate only the first N tasks."),
    ] = None,
    seeds: Annotated[
        list[int] | None,
        typer.Option(
            "--seed", help="Seed crossed with every task; repeat it (default: 0)."
        ),
    ] = None,
    samples_per_task: Annotated[
        int,
        typer.Option(
            "--samples-per-task", min=1, help="Episodes per task and seed, for pass@k."
        ),
    ] = 1,
    temperature: Annotated[
        float,
        typer.Option(
            "--temperature",
            min=0.0,
            help="Sampling temperature; must be > 0 with several samples per task.",
        ),
    ] = 0.0,
    success_threshold: Annotated[
        float,
        typer.Option(
            "--success-threshold",
            help="Reward at or above which an episode is solved.",
        ),
    ] = 1.0,
    max_tokens: Annotated[
        int, typer.Option("--max-tokens", min=1, help="Max completion tokens.")
    ] = 200,
    max_turns: Annotated[
        int,
        typer.Option("--max-turns", min=1, help="Max tool/model turns per episode."),
    ] = 9,
    system_prompt: Annotated[
        str | None,
        typer.Option(
            "--system-prompt",
            help="System prompt for the model (default: the one of openenv collect).",
        ),
    ] = None,
    compare: Annotated[
        Path | None,
        typer.Option(
            "--compare",
            exists=True,
            dir_okay=False,
            help="An earlier result.json: print the tasks fixed, regressed and "
            "left out since then.",
        ),
    ] = None,
) -> None:
    """
    Evaluate a model on a split of an environment's tasks.

    The tasks come from the environment's Task API, and each episode runs through
    the collect pipeline with [`~openenv.core.evals.EnvEvalHarness`]. The episodes
    go to `run-0000/` in `--output-dir`, and the scores, with the whole
    configuration, to `result.json` beside it. With `--compare`, the tasks whose
    outcome changed since an earlier result are printed, as
    [`~openenv.core.evals.compare_eval_results`] lists them.
    """
    if not llm_endpoint:
        # Collect's client builder would fall back to hosted OpenAI.
        raise typer.BadParameter("must not be empty.", param_hint="--llm-endpoint")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise typer.BadParameter(
            f"{output_dir} is not empty; use a new directory.",
            param_hint="--output-dir",
        )
    before = None
    if compare is not None:
        try:
            before = EvalResult.model_validate_json(compare.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise typer.BadParameter(
                f"{compare} is not a result of openenv eval.", param_hint="--compare"
            ) from exc
        # Checked here so that a wrong file fails before the evaluation runs.
        if not isinstance(before.scores.get("per_task"), dict):
            raise typer.BadParameter(
                f"{compare} has no per_task scores.", param_hint="--compare"
            )
    factory = _build_session_factory(env, base_url)
    system_prompt = system_prompt or _SYSTEM_PROMPT
    model_step = _build_llm_model_step(
        provider="openai",
        model=model,
        llm_endpoint=llm_endpoint,
        llm_port=None,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )

    client = TaskAPIClient(base_url, env_name=env.partition(":")[0] + "_env")
    try:
        splits = [entry["name"] for entry in client.list_splits()]
    except NotImplementedError as exc:
        raise typer.BadParameter(
            f"the environment at {base_url} does not implement the Task API.",
            param_hint="ENV",
        ) from exc
    if split not in splits:
        raise typer.BadParameter(
            f"unknown split {split!r}; available: {', '.join(splits)}.",
            param_hint="--split",
        )
    # Paged, so an environment that caps its ranges is read up to the limit.
    tasks = list(islice(client.iter_tasks(split), limit))
    if not tasks:
        raise typer.BadParameter(f"split {split!r} has no tasks.", param_hint="--split")

    harness = EnvEvalHarness(
        factory,
        MCPHarnessAdapter(),
        tasks,
        model_step=model_step,
        output_dir=output_dir,
    )
    config = EvalConfig(
        harness_name=harness.name,
        harness_version=__version__,
        library_versions={"openenv": __version__},
        dataset=env,
        eval_parameters={
            "split": split,
            "limit": limit,
            "seeds": seeds or [0],
            "samples_per_task": samples_per_task,
            "temperature": temperature,
            "success_threshold": success_threshold,
            "max_tokens": max_tokens,
            "max_turns": max_turns,
            "model": model,
            "llm_endpoint": llm_endpoint,
            "system_prompt": system_prompt,
            "base_url": base_url,
        },
    )
    result = harness.run_from_config(config)

    path = output_dir / "result.json"
    path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    for key, value in result.scores.items():
        if not isinstance(value, (dict, list)):
            console.print(f"{key}={value}")
    console.print(f"[green]Result:[/green] {path}")

    if before is not None:
        diff = compare_eval_results(before, result)
        for name, ids in (
            ("fixed", diff.fixed),
            ("regressed", diff.regressed),
            ("left_out", diff.left_out),
        ):
            console.print(f"{name} ({len(ids)}): {', '.join(ids) or '-'}", markup=False)


__all__ = ["evaluate"]
