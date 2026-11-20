# SPDX-License-Identifier: BSD-3-Clause

"""
A GRPO training loop that draws its tasks from a task sampler.

The loop does not depend on a training framework: it needs a policy that
generates answers in batches and takes optimizer steps, and an environment that
asks and scores each task. `train_mlx.py` gives both for MLX and
`reasoning_gym_env`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from openenv.core.evals import EvalSchedule
    from openenv.core.task_sampler import TaskSampler


@dataclass
class Completion:
    """
    One generated answer.

    Attributes:
        tokens (`list[int]`):
            The generated token ids; their number is the answer's cost.
        text (`str`):
            The decoded answer.
    """

    tokens: list[int]
    text: str


class Policy(Protocol):
    """What the loop needs from a model."""

    def encode_prompt(self, question: str) -> list[int]:
        """Return the prompt of a question, as token ids."""
        ...

    def generate(
        self, prompts: list[list[int]], max_tokens: int, temperature: float
    ) -> list[Completion]:
        """Generate one answer per prompt, in order."""
        ...

    def update(self, samples: list[tuple[list[int], list[int], float]]) -> float:
        """Take one optimizer step on (prompt ids, answer ids, advantage) samples."""
        ...


class TaskEnv(Protocol):
    """What the loop needs from an environment."""

    def question(self, task: Any) -> str:
        """Return the question of a task."""
        ...

    def reward(self, task: Any, reply: str) -> float | None:
        """Return the environment's reward for a reply, None if it could not score it."""
        ...


def train(
    policy: Policy,
    env: TaskEnv,
    sampler: TaskSampler,
    *,
    steps: int,
    prompts_per_step: int,
    group_size: int,
    max_tokens: int,
    temperature: float,
    log: Callable[[dict[str, Any]], None],
    schedule: EvalSchedule | None = None,
) -> None:
    """
    Train for `steps` steps, or until a finite sampler runs out of tasks.

    Each step draws `prompts_per_step` tasks, generates `group_size` answers for
    each, and reports every group's rewards and token counts to the sampler before
    any group is dropped. A group whose scored rewards are all equal, or that has
    fewer than two, has no signal and gives no gradient; the others give each
    scored answer the advantage `(reward - mean) / (std + 1e-4)` over the group's
    scored rewards. A reward of None (the environment could not score the answer)
    is reported to the sampler and left out of the gradient.

    With `schedule`, the policy is evaluated before the first step, at the
    schedule's interval, and after the last step.

    Args:
        policy (`Policy`):
            The model.
        env (`TaskEnv`):
            Asks and scores the tasks.
        sampler (`TaskSampler`):
            Chooses the tasks.
        steps (`int`):
            The number of optimizer steps.
        prompts_per_step (`int`):
            Tasks per step.
        group_size (`int`):
            Answers per task.
        max_tokens (`int`):
            Length limit of each answer.
        temperature (`float`):
            Sampling temperature.
        log (`Callable[[dict[str, Any]], None]`):
            Receives one record per step: the step, the generated tokens, the mean
            scored reward, each group's task id, rewards and token counts, the
            number of groups with signal, the loss (None without a step) and the
            sampler's `metrics()`.
        schedule ([`~openenv.core.evals.EvalSchedule`], *optional*):
            The evaluations to run while training.
    """
    evaluated = None
    if schedule is not None and schedule.maybe_run(0) is not None:
        evaluated = 0
    step = 0
    while step < steps:
        batch = sampler.sample(prompts_per_step)
        if not batch:
            break
        step += 1
        prompts = [policy.encode_prompt(env.question(task)) for task in batch]
        completions = policy.generate(
            [prompt for prompt in prompts for _ in range(group_size)],
            max_tokens,
            temperature,
        )

        samples: list[tuple[list[int], list[int], float]] = []
        groups: list[dict[str, Any]] = []
        signal_groups = 0
        for i, task in enumerate(batch):
            group = completions[i * group_size : (i + 1) * group_size]
            rewards = [env.reward(task, completion.text) for completion in group]
            tokens = [len(completion.tokens) for completion in group]
            sampler.update(task, rewards, costs=tokens)
            groups.append(
                {"task": task.get("id"), "rewards": rewards, "tokens": tokens}
            )
            scored = [reward for reward in rewards if reward is not None]
            if len(set(scored)) < 2:  # fewer than two scored rewards, or all equal
                continue
            signal_groups += 1
            mean = sum(scored) / len(scored)
            std = math.sqrt(sum((r - mean) ** 2 for r in scored) / len(scored))
            samples += [
                (prompts[i], completion.tokens, (reward - mean) / (std + 1e-4))
                for completion, reward in zip(group, rewards)
                if reward is not None
            ]

        loss = policy.update(samples) if samples else None
        scored = [r for group in groups for r in group["rewards"] if r is not None]
        log(
            {
                "step": step,
                "generated_tokens": sum(len(c.tokens) for c in completions),
                "mean_reward": sum(scored) / len(scored) if scored else None,
                "groups": groups,
                "signal_groups": signal_groups,
                "loss": loss,
                "sampler": sampler.metrics(),
            }
        )
        if schedule is not None and schedule.maybe_run(step) is not None:
            evaluated = step
    if schedule is not None and schedule.every_n_steps and evaluated != step:
        schedule.run(step)


__all__ = ["Completion", "Policy", "TaskEnv", "train"]
