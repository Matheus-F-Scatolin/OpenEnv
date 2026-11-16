# SPDX-License-Identifier: BSD-3-Clause

"""Run an evaluation harness at explicit steps or fixed step intervals."""

from __future__ import annotations

from collections.abc import Callable

from openenv.core.evals import EvalConfig, EvalHarness, EvalResult


class EvalSchedule:
    """
    Run a harness and pass each result to the trainer's callback.

    Args:
        harness (`EvalHarness`):
            Evaluation implementation.
        config (`EvalConfig`):
            Configuration passed to the harness on each evaluation.
        every_n_steps (`int`):
            Step interval; zero disables automatic evaluations.
        on_result (`Callable[[int, EvalResult], None]`):
            Callback receiving the step and result after each evaluation.

    Raises:
        `ValueError`: If `every_n_steps` is negative.
    """

    def __init__(
        self,
        harness: EvalHarness,
        config: EvalConfig,
        *,
        every_n_steps: int,
        on_result: Callable[[int, EvalResult], None],
    ) -> None:
        if every_n_steps < 0:
            raise ValueError("every_n_steps must not be negative")
        self.harness = harness
        self.config = config
        self.every_n_steps = every_n_steps
        self.on_result = on_result

    def run(self, step: int) -> EvalResult:
        """
        Evaluate immediately and notify the callback.

        Args:
            step (`int`):
                Current training step.

        Returns:
            `EvalResult`: The harness result.

        Raises:
            `ValueError`: If `step` is negative.
        """
        if step < 0:
            raise ValueError("step must not be negative")
        result = self.harness.run_from_config(self.config)
        self.on_result(step, result)
        return result

    def maybe_run(self, step: int) -> EvalResult | None:
        """
        Evaluate on interval boundaries, including step zero.

        Args:
            step (`int`):
                Current training step.

        Returns:
            `EvalResult` or `None`: Result when due, otherwise `None`.

        Raises:
            `ValueError`: If `step` is negative.
        """
        if step < 0:
            raise ValueError("step must not be negative")
        if self.every_n_steps and step % self.every_n_steps == 0:
            return self.run(step)
        return None


__all__ = ["EvalSchedule"]
