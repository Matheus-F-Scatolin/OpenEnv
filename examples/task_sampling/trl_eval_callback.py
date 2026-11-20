# SPDX-License-Identifier: BSD-3-Clause

"""A Transformers `TrainerCallback` that runs an `EvalSchedule` during training."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TYPE_CHECKING

from transformers import TrainerCallback

if TYPE_CHECKING:
    from openenv.core.evals import EvalSchedule
    from transformers import TrainerControl, TrainerState, TrainingArguments


class EvalScheduleCallback(TrainerCallback):
    """
    Run an `EvalSchedule` on the starting weights and after every `every_n_steps`
    optimizer steps.

    Before each evaluation, every process calls `sync`, which must leave the policy
    endpoint serving the weights of the current step. Then the main process runs the
    schedule, which passes the step and the harness's result to its `on_result`.

    Args:
        schedule (`EvalSchedule`):
            The schedule to run; its `every_n_steps` picks the steps, and zero turns
            the evaluations off.
        sync (`Callable[[], None]`):
            Called on every process before each evaluation. With TRL's vLLM server
            mode, `trainer.vllm_generation.sync_weights`. Pass `lambda: None` when the
            endpoint already serves the current weights.

    Examples:

    ```python
    schedule = EvalSchedule(harness, config, every_n_steps=50, on_result=log_scores)
    trainer.add_callback(
        EvalScheduleCallback(schedule, sync=trainer.vllm_generation.sync_weights)
    )
    trainer.train()
    ```
    """

    def __init__(self, schedule: EvalSchedule, *, sync: Callable[[], None]) -> None:
        self.schedule = schedule
        self.sync = sync

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Evaluate the starting weights, unless the run resumes from a checkpoint."""
        if state.global_step == 0:
            self._maybe_evaluate(state)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Evaluate after the optimizer steps that the schedule marks."""
        self._maybe_evaluate(state)

    def _maybe_evaluate(self, state: TrainerState) -> None:
        # The same steps as `EvalSchedule.maybe_run`. Syncing gathers the weights
        # across processes, so every process takes part before rank 0 evaluates.
        every = self.schedule.every_n_steps
        if every and state.global_step % every == 0:
            self.sync()
            if state.is_world_process_zero:
                self.schedule.run(state.global_step)


__all__ = ["EvalScheduleCallback"]
