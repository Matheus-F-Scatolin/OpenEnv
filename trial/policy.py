# SPDX-License-Identifier: BSD-3-Clause

"""The contract between the training loop and the model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass
class Completion:
    """
    One generated answer.

    Attributes:
        tokens (`list[int]`):
            The generated token ids, including the stop token when there is one.
        text (`str`):
            The decoded answer, without the stop token.
        hit_stop (`bool`):
            Whether generation ended on a stop token and not on the length limit.
    """

    tokens: list[int]
    text: str
    hit_stop: bool


class Policy(Protocol):
    """What the training loop needs from a model."""

    def encode_prompt(self, question: str) -> list[int]:
        """
        Return the prompt of a question, as token ids.

        Args:
            question (`str`):
                The question text from the environment.

        Returns:
            `list[int]`: The token ids the model receives.
        """
        ...

    def prompt_text(self, question: str) -> str:
        """
        Return the prompt of a question as text, for the log.

        Args:
            question (`str`):
                The question text from the environment.

        Returns:
            `str`: The prompt exactly as the model receives it.
        """
        ...

    def generate(
        self, prompts: list[list[int]], max_tokens: int, temperature: float
    ) -> list[Completion]:
        """
        Generate one answer per prompt.

        Args:
            prompts (`list[list[int]]`):
                The prompts, as token ids.
            max_tokens (`int`):
                Length limit of each answer.
            temperature (`float`):
                Sampling temperature; `0.0` is greedy.

        Returns:
            `list[Completion]`: One completion per prompt, in the same order.
        """
        ...

    def update(self, samples: list[tuple[list[int], list[int], float]]) -> float:
        """
        Take one optimizer step.

        Args:
            samples (`list[tuple[list[int], list[int], float]]`):
                For each answer: the prompt tokens, the answer tokens and the
                advantage. Never empty.

        Returns:
            `float`: The loss.
        """
        ...

    def save(self, path: Path) -> None:
        """
        Write the trained weights to exactly this file.

        Args:
            path (`Path`):
                The file to write.
        """
        ...

    def describe(self) -> dict[str, Any]:
        """
        Describe the model, for the log.

        Returns:
            `dict[str, Any]`: `trainable_params` and `model_snapshot`.
        """
        ...

    def peak_memory_gb(self) -> float:
        """
        Return the peak memory used so far.

        Returns:
            `float`: Gigabytes.
        """
        ...


__all__ = ["Completion", "Policy"]
