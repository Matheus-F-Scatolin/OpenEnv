"""A stand-in for the model: answers arithmetic questions with a fixed chance."""

from __future__ import annotations

import random
from pathlib import Path

from trial.policy import Completion

# Chance of a right answer by number of terms in the question (the pilot's levels).
CHANCE = {2: 0.95, 3: 0.8, 4: 0.5, 5: 0.2, 6: 0.05}


def solve(question: str) -> tuple[int, int]:
    """Return the sum the question asks for, and its number of terms."""
    words = question.split(":")[-1].replace("=", " ").split()
    total, sign, terms = 0, 1, 0
    for word in words:
        if word in "+-":
            sign = 1 if word == "+" else -1
        else:
            total += sign * int(word)
            terms += 1
    return total, terms


class FakePolicy:
    """Satisfies `trial.policy.Policy` without a model or a GPU."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.updates = 0
        self.saved: list[Path] = []

    def encode_prompt(self, question: str) -> list[int]:
        return [ord(char) for char in question]

    def prompt_text(self, question: str) -> str:
        return f"Today Date: 26 Sep 2026\n\n{question}"

    def generate(
        self, prompts: list[list[int]], max_tokens: int, temperature: float
    ) -> list[Completion]:
        out = []
        for prompt in prompts:
            question = "".join(chr(code) for code in prompt)
            answer, terms = solve(question)
            # Greedy answers depend on the question only, so evaluation draws nothing.
            rng = self.rng if temperature > 0 else random.Random(question)
            right = rng.random() < CHANCE.get(terms, 0.3)
            length = min(max_tokens, rng.randint(4, 6) + terms)
            shown = answer if right else answer + rng.choice([1, 10, -7])
            tagged = rng.random() < 0.7
            text = f"<answer>{shown}</answer>" if tagged else f"so it is {shown}"
            hit_stop = rng.random() < 0.9
            out.append(Completion([7] * length, text, hit_stop))
        return out

    def update(self, samples: list[tuple[list[int], list[int], float]]) -> float:
        assert samples
        self.updates += 1
        return sum(advantage for _, _, advantage in samples) / len(samples)

    def save(self, path: Path) -> None:
        Path(path).write_bytes(f"adapters after {self.updates} updates".encode())
        self.saved.append(Path(path))

    def describe(self) -> dict:
        return {"trainable_params": 0, "model_snapshot": "fake"}

    def peak_memory_gb(self) -> float:
        return 0.0
