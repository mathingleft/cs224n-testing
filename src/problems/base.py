from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Problem:
    id: str
    statement: str
    answer: Any
    split: str = "train"
    metadata: dict = field(default_factory=dict)


class ProblemSet(ABC):
    @abstractmethod
    def load(self, split: str | None = None) -> list[Problem]:
        """Load problems, optionally filtered by split."""
        ...

    def train(self) -> list[Problem]:
        return self.load(split="train")

    def test(self) -> list[Problem]:
        return self.load(split="test")
