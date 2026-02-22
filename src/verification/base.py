from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.problems.base import Problem


@dataclass
class VerifyResult:
    correct: bool
    extracted_answer: Any
    details: dict = field(default_factory=dict)
    """Additional verification details (verifier-specific).

    Current usage by verifier type:

    NumericVerifier:
        - "expected": int - The expected answer from the problem

    LeanVerifier:
        - "stdout": str - Lean compiler stdout (truncated to 2000 chars)
        - "stderr": str - Lean compiler stderr (truncated to 2000 chars)
        - "error": str - Error type ("timeout" | "lean not installed")

    Last updated: 02/15/2026
    """


class Verifier(ABC):
    @abstractmethod
    def verify(self, problem: Problem, solution: str) -> VerifyResult:
        ...

    def verify_batch(
        self, problem: Problem, solutions: list[str]
    ) -> list[VerifyResult]:
        return [self.verify(problem, s) for s in solutions]
