from __future__ import annotations
import re

from src.problems.base import Problem
from .base import Verifier, VerifyResult


class NumericVerifier(Verifier):
    """Verifier for AIME-style problems where the answer is an integer 0-999."""

    def verify(self, problem: Problem, solution: str) -> VerifyResult:
        extracted = self._extract_answer(solution)
        correct = (extracted is not None) and (extracted == int(problem.answer))
        return VerifyResult(
            correct=correct,
            extracted_answer=extracted,
            details={"expected": int(problem.answer)},
        )

    def _extract_answer(self, solution: str) -> int | None:
        # Strategy 1: look for "ANSWER: <number>"
        match = re.search(r"ANSWER\s*:\s*(\d+)", solution, re.IGNORECASE)
        if match:
            return int(match.group(1))

        # Strategy 2: look for "the answer is <number>"
        match = re.search(
            r"the\s+answer\s+is\s*:?\s*(\d+)", solution, re.IGNORECASE
        )
        if match:
            return int(match.group(1))

        # Strategy 3: look for boxed answer (LaTeX)
        match = re.search(r"\\boxed\{(\d+)\}", solution)
        if match:
            return int(match.group(1))

        # Strategy 4: last number in the solution
        numbers = re.findall(r"\b(\d+)\b", solution)
        if numbers:
            return int(numbers[-1])

        return None
