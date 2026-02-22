"""Tests for the verification module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.problems.base import Problem
from src.verification.numeric import NumericVerifier


def make_problem(answer: int = 42) -> Problem:
    return Problem(id="test_p1", statement="Test problem", answer=answer)


class TestNumericVerifier:
    def setup_method(self):
        self.verifier = NumericVerifier()

    def test_answer_prefix(self):
        result = self.verifier.verify(make_problem(42), "blah blah ANSWER: 42")
        assert result.correct
        assert result.extracted_answer == 42

    def test_answer_prefix_case_insensitive(self):
        result = self.verifier.verify(make_problem(42), "answer: 42")
        assert result.correct

    def test_answer_prefix_no_space(self):
        result = self.verifier.verify(make_problem(42), "ANSWER:42")
        assert result.correct

    def test_the_answer_is(self):
        result = self.verifier.verify(
            make_problem(100), "So the answer is 100."
        )
        assert result.correct
        assert result.extracted_answer == 100

    def test_boxed_latex(self):
        result = self.verifier.verify(
            make_problem(7), "Therefore $\\boxed{7}$"
        )
        assert result.correct

    def test_last_number_fallback(self):
        result = self.verifier.verify(
            make_problem(999), "We get 3 and then 5 and finally 999"
        )
        assert result.correct
        assert result.extracted_answer == 999

    def test_wrong_answer(self):
        result = self.verifier.verify(make_problem(42), "ANSWER: 43")
        assert not result.correct
        assert result.extracted_answer == 43

    def test_no_number(self):
        result = self.verifier.verify(make_problem(42), "I don't know")
        assert not result.correct
        assert result.extracted_answer is None

    def test_leading_zeros(self):
        result = self.verifier.verify(make_problem(7), "ANSWER: 007")
        assert result.correct
        assert result.extracted_answer == 7

    def test_zero_answer(self):
        result = self.verifier.verify(make_problem(0), "ANSWER: 0")
        assert result.correct

    def test_verify_batch(self):
        problem = make_problem(10)
        solutions = ["ANSWER: 10", "ANSWER: 5", "ANSWER: 10"]
        results = self.verifier.verify_batch(problem, solutions)
        assert len(results) == 3
        assert results[0].correct
        assert not results[1].correct
        assert results[2].correct
