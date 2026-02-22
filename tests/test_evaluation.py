"""Tests for the evaluation module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluate import pass_at_k


class TestPassAtK:
    def test_all_correct(self):
        # If all samples are correct, pass@k should be 1.0
        assert pass_at_k(n=10, c=10, k=1) == 1.0
        assert pass_at_k(n=10, c=10, k=5) == 1.0

    def test_none_correct(self):
        # If no samples are correct, pass@k should be 0.0
        assert pass_at_k(n=10, c=0, k=1) == 0.0
        assert pass_at_k(n=10, c=0, k=5) == 0.0

    def test_pass_at_1(self):
        # pass@1 with 5/10 correct = 0.5
        result = pass_at_k(n=10, c=5, k=1)
        assert abs(result - 0.5) < 1e-9

    def test_pass_at_k_increases_with_k(self):
        # pass@k should increase as k increases
        p1 = pass_at_k(n=10, c=3, k=1)
        p5 = pass_at_k(n=10, c=3, k=5)
        p10 = pass_at_k(n=10, c=3, k=10)
        assert p1 < p5 < p10

    def test_pass_at_k_when_k_equals_n(self):
        # When k=n and c>0, pass@k should be 1.0
        assert pass_at_k(n=5, c=1, k=5) == 1.0

    def test_known_value(self):
        # pass@2 with n=4, c=2
        # = 1 - C(2,2)/C(4,2) = 1 - 1/6 = 5/6
        result = pass_at_k(n=4, c=2, k=2)
        assert abs(result - 5 / 6) < 1e-9
