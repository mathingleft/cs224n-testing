"""Tests for the training data construction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.problems.base import Problem
from src.verification.base import VerifyResult
from src.training.data import build_sft_dataset


class TestBuildSFTDataset:
    def test_basic(self):
        problems = [
            Problem(id="p1", statement="problem 1", answer=1),
            Problem(id="p2", statement="problem 2", answer=2),
        ]
        solutions = {
            "p1": ["sol1_correct", "sol1_wrong"],
            "p2": ["sol2_wrong", "sol2_correct"],
        }
        results = {
            "p1": [
                VerifyResult(correct=True, extracted_answer=1),
                VerifyResult(correct=False, extracted_answer=3),
            ],
            "p2": [
                VerifyResult(correct=False, extracted_answer=5),
                VerifyResult(correct=True, extracted_answer=2),
            ],
        }

        dataset = build_sft_dataset(problems, solutions, results)
        assert len(dataset) == 2
        assert "sol1_correct" in dataset[0]["completion"]
        assert "sol2_correct" in dataset[1]["completion"]

    def test_empty_when_none_correct(self):
        problems = [Problem(id="p1", statement="problem", answer=1)]
        solutions = {"p1": ["wrong1", "wrong2"]}
        results = {
            "p1": [
                VerifyResult(correct=False, extracted_answer=2),
                VerifyResult(correct=False, extracted_answer=3),
            ]
        }

        dataset = build_sft_dataset(problems, solutions, results)
        assert len(dataset) == 0

    def test_min_correct_ratio_filter(self):
        problems = [Problem(id="p1", statement="problem", answer=1)]
        solutions = {"p1": ["correct", "wrong1", "wrong2", "wrong3"]}
        results = {
            "p1": [
                VerifyResult(correct=True, extracted_answer=1),
                VerifyResult(correct=False, extracted_answer=2),
                VerifyResult(correct=False, extracted_answer=3),
                VerifyResult(correct=False, extracted_answer=4),
            ]
        }

        # 1/4 = 25% correct, filter requires >50%
        dataset = build_sft_dataset(
            problems, solutions, results, min_correct_ratio=0.5
        )
        assert len(dataset) == 0

    def test_max_correct_ratio_filter(self):
        problems = [Problem(id="p1", statement="problem", answer=1)]
        solutions = {"p1": ["c1", "c2", "c3", "c4"]}
        results = {
            "p1": [
                VerifyResult(correct=True, extracted_answer=1),
                VerifyResult(correct=True, extracted_answer=1),
                VerifyResult(correct=True, extracted_answer=1),
                VerifyResult(correct=True, extracted_answer=1),
            ]
        }

        # 100% correct, filter caps at 90%
        dataset = build_sft_dataset(
            problems, solutions, results, max_correct_ratio=0.9
        )
        assert len(dataset) == 0
