"""Tests for the generation module (prompt formatting)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.problems.base import Problem
from src.generation.prompts import format_prompt, DEFAULT_AIME_TEMPLATE, DEFAULT_LEAN_TEMPLATE


class TestPromptFormatting:
    def test_default_template(self):
        problem = Problem(
            id="test", statement="Find the value of x.", answer=42
        )
        prompt = format_prompt(problem)
        assert "Find the value of x." in prompt
        assert "AIME" in prompt
        assert "ANSWER:" in prompt

    def test_custom_template(self):
        problem = Problem(
            id="test", statement="What is 2+2?", answer=4
        )
        template = "Q: {problem}\nA:"
        prompt = format_prompt(problem, template)
        assert prompt == "Q: What is 2+2?\nA:"

    def test_statement_preserved_exactly(self):
        statement = "Let $f(x) = x^2 + 3x + 1$. Find $f(5)$."
        problem = Problem(id="test", statement=statement, answer=41)
        prompt = format_prompt(problem)
        assert statement in prompt

    def test_default_lean_template(self):
        problem = Problem(
            id="test",
            statement="theorem example : 1 + 1 = 2 := by",
            answer="rfl"
        )
        prompt = format_prompt(problem, DEFAULT_LEAN_TEMPLATE)
        assert "theorem example : 1 + 1 = 2 := by" in prompt
        assert "LEAN 4" in prompt
        assert "proof term" in prompt
