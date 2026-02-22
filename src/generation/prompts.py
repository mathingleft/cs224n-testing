from __future__ import annotations
from src.problems.base import Problem


DEFAULT_AIME_TEMPLATE = (
    "Solve the following AIME problem. Show your work step-by-step, "
    "then give your final answer as an integer from 000 to 999 on its own "
    'line prefixed with "ANSWER:".\n\n'
    "Problem: {problem}\n\n"
    "Solution:"
)

DEFAULT_LEAN_TEMPLATE = (
    "Complete the following LEAN 4 theorem proof. "
    "Provide only the proof term.\n\n{problem}"
)


def format_prompt(problem: Problem, template: str | None = None) -> str:
    if template is None:
        template = DEFAULT_AIME_TEMPLATE
    return template.format(problem=problem.statement)
