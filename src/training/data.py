from __future__ import annotations
from datasets import Dataset

from src.problems.base import Problem
from src.verification.base import VerifyResult
from src.generation.prompts import format_prompt


def build_sft_dataset(
    problems: list[Problem],
    solutions: dict[str, list[str]],
    results: dict[str, list[VerifyResult]],
    prompt_template: str | None = None,
    min_correct_ratio: float = 0.0,
    max_correct_ratio: float = 1.0,
) -> Dataset:
    """Build a HuggingFace Dataset from verified correct solutions.

    Args:
        problems: list of Problem objects
        solutions: {problem_id: [solution_strings]}
        results: {problem_id: [VerifyResult]}
        prompt_template: prompt format string
        min_correct_ratio: skip problems with fewer correct solutions than this
        max_correct_ratio: skip problems with more correct solutions than this
    """
    records = []

    for problem in problems:
        pid = problem.id
        if pid not in solutions or pid not in results:
            continue

        correct_indices = [
            i for i, r in enumerate(results[pid]) if r.correct
        ]
        ratio = len(correct_indices) / len(results[pid]) if results[pid] else 0

        if ratio < min_correct_ratio or ratio > max_correct_ratio:
            continue

        prompt = format_prompt(problem, prompt_template)

        for idx in correct_indices:
            records.append({
                "prompt": prompt,
                "completion": solutions[pid][idx],
                "problem_id": pid,
            })

    if not records:
        return Dataset.from_dict({"prompt": [], "completion": [], "problem_id": []})

    return Dataset.from_list(records)
