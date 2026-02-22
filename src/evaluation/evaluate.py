from __future__ import annotations
import logging
import math

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.problems.base import Problem
from src.generation.sampler import Sampler
from src.verification.base import Verifier

logger = logging.getLogger("self_distill")


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k from the Codex paper (Chen et al., 2021).

    Args:
        n: total number of samples
        c: number of correct samples
        k: k value for pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    problems: list[Problem],
    verifier: Verifier,
    num_samples: int = 8,
    prompt_template: str | None = None,
    gen_config: dict | None = None,
    pass_at_k_values: list[int] | None = None,
) -> dict:
    """Evaluate a model on a set of problems.

    Returns a dict with pass@k metrics and per-problem breakdown.
    """
    if gen_config is None:
        gen_config = {}
    if pass_at_k_values is None:
        pass_at_k_values = gen_config.get("pass_at_k", [1])
    # Also accept from top-level call
    if pass_at_k_values is None:
        pass_at_k_values = [1]

    sampler = Sampler(
        model=model,
        tokenizer=tokenizer,
        temperature=gen_config.get("temperature", 0.7),
        top_p=gen_config.get("top_p", 0.95),
        max_new_tokens=gen_config.get("max_new_tokens", 1024),
        batch_size=gen_config.get("batch_size", 4),
    )

    per_problem = []
    total_correct = 0
    total_samples = 0

    for i, problem in enumerate(problems):
        logger.info(f"  Evaluating [{i+1}/{len(problems)}] {problem.id}")
        solutions = sampler.sample(problem, num_samples, prompt_template)
        results = verifier.verify_batch(problem, solutions)

        c = sum(1 for r in results if r.correct)
        n = len(results)
        total_correct += c
        total_samples += n

        problem_metrics = {
            "problem_id": problem.id,
            "n": n,
            "correct": c,
        }
        for k in pass_at_k_values:
            if k <= n:
                problem_metrics[f"pass@{k}"] = pass_at_k(n, c, k)

        per_problem.append(problem_metrics)

    # Aggregate
    metrics = {"num_problems": len(problems), "num_samples_per_problem": num_samples}
    for k in pass_at_k_values:
        scores = [p[f"pass@{k}"] for p in per_problem if f"pass@{k}" in p]
        if scores:
            metrics[f"pass@{k}"] = sum(scores) / len(scores)

    metrics["per_problem"] = per_problem
    return metrics
