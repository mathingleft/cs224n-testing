#!/usr/bin/env python3
"""Entry point: compare multiple model checkpoints."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.problems.aime import AIMEProblemSet
from src.verification.numeric import NumericVerifier
from src.evaluation.compare import compare_checkpoints, print_comparison_table


def main():
    parser = argparse.ArgumentParser(description="Compare model checkpoints")
    parser.add_argument(
        "--config", type=str, default="config/aime.yaml", help="Config file"
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="Checkpoint directories to compare",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name (for LoRA checkpoints)",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default="data/aime")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    gen_cfg = config.get("generation", {})
    eval_cfg = config.get("evaluation", {})
    num_samples = args.num_samples or gen_cfg.get("num_samples", 8)
    pass_at_k_values = eval_cfg.get("pass_at_k", [1])

    problem_set = AIMEProblemSet(args.data_dir)
    problems = problem_set.load(split=args.split)
    print(f"Loaded {len(problems)} problems (split={args.split})")

    if not problems:
        print("No problems found.")
        return

    verifier = NumericVerifier()
    prompt_template = gen_cfg.get("prompt_template")

    results = compare_checkpoints(
        checkpoint_dirs=args.checkpoints,
        problems=problems,
        verifier=verifier,
        num_samples=num_samples,
        prompt_template=prompt_template,
        gen_config=gen_cfg,
        pass_at_k_values=pass_at_k_values,
        base_model_name=args.base_model,
    )

    print_comparison_table(results, pass_at_k_values)

    if args.output:
        # Strip per_problem from saved output for readability
        save_results = []
        for r in results:
            sr = {k: v for k, v in r.items() if k != "per_problem"}
            save_results.append(sr)
        with open(args.output, "w") as f:
            json.dump(save_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
