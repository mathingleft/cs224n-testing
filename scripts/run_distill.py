#!/usr/bin/env python3
"""Entry point: run the full self-distillation pipeline."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config, apply_overrides
from src.problems.aime import AIMEProblemSet
from src.verification.numeric import NumericVerifier
from src.training.distill import DistillationPipeline


def main():
    parser = argparse.ArgumentParser(description="Run self-distillation pipeline")
    parser.add_argument(
        "--config", type=str, default="config/aime.yaml", help="Path to config file"
    )
    parser.add_argument("--model", type=str, help="Override model name")
    parser.add_argument("--rounds", type=int, help="Override number of rounds")
    parser.add_argument("--num-samples", type=int, help="Override num_samples")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiments/default",
        help="Output directory",
    )
    parser.add_argument("--data-dir", type=str, default="data/aime", help="Data directory")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    config = load_config(args.config)

    # Apply CLI overrides
    overrides = {}
    if args.model:
        overrides["model.name"] = args.model
    if args.rounds:
        overrides["distillation.rounds"] = args.rounds
    if args.num_samples:
        overrides["generation.num_samples"] = args.num_samples
    if overrides:
        config = apply_overrides(config, overrides)

    # Set up problem set and verifier based on problem_type
    problem_type = config.get("problem_type", "aime")
    if problem_type == "aime":
        problem_set = AIMEProblemSet(args.data_dir)
        verifier = NumericVerifier()
    elif problem_type == "lean":
        from src.problems.lean import LeanProblemSet
        from src.verification.lean_check import LeanVerifier

        problem_set = LeanProblemSet(args.data_dir)
        verifier = LeanVerifier(
            timeout_sec=config.get("verification", {}).get("timeout_sec", 60)
        )
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")

    pipeline = DistillationPipeline(
        config=config,
        problem_set=problem_set,
        verifier=verifier,
        output_dir=args.output_dir,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
