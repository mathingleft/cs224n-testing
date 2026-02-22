#!/usr/bin/env python3
"""Entry point: evaluate a single model checkpoint."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.problems.aime import AIMEProblemSet
from src.verification.numeric import NumericVerifier
from src.evaluation.evaluate import evaluate_model
from src.evaluation.compare import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model checkpoint")
    parser.add_argument(
        "--config", type=str, default="config/aime.yaml", help="Config file"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model name or checkpoint path"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name (for LoRA checkpoints)",
    )
    parser.add_argument("--split", type=str, default="test", help="Problem split")
    parser.add_argument(
        "--num-samples", type=int, default=None, help="Samples per problem"
    )
    parser.add_argument("--data-dir", type=str, default="data/aime")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
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

    # Load problems
    problem_set = AIMEProblemSet(args.data_dir)
    problems = problem_set.load(split=args.split)
    print(f"Loaded {len(problems)} problems (split={args.split})")

    if not problems:
        print("No problems found. Check data directory and split.")
        return

    # Load model
    model_path = Path(args.model)
    if model_path.exists():
        model, tokenizer = load_checkpoint(model_path, args.base_model)
    else:
        # Treat as HF model name
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    verifier = NumericVerifier()
    prompt_template = gen_cfg.get("prompt_template")

    metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        problems=problems,
        verifier=verifier,
        num_samples=num_samples,
        prompt_template=prompt_template,
        gen_config=gen_cfg,
        pass_at_k_values=pass_at_k_values,
    )

    # Print results
    print(f"\nModel: {args.model}")
    print(f"Problems: {metrics['num_problems']} ({args.split} split)")
    for k in pass_at_k_values:
        key = f"pass@{k}"
        if key in metrics:
            print(f"{key}: {metrics[key]:.1%}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
