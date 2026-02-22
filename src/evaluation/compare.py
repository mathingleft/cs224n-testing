from __future__ import annotations
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

from src.problems.base import Problem
from src.verification.base import Verifier
from src.evaluation.evaluate import evaluate_model

logger = logging.getLogger("self_distill")


def load_checkpoint(
    checkpoint_dir: str | Path,
    base_model_name: str | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a model checkpoint. Handles both full models and LoRA adapters."""
    checkpoint_dir = Path(checkpoint_dir)

    # Check if this is a LoRA adapter
    adapter_config_path = checkpoint_dir / "adapter_config.json"
    if adapter_config_path.exists():
        peft_config = PeftConfig.from_pretrained(str(checkpoint_dir))
        model_name = base_model_name or peft_config.base_model_name_or_path
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model = PeftModel.from_pretrained(model, str(checkpoint_dir))
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    else:
        # Full model checkpoint or base model name
        model_name = base_model_name or str(checkpoint_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def compare_checkpoints(
    checkpoint_dirs: list[str | Path],
    problems: list[Problem],
    verifier: Verifier,
    num_samples: int = 8,
    prompt_template: str | None = None,
    gen_config: dict | None = None,
    pass_at_k_values: list[int] | None = None,
    base_model_name: str | None = None,
) -> list[dict]:
    """Evaluate and compare multiple checkpoints on the same problems."""
    if pass_at_k_values is None:
        pass_at_k_values = [1]

    results = []
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_dir = Path(checkpoint_dir)
        name = checkpoint_dir.name
        logger.info(f"\nEvaluating checkpoint: {name}")

        model, tokenizer = load_checkpoint(checkpoint_dir, base_model_name)
        metrics = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            problems=problems,
            verifier=verifier,
            num_samples=num_samples,
            prompt_template=prompt_template,
            gen_config=gen_config,
            pass_at_k_values=pass_at_k_values,
        )

        results.append({"name": name, "path": str(checkpoint_dir), **metrics})

        # Free memory
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def print_comparison_table(results: list[dict], pass_at_k_values: list[int] | None = None):
    """Pretty-print a comparison table."""
    if pass_at_k_values is None:
        pass_at_k_values = [1]

    # Header
    k_cols = [f"pass@{k}" for k in pass_at_k_values]
    header = f"{'Model':<25}" + "".join(f"{col:>10}" for col in k_cols)
    separator = "-" * len(header)

    print(f"\n{separator}")
    print(header)
    print(separator)

    for r in results:
        row = f"{r['name']:<25}"
        for k in pass_at_k_values:
            key = f"pass@{k}"
            val = r.get(key, 0.0)
            row += f"{val:>9.1%} "
        print(row)

    print(separator)
