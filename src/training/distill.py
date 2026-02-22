from __future__ import annotations
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.problems.base import Problem, ProblemSet
from src.generation.sampler import Sampler
from src.verification.base import Verifier
from src.training.trainer import SFTLoRATrainer
from src.training.data import build_sft_dataset
from src.evaluation.evaluate import evaluate_model
from src.utils.logging import RunLogger

logger = logging.getLogger("self_distill")


class DistillationPipeline:
    def __init__(
        self,
        config: dict,
        problem_set: ProblemSet,
        verifier: Verifier,
        output_dir: str | Path,
    ):
        self.config = config
        self.problem_set = problem_set
        self.verifier = verifier
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.run_logger = RunLogger(self.output_dir / "logs")

    def run(self):
        """Run the full distillation loop."""
        model_cfg = self.config["model"]
        gen_cfg = self.config["generation"]
        dist_cfg = self.config["distillation"]
        eval_cfg = self.config.get("evaluation", {})

        rounds = dist_cfg.get("rounds", 3)
        num_samples = gen_cfg.get("num_samples", 8)
        prompt_template = gen_cfg.get("prompt_template")

        # Load base model
        logger.info(f"Loading base model: {model_cfg['name']}")
        model, tokenizer = self._load_model(model_cfg)

        # Save base checkpoint
        base_dir = self.output_dir / "checkpoints" / "round_000_base"
        base_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(base_dir))
        model.config.to_json_file(str(base_dir / "config.json"))

        train_problems = self.problem_set.train()
        test_problems = self.problem_set.test()

        logger.info(
            f"Loaded {len(train_problems)} train, {len(test_problems)} test problems"
        )

        all_eval_results = []

        # Evaluate base model
        if test_problems:
            logger.info("Evaluating base model...")
            base_metrics = evaluate_model(
                model=model,
                tokenizer=tokenizer,
                problems=test_problems,
                verifier=self.verifier,
                num_samples=num_samples,
                prompt_template=prompt_template,
                gen_config=gen_cfg,
                pass_at_k_values=eval_cfg.get("pass_at_k", [1]),
            )
            self.run_logger.log_evaluation(0, base_metrics)
            all_eval_results.append({"round": 0, "metrics": base_metrics})
            logger.info(f"Base model metrics: {base_metrics}")

        for round_num in range(1, rounds + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"DISTILLATION ROUND {round_num}/{rounds}")
            logger.info(f"{'='*60}")

            # Step 1: Generate solutions
            logger.info("Generating solutions...")
            sampler = Sampler(
                model=model,
                tokenizer=tokenizer,
                temperature=gen_cfg.get("temperature", 0.7),
                top_p=gen_cfg.get("top_p", 0.95),
                max_new_tokens=gen_cfg.get("max_new_tokens", 1024),
                batch_size=gen_cfg.get("batch_size", 4),
            )

            all_solutions = {}
            all_results = {}

            for i, problem in enumerate(train_problems):
                logger.info(
                    f"  [{i+1}/{len(train_problems)}] Generating for {problem.id}"
                )
                solutions = sampler.sample(problem, num_samples, prompt_template)
                all_solutions[problem.id] = solutions

                # Log generations
                for idx, sol in enumerate(solutions):
                    self.run_logger.log_generation(
                        round_num, problem.id, idx, sol
                    )

                # Step 2: Verify solutions
                verify_results = self.verifier.verify_batch(problem, solutions)
                all_results[problem.id] = verify_results

                correct_count = sum(1 for r in verify_results if r.correct)
                logger.info(
                    f"    {correct_count}/{len(solutions)} correct"
                )

                # Log verifications
                for idx, vr in enumerate(verify_results):
                    self.run_logger.log_verification(
                        round_num,
                        problem.id,
                        idx,
                        vr.correct,
                        vr.extracted_answer,
                        vr.details.get("expected"),
                    )

            # Step 3: Build SFT dataset from correct solutions
            logger.info("Building training dataset...")
            sft_dataset = build_sft_dataset(
                problems=train_problems,
                solutions=all_solutions,
                results=all_results,
                prompt_template=prompt_template,
                min_correct_ratio=dist_cfg.get("min_correct_ratio", 0.0),
                max_correct_ratio=dist_cfg.get("max_correct_ratio", 1.0),
            )
            logger.info(f"Training dataset size: {len(sft_dataset)} examples")

            if len(sft_dataset) == 0:
                logger.warning(
                    "No correct solutions found — skipping training this round."
                )
                continue

            # Step 4: Fine-tune
            logger.info("Fine-tuning...")
            checkpoint_dir = (
                self.output_dir / "checkpoints" / f"round_{round_num:03d}"
            )
            trainer = SFTLoRATrainer(self.config)
            model, tokenizer = trainer.train(
                model=model,
                tokenizer=tokenizer,
                dataset=sft_dataset,
                output_dir=checkpoint_dir,
                round_num=round_num,
                run_logger=self.run_logger,
            )

            # Step 5: Evaluate
            if test_problems:
                logger.info("Evaluating...")
                metrics = evaluate_model(
                    model=model,
                    tokenizer=tokenizer,
                    problems=test_problems,
                    verifier=self.verifier,
                    num_samples=num_samples,
                    prompt_template=prompt_template,
                    gen_config=gen_cfg,
                    pass_at_k_values=eval_cfg.get("pass_at_k", [1]),
                )
                self.run_logger.log_evaluation(round_num, metrics)
                all_eval_results.append(
                    {"round": round_num, "metrics": metrics}
                )
                logger.info(f"Round {round_num} metrics: {metrics}")

        # Save overall results
        results_path = self.output_dir / "eval_results.json"
        with open(results_path, "w") as f:
            json.dump(all_eval_results, f, indent=2)

        logger.info(f"\nDistillation complete. Results saved to {results_path}")
        return all_eval_results

    def _load_model(self, model_cfg: dict):
        model_name = model_cfg["name"]
        load_in_4bit = model_cfg.get("load_in_4bit", False)

        kwargs = {
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "device_map": "auto" if torch.cuda.is_available() else None,
        }

        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id

        return model, tokenizer
