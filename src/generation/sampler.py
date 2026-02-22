from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.problems.base import Problem
from src.generation.prompts import format_prompt


class Sampler:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 1024,
        batch_size: int = 4,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

    @torch.no_grad()
    def sample(
        self,
        problem: Problem,
        num_samples: int,
        prompt_template: str | None = None,
    ) -> list[str]:
        """Generate num_samples solutions for a single problem.

        Returns a list of solution strings, problem ID not included.
        """
        prompt = format_prompt(problem, prompt_template)

        # For chat models, wrap in chat template if available
        assert (hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template)
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_text = prompt

        solutions = []
        remaining = num_samples

        while remaining > 0:
            batch = min(remaining, self.batch_size)
            inputs = self.tokenizer(
                [prompt_text] * batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # Decode only the generated part (strip the prompt)
            prompt_len = inputs["input_ids"].shape[1]
            generated_tokens = outputs[:, prompt_len:]
            texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            solutions.extend(texts)

            remaining -= batch

        return solutions[:num_samples]

    def sample_batch(
        self,
        problems: list[Problem],
        num_samples: int,
        prompt_template: str | None = None,
    ) -> dict[str, list[str]]:
        """Generate solutions for multiple problems.

        Returns dict mapping problem IDs to solutions: {problem_id: [solutions]}.
        """
        results = {}
        for problem in problems:
            results[problem.id] = self.sample(problem, num_samples, prompt_template)
        return results
