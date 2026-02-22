from __future__ import annotations
import logging
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
)
from trl import SFTTrainer, SFTConfig

from src.utils.logging import RunLogger

logger = logging.getLogger("self_distill")


class LoggingCallback(TrainerCallback):
    """Callback that logs training metrics to our RunLogger."""

    def __init__(self, run_logger: RunLogger, round_num: int):
        self.run_logger = run_logger
        self.round_num = round_num

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.run_logger.log_training(
                round_num=self.round_num,
                step=state.global_step,
                loss=logs.get("loss", 0.0),
                lr=logs.get("learning_rate", 0.0),
                epoch=logs.get("epoch", 0.0),
            )


class SFTLoRATrainer:
    def __init__(self, config: dict):
        self.config = config

    def train(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        dataset: Dataset,
        output_dir: str | Path,
        round_num: int,
        run_logger: RunLogger | None = None,
    ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Fine-tune the model on the dataset using LoRA SFT.

        Returns the updated model and tokenizer.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = self.config.get("training", {})
        lora_cfg = train_cfg.get("lora", {})

        if len(dataset) == 0:
            logger.warning("Empty dataset — skipping training for this round.")
            return model, tokenizer

        # Apply LoRA if not already a PeftModel
        if not isinstance(model, PeftModel):
            lora_config = LoraConfig(
                r=lora_cfg.get("r", 16),
                lora_alpha=lora_cfg.get("alpha", 32),
                target_modules=lora_cfg.get(
                    "target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]
                ),
                lora_dropout=lora_cfg.get("dropout", 0.05),
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        # Format dataset for SFT: combine prompt + completion
        def format_example(example):
            if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
                messages = [
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": example["completion"]},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            else:
                text = example["prompt"] + " " + example["completion"]
            return {"text": text}

        formatted = dataset.map(format_example)

        callbacks = []
        if run_logger:
            callbacks.append(LoggingCallback(run_logger, round_num))

        sft_config = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=train_cfg.get("epochs", 2),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=train_cfg.get(
                "gradient_accumulation_steps", 4
            ),
            learning_rate=train_cfg.get("lr", 2e-5),
            warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
            max_seq_length=train_cfg.get("max_seq_len", 1024),
            logging_steps=1,
            save_strategy="epoch",
            fp16=torch.cuda.is_available(),
            dataset_text_field="text",
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=formatted,
            args=sft_config,
            callbacks=callbacks,
        )

        trainer.train()

        # Save the LoRA adapter
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        logger.info(f"Saved checkpoint to {output_dir}")
        return model, tokenizer
