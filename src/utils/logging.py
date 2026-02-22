from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path


logger = logging.getLogger("self_distill")


class RunLogger:
    """Structured JSONL logger for generation, verification, and training events."""

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._files: dict[str, Path] = {}

    def _get_path(self, round_num: int, category: str) -> Path:
        round_dir = self.log_dir / f"round_{round_num:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        return round_dir / f"{category}.jsonl"

    def _write(self, round_num: int, category: str, data: dict):
        path = self._get_path(round_num, category)
        data["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(path, "a") as f:
            f.write(json.dumps(data) + "\n")

    def log_generation(
        self,
        round_num: int,
        problem_id: str,
        sample_idx: int,
        solution: str,
    ):
        self._write(
            round_num,
            "generation",
            {
                "problem_id": problem_id,
                "sample_idx": sample_idx,
                "solution": solution,
            },
        )

    def log_verification(
        self,
        round_num: int,
        problem_id: str,
        sample_idx: int,
        correct: bool,
        extracted_answer,
        expected_answer,
    ):
        self._write(
            round_num,
            "verification",
            {
                "problem_id": problem_id,
                "sample_idx": sample_idx,
                "correct": correct,
                "extracted_answer": extracted_answer,
                "expected_answer": expected_answer,
            },
        )

    def log_training(self, round_num: int, step: int, loss: float, lr: float, epoch: float):
        self._write(
            round_num,
            "training",
            {"step": step, "loss": loss, "lr": lr, "epoch": epoch},
        )

    def log_evaluation(self, round_num: int, metrics: dict):
        self._write(round_num, "evaluation", {"metrics": metrics})

    @staticmethod
    def read_log(path: str | Path) -> list[dict]:
        entries = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries
