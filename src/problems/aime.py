from __future__ import annotations
import json
from pathlib import Path

from .base import Problem, ProblemSet


class AIMEProblemSet(ProblemSet):
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def load(self, split: str | None = None) -> list[Problem]:
        problems = []
        for path in sorted(self.data_dir.glob("*.json")):
            with open(path) as f:
                entries = json.load(f)
            for entry in entries:
                p = Problem(
                    id=entry["id"],
                    statement=entry["statement"],
                    answer=int(entry["answer"]),
                    split=entry.get("split", "train"),
                    metadata={
                        k: v
                        for k, v in entry.items()
                        if k not in ("id", "statement", "answer", "split")
                    },
                )
                if split is None or p.split == split:
                    problems.append(p)
        return problems
