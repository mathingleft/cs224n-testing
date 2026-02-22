from __future__ import annotations
import json
from pathlib import Path

from .base import Problem, ProblemSet


class LeanProblemSet(ProblemSet):
    """Loader for miniF2F / LEAN 4 problems. Future use."""

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
                    answer=None,
                    split=entry.get("split", "test"),
                    metadata=entry.get("metadata", {}),
                )
                if split is None or p.split == split:
                    problems.append(p)
        return problems
