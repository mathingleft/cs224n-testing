from __future__ import annotations
import subprocess
import tempfile
from pathlib import Path

from src.problems.base import Problem
from .base import Verifier, VerifyResult


class LeanVerifier(Verifier):
    """Verifier that checks LEAN 4 proofs via type-checking. Future use."""

    def __init__(self, timeout_sec: int = 60):
        self.timeout_sec = timeout_sec

    def verify(self, problem: Problem, solution: str) -> VerifyResult:
        # Combine the theorem statement with the proposed proof
        lean_code = problem.statement + "\n" + solution

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".lean", delete=False
        ) as f:
            f.write(lean_code)
            lean_file = Path(f.name)

        try:
            result = subprocess.run(
                ["lake", "env", "lean", str(lean_file)],
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
            correct = result.returncode == 0
            details = {
                "stdout": result.stdout[:2000],
                "stderr": result.stderr[:2000],
            }
        except subprocess.TimeoutExpired:
            correct = False
            details = {"error": "timeout"}
        except FileNotFoundError:
            correct = False
            details = {"error": "lean not installed"}
        finally:
            lean_file.unlink(missing_ok=True)

        return VerifyResult(
            correct=correct,
            extracted_answer=None,
            details=details,
        )
