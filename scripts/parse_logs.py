#!/usr/bin/env python3
"""Entry point: parse and analyze distillation logs."""

import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import RunLogger


def get_rounds(log_dir: Path) -> list[int]:
    """Find all round directories."""
    rounds = []
    for d in sorted(log_dir.iterdir()):
        if d.is_dir() and d.name.startswith("round_"):
            try:
                rounds.append(int(d.name.split("_")[1]))
            except (ValueError, IndexError):
                pass
    return rounds


def summary(log_dir: Path):
    """Print a summary of the distillation run."""
    rounds = get_rounds(log_dir)
    if not rounds:
        print("No rounds found.")
        return

    print(f"Distillation run: {len(rounds)} rounds found")
    print(f"Rounds: {rounds}\n")

    for r in rounds:
        round_dir = log_dir / f"round_{r:03d}"
        print(f"--- Round {r} ---")

        # Verification stats
        verif_path = round_dir / "verification.jsonl"
        if verif_path.exists():
            entries = RunLogger.read_log(verif_path)
            total = len(entries)
            correct = sum(1 for e in entries if e.get("correct"))
            problems = len(set(e["problem_id"] for e in entries))
            print(f"  Verification: {correct}/{total} correct across {problems} problems")

        # Training stats
        train_path = round_dir / "training.jsonl"
        if train_path.exists():
            entries = RunLogger.read_log(train_path)
            if entries:
                final_loss = entries[-1].get("loss", "?")
                steps = entries[-1].get("step", "?")
                print(f"  Training: {steps} steps, final loss = {final_loss}")

        # Evaluation stats
        eval_path = round_dir / "evaluation.jsonl"
        if eval_path.exists():
            entries = RunLogger.read_log(eval_path)
            if entries:
                metrics = entries[-1].get("metrics", {})
                metric_strs = []
                for k, v in metrics.items():
                    if k.startswith("pass@"):
                        metric_strs.append(f"{k}={v:.1%}")
                if metric_strs:
                    print(f"  Evaluation: {', '.join(metric_strs)}")

        print()


def per_problem_breakdown(log_dir: Path, output_path: str | None = None):
    """Show per-problem results across rounds."""
    rounds = get_rounds(log_dir)

    # Collect per-problem stats across rounds
    # {problem_id: {round: {correct, total}}}
    data = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

    for r in rounds:
        verif_path = log_dir / f"round_{r:03d}" / "verification.jsonl"
        if not verif_path.exists():
            continue
        entries = RunLogger.read_log(verif_path)
        for e in entries:
            pid = e["problem_id"]
            data[pid][r]["total"] += 1
            if e.get("correct"):
                data[pid][r]["correct"] += 1

    if not data:
        print("No verification data found.")
        return

    # Print or save
    rows = []
    for pid in sorted(data.keys()):
        row = {"problem_id": pid}
        for r in rounds:
            d = data[pid][r]
            row[f"round_{r}_correct"] = d["correct"]
            row[f"round_{r}_total"] = d["total"]
            row[f"round_{r}_rate"] = (
                f"{d['correct']/d['total']:.0%}" if d["total"] > 0 else "N/A"
            )
        rows.append(row)

    if output_path:
        fieldnames = ["problem_id"]
        for r in rounds:
            fieldnames.extend(
                [f"round_{r}_correct", f"round_{r}_total", f"round_{r}_rate"]
            )
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Per-problem breakdown saved to {output_path}")
    else:
        # Print table
        print(f"{'Problem':<25}", end="")
        for r in rounds:
            print(f"  Round {r:>3}", end="")
        print()
        print("-" * (25 + 10 * len(rounds)))
        for row in rows:
            print(f"{row['problem_id']:<25}", end="")
            for r in rounds:
                print(f"  {row[f'round_{r}_rate']:>8}", end="")
            print()


def find_regressions(log_dir: Path):
    """Find problems where performance got worse between consecutive rounds."""
    rounds = get_rounds(log_dir)
    if len(rounds) < 2:
        print("Need at least 2 rounds to find regressions.")
        return

    # Collect per-problem accuracy by round
    data = defaultdict(dict)  # {pid: {round: accuracy}}

    for r in rounds:
        verif_path = log_dir / f"round_{r:03d}" / "verification.jsonl"
        if not verif_path.exists():
            continue
        entries = RunLogger.read_log(verif_path)
        by_problem = defaultdict(lambda: {"correct": 0, "total": 0})
        for e in entries:
            pid = e["problem_id"]
            by_problem[pid]["total"] += 1
            if e.get("correct"):
                by_problem[pid]["correct"] += 1
        for pid, d in by_problem.items():
            data[pid][r] = d["correct"] / d["total"] if d["total"] > 0 else 0

    regressions = []
    for pid in sorted(data.keys()):
        for i in range(len(rounds) - 1):
            r_prev, r_next = rounds[i], rounds[i + 1]
            if r_prev in data[pid] and r_next in data[pid]:
                if data[pid][r_next] < data[pid][r_prev]:
                    regressions.append(
                        {
                            "problem_id": pid,
                            "round_from": r_prev,
                            "round_to": r_next,
                            "acc_before": data[pid][r_prev],
                            "acc_after": data[pid][r_next],
                        }
                    )

    if not regressions:
        print("No regressions found.")
    else:
        print(f"Found {len(regressions)} regressions:\n")
        for reg in regressions:
            print(
                f"  {reg['problem_id']}: "
                f"round {reg['round_from']} ({reg['acc_before']:.0%}) -> "
                f"round {reg['round_to']} ({reg['acc_after']:.0%})"
            )


def main():
    parser = argparse.ArgumentParser(description="Parse distillation logs")
    parser.add_argument(
        "--log-dir", type=str, required=True, help="Log directory"
    )
    parser.add_argument("--summary", action="store_true", help="Print run summary")
    parser.add_argument(
        "--per-problem", action="store_true", help="Per-problem breakdown"
    )
    parser.add_argument(
        "--regressions", action="store_true", help="Find regressions"
    )
    parser.add_argument("--output", type=str, default=None, help="Output file (CSV)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Log directory not found: {log_dir}")
        sys.exit(1)

    if args.summary:
        summary(log_dir)
    if args.per_problem:
        per_problem_breakdown(log_dir, args.output)
    if args.regressions:
        find_regressions(log_dir)

    if not (args.summary or args.per_problem or args.regressions):
        # Default to summary
        summary(log_dir)


if __name__ == "__main__":
    main()
