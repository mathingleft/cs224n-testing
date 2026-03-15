"""
pass@1.py

Walk all compare_results.json files under sophia_results/, compute the
unbiased pass@1 estimate for each model using the per-sample status fields,
and write a compact summary to sophia_results/pass_at_1_summary.json.

Usage:
    python model-testing/pass@1.py
    python model-testing/pass@1.py --results-dir path/to/sophia_results
"""

import argparse
import collections
import json
import os
from math import comb
from pathlib import Path


def unbiased_pass_at_1(statuses: list[str]) -> float:
    """c/n — unbiased estimator of pass@1 given n samples with c passing."""
    n = len(statuses)
    c = statuses.count("PASS")
    if n == 0:
        return 0.0
    # General formula: 1 - C(n-c, k) / C(n, k), with k=1 simplifies to c/n
    if n - c < 1:
        return 1.0
    return 1.0 - (n - c) / n


def compute_pass_at_1(results: list[dict]) -> dict:
    """Given a flat list of per-sample result dicts, return pass@1 stats."""
    by_example = collections.defaultdict(list)
    for r in results:
        by_example[r["example_idx"]].append(r["status"])

    estimates = []
    for idx, statuses in sorted(by_example.items()):
        estimates.append(unbiased_pass_at_1(statuses))

    total = len(estimates)
    mean_est = sum(estimates) / max(total, 1)
    pass_count = sum(1 for e in estimates if e > 0)
    return {
        "pass_at_1_pct": round(100 * mean_est, 3),
        "problems_with_any_pass": pass_count,
        "total_problems": total,
        "n_samples_per_problem": len(results) // max(total, 1),
    }


def iter_compare_files(results_dir: Path):
    """Yield (run_name, path) for every compare_results.json found."""
    for path in sorted(results_dir.rglob("compare_results.json")):
        # Skip data_filtering subtrees
        parts = path.relative_to(results_dir).parts
        if any(p.startswith("data_filtering") for p in parts):
            continue
        run_name = str(path.relative_to(results_dir).parent)
        yield run_name, path


def extract_models(data: dict, run_name: str) -> list[dict]:
    """
    Handle both schema variants:
      - dict models: {"models": {"model_name": {"results": [...], ...}}}
      - list models: {"models": [{"results": [...], ...}]}
    Returns list of {"model": str, "results": list, "config": dict, "orig_pass_at_k_pct": float}
    """
    config = data.get("config", {})
    models_raw = data.get("models", {})
    out = []

    if isinstance(models_raw, dict):
        for model_name, model_data in models_raw.items():
            if not isinstance(model_data, dict) or "results" not in model_data:
                continue
            out.append({
                "model": model_name,
                "results": model_data["results"],
                "config": config,
                "orig_pass_count": model_data.get("pass_count"),
                "orig_total": model_data.get("total"),
                "orig_pass_at_k_pct": model_data.get("pass_at_k_pct"),
            })
    elif isinstance(models_raw, list):
        for model_data in models_raw:
            if not isinstance(model_data, dict) or "results" not in model_data:
                continue
            # Infer model name from results entries
            labels = {r.get("model_label") for r in model_data["results"] if r.get("model_label")}
            model_name = labels.pop() if len(labels) == 1 else (labels.pop() if labels else run_name)
            out.append({
                "model": model_name,
                "results": model_data["results"],
                "config": config,
                "orig_pass_count": model_data.get("pass_count"),
                "orig_total": model_data.get("total"),
                "orig_pass_at_k_pct": model_data.get("pass_at_k_pct"),
            })

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=None,
                        help="Path to sophia_results (default: ../sophia_results relative to this script)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    results_dir = Path(args.results_dir) if args.results_dir else script_dir.parent / "sophia_results"

    if not results_dir.exists():
        print(f"ERROR: results dir not found: {results_dir}")
        return

    summary = []  # list of entries, one per (run, model)

    for run_name, path in iter_compare_files(results_dir):
        with open(path) as f:
            data = json.load(f)

        models = extract_models(data, run_name)
        if not models:
            print(f"  [skip] {run_name}: no models found")
            continue

        for m in models:
            stats = compute_pass_at_1(m["results"])
            entry = {
                "run": run_name,
                "model": m["model"],
                "pass_at_1_pct": stats["pass_at_1_pct"],
                "problems_with_any_pass": stats["problems_with_any_pass"],
                "total_problems": stats["total_problems"],
                "n_samples_per_problem": stats["n_samples_per_problem"],
                "orig_pass_at_k_pct": m["orig_pass_at_k_pct"],
                "orig_pass_count": m["orig_pass_count"],
                "config": m["config"],
            }
            summary.append(entry)
            k = m["config"].get("K", "?")
            print(
                f"  {run_name} | {m['model']}\n"
                f"    pass@1={stats['pass_at_1_pct']:.1f}%  "
                f"(orig pass@{k}={m['orig_pass_at_k_pct']}%  "
                f"{stats['problems_with_any_pass']}/{stats['total_problems']} problems  "
                f"n={stats['n_samples_per_problem']})"
            )

    # Sort by pass@1 descending
    summary.sort(key=lambda e: e["pass_at_1_pct"], reverse=True)

    out_path = results_dir / "pass_at_1_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {len(summary)} entries → {out_path}")


if __name__ == "__main__":
    main()
