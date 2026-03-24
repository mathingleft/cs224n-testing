"""
temp_plot.py

One subplot per dataset group, each bar labelled by model name only.

Usage:
    python model-testing/temp_plot.py sophia_results/pass_at_1_summary.json
    python model-testing/temp_plot.py sophia_results/pass_at_1_summary.json --out my_chart.png
"""

import argparse
import json
from pathlib import Path

# ── Hard-coded groups ─────────────────────────────────────────────────────────
GROUPS = [
    {
        "title": "Numina_ones",
        "runs": ["compare_kms_kms_numinaones", "distill_big32_gemini_1s"],
    },
    {
        "title": "MiniF2F_entire",
        "runs": ["compare_all1"],
    },
    {
        "title": "MiniF2F_aimes",
        "runs": ["compare_base_sdft_aime", "compare_feedback_help", "compare_kms_base_numinaones"],
    },
    {
        "title": "MiniF2F_good20",
        "runs": ["distill_big32_gemini", "compare_trainfeed_20", "cover1", "teacherstudent"],
    },
    {
        "title": "MiniF2F_50",
        "runs": ["compare_50", "compare_50_sft"],
    },
]
# ─────────────────────────────────────────────────────────────────────────────


def model_short(model: str) -> str:
    """Last two path components of the model path."""
    parts = model.replace("\\", "/").strip("/").split("/")
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to pass_at_1_summary.json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(args.input_file) as f:
        raw = json.load(f)

    # Index by run name → list of entries (strip stray quote characters)
    by_run: dict[str, list] = {}
    for entry in raw:
        key = entry["run"].strip("\"'\u201c\u201d\u2018\u2019")
        by_run.setdefault(key, []).append(entry)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_groups = len(GROUPS)
    fig, axes = plt.subplots(1, n_groups, figsize=(4 * n_groups, 5),
                             sharey=False, constrained_layout=True)
    if n_groups == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors

    for ax, group in zip(axes, GROUPS):
        labels, scores = [], []
        for run_name in group["runs"]:
            for entry in by_run.get(run_name, []):
                labels.append(model_short(entry["model"]))
                scores.append(entry["pass_at_1_pct"])

        if not scores:
            ax.set_title(group["title"])
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue

        x = range(len(labels))
        bars = ax.bar(x, scores, color=[colors[i % len(colors)] for i in range(len(scores))],
                      edgecolor="white")

        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(group["title"], fontsize=10, fontweight="bold")
        ax.set_ylabel("pass@1 (%)")
        ax.set_ylim(0, max(scores + [5]) * 1.25)

        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{score:.1f}%", ha="center", va="bottom", fontsize=8)

    fig.suptitle("pass@1 by dataset", fontsize=13, fontweight="bold")

    out_path = args.out or str(Path(args.input_file).parent / "pass_at_1_chart.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
