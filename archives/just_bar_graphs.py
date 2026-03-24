"""
Plot pass@k bar graphs for Goedel and Kimina models on the same chart.

Goedel results: local files in model-testing/goedel_results/
Kimina results: downloaded from Modal volume (results/Kimina-Prover-Preview-Distill-7B/)
"""

import json
import os
import collections
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODAL = "/Users/sophia/anaconda3/envs/cs224n/bin/modal"
KIMINA_CACHE = os.path.join(SCRIPT_DIR, "kimina_results")
os.makedirs(KIMINA_CACHE, exist_ok=True)


def compute_pass_at_k(results):
    """pass@k: a problem passes if ANY of its K samples compiles."""
    by_example = collections.defaultdict(list)
    for r in results:
        by_example[r["example_idx"]].append(r)
    pass_count = sum(
        1 for samples in by_example.values()
        if "PASS" in [s["status"] for s in samples]
    )
    total = len(by_example)
    return pass_count, total


def load_results_list(data):
    """Extract flat list of result entries from either file format."""
    if "results" in data and isinstance(data["results"], list):
        return data["results"]
    if "models" in data:
        models = data["models"]
        if isinstance(models, list):
            out = []
            for m in models:
                out.extend(m.get("results", []))
            return out
        if isinstance(models, dict):
            out = []
            for m in models.values():
                out.extend(m.get("results", []))
            return out
    return []


def download_from_volume(vol_path, local_name):
    """Download a file from Modal volume, return local path or None."""
    local_path = os.path.join(KIMINA_CACHE, local_name)
    result = subprocess.run(
        [MODAL, "volume", "get", "my-volume-1", vol_path, local_path],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and os.path.exists(local_path):
        return local_path
    return None


# ── Goedel (local) ───────────────────────────────────────────────────────────
goedel_dir = os.path.join(SCRIPT_DIR, "goedel_results")
goedel_models = {
    "Base":              "default.json",
    "Compiler Feedback": "compiler-error.json",
    # "run2":              "run2.json",
    "SDFT":              "run3.json",
    # "run4":              "run4.json",
}

goedel_labels = []
goedel_scores = []
for short_label, fname in goedel_models.items():
    path = os.path.join(goedel_dir, fname)
    if not os.path.exists(path):
        print(f"  skip (not found): {path}")
        continue
    data = json.load(open(path))
    results = load_results_list(data)
    passed, total = compute_pass_at_k(results)
    pct = 100 * passed / max(total, 1)
    goedel_labels.append(short_label)
    goedel_scores.append(pct)
    print(f"  Goedel {short_label}: {passed}/{total} ({pct:.1f}%)")


# ── Kimina (volume) ──────────────────────────────────────────────────────────
kimina_volume = {
    "Base":              "results/Kimina-Prover-Preview-Distill-7B/base/compare_results.json",
    "SDFT":              "results/Kimina-Prover-Preview-Distill-7B/run_sdft/epoch-19/compare_results.json",
    "SDFT + Feedback":   "results/Kimina-Prover-Preview-Distill-7B/run_notarg_kimi_feedback/epoch-9/compare_results.json",
}

kimina_local = {
    "Base":              "model-testing/kimina_results/base.json",
    "SDFT":              "model-testing/kimina_results/kimina_sdft.json",
    "SDFT + Feedback":   "model-testing/kimina_results/kimina_sdft_+_feedback.json",
}

kimina_labels = []
kimina_scores = []
for short_label, vol_path in kimina_volume.items():
    # local = download_from_volume(vol_path, f"kimina_{short_label.replace(' ', '_').lower()}.json")
    # if local is None:
    #     print(f"  skip (download failed): {vol_path}")
    #     continue
    local_path = kimina_local[short_label]
    if not os.path.exists(local_path):
        print(f"  skip (local file not found): {local_path}")
        continue
    data = json.load(open(local_path))
    results = load_results_list(data)
    passed, total = compute_pass_at_k(results)
    pct = 100 * passed / max(total, 1)
    kimina_labels.append(short_label)
    kimina_scores.append(pct)
    print(f"  Kimina {short_label}: {passed}/{total} ({pct:.1f}%)")


# ── Combined bar chart ───────────────────────────────────────────────────────
all_labels = [f"Goedel\n{l}" for l in goedel_labels] + [f"Kimina\n{l}" for l in kimina_labels]
all_scores = goedel_scores + kimina_scores
colors = ["blue"] * len(goedel_scores) + ["orange"] * len(kimina_scores)

fig, ax = plt.subplots(figsize=(max(8, len(all_labels) * 1.4), 6))
bars = ax.bar(range(len(all_labels)), all_scores, color=colors, edgecolor="white", linewidth=0.5)

ax.set_xticks(range(len(all_labels)))
ax.set_xticklabels(all_labels, rotation=0, ha="center", fontsize=9)
ax.set_ylabel("pass@k (%)")
ax.set_title("pass@k: Goedel-Prover-SFT vs Kimina-Prover-Preview-Distill-7B")
ax.set_ylim(0, max(all_scores + [5]) * 1.3 + 3)

for bar, score in zip(bars, all_scores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{score:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.legend(
    handles=[
        Patch(facecolor="blue", label="Goedel-Prover-SFT"),
        Patch(facecolor="orange", label="Kimina-Prover-Preview-Distill-7B"),
    ],
    loc="upper right",
)

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "comparison_goedel_kimina.png")
fig.savefig(out_path, dpi=150)
print(f"\nSaved → {out_path}")
