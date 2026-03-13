import json
import os
import collections
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def plot(labels, scores, title, outpath):
    all_labels = labels
    all_scores = scores
    colors = ["steelblue"] * len(scores)

    fig, ax = plt.subplots(figsize=(max(8, len(all_labels) * 1.4), 6))
    bars = ax.bar(range(len(all_labels)), all_scores, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=0, ha="center", fontsize=9)
    ax.set_ylabel("pass@k (%)")
    ax.set_title(title)
    ax.set_ylim(0, max(all_scores + [5]) * 1.3 + 3)

    for bar, score in zip(bars, all_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{score:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out_path = outpath
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved → {out_path}")

plot(["SFT", "Compiler", "SDFT", "Base"], [52, 55, 57, 58], "All MiniF2F", "all_mini.png")
plot(["SFT", "Compiler", "SDFT", "Base"], [52, 55, 57, 58], "Sample 20 MiniF2F", "all_mini.png")
plot(["Compiler + Feedback", "Compiler", "SDFT", ""])
plot(["Distill@10", "Compiler feedback", "Compiler", "Base"], [21, 39.3, 35], "Numina ones", "numina_ones.png")