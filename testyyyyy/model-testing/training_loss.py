"""
training_loss.py  —  visualize training logs from epoch-*.json files

Usage:
    python model-testing/training_loss.py <folder> [--out output.png]

Produces a figure with:
  1. Top: heatmap — PASS (green) / not-PASS (red) per (epoch, example)
  2. Bottom-left: pass rate per epoch (fraction of examples that passed)
  3. Bottom-right: mean training loss per epoch
"""

import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_epochs(folder):
    files = []
    for fname in os.listdir(folder):
        m = re.fullmatch(r"epoch-(\d+)\.json", fname)
        if m:
            files.append((int(m.group(1)), os.path.join(folder, fname)))
    files.sort()
    return [json.load(open(path)) for _, path in files]


def compute_stats(epochs):
    n_epochs = len(epochs)
    n_examples = max(len(d["examples"]) for d in epochs)
    pass_grid = np.zeros((n_examples, n_epochs), dtype=float)
    pass_rate, mean_loss = [], []

    for col, data in enumerate(epochs):
        examples = data["examples"]
        passed, losses = 0, []
        for row, ex in enumerate(examples):
            is_pass = ex["verification_status"] == "PASS"
            pass_grid[row, col] = 1.0 if is_pass else 0.0
            if is_pass:
                passed += 1
            if ex["loss"] is not None:
                losses.append(ex["loss"])
        pass_rate.append(passed / len(examples) if examples else 0)
        mean_loss.append(np.mean(losses) if losses else float("nan"))

    return pass_grid, pass_rate, mean_loss


def plot_heatmap(ax, pass_grid, epoch_nums):
    n_examples = pass_grid.shape[0]
    cmap = matplotlib.colors.ListedColormap(["#d73027", "#1a9850"])
    ax.imshow(pass_grid, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(epoch_nums)))
    ax.set_xticklabels(epoch_nums, fontsize=8)
    ax.set_yticks(range(n_examples))
    ax.set_yticklabels([f"ex {i}" for i in range(n_examples)], fontsize=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Example")
    ax.set_title("PASS / not-PASS per epoch per example")
    ax.legend(handles=[
        mpatches.Patch(color="#1a9850", label="PASS"),
        mpatches.Patch(color="#d73027", label="not PASS"),
    ], loc="upper right", fontsize=8)


def plot_pass_rate(ax, epoch_nums, pass_rate):
    ax.plot(epoch_nums, [r * 100 for r in pass_rate], marker="o", color="#1a9850")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pass rate (%)")
    ax.set_title("Pass@1 per epoch")
    ax.set_ylim(0, 105)
    ax.set_xticks(epoch_nums)
    ax.tick_params(axis="x", labelsize=7)


def plot_mean_loss(ax, epoch_nums, mean_loss):
    ax.plot(epoch_nums, mean_loss, marker="o", color="steelblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean loss")
    ax.set_title("Mean training loss per epoch")
    ax.set_xticks(epoch_nums)
    ax.tick_params(axis="x", labelsize=7)


def make_figure(epochs, folder_name):
    epoch_nums = [d["epoch"] for d in epochs]
    pass_grid, pass_rate, mean_loss = compute_stats(epochs)

    fig = plt.figure(figsize=(max(12, len(epochs) * 0.6), 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1], hspace=0.45, wspace=0.35)
    fig.suptitle(folder_name, fontsize=13, fontweight="bold")

    plot_heatmap(fig.add_subplot(gs[0, :]), pass_grid, epoch_nums)
    plot_pass_rate(fig.add_subplot(gs[1, 0]), epoch_nums, pass_rate)
    plot_mean_loss(fig.add_subplot(gs[1, 1]), epoch_nums, mean_loss)

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder containing epoch-*.json files")
    parser.add_argument("--out", default="training_plot.png", help="Output PNG path")
    args = parser.parse_args()

    epochs = load_epochs(args.folder)
    if not epochs:
        print(f"No epoch-*.json files found in {args.folder}")
        return

    folder_name = os.path.basename(os.path.abspath(args.folder))
    fig = make_figure(epochs, folder_name)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
