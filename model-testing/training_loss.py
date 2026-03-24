"""
training_loss.py  —  visualize training logs from epoch-*.json files

Usage:
    python model-testing/training_loss.py <folder> [--out output.png]

Produces a figure with:
  1. Top-left:  Pass@1 per epoch (fraction of examples that passed)
  2. Top-right: Mean training loss per epoch
  3. Bottom:    Mean gradient norm per epoch
"""

import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    pass_rate, mean_loss, mean_grad_norm = [], [], []

    for data in epochs:
        examples = data["examples"]
        passed, losses, grad_norms = 0, [], []
        for ex in examples:
            if ex["verification_status"] == "PASS":
                passed += 1
            if ex.get("loss") is not None:
                losses.append(ex["loss"])
            if ex.get("grad_norm") is not None:
                grad_norms.append(ex["grad_norm"])
        pass_rate.append(passed / len(examples) if examples else 0)
        mean_loss.append(np.mean(losses) if losses else float("nan"))
        mean_grad_norm.append(np.mean(grad_norms) if grad_norms else float("nan"))

    return pass_rate, mean_loss, mean_grad_norm


def plot_pass_rate(ax, epoch_nums, pass_rate):
    ax.plot(epoch_nums, [r * 100 for r in pass_rate], marker="o", color="#1a9850")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pass@1 (%)")
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


def plot_grad_norm(ax, epoch_nums, mean_grad_norm):
    ax.plot(epoch_nums, mean_grad_norm, marker="o", color="darkorange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean grad norm")
    ax.set_title("Mean gradient norm per epoch")
    ax.set_xticks(epoch_nums)
    ax.tick_params(axis="x", labelsize=7)


def make_figure(epochs, folder_name):
    epoch_nums = [d["epoch"] for d in epochs]
    pass_rate, mean_loss, mean_grad_norm = compute_stats(epochs)

    fig, axes = plt.subplots(1, 3, figsize=(max(15, len(epochs) * 0.7), 5))
    fig.suptitle(folder_name, fontsize=13, fontweight="bold")
    fig.subplots_adjust(wspace=0.35, top=0.88)

    plot_pass_rate(axes[0], epoch_nums, pass_rate)
    plot_mean_loss(axes[1], epoch_nums, mean_loss)
    plot_grad_norm(axes[2], epoch_nums, mean_grad_norm)

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
