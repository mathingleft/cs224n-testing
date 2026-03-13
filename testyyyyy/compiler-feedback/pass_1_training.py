"""
Graph pass@1 over training epochs from Modal training logs.

Reads cached logs from compiler-feedback/training_logs_cache/.
To refresh cache, run:
  modal volume get my-volume-1 training_logs/Goedel-LM/Goedel-Prover-V2-8B/<run_name>/ compiler-feedback/training_logs_cache/<run_name>/
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE_DIR = "compiler-feedback/training_logs_cache"
OUT_DIR = "compiler-feedback"

RUNS = {
    "Base (no training)": f"{CACHE_DIR}/base_eval",
    "SDFT + Compiler Error": f"{CACHE_DIR}/sdft_comperror_trainlog",
    "SDFT + Gemini Feedback": f"{CACHE_DIR}/sdft_gemini_feed_trainlog",
}

COLORS = {
    "Base (no training)": "gray",
    "SDFT + Compiler Error": "steelblue",
    "SDFT + Gemini Feedback": "darkorange",
}


def load_epochs(run_dir):
    """Load all epoch-*.json files, return sorted list of (epoch_num, data)."""
    epochs = []
    for fname in os.listdir(run_dir):
        if fname.startswith("epoch-") and fname.endswith(".json"):
            epoch_num = int(fname.replace("epoch-", "").replace(".json", ""))
            with open(os.path.join(run_dir, fname)) as f:
                data = json.load(f)
            epochs.append((epoch_num, data))
    epochs.sort(key=lambda x: x[0])
    return epochs


def flatten_samples(epochs):
    """Flatten all per-example results across epochs into a list of (sample_number, passed)."""
    samples = []
    sample_num = 0
    for _, data in epochs:
        for ex in data["examples"]:
            passed = 1 if ex["verification_status"] == "PASS" else 0
            samples.append((sample_num, passed))
            sample_num += 1
    return samples


def rolling_sample_avg(samples, window=20):
    """Rolling average pass rate over individual samples."""
    result = []
    for i, (sample_num, _) in enumerate(samples):
        start = max(0, i - window + 1)
        vals = [samples[j][1] for j in range(start, i + 1)]
        result.append((sample_num, sum(vals) / len(vals)))
    return result


def cumulative_pass_at_1(epochs):
    """Cumulative pass@1: total passes so far / total examples so far."""
    cum_pass, cum_total = 0, 0
    result = []
    for epoch_num, data in epochs:
        cum_pass += data["n_pass"]
        cum_total += data["batch_size"]
        result.append((epoch_num, cum_pass / cum_total))
    return result


def rolling_pass_at_1(epochs, window=3):
    """Rolling average pass@1 over a window of epochs."""
    raw = [(e, d["pass_at_1"]) for e, d in epochs]
    result = []
    for i, (epoch_num, _) in enumerate(raw):
        start = max(0, i - window + 1)
        vals = [raw[j][1] for j in range(start, i + 1)]
        result.append((epoch_num, sum(vals) / len(vals)))
    return result


# ── Load all data ─────────────────────────────────────────────────────────────
all_runs = {}
for label, run_dir in RUNS.items():
    if not os.path.isdir(run_dir):
        print(f"  skip (not found): {run_dir}")
        continue
    epochs = load_epochs(run_dir)
    if not epochs:
        print(f"  skip (no epochs): {run_dir}")
        continue
    all_runs[label] = epochs
    print(f"  {label}: {len(epochs)} epochs, pass@1 range: "
          f"{min(d['pass_at_1'] for _, d in epochs):.2f} - {max(d['pass_at_1'] for _, d in epochs):.2f}")


# ── Plot 1: pass@1 per epoch (3 views) ───────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1a: Raw per-batch pass@1
ax = axes[0]
for label, epochs in all_runs.items():
    xs = [e for e, _ in epochs]
    ys = [d["pass_at_1"] for _, d in epochs]
    ax.plot(xs, ys, marker="o", markersize=4, label=label, color=COLORS[label], alpha=0.7)
ax.set_xlabel("Epoch (batch of 10)")
ax.set_ylabel("pass@1")
ax.set_title("pass@1 per Batch (raw)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 1b: Rolling average pass@1
ax = axes[1]
for label, epochs in all_runs.items():
    rolling = rolling_pass_at_1(epochs, window=3)
    xs = [e for e, _ in rolling]
    ys = [v for _, v in rolling]
    ax.plot(xs, ys, marker="o", markersize=4, label=label, color=COLORS[label])
ax.set_xlabel("Epoch (batch of 10)")
ax.set_ylabel("pass@1")
ax.set_title("pass@1 Rolling Average (window=3 epochs)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 1c: Cumulative pass@1
ax = axes[2]
for label, epochs in all_runs.items():
    cum = cumulative_pass_at_1(epochs)
    xs = [e for e, _ in cum]
    ys = [v for _, v in cum]
    ax.plot(xs, ys, marker="o", markersize=4, label=label, color=COLORS[label])
ax.set_xlabel("Epoch (batch of 10)")
ax.set_ylabel("pass@1")
ax.set_title("pass@1 Cumulative")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle("pass@1 Over Training — Goedel-Prover-V2-8B", fontsize=13, fontweight="bold")
plt.tight_layout()
out1 = os.path.join(OUT_DIR, "pass1_over_training.png")
fig.savefig(out1, dpi=150)
print(f"\nSaved → {out1}")


# ── Plot 2: Sample-level rolling average ──────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(12, 5))
for label, epochs in all_runs.items():
    samples = flatten_samples(epochs)
    rolling = rolling_sample_avg(samples, window=20)
    xs = [s for s, _ in rolling]
    ys = [v for _, v in rolling]
    ax3.plot(xs, ys, label=label, color=COLORS[label], alpha=0.85)
ax3.set_xlabel("Sample number")
ax3.set_ylabel("pass@1 (rolling avg)")
ax3.set_title("pass@1 Rolling Average by Sample (window=20)")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

plt.suptitle("pass@1 Over Training (per sample) — Goedel-Prover-V2-8B", fontsize=13, fontweight="bold")
plt.tight_layout()
out3 = os.path.join(OUT_DIR, "pass1_by_sample.png")
fig3.savefig(out3, dpi=150)
print(f"Saved → {out3}")


# ── Plot 3: Difference from base model ───────────────────────────────────────
base_label = "Base (no training)"
if base_label in all_runs:
    base_epochs = all_runs[base_label]
    base_pass = {e: d["pass_at_1"] for e, d in base_epochs}

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    # 3a: Raw difference
    ax = axes2[0]
    for label, epochs in all_runs.items():
        if label == base_label:
            continue
        xs, ys = [], []
        for e, d in epochs:
            if e in base_pass:
                xs.append(e)
                ys.append(d["pass_at_1"] - base_pass[e])
        ax.plot(xs, ys, marker="o", markersize=4, label=label, color=COLORS[label], alpha=0.7)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch (batch of 10)")
    ax.set_ylabel("pass@1 difference vs base")
    ax.set_title("pass@1 Difference from Base (raw)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3b: Rolling average difference
    ax = axes2[1]
    for label, epochs in all_runs.items():
        if label == base_label:
            continue
        raw_diff = []
        for e, d in epochs:
            if e in base_pass:
                raw_diff.append((e, d["pass_at_1"] - base_pass[e]))
        result = []
        window = 3
        for i, (epoch_num, _) in enumerate(raw_diff):
            start = max(0, i - window + 1)
            vals = [raw_diff[j][1] for j in range(start, i + 1)]
            result.append((epoch_num, sum(vals) / len(vals)))
        xs = [e for e, _ in result]
        ys = [v for _, v in result]
        ax.plot(xs, ys, marker="o", markersize=4, label=label, color=COLORS[label])
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch (batch of 10)")
    ax.set_ylabel("pass@1 difference vs base")
    ax.set_title("pass@1 Difference from Base (rolling avg, window=3)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("pass@1 Improvement Over Base — Goedel-Prover-V2-8B", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, "pass1_diff_from_base.png")
    fig2.savefig(out2, dpi=150)
    print(f"Saved → {out2}")
else:
    print("Base model data not found, skipping difference plot.")
