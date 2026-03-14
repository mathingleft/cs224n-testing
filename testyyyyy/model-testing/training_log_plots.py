"""
Graph pass@1 over training epochs from local training logs.

Reads logs from local-volume/Qwen3.5-4B/<run_name>/
To download a run from Modal:
  modal volume get my-volume-2 training_logs/Qwen/Qwen3.5-4B/<run_name> local-volume/Qwen3.5-4B/
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGS_DIR = "local-volume/Qwen3.5-4B"
OUT_DIR  = "local-volume/Qwen3.5-4B"
ALPHA     = 0.09

RUNS = {
    "SDFT + Compiler Error":  f"{LOGS_DIR}/qwen3.5-4b_qwen3.5-4b_compiler-error",
    "SDFT + Gemini Feedback": f"{LOGS_DIR}/qwen3.5-4b_qwen3.5-4b_compiler-error_gemini-feedback",
    "SDFT (Vanilla)":         f"{LOGS_DIR}/qwen3.5-4b_qwen3.5-4b_no-compiler-context_no-gemini",
}

COLORS = {
    "SDFT + Compiler Error":  "steelblue",
    "SDFT + Gemini Feedback": "darkorange",
    "SDFT (Vanilla)":         "pink",
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
        cum_pass  += data["n_pass"]
        cum_total += data["batch_size"]
        result.append((epoch_num, cum_pass / cum_total))
    return result


def ema_pass_at_1(epochs, alpha=0.09):
    """Exponential moving average pass@1. Higher alpha = more weight on recent epochs."""
    result = []
    ema = None
    for epoch_num, data in epochs:
        val = data["pass_at_1"]
        ema = val if ema is None else alpha * val + (1 - alpha) * ema
        result.append((epoch_num, ema))
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


def wandb_ema(values, smoothing_param=0.8):
    """W&B-style EMA with de-bias correction.

    smoothing_param: higher = smoother (more history retained).
    Equivalent alpha (new-point weight) ≈ 1 - sqrt(smoothing_param).
    De-bias divides by accumulated weight so early estimates aren't
    pulled toward the initial value.
    """
    import math
    sw = min(math.sqrt(smoothing_param), 0.999)
    last_y, debias = 0.0, 0.0
    result = []
    for x, y in values:
        last_y = last_y * sw + y
        debias = debias * sw + 1
        result.append((x, last_y / debias))
    return result


def mean_grad_norm(epochs):
    """Mean gradient norm per epoch (skips None values)."""
    result = []
    for epoch_num, data in epochs:
        norms = [ex["grad_norm"] for ex in data["examples"] if ex.get("grad_norm") is not None]
        if norms:
            result.append((epoch_num, sum(norms) / len(norms)))
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


# ── Plot 1: pass@1 per epoch (4 views) ───────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(24, 5))

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

# 1d: Simple EMA vs W&B EMA pass@1
ax = axes[3]
for label, epochs in all_runs.items():
    raw = [(e, d["pass_at_1"]) for e, d in epochs]
    ema = ema_pass_at_1(epochs, alpha=ALPHA)
    wb  = wandb_ema(raw, smoothing_param=0.8)
    xs_e = [e for e, _ in ema];  ys_e = [v for _, v in ema]
    xs_w = [e for e, _ in wb];   ys_w = [v for _, v in wb]
    ax.plot(xs_e, ys_e, marker="o", markersize=3, linestyle="--",
            label=f"{label} (EMA α={ALPHA})", color=COLORS[label], alpha=0.6)
    ax.plot(xs_w, ys_w, marker="o", markersize=3,
            label=f"{label} (W&B EMA)", color=COLORS[label])
ax.set_xlabel("Epoch (batch of 10)")
ax.set_ylabel("pass@1")
ax.set_title("Simple EMA (dashed) vs W&B EMA (solid)")
ax.legend(fontsize=6)
ax.grid(True, alpha=0.3)

plt.suptitle("pass@1 Over Training — Qwen3.5-4B", fontsize=13, fontweight="bold")
plt.tight_layout()
out1 = os.path.join(OUT_DIR, "pass1_over_training.png")
fig.savefig(out1, dpi=150)
print(f"\nSaved → {out1}")


# ── Plot 2: Sample-level rolling avg vs W&B EMA ───────────────────────────────
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(18, 5))

for label, epochs in all_runs.items():
    samples = flatten_samples(epochs)
    rolling = rolling_sample_avg(samples, window=20)
    xs = [s for s, _ in rolling]
    ys = [v for _, v in rolling]
    ax2a.plot(xs, ys, label=label, color=COLORS[label], alpha=0.85)

ax2a.set_xlabel("Sample number")
ax2a.set_ylabel("pass@1")
ax2a.set_title("Rolling Average by Sample (window=20)")
ax2a.legend(fontsize=9)
ax2a.grid(True, alpha=0.3)

for label, epochs in all_runs.items():
    samples = flatten_samples(epochs)
    smoothed = wandb_ema(samples, smoothing_param=0.8)
    xs = [s for s, _ in smoothed]
    ys = [v for _, v in smoothed]
    ax2b.plot(xs, ys, label=label, color=COLORS[label], alpha=0.85)

ax2b.set_xlabel("Sample number")
ax2b.set_ylabel("pass@1")
ax2b.set_title("W&B EMA by Sample (smoothing=0.8)")
ax2b.legend(fontsize=9)
ax2b.grid(True, alpha=0.3)

plt.suptitle("pass@1 Over Training (per sample) — Qwen3.5-4B", fontsize=13, fontweight="bold")
plt.tight_layout()
out2 = os.path.join(OUT_DIR, "pass1_by_sample.png")
fig2.savefig(out2, dpi=150)
print(f"Saved → {out2}")


# ── Plot 3: Gradient norms ────────────────────────────────────────────────────
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 5))

for label, epochs in all_runs.items():
    raw = mean_grad_norm(epochs)
    xs = [e for e, _ in raw]
    ys = [v for _, v in raw]
    ax3a.plot(xs, ys, marker="o", markersize=4, label=label, color=COLORS[label], alpha=0.7)

ax3a.set_xlabel("Epoch (batch of 10)")
ax3a.set_ylabel("Mean grad norm")
ax3a.set_title("Mean Gradient Norm per Epoch (raw)")
ax3a.legend(fontsize=8)
ax3a.grid(True, alpha=0.3)

for label, epochs in all_runs.items():
    raw = mean_grad_norm(epochs)
    smoothed = wandb_ema(raw, smoothing_param=0.8)
    xs = [e for e, _ in smoothed]
    ys = [v for _, v in smoothed]
    ax3b.plot(xs, ys, marker="o", markersize=4, label=label, color=COLORS[label])

ax3b.set_xlabel("Epoch (batch of 10)")
ax3b.set_ylabel("Mean grad norm")
ax3b.set_title("Mean Gradient Norm per Epoch (W&B EMA)")
ax3b.legend(fontsize=8)
ax3b.grid(True, alpha=0.3)

plt.suptitle("Gradient Norms — Qwen3.5-4B", fontsize=13, fontweight="bold")
plt.tight_layout()
out3 = os.path.join(OUT_DIR, "grad_norms.png")
fig3.savefig(out3, dpi=150)
print(f"Saved → {out3}")
