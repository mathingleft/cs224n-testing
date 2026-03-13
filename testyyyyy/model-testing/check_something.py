import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

filename = "results/compare_all1/compare_results.json"

with open(filename, "r") as f:
    data = json.load(f)

names = {
    "sft": "SFT",
    "base": "Base",
    "good_data_2": "Compiler error",
    "good_data_no_feed": "Vanilla SDFT"
}

# Filter for algebra/numbertheory problems and count pass/total per model
filtered = {}
for model, model_data in data["models"].items():
    results = model_data["results"]
    filtered[model] = {"pass": 0, "total": 0}
    for result in results:
        name = result["statement"]
        if ("imo" in name):
            filtered[model]["total"] += 1
            if result["status"] == "PASS":
                filtered[model]["pass"] += 1

# Short display names
short_names = {}
for full_name in filtered:
    short = full_name.split("/")[-1]
    for seed, label in names.items():
        if seed in full_name:
            short = label
            break
    short_names[full_name] = short

labels = [short_names[m] for m in filtered]
pass_counts = [filtered[m]["pass"] for m in filtered]
totals = [filtered[m]["total"] for m in filtered]
pcts = [100 * p / max(t, 1) for p, t in zip(pass_counts, totals)]

for model in filtered:
    p, t = filtered[model]["pass"], filtered[model]["total"]
    print(f"  {short_names[model]}: {p}/{t} ({100*p/max(t,1):.1f}%)")

# Bar chart
fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2), 6))
bars = ax.bar(range(len(labels)), pcts, color="steelblue", edgecolor="white", linewidth=0.5)

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("pass@1 (%)")
ax.set_title("pass@1 on Algebra/Number Theory problems")
ax.set_ylim(0, max(pcts + [5]) * 1.3 + 3)

for bar, score, p, t in zip(bars, pcts, pass_counts, totals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{score:.1f}%\n({p}/{t})", ha="center", va="bottom", fontsize=8, fontweight="bold")

plt.tight_layout()
out_path = "model-testing/imo.png"
fig.savefig(out_path, dpi=150)
print(f"\nSaved → {out_path}")
