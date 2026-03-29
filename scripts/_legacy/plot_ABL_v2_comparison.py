#!/usr/bin/env python3
"""
ABL Ablation Comparison Plots — corrected data (v2 simplex runs)
================================================================
Produces 4 publication-quality figures:

1. Grouped bar chart: R² by configuration (8 bars)
2. Factor effect plot: marginal effect of each factor
3. Interaction heatmaps: 2D slices of the 2×2×2 factorial
4. Head-to-head simplex delta plot

Uses v2 data for simplex experiments (config bug fixed).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Data loading ──────────────────────────────────────────────────────────────
# Use v1 for "none" experiments, v2 for "simplex" experiments
EXPERIMENTS = {
    # (transform, simplex, align) → path
    ("raw",  "none",    "noAlign"): "oscar_output/ABL1_N200_raw_none_noAlign_H300/test/test_results.csv",
    ("raw",  "none",    "align"):   "oscar_output/ABL2_N200_raw_none_align_H300/test/test_results.csv",
    ("raw",  "simplex", "noAlign"): "oscar_output/ABL3_N200_raw_simplex_noAlign_H300_v2/test/test_results.csv",
    ("raw",  "simplex", "align"):   "oscar_output/ABL4_N200_raw_simplex_align_H300_v2/test/test_results.csv",
    ("sqrt", "none",    "noAlign"): "oscar_output/ABL5_N200_sqrt_none_noAlign_H300/test/test_results.csv",
    ("sqrt", "none",    "align"):   "oscar_output/ABL6_N200_sqrt_none_align_H300/test/test_results.csv",
    ("sqrt", "simplex", "noAlign"): "oscar_output/ABL7_N200_sqrt_simplex_noAlign_H300_v2/test/test_results.csv",
    ("sqrt", "simplex", "align"):   "oscar_output/ABL8_N200_sqrt_simplex_align_H300_v2/test/test_results.csv",
}

rows = []
for (xform, mass_pp, align), path in EXPERIMENTS.items():
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        rows.append({
            "xform": xform,
            "mass_pp": mass_pp,
            "align": align,
            "r2": r["r2_reconstructed"],
            "neg_pct": r["negativity_frac"],
            "rmse": r["rmse_recon"],
        })

data = pd.DataFrame(rows)

# Summary table
summary = data.groupby(["xform", "mass_pp", "align"]).agg(
    r2_mean=("r2", "mean"),
    r2_std=("r2", "std"),
    neg_mean=("neg_pct", "mean"),
    rmse_mean=("rmse", "mean"),
    n=("r2", "count"),
).reset_index()

print("\n" + "=" * 85)
print("CORRECTED ABL ABLATION RESULTS (v2 simplex data)")
print("=" * 85)
print(f"{'transform':<10} {'mass_pp':<10} {'align':<10} {'R²':>9} {'±std':>8} {'neg%':>7} {'RMSE':>9}")
print("-" * 85)
for _, s in summary.iterrows():
    print(f"{s['xform']:<10} {s['mass_pp']:<10} {s['align']:<10} "
          f"{s['r2_mean']:>+9.4f} {s['r2_std']:>8.4f} {s['neg_mean']:>6.1f}% {s['rmse_mean']:>9.4f}")

OUT = Path("oscar_output/ABL_comparison_v2")
OUT.mkdir(parents=True, exist_ok=True)

# ── Color palette ─────────────────────────────────────────────────────────────
C_RAW = "#2196F3"       # blue
C_SQRT = "#FF9800"      # orange
C_ALIGN = "#4CAF50"     # green
C_NOALIGN = "#F44336"   # red
C_NONE = "#90A4AE"      # grey
C_SIMPLEX = "#9C27B0"   # purple

# ── Figure 1: Grouped bar chart ──────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(12, 5.5))

labels = []
means = []
stds = []
colors = []
hatches = []

for _, s in summary.sort_values(["xform", "mass_pp", "align"]).iterrows():
    label = f"{s['xform']}\n{s['mass_pp']}\n{s['align']}"
    labels.append(label)
    means.append(s["r2_mean"])
    stds.append(s["r2_std"])
    colors.append(C_RAW if s["xform"] == "raw" else C_SQRT)
    hatches.append("///" if s["align"] == "align" else "")

x = np.arange(len(labels))
bars = ax1.bar(x, means, yerr=stds, width=0.65, color=colors, edgecolor="black",
               linewidth=0.8, capsize=4, error_kw={"linewidth": 1.2})
for bar, h in zip(bars, hatches):
    bar.set_hatch(h)

ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=8)
ax1.set_ylabel("R² (reconstructed density)", fontsize=12)
ax1.set_title("ABL Ablation: R² by Pipeline Configuration", fontsize=14, fontweight="bold")
ax1.axhline(0, color="black", linewidth=0.5, linestyle="-")
ax1.set_ylim(-0.15, 1.1)

# Legend
legend_elements = [
    mpatches.Patch(facecolor=C_RAW, edgecolor="black", label="raw transform"),
    mpatches.Patch(facecolor=C_SQRT, edgecolor="black", label="√ρ transform"),
    mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="shift-aligned"),
    mpatches.Patch(facecolor="white", edgecolor="black", label="no alignment"),
]
ax1.legend(handles=legend_elements, loc="upper left", fontsize=9)

# Annotate R² values on bars
for i, (m, s) in enumerate(zip(means, stds)):
    y = max(m, 0) + s + 0.02
    ax1.text(i, y, f"{m:+.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

fig1.tight_layout()
fig1.savefig(OUT / "1_bar_chart.png", dpi=200)
print(f"\n✓ Saved {OUT / '1_bar_chart.png'}")

# ── Figure 2: Factor effects (main effects plot) ─────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)

factors = [
    ("Shift Alignment", "align", ["noAlign", "align"], [C_NOALIGN, C_ALIGN]),
    ("Simplex Projection", "mass_pp", ["none", "simplex"], [C_NONE, C_SIMPLEX]),
    ("Density Transform", "xform", ["raw", "sqrt"], [C_RAW, C_SQRT]),
]

for ax, (title, col, levels, cols) in zip(axes2, factors):
    level_means = []
    level_stds = []
    for lev in levels:
        vals = data[data[col] == lev]["r2"]
        level_means.append(vals.mean())
        level_stds.append(vals.std() / np.sqrt(len(vals)))  # SEM

    x = [0, 1]
    bars = ax.bar(x, level_means, yerr=level_stds, width=0.5, color=cols,
                  edgecolor="black", linewidth=0.8, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Effect size annotation
    effect = level_means[1] - level_means[0]
    ax.annotate(f"Δ = {effect:+.4f}", xy=(0.5, max(level_means) + 0.06),
                ha="center", fontsize=11, fontweight="bold",
                color="darkgreen" if effect > 0.01 else ("darkred" if effect < -0.01 else "gray"))

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylim(-0.1, 1.15)

axes2[0].set_ylabel("Mean R²", fontsize=12)
fig2.suptitle("Factor Main Effects on R²", fontsize=14, fontweight="bold", y=1.02)
fig2.tight_layout()
fig2.savefig(OUT / "2_factor_effects.png", dpi=200, bbox_inches="tight")
print(f"✓ Saved {OUT / '2_factor_effects.png'}")

# ── Figure 3: Interaction heatmaps (3 panels: each pair of factors) ──────────
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4.5))

pairs = [
    ("align", "mass_pp",   "Alignment × Simplex"),
    ("align", "xform", "Alignment × Transform"),
    ("mass_pp", "xform", "Simplex × Transform"),
]

for ax, (f1, f2, title) in zip(axes3, pairs):
    pivot = data.groupby([f1, f2])["r2"].mean().unstack()
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-0.1, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_xlabel(f2, fontsize=11)
    ax.set_ylabel(f1, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "white" if val < 0.3 else "black"
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

fig3.colorbar(im, ax=axes3, label="R²", shrink=0.8)
fig3.suptitle("Two-Way Interaction Effects", fontsize=14, fontweight="bold", y=1.02)
fig3.tight_layout()
fig3.savefig(OUT / "3_interaction_heatmaps.png", dpi=200, bbox_inches="tight")
print(f"✓ Saved {OUT / '3_interaction_heatmaps.png'}")

# ── Figure 4: Head-to-head simplex delta ─────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(9, 5))

# Compute simplex delta for each (transform, align) combo
conditions = [
    ("raw / noAlign", "raw", "noAlign"),
    ("raw / align",   "raw", "align"),
    ("√ρ / noAlign",  "sqrt", "noAlign"),
    ("√ρ / align",    "sqrt", "align"),
]

deltas = []
delta_labels = []
for label, t, a in conditions:
    r2_none = data[(data["xform"] == t) & (data["mass_pp"] == "none") & (data["align"] == a)]["r2"].mean()
    r2_simp = data[(data["xform"] == t) & (data["mass_pp"] == "simplex") & (data["align"] == a)]["r2"].mean()
    delta = r2_simp - r2_none
    deltas.append(delta)
    delta_labels.append(label)

colors4 = [C_SIMPLEX if d > 0.001 else (C_NONE if d < -0.001 else "gray") for d in deltas]
bars4 = ax4.barh(range(len(deltas)), deltas, color=colors4, edgecolor="black", height=0.5)

ax4.set_yticks(range(len(delta_labels)))
ax4.set_yticklabels(delta_labels, fontsize=12)
ax4.set_xlabel("ΔR² (simplex − none)", fontsize=12)
ax4.set_title("Effect of Simplex Projection on R²\n(positive = simplex helps)", fontsize=13, fontweight="bold")
ax4.axvline(0, color="black", linewidth=1)

for i, d in enumerate(deltas):
    offset = 0.002 if d >= 0 else -0.002
    ha = "left" if d >= 0 else "right"
    ax4.text(d + offset, i, f"{d:+.4f}", ha=ha, va="center", fontsize=11, fontweight="bold")

darr = np.array(deltas, dtype=float)
darr = darr[np.isfinite(darr)]
if len(darr) > 0:
    margin = max(0.005, (darr.max() - darr.min()) * 0.15)
    ax4.set_xlim(darr.min() - margin, darr.max() + margin)
fig4.tight_layout()
fig4.savefig(OUT / "4_simplex_delta.png", dpi=200)
print(f"✓ Saved {OUT / '4_simplex_delta.png'}")

# ── Summary ───────────────────────────────────────────────────────────────────
# Compute for summary
align_effect = data[data["align"] == "align"]["r2"].mean() - data[data["align"] == "noAlign"]["r2"].mean()
sqrt_effect = data[data["xform"] == "sqrt"]["r2"].mean() - data[data["xform"] == "raw"]["r2"].mean()

print(f"\n{'=' * 60}")
print("KEY TAKEAWAYS:")
print(f"{'=' * 60}")
print(f"  1. Shift alignment is THE factor: Δ = +{align_effect:.3f} R²")
print(f"  2. Simplex helps √ρ/noAlign by +{deltas[2]:.4f}")
print(f"     but is negligible for raw (+{deltas[0]:.4f}) or when aligned (+{deltas[1]:.4f}, +{deltas[3]:.4f})")
print(f"  3. Transform choice (raw vs √ρ) barely matters: Δ = {sqrt_effect:+.4f}")
print(f"  4. Best config: raw + none + align (simplest!)")
print(f"\nAll plots saved to {OUT}/")
