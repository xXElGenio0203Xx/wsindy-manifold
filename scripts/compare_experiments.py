#!/usr/bin/env python3
"""
compare_experiments.py — Cross-experiment suite comparison
==========================================================

Reads summary.json and test_results.csv from each experiment and generates:
  1. r2_heatmap.png           — experiments × methods heatmap
  2. r2_barplot.png           — grouped bar chart
  3. degradation_comparison.png — R²(t) decay overlaid per experiment
  4. wsindy_equation_table.png — discovered PDE terms per experiment
  5. method_ranking.png       — which method wins per regime
  6. suite_summary.csv        — master table
  7. suite_summary.json       — machine-readable summary

Usage:
  python scripts/compare_experiments.py
  python scripts/compare_experiments.py --experiments DYN1_gentle DYN4_blackhole
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OSCAR_DIR = ROOT / "oscar_output"
OUTPUT_DIR = ROOT / "predictions" / "SUITE_COMPARISON"

ALL_EXPERIMENTS = [
    "DYN1_gentle",
    "DYN2_hypervelocity",
    "DYN3_hypernoisy",
    "DYN4_blackhole",
    "DYN5_supernova",
    "DYN6_varspeed",
    "DYN7_pure_vicsek",
]

METHODS = ["MVAR", "LSTM", "WSINDy"]
METHOD_COLORS = {"MVAR": "#2196F3", "LSTM": "#FF9800", "WSINDy": "#4CAF50"}
SHORT_NAMES = {
    "DYN1_gentle": "Gentle",
    "DYN2_hypervelocity": "Hypervel.",
    "DYN3_hypernoisy": "Noisy",
    "DYN4_blackhole": "Blackhole",
    "DYN5_supernova": "Supernova",
    "DYN6_varspeed": "Var-speed",
    "DYN7_pure_vicsek": "Pure Vicsek",
}


def load_experiment(exp_name):
    """Load summary + per-method test results for one experiment."""
    exp_dir = OSCAR_DIR / exp_name
    info = {"name": exp_name, "short": SHORT_NAMES.get(exp_name, exp_name)}

    # Load master summary
    summary_path = exp_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            info["summary"] = json.load(f)
    else:
        # Try predictions dir
        alt = ROOT / "predictions" / exp_name / "pipeline_summary.json"
        if alt.exists():
            with open(alt) as f:
                info["summary"] = json.load(f)
        else:
            info["summary"] = {}

    # Load per-method test results
    for method in METHODS:
        method_dir_name = method if method != "WSINDy" else "WSINDy"
        csv_path = exp_dir / method_dir_name / "test_results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            info[method] = df
        else:
            info[method] = None

    # Try loading time-resolved R² per method
    info["r2_vs_time"] = {}
    test_dirs = sorted(exp_dir.glob("test/test_*"))
    for method in METHODS:
        suffix = method.lower()
        if suffix == "wsindy":
            suffix = "wsindy"  # r2_vs_time_wsindy.csv
        all_curves = []
        for td in test_dirs:
            csv = td / f"r2_vs_time_{suffix}.csv"
            if not csv.exists():
                csv = td / "r2_vs_time.csv"  # MVAR default name
            if csv.exists():
                try:
                    curve = pd.read_csv(csv)
                    all_curves.append(curve)
                except Exception:
                    pass
        if all_curves:
            info["r2_vs_time"][method] = all_curves

    # WSINDy equation info
    mf_path = exp_dir / "WSINDy" / "multifield_model.json"
    if mf_path.exists():
        with open(mf_path) as f:
            info["wsindy_model"] = json.load(f)
    else:
        info["wsindy_model"] = None

    return info


def get_mean_r2(info, method):
    """Extract mean R² for a method from experiment info."""
    # Try from test_results CSV
    df = info.get(method)
    if df is not None:
        for col in ["r2_reconstructed", "r2", "mean_r2"]:
            if col in df.columns:
                val = df[col].mean()
                if not np.isnan(val):
                    return val

    # Try from summary.json
    s = info.get("summary", {})
    method_key = method.lower()
    if method_key in s:
        for key in ["mean_r2_test", "mean_r2"]:
            if key in s[method_key]:
                val = s[method_key][key]
                if val is not None:
                    return val

    # WSINDy nested
    if method == "WSINDy" and "wsindy" in s:
        te = s["wsindy"].get("test_evaluation", {})
        if "mean_r2" in te and te["mean_r2"] is not None:
            return te["mean_r2"]

    return np.nan


def build_results_table(experiments_info):
    """Build a DataFrame: experiments × methods → mean R²."""
    rows = []
    for info in experiments_info:
        row = {"experiment": info["name"], "short_name": info["short"]}
        for method in METHODS:
            row[method] = get_mean_r2(info, method)
        # Winner
        vals = {m: row[m] for m in METHODS if not np.isnan(row[m])}
        row["winner"] = max(vals, key=vals.get) if vals else "N/A"
        rows.append(row)
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════
#  PLOT 1: R² Heatmap
# ════════════════════════════════════════════════════════════════
def plot_r2_heatmap(df, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))

    data = df[METHODS].values.astype(float)
    short_names = df["short_name"].tolist()

    im = ax.imshow(data, cmap="RdYlGn", vmin=-0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(METHODS, fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(short_names)))
    ax.set_yticklabels(short_names, fontsize=11)

    # Annotate cells
    for i in range(len(short_names)):
        for j in range(len(METHODS)):
            val = data[i, j]
            if np.isnan(val):
                text = "N/A"
                color = "gray"
            else:
                text = f"{val:.3f}"
                color = "white" if val < 0.3 else "black"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Mean R²", shrink=0.8)
    ax.set_title("Test R² Across Experiments and Methods",
                 fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(output_dir / "r2_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: r2_heatmap.png")


# ════════════════════════════════════════════════════════════════
#  PLOT 2: Grouped Bar Chart
# ════════════════════════════════════════════════════════════════
def plot_r2_barplot(df, output_dir):
    fig, ax = plt.subplots(figsize=(12, 5))

    n_exp = len(df)
    n_methods = len(METHODS)
    x = np.arange(n_exp)
    width = 0.25

    for i, method in enumerate(METHODS):
        vals = df[method].values.astype(float)
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      label=method, color=METHOD_COLORS[method], alpha=0.85)
        # Add value labels on bars
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(df["short_name"].tolist(), fontsize=10)
    ax.set_ylabel("Mean Test R²", fontsize=12)
    ax.set_title("MVAR vs LSTM vs WSINDy Across Dynamical Regimes",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower left")
    ax.set_ylim([-0.5, 1.1])
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "r2_barplot.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: r2_barplot.png")


# ════════════════════════════════════════════════════════════════
#  PLOT 3: R²(t) Degradation Curves
# ════════════════════════════════════════════════════════════════
def plot_degradation(experiments_info, output_dir):
    n_exp = len(experiments_info)
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharey=True)
    axes = axes.flatten()

    for idx, info in enumerate(experiments_info):
        ax = axes[idx]
        ax.set_title(info["short"], fontsize=11, fontweight="bold")

        for method in METHODS:
            curves = info["r2_vs_time"].get(method, [])
            if not curves:
                continue
            # Average across test runs
            # Find common time column
            try:
                all_r2 = []
                for c in curves:
                    r2_col = [col for col in c.columns if "r2" in col.lower()]
                    if r2_col:
                        all_r2.append(c[r2_col[0]].values)
                if all_r2:
                    min_len = min(len(r) for r in all_r2)
                    stacked = np.array([r[:min_len] for r in all_r2])
                    mean_r2 = np.nanmean(stacked, axis=0)
                    time_frac = np.linspace(0, 1, len(mean_r2))
                    ax.plot(time_frac, mean_r2, label=method,
                            color=METHOD_COLORS[method], linewidth=2)
                    # Shade std
                    if stacked.shape[0] > 1:
                        std_r2 = np.nanstd(stacked, axis=0)
                        ax.fill_between(time_frac, mean_r2 - std_r2,
                                        mean_r2 + std_r2,
                                        color=METHOD_COLORS[method], alpha=0.15)
            except Exception:
                pass

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        ax.set_ylim([-0.5, 1.05])
        if idx >= 4:
            ax.set_xlabel("Forecast fraction", fontsize=10)
        if idx % 4 == 0:
            ax.set_ylabel("R²", fontsize=10)
        ax.grid(alpha=0.3)

    # Hide unused subplot
    if n_exp < len(axes):
        for i in range(n_exp, len(axes)):
            axes[i].axis("off")

    axes[0].legend(fontsize=9)
    fig.suptitle("R² Degradation Over Forecast Horizon",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "degradation_comparison.png",
                dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: degradation_comparison.png")


# ════════════════════════════════════════════════════════════════
#  PLOT 4: WSINDy Equation Table
# ════════════════════════════════════════════════════════════════
def plot_wsindy_equations(experiments_info, output_dir):
    """Show which terms survived in each experiment's discovered PDE."""
    all_terms = set()
    exp_terms = {}

    for info in experiments_info:
        model = info.get("wsindy_model")
        if model is None:
            exp_terms[info["short"]] = {}
            continue

        terms_dict = {}
        # Extract active terms per equation from model JSON
        for eq_name in ["rho", "px", "py"]:
            eq_info = model.get(eq_name, model.get(f"{eq_name}_model", {}))
            if isinstance(eq_info, dict):
                active = eq_info.get("active_terms", [])
                coeffs = eq_info.get("coefficients", [])
                for t, c in zip(active, coeffs):
                    key = f"{eq_name}:{t}"
                    all_terms.add(key)
                    terms_dict[key] = c

        exp_terms[info["short"]] = terms_dict

    if not all_terms:
        print("  No WSINDy models found, skipping equation table")
        return

    all_terms = sorted(all_terms)
    exp_names = [info["short"] for info in experiments_info]

    fig, ax = plt.subplots(figsize=(max(12, len(all_terms) * 0.8),
                                    max(4, len(exp_names) * 0.6)))

    data = np.zeros((len(exp_names), len(all_terms)))
    for i, name in enumerate(exp_names):
        for j, term in enumerate(all_terms):
            c = exp_terms.get(name, {}).get(term, 0)
            data[i, j] = c

    # Binary: active (1) or not (0)
    active_mask = np.abs(data) > 1e-10

    im = ax.imshow(active_mask.astype(float), cmap="Greens",
                   vmin=0, vmax=1, aspect="auto")

    # Annotate with coefficients
    for i in range(len(exp_names)):
        for j in range(len(all_terms)):
            if active_mask[i, j]:
                ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, fontweight="bold")

    ax.set_xticks(range(len(all_terms)))
    ax.set_xticklabels([t.replace("_", " ") for t in all_terms],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(exp_names)))
    ax.set_yticklabels(exp_names, fontsize=10)
    ax.set_title("Discovered PDE Terms Across Experiments",
                 fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(output_dir / "wsindy_equation_table.png",
                dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: wsindy_equation_table.png")


# ════════════════════════════════════════════════════════════════
#  PLOT 5: Method Ranking
# ════════════════════════════════════════════════════════════════
def plot_method_ranking(df, output_dir):
    fig, ax = plt.subplots(figsize=(8, 4))

    winners = df["winner"].value_counts()
    colors = [METHOD_COLORS.get(w, "#999") for w in winners.index]

    ax.bar(winners.index, winners.values, color=colors, alpha=0.85)
    ax.set_ylabel("Number of experiments won", fontsize=12)
    ax.set_title("Method Wins Across Dynamical Regimes",
                 fontsize=14, fontweight="bold")

    # Annotate which experiments each method won
    for method in winners.index:
        exps = df[df["winner"] == method]["short_name"].tolist()
        idx = list(winners.index).index(method)
        ax.text(idx, winners[method] + 0.1,
                "\n".join(exps), ha="center", va="bottom", fontsize=8)

    ax.set_ylim([0, max(winners.values) + 2])
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "method_ranking.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: method_ranking.png")


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Cross-experiment suite comparison")
    parser.add_argument("--experiments", nargs="+", default=ALL_EXPERIMENTS,
                        help="Experiment names to compare")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  SUITE COMPARISON: {len(args.experiments)} experiments")
    print(f"{'='*60}\n")

    # Load all experiments
    experiments_info = []
    for exp_name in args.experiments:
        print(f"  Loading {exp_name}...")
        info = load_experiment(exp_name)
        experiments_info.append(info)

    # Build master table
    df = build_results_table(experiments_info)
    print(f"\n{df.to_string(index=False)}\n")

    # Save CSV + JSON
    df.to_csv(OUTPUT_DIR / "suite_summary.csv", index=False)
    print(f"  Saved: suite_summary.csv")

    summary = {
        "experiments": args.experiments,
        "results": df.to_dict(orient="records"),
        "methods": METHODS,
    }
    with open(OUTPUT_DIR / "suite_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: suite_summary.json")

    # Generate plots
    print(f"\n  Generating plots...")
    plot_r2_heatmap(df, OUTPUT_DIR)
    plot_r2_barplot(df, OUTPUT_DIR)
    plot_degradation(experiments_info, OUTPUT_DIR)
    plot_wsindy_equations(experiments_info, OUTPUT_DIR)
    plot_method_ranking(df, OUTPUT_DIR)

    print(f"\n{'='*60}")
    print(f"  All outputs in: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
