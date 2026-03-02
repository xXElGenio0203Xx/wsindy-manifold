#!/usr/bin/env python3
"""
Analyse VDYN (variable-speed) suite results — MVAR vs LSTM.
Produces:
  1. Console summary table (per-experiment mean ± std)
  2. Bar chart: MVAR vs LSTM R² (reconstructed) across regimes
  3. Box plot: per-test-sim R² distributions
  4. Negativity fraction comparison
"""
import os, json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "oscar_output")
FIG_DIR = os.path.join(ROOT, "artifacts", "VDYN_analysis")
os.makedirs(FIG_DIR, exist_ok=True)

EXPERIMENTS = [
    "VDYN1_gentle_varspeed",
    "VDYN2_hypervelocity_varspeed",
    "VDYN3_hypernoisy_varspeed",
    "VDYN4_blackhole_varspeed",
    "VDYN5_supernova_varspeed",
    "VDYN6_baseline_varspeed",
    "VDYN7_pure_vicsek_varspeed",
]

SHORT_NAMES = [
    "VDYN1\nGentle",
    "VDYN2\nHyperVel",
    "VDYN3\nNoisy",
    "VDYN4\nBlackhole",
    "VDYN5\nSupernova",
    "VDYN6\nBaseline",
    "VDYN7\nPureVicsek",
]

def load_results():
    """Load test_results.csv for MVAR and LSTM across all VDYN experiments."""
    rows = []
    for exp in EXPERIMENTS:
        for method in ["MVAR", "LSTM"]:
            csv_path = os.path.join(OUT_DIR, exp, method, "test_results.csv")
            if not os.path.isfile(csv_path):
                print(f"  [SKIP] {csv_path}")
                continue
            df = pd.read_csv(csv_path)
            df["experiment"] = exp
            df["method"] = method
            rows.append(df)
    if not rows:
        print("No results found!")
        sys.exit(1)
    return pd.concat(rows, ignore_index=True)


def print_summary(df):
    """Console table: mean ± std of key metrics per experiment × method."""
    print("\n" + "=" * 100)
    print(f"{'Experiment':<30} {'Method':<6} {'R²_recon':>12} {'R²_latent':>12} "
          f"{'R²_1step':>10} {'Neg%':>8} {'RMSE_recon':>12}")
    print("-" * 100)
    for exp in EXPERIMENTS:
        for method in ["MVAR", "LSTM"]:
            sub = df[(df["experiment"] == exp) & (df["method"] == method)]
            if sub.empty:
                continue
            r2r = sub["r2_reconstructed"]
            r2l = sub["r2_latent"]
            r21 = sub["r2_1step"]
            neg = sub["negativity_frac"]
            rmse = sub["rmse_recon"]
            tag = exp.replace("_varspeed", "")
            print(f"  {tag:<28} {method:<6} "
                  f"{r2r.mean():>6.3f}±{r2r.std():>5.3f} "
                  f"{r2l.mean():>6.3f}±{r2l.std():>5.3f} "
                  f"{r21.mean():>6.3f}±{r21.std():.3f} "
                  f"{neg.mean():>6.1f}% "
                  f"{rmse.mean():>8.4f}")
        print()
    print("=" * 100)

    # Also compute overall winner count
    mvar_wins = 0
    lstm_wins = 0
    for exp in EXPERIMENTS:
        mvar_mean = df[(df["experiment"] == exp) & (df["method"] == "MVAR")]["r2_reconstructed"].mean()
        lstm_mean = df[(df["experiment"] == exp) & (df["method"] == "LSTM")]["r2_reconstructed"].mean()
        if mvar_mean > lstm_mean:
            mvar_wins += 1
        else:
            lstm_wins += 1
    print(f"\n  MVAR wins: {mvar_wins}/7,  LSTM wins: {lstm_wins}/7  (by mean R² reconstructed)")


def plot_r2_bars(df):
    """Grouped bar chart: MVAR vs LSTM mean R² (reconstructed) ± std."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(EXPERIMENTS))
    width = 0.35

    mvar_means, mvar_stds = [], []
    lstm_means, lstm_stds = [], []
    for exp in EXPERIMENTS:
        for method, means, stds in [("MVAR", mvar_means, mvar_stds),
                                     ("LSTM", lstm_means, lstm_stds)]:
            sub = df[(df["experiment"] == exp) & (df["method"] == method)]["r2_reconstructed"]
            means.append(sub.mean() if len(sub) else 0)
            stds.append(sub.std() if len(sub) else 0)

    bars_mvar = ax.bar(x - width/2, mvar_means, width, yerr=mvar_stds,
                       label="MVAR", color="#2196F3", alpha=0.85, capsize=3)
    bars_lstm = ax.bar(x + width/2, lstm_means, width, yerr=lstm_stds,
                       label="LSTM", color="#FF5722", alpha=0.85, capsize=3)

    ax.set_ylabel("R² (reconstructed density)", fontsize=12)
    ax.set_title("VDYN Suite: MVAR vs LSTM — Variable Speed Experiments", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_NAMES, fontsize=9)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=11)
    ax.set_ylim(min(min(lstm_means) - 1, -2), max(max(mvar_means) + 0.3, 1.1))
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "vdyn_r2_recon_bars.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)


def plot_r2_boxplots(df):
    """Box plot: per-test-sim R² distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, metric, title in [
        (axes[0], "r2_reconstructed", "R² Reconstructed Density"),
        (axes[1], "r2_latent", "R² Latent Space"),
    ]:
        data_mvar, data_lstm = [], []
        for exp in EXPERIMENTS:
            data_mvar.append(
                df[(df["experiment"] == exp) & (df["method"] == "MVAR")][metric].values
            )
            data_lstm.append(
                df[(df["experiment"] == exp) & (df["method"] == "LSTM")][metric].values
            )

        positions = np.arange(len(EXPERIMENTS))
        bp1 = ax.boxplot(data_mvar, positions=positions - 0.18, widths=0.3,
                         patch_artist=True, showfliers=False,
                         boxprops=dict(facecolor="#2196F3", alpha=0.6),
                         medianprops=dict(color="navy"))
        bp2 = ax.boxplot(data_lstm, positions=positions + 0.18, widths=0.3,
                         patch_artist=True, showfliers=False,
                         boxprops=dict(facecolor="#FF5722", alpha=0.6),
                         medianprops=dict(color="darkred"))

        ax.set_xticks(positions)
        ax.set_xticklabels(SHORT_NAMES, fontsize=8)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=10)
        ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["MVAR", "LSTM"], fontsize=10)

    fig.suptitle("VDYN Suite: Per-Test-Sim R² Distributions", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "vdyn_r2_boxplots.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


def plot_negativity(df):
    """Bar chart: negativity fraction comparison."""
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(EXPERIMENTS))
    width = 0.35

    mvar_neg = [df[(df["experiment"] == exp) & (df["method"] == "MVAR")]["negativity_frac"].mean()
                for exp in EXPERIMENTS]
    lstm_neg = [df[(df["experiment"] == exp) & (df["method"] == "LSTM")]["negativity_frac"].mean()
                for exp in EXPERIMENTS]

    ax.bar(x - width/2, mvar_neg, width, label="MVAR", color="#2196F3", alpha=0.85)
    ax.bar(x + width/2, lstm_neg, width, label="LSTM", color="#FF5722", alpha=0.85)

    ax.set_ylabel("Negativity Fraction (%)", fontsize=12)
    ax.set_title("VDYN Suite: Negative Density Production (MVAR vs LSTM)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_NAMES, fontsize=9)
    ax.legend(fontsize=11)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "vdyn_negativity.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)


def plot_runtime(df):
    """Runtime comparison from runtime_comparison.json files."""
    runtimes = {}
    for exp in EXPERIMENTS:
        rt_path = os.path.join(OUT_DIR, exp, "runtime_comparison.json")
        if os.path.isfile(rt_path):
            with open(rt_path) as f:
                runtimes[exp] = json.load(f)

    if not runtimes:
        print("  [SKIP] No runtime_comparison.json files found")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(EXPERIMENTS))
    width = 0.35

    mvar_rt, lstm_rt = [], []
    for exp in EXPERIMENTS:
        rt = runtimes.get(exp, {})
        mvar_rt.append(rt.get("MVAR", {}).get("total_s", 0) / 60)  # to minutes
        lstm_rt.append(rt.get("LSTM", {}).get("total_s", 0) / 60)

    ax.bar(x - width/2, mvar_rt, width, label="MVAR", color="#2196F3", alpha=0.85)
    ax.bar(x + width/2, lstm_rt, width, label="LSTM", color="#FF5722", alpha=0.85)

    ax.set_ylabel("Runtime (minutes)", fontsize=12)
    ax.set_title("VDYN Suite: Training + Evaluation Runtime", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_NAMES, fontsize=9)
    ax.legend(fontsize=11)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "vdyn_runtime.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)


def plot_latent_vs_recon(df):
    """Scatter: R² latent vs R² reconstructed, colored by method."""
    fig, ax = plt.subplots(figsize=(8, 7))
    for method, color, marker in [("MVAR", "#2196F3", "o"), ("LSTM", "#FF5722", "^")]:
        for i, exp in enumerate(EXPERIMENTS):
            sub = df[(df["experiment"] == exp) & (df["method"] == method)]
            if sub.empty:
                continue
            ax.scatter(sub["r2_latent"].mean(), sub["r2_reconstructed"].mean(),
                       c=color, marker=marker, s=120, edgecolors="k", linewidths=0.5,
                       zorder=5, label=f"{method}" if i == 0 else "")
            ax.annotate(exp.replace("_varspeed", "").replace("VDYN", "V"),
                        (sub["r2_latent"].mean(), sub["r2_reconstructed"].mean()),
                        textcoords="offset points", xytext=(6, 6), fontsize=7)

    ax.axhline(0, color="gray", ls="--", lw=0.7)
    ax.axvline(0, color="gray", ls="--", lw=0.7)
    ax.plot([-10, 1], [-10, 1], "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("R² Latent", fontsize=12)
    ax.set_ylabel("R² Reconstructed", fontsize=12)
    ax.set_title("Latent vs Reconstructed R² — VDYN Suite", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim(-9, 1)
    ax.set_ylim(-9, 1)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "vdyn_latent_vs_recon.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    print("Loading VDYN varspeed results...")
    df = load_results()
    print(f"  Loaded {len(df)} rows ({df['experiment'].nunique()} experiments, "
          f"{df['method'].nunique()} methods)")

    print_summary(df)

    print("\nGenerating plots...")
    plot_r2_bars(df)
    plot_r2_boxplots(df)
    plot_negativity(df)
    plot_runtime(df)
    plot_latent_vs_recon(df)
    print(f"\nAll figures saved to: {FIG_DIR}")
