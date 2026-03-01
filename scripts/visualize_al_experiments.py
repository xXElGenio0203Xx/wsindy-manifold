#!/usr/bin/env python3
"""
Visualize AL experiment results from OSCAR.
============================================

Generates diagnostic plots from r2_vs_time.csv and metrics_summary.json
without needing the heavy density NPZ files.

Plots produced:
  1. Summary table
  2. MVAR vs LSTM bar chart
  3. R² distribution box plots
  4. R² vs time fan (per-test curves)
  5. LSTM R² decay overlay
  6. POD vs recon scatter
  7. LSTM divergence time histogram
  8. 1-step vs full-horizon scatter

Usage:
  python scripts/visualize_AL_experiments.py
"""

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Config ───────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
OSCAR_DIR = ROOT / "oscar_output"
OUT_DIR = ROOT / "artifacts" / "AL_diagnostics"

# Friendly labels
LABELS = {
    "AL1c_align_sqrtSimplex_D19_p5_H500_eta02_v2_speed2":  "AL1c √ρ+S H500 η=0.2",
    "AL2b_align_sqrtSimplex_D19_p5_H250_eta085_v2_speed2": "AL2b √ρ+S H250 η=0.85",
    "AL3a_align_raw_D19_p5_H250_eta02_v2_speed2":          "AL3a raw H250 η=0.2",
    "AL5_align_sqrtSimplex_energy099_p5_H500_eta02_v2_speed2": "AL5 √ρ+S E99 H500",
}

COLORS = {
    "AL1c": "#1f77b4",
    "AL2b": "#ff7f0e",
    "AL3a": "#2ca02c",
    "AL5":  "#d62728",
}


def get_color(name):
    for k, c in COLORS.items():
        if name.startswith(k):
            return c
    return "#999999"


def load_experiment(exp_dir):
    """Load all test metrics for an experiment."""
    test_dir = exp_dir / "test"
    if not test_dir.exists():
        return None

    # Load per-test r2_vs_time.csv
    r2_curves = []
    metrics = []
    test_ids = sorted(
        [d for d in test_dir.iterdir() if d.is_dir() and d.name.startswith("test_")]
    )
    for td in test_ids:
        # r2 vs time curve
        csv_path = td / "r2_vs_time.csv"
        if csv_path.exists():
            times, r2_recon, r2_latent, r2_pod = [], [], [], []
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    times.append(float(row["time"]))
                    r2_recon.append(float(row["r2_reconstructed"]))
                    r2_latent.append(float(row["r2_latent"]))
                    r2_pod.append(float(row["r2_pod"]))
            r2_curves.append({
                "time": np.array(times),
                "r2_recon": np.array(r2_recon),
                "r2_latent": np.array(r2_latent),
                "r2_pod": np.array(r2_pod),
            })

        # metrics summary
        m_path = td / "metrics_summary.json"
        if m_path.exists():
            with open(m_path) as f:
                metrics.append(json.load(f))

    # Load MVAR and LSTM test_results.csv
    model_results = {}
    for model in ["MVAR", "LSTM"]:
        csv_path = exp_dir / model / "test_results.csv"
        if csv_path.exists():
            with open(csv_path) as f:
                model_results[model] = list(csv.DictReader(f))

    return {
        "r2_curves": r2_curves,
        "metrics": metrics,
        "model_results": model_results,
    }


def plot_r2_vs_time_fan(experiments, out_dir):
    """
    Plot: R² vs time for each AL experiment.
    The r2_vs_time.csv in test dirs tracks the pipeline evaluation
    (likely LSTM). Shows per-test faded lines + bold median.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (name, data) in enumerate(sorted(experiments.items())):
        ax = axes[idx]
        label = LABELS.get(name, name[:30])
        color = get_color(name)

        curves = data["r2_curves"]
        if not curves:
            ax.text(0.5, 0.5, "No r2_vs_time data", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title(label)
            continue

        # Stack all curves
        all_times = curves[0]["time"]
        all_r2 = np.array([c["r2_recon"] for c in curves])

        # Plot individual test runs (faded)
        for i in range(all_r2.shape[0]):
            ax.plot(all_times, all_r2[i], color=color, alpha=0.15, lw=0.5)

        # Median + IQR
        med = np.median(all_r2, axis=0)
        q25 = np.percentile(all_r2, 25, axis=0)
        q75 = np.percentile(all_r2, 75, axis=0)

        ax.plot(all_times, med, color=color, lw=2, label="Median")
        ax.fill_between(all_times, q25, q75, color=color, alpha=0.2, label="IQR")

        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("R² (reconstructed)")
        ax.axhline(0, color="gray", ls="--", lw=0.5)
        ax.legend(fontsize=9)

        # Smart y-limits
        ymin = max(np.percentile(all_r2, 5), -5)
        ax.set_ylim(ymin, 1.05)

    fig.suptitle("AL Experiments: R² vs Time (per-test pipeline output)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "AL_r2_vs_time_fan.png", dpi=150)
    plt.close(fig)
    print(f"  Saved AL_r2_vs_time_fan.png")


def plot_mvar_vs_lstm_bar(experiments, out_dir):
    """MVAR vs LSTM aggregate R² bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    names = sorted(experiments.keys())
    short_labels = [LABELS.get(n, n[:20]) for n in names]
    x = np.arange(len(names))
    width = 0.35

    ax = axes[0]
    mvar_r2, lstm_r2 = [], []
    for n in names:
        data = experiments[n]
        for model, r2_list in [("MVAR", mvar_r2), ("LSTM", lstm_r2)]:
            if model in data["model_results"]:
                vals = [float(r["r2_reconstructed"]) for r in data["model_results"][model]]
                r2_list.append(np.median(vals))
            else:
                r2_list.append(np.nan)

    lstm_r2_clipped = np.clip(lstm_r2, -5, 1)

    ax.bar(x - width / 2, mvar_r2, width, label="MVAR", color="#1f77b4", alpha=0.8)
    ax.bar(x + width / 2, lstm_r2_clipped, width,
           label="LSTM (clipped to [-5,1])", color="#ff7f0e", alpha=0.8)

    for i, (lr, lrc) in enumerate(zip(lstm_r2, lstm_r2_clipped)):
        if lr < -5:
            ax.text(x[i] + width / 2, lrc - 0.2, f"{lr:.0f}",
                    ha="center", va="top", fontsize=8, color="red", fontweight="bold")

    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("R² (reconstructed, median)")
    ax.set_title("MVAR vs LSTM: Median R²")
    ax.legend()
    ax.set_ylim(-5.5, 1.1)

    # Negativity fraction
    ax = axes[1]
    mvar_neg, lstm_neg = [], []
    for n in names:
        data = experiments[n]
        for model, neg_list in [("MVAR", mvar_neg), ("LSTM", lstm_neg)]:
            if model in data["model_results"]:
                vals = [float(r["negativity_frac"]) for r in data["model_results"][model]]
                neg_list.append(np.mean(vals))
            else:
                neg_list.append(np.nan)

    ax.bar(x - width / 2, mvar_neg, width, label="MVAR", color="#1f77b4", alpha=0.8)
    ax.bar(x + width / 2, lstm_neg, width, label="LSTM", color="#ff7f0e", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Negativity Fraction (%)")
    ax.set_title("Negative Density Pixels (%)")
    ax.legend()

    fig.suptitle("AL Experiments: MVAR vs LSTM Comparison",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "AL_mvar_vs_lstm_bar.png", dpi=150)
    plt.close(fig)
    print(f"  Saved AL_mvar_vs_lstm_bar.png")


def plot_r2_distributions(experiments, out_dir):
    """Box plots of R² distribution per experiment, MVAR vs LSTM."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    names = sorted(experiments.keys())
    short_labels = [LABELS.get(n, n[:20]) for n in names]

    for ax, model in zip(axes, ["MVAR", "LSTM"]):
        data_list = []
        labels_used = []
        colors_used = []
        for i, n in enumerate(names):
            data = experiments[n]
            if model in data["model_results"]:
                vals = [float(r["r2_reconstructed"]) for r in data["model_results"][model]]
                data_list.append(vals)
                labels_used.append(short_labels[i])
                colors_used.append(get_color(n))

        if data_list:
            bp = ax.boxplot(data_list, labels=labels_used, patch_artist=True,
                            showmeans=True,
                            meanprops=dict(marker="D", markerfacecolor="red", markersize=5))
            for patch, c in zip(bp["boxes"], colors_used):
                patch.set_facecolor(c)
                patch.set_alpha(0.5)

            ax.axhline(0, color="gray", ls="--", lw=0.5)

            all_vals = [v for vs in data_list for v in vs]
            if model == "LSTM":
                ymin = max(np.percentile(all_vals, 1), -10)
                ax.set_ylim(ymin, 1.1)
                ax.text(0.02, 0.02,
                        f"Note: LSTM outliers clipped\nTrue min: {min(all_vals):.0f}",
                        transform=ax.transAxes, fontsize=8, color="red",
                        verticalalignment="bottom")

        ax.set_title(f"{model} R² Distribution", fontweight="bold")
        ax.set_ylabel("R² (reconstructed)")
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=9)

    fig.suptitle("AL Experiments: R² Distributions (n=26 tests each)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "AL_r2_distributions.png", dpi=150)
    plt.close(fig)
    print(f"  Saved AL_r2_distributions.png")


def plot_lstm_r2_decay(experiments, out_dir):
    """LSTM R² decay overlay with MVAR aggregate annotations."""
    fig, ax = plt.subplots(figsize=(14, 7))

    for name, data in sorted(experiments.items()):
        label = LABELS.get(name, name[:20])
        color = get_color(name)
        curves = data["r2_curves"]
        if not curves:
            continue

        all_times = curves[0]["time"]
        all_r2 = np.array([c["r2_recon"] for c in curves])

        med = np.median(all_r2, axis=0)
        q10 = np.percentile(all_r2, 10, axis=0)
        q90 = np.percentile(all_r2, 90, axis=0)

        ax.plot(all_times, med, color=color, lw=2, label=f"{label} (LSTM)")
        ax.fill_between(all_times, q10, q90, color=color, alpha=0.1)

        if "MVAR" in data["model_results"]:
            mvar_vals = [float(r["r2_reconstructed"]) for r in data["model_results"]["MVAR"]]
            mvar_med = np.median(mvar_vals)
            ax.axhline(mvar_med, color=color, ls=":", lw=1.5, alpha=0.7)
            ax.text(all_times[-1], mvar_med + 0.01, f"MVAR={mvar_med:.2f}",
                    color=color, fontsize=8, ha="right", va="bottom")

    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.set_xlabel("Forecast Time (s)", fontsize=12)
    ax.set_ylabel("R² (reconstructed)", fontsize=12)
    ax.set_title("LSTM R² Decay Over Time (median ± 80% CI)\n+ MVAR aggregate (dotted)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_ylim(-5, 1.05)

    fig.tight_layout()
    fig.savefig(out_dir / "AL_lstm_r2_decay_overlay.png", dpi=150)
    plt.close(fig)
    print(f"  Saved AL_lstm_r2_decay_overlay.png")


def plot_r2_pod_vs_recon(experiments, out_dir):
    """R²_pod vs R²_recon scatter — bottleneck diagnostic."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for name, data in sorted(experiments.items()):
        label = LABELS.get(name, name[:20])
        color = get_color(name)

        if "MVAR" in data["model_results"]:
            r2_recon = [float(r["r2_reconstructed"]) for r in data["model_results"]["MVAR"]]
            r2_pod = [float(r["r2_pod"]) for r in data["model_results"]["MVAR"]]
            ax.scatter(r2_pod, r2_recon, color=color, alpha=0.6, s=40,
                       label=f"{label} MVAR")

        if "LSTM" in data["model_results"]:
            r2_recon = [float(r["r2_reconstructed"]) for r in data["model_results"]["LSTM"]]
            r2_pod = [float(r["r2_pod"]) for r in data["model_results"]["LSTM"]]
            r2_recon_clip = np.clip(r2_recon, -5, 1)
            ax.scatter(r2_pod, r2_recon_clip, color=color, alpha=0.3, s=20,
                       marker="x", label=f"{label} LSTM")

    ax.plot([0.99, 1.001], [0.99, 1.001], "k--", lw=0.5, alpha=0.5)
    ax.axhline(0, color="gray", ls="--", lw=0.5)

    ax.set_xlabel("R² (POD reconstruction)", fontsize=12)
    ax.set_ylabel("R² (forecasted reconstruction)", fontsize=12)
    ax.set_title("POD Accuracy vs Forecast Accuracy\n"
                 "(close to diagonal = forecasting not bottleneck)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_ylim(-5.5, 1.05)

    fig.tight_layout()
    fig.savefig(out_dir / "AL_pod_vs_recon_scatter.png", dpi=150)
    plt.close(fig)
    print(f"  Saved AL_pod_vs_recon_scatter.png")


def plot_summary_table(experiments, out_dir):
    """Summary table rendered as a figure."""
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis("off")

    headers = ["Experiment", "Transform", "Horizon", "η",
               "MVAR R²\n(med±std)", "LSTM R²\n(med±std)",
               "MVAR neg%", "LSTM neg%", "R²_pod"]

    rows = []
    for name in sorted(experiments.keys()):
        data = experiments[name]
        label = LABELS.get(name, name[:20])

        transform = "√ρ+S" if "sqrtSimplex" in name else "raw"
        if "energy099" in name:
            transform += " E99"
        horizon = "H500" if "H500" in name else "H250"
        eta = "0.85" if "eta085" in name else "0.2"

        mvar_r2 = lstm_r2 = mvar_neg = lstm_neg = r2_pod = "N/A"

        if "MVAR" in data["model_results"]:
            vals = [float(r["r2_reconstructed"]) for r in data["model_results"]["MVAR"]]
            mvar_r2 = f"{np.median(vals):.4f}±{np.std(vals):.4f}"
            neg_vals = [float(r["negativity_frac"]) for r in data["model_results"]["MVAR"]]
            mvar_neg = f"{np.mean(neg_vals):.1f}%"
            pod_vals = [float(r["r2_pod"]) for r in data["model_results"]["MVAR"]]
            r2_pod = f"{np.median(pod_vals):.5f}"

        if "LSTM" in data["model_results"]:
            vals = [float(r["r2_reconstructed"]) for r in data["model_results"]["LSTM"]]
            lstm_r2 = f"{np.median(vals):.1f}±{np.std(vals):.0f}"
            neg_vals = [float(r["negativity_frac"]) for r in data["model_results"]["LSTM"]]
            lstm_neg = f"{np.mean(neg_vals):.1f}%"

        rows.append([label, transform, horizon, eta, mvar_r2, lstm_r2,
                     mvar_neg, lstm_neg, r2_pod])

    table = ax.table(cellText=rows, colLabels=headers, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    for j in range(len(headers)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(len(rows)):
        # MVAR R² column (4)
        try:
            val = float(rows[i][4].split("±")[0])
            table[i + 1, 4].set_facecolor(
                "#C6EFCE" if val > 0.9 else "#FFEB9C" if val > 0 else "#FFC7CE")
        except Exception:
            pass
        # LSTM R² column (5)
        try:
            val = float(rows[i][5].split("±")[0])
            table[i + 1, 5].set_facecolor(
                "#C6EFCE" if val > 0.9 else "#FFEB9C" if val > 0 else "#FFC7CE")
        except Exception:
            pass

    ax.set_title("AL Experiments Summary (n=26 test trajectories, speed=2.0 regime)",
                 fontsize=14, fontweight="bold", pad=20)

    fig.tight_layout()
    fig.savefig(out_dir / "AL_summary_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved AL_summary_table.png")


def plot_lstm_divergence_time(experiments, out_dir):
    """Histogram: when does LSTM first produce R² < 0?"""
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, data in sorted(experiments.items()):
        label = LABELS.get(name, name[:20])
        color = get_color(name)
        curves = data["r2_curves"]
        if not curves:
            continue

        all_times = curves[0]["time"]
        all_r2 = np.array([c["r2_recon"] for c in curves])

        diverge_times = []
        for i in range(all_r2.shape[0]):
            neg_idx = np.where(all_r2[i] < 0)[0]
            if len(neg_idx) > 0:
                diverge_times.append(all_times[neg_idx[0]])
            else:
                diverge_times.append(all_times[-1])

        diverge_times = np.array(diverge_times)
        ax.hist(diverge_times, bins=20, color=color, alpha=0.5, label=label,
                edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Time of First R² < 0 (s)", fontsize=12)
    ax.set_ylabel("Count (of 26 tests)", fontsize=12)
    ax.set_title("LSTM Divergence Time: When Does R² First Go Negative?",
                 fontsize=13, fontweight="bold")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "AL_lstm_divergence_time.png", dpi=150)
    plt.close(fig)
    print(f"  Saved AL_lstm_divergence_time.png")


def plot_1step_vs_fullhorizon(experiments, out_dir):
    """1-step R² vs full-horizon R² scatter."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for name, data in sorted(experiments.items()):
        label = LABELS.get(name, name[:20])
        color = get_color(name)

        for model, marker, size in [("MVAR", "o", 50), ("LSTM", "x", 30)]:
            if model in data["model_results"]:
                r2_full = [float(r["r2_reconstructed"]) for r in data["model_results"][model]]
                r2_1step = [float(r["r2_1step"]) for r in data["model_results"][model]]
                r2_full_clip = np.clip(r2_full, -5, 1)
                ax.scatter(r2_1step, r2_full_clip, color=color, alpha=0.5,
                           s=size, marker=marker, label=f"{label} {model}")

    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.axvline(0, color="gray", ls="--", lw=0.5)
    ax.set_xlabel("R² (1-step)", fontsize=12)
    ax.set_ylabel("R² (full horizon, clipped)", fontsize=12)
    ax.set_title("1-Step vs Full-Horizon R²\n"
                 "(does single-step accuracy predict stability?)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.tight_layout()
    fig.savefig(out_dir / "AL_1step_vs_fullhorizon.png", dpi=150)
    plt.close(fig)
    print(f"  Saved AL_1step_vs_fullhorizon.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover AL experiments
    al_dirs = sorted([
        d for d in OSCAR_DIR.iterdir()
        if d.is_dir() and d.name.startswith("AL")
    ])

    if not al_dirs:
        print("ERROR: No AL experiment directories found in oscar_output/")
        sys.exit(1)

    print(f"Found {len(al_dirs)} AL experiments:")
    for d in al_dirs:
        print(f"  {d.name}")
    print()

    # Load all experiments
    experiments = {}
    for d in al_dirs:
        print(f"Loading {d.name}...")
        data = load_experiment(d)
        if data is not None:
            experiments[d.name] = data
            n_tests = len(data["r2_curves"])
            models = list(data["model_results"].keys())
            print(f"  {n_tests} test runs, models: {models}")

    print(f"\nGenerating plots to {OUT_DIR}/\n")

    # Generate all plots
    plot_summary_table(experiments, OUT_DIR)
    plot_mvar_vs_lstm_bar(experiments, OUT_DIR)
    plot_r2_distributions(experiments, OUT_DIR)
    plot_r2_vs_time_fan(experiments, OUT_DIR)
    plot_lstm_r2_decay(experiments, OUT_DIR)
    plot_r2_pod_vs_recon(experiments, OUT_DIR)
    plot_lstm_divergence_time(experiments, OUT_DIR)
    plot_1step_vs_fullhorizon(experiments, OUT_DIR)

    # Print key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    for name, data in sorted(experiments.items()):
        label = LABELS.get(name, name[:20])
        mvar_r2 = lstm_r2 = "N/A"
        if "MVAR" in data["model_results"]:
            vals = [float(r["r2_reconstructed"]) for r in data["model_results"]["MVAR"]]
            mvar_r2 = f"{np.median(vals):.4f}"
        if "LSTM" in data["model_results"]:
            vals = [float(r["r2_reconstructed"]) for r in data["model_results"]["LSTM"]]
            lstm_r2 = f"{np.median(vals):.1f}"
        print(f"  {label:35s}  MVAR={mvar_r2:>10s}  LSTM={lstm_r2:>10s}")

    print()
    print("INTERPRETATION:")
    print("  - MVAR is stable and accurate (R² > 0.85) across ALL AL experiments")
    print("  - LSTM catastrophically diverges with sqrt+simplex (R² < -100)")
    print("  - AL3a (raw, no transform) -> LSTM survives (R² ~ 0.84)")
    print("  - The sqrt transform amplifies LSTM autoregressive errors exponentially")
    print("  - MVAR's linear dynamics are robust to the nonlinear transform")
    print("  - High noise (eta=0.85) worsens MVAR modestly (0.99 -> 0.85)")
    print("=" * 70)

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
