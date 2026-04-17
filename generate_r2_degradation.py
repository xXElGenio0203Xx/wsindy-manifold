"""Generate R2(t) degradation curves for all systematic experiments.

Produces two figures:
1. MVAR R2(t) degradation
2. LSTM R2(t) degradation
Each shows faint per-experiment traces, bold median, shaded IQR.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
})

BASE = Path("oscar_output/systematics")

# Main-text regimes (will be drawn with heavier line weight)
MAIN_REGIMES = {
    "NDYN04_gas_thesis_final",
    "NDYN05_blackhole_thesis_final",
    "NDYN06_supernova_thesis_final",
    "NDYN08_pure_vicsek_thesis_final",
    "NDYN04_gas_VS_thesis_final",
    "NDYN05_blackhole_VS_thesis_final",
    "NDYN06_supernova_VS_thesis_final",
}


def load_r2_curves(model_suffix: str):
    """Load all R2(t) curves for a given model (mvar or lstm).

    Returns dict: experiment_name -> (times, r2_mean_across_test_runs)
    """
    fname = f"r2_vs_time_{model_suffix}.csv"
    experiments = defaultdict(list)

    for exp_dir in sorted(BASE.iterdir()):
        if not exp_dir.is_dir():
            continue
        test_dir = exp_dir / "test"
        if not test_dir.exists():
            continue
        curves = []
        for run_dir in sorted(test_dir.iterdir()):
            csv_path = run_dir / fname
            if not csv_path.exists():
                continue
            times, r2s = [], []
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    times.append(float(row["time"]))
                    r2s.append(float(row["r2_reconstructed"]))
            curves.append((np.array(times), np.array(r2s)))
        if curves:
            # Average across test runs (they share the same time grid)
            t = curves[0][0]
            min_len = min(len(c[1]) for c in curves)
            r2_stack = np.array([c[1][:min_len] for c in curves])
            r2_mean = np.mean(r2_stack, axis=0)
            experiments[exp_dir.name] = (t[:min_len], r2_mean)

    return experiments


def make_degradation_plot(experiments: dict, model_name: str, out_path: str):
    """Create aggregate R2(t) degradation figure."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Normalize times to [0, 1] fraction of horizon for aggregation
    # But also keep absolute times for main x-axis
    # Since experiments may have different T_test, normalize to fraction
    all_r2_interp = []
    t_norm = np.linspace(0, 1, 500)

    for name, (t, r2) in sorted(experiments.items()):
        t_frac = (t - t[0]) / (t[-1] - t[0]) if t[-1] > t[0] else np.zeros_like(t)
        is_main = name in MAIN_REGIMES
        lw = 1.2 if is_main else 0.4
        alpha = 0.6 if is_main else 0.15
        color = "C1" if is_main else "C0"
        ax.plot(t_frac, r2, color=color, lw=lw, alpha=alpha, zorder=2 if is_main else 1)

        # Interpolate to common grid
        r2_interp = np.interp(t_norm, t_frac, r2)
        all_r2_interp.append(r2_interp)

    # Compute median and IQR
    r2_stack = np.array(all_r2_interp)
    median = np.median(r2_stack, axis=0)
    q25 = np.percentile(r2_stack, 25, axis=0)
    q75 = np.percentile(r2_stack, 75, axis=0)

    ax.fill_between(t_norm, q25, q75, color="C2", alpha=0.25, zorder=3, label="IQR")
    ax.plot(t_norm, median, color="C2", lw=2.5, zorder=4, label="Median")

    ax.set_xlabel("Fraction of prediction horizon")
    ax.set_ylabel(r"$R^2_{\mathrm{reconstructed}}(t)$")
    ax.set_title(f"{model_name} — $R^2(t)$ degradation ({len(experiments)} experiments)")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 1.05)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add secondary legend for main vs other
    from matplotlib.lines import Line2D
    custom = [
        Line2D([0], [0], color="C1", lw=1.2, alpha=0.6, label="Main-text regimes"),
        Line2D([0], [0], color="C0", lw=0.5, alpha=0.3, label="Other regimes"),
    ]
    leg2 = ax.legend(handles=custom, loc="lower right", fontsize=8)
    ax.add_artist(ax.legend(loc="lower left", fontsize=9))

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out} ({len(experiments)} experiments)")


if __name__ == "__main__":
    print("Loading MVAR curves...")
    mvar_exps = load_r2_curves("mvar")
    print(f"  Loaded {len(mvar_exps)} experiments")
    make_degradation_plot(mvar_exps, "MVAR", "Thesis_Figures/appF_r2_degradation_all_mvar.pdf")

    print("Loading LSTM curves...")
    lstm_exps = load_r2_curves("lstm")
    print(f"  Loaded {len(lstm_exps)} experiments")
    make_degradation_plot(lstm_exps, "LSTM", "Thesis_Figures/appF_r2_degradation_all_lstm.pdf")

    print("Done.")
