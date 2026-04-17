"""Generate combined MVAR+LSTM R2(t) degradation figure.

Single 2-panel (side-by-side) figure replacing the two separate ones.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
})

BASE = Path("oscar_output/systematics")

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
            t = curves[0][0]
            min_len = min(len(c[1]) for c in curves)
            r2_stack = np.array([c[1][:min_len] for c in curves])
            r2_mean = np.mean(r2_stack, axis=0)
            experiments[exp_dir.name] = (t[:min_len], r2_mean)
    return experiments


def plot_panel(ax, experiments, model_name, color_main, color_other, color_agg):
    t_norm = np.linspace(0, 1, 500)
    all_r2_interp = []
    n_diverged = 0
    for name, (t, r2) in sorted(experiments.items()):
        t_frac = (t - t[0]) / (t[-1] - t[0]) if t[-1] > t[0] else np.zeros_like(t)
        is_main = name in MAIN_REGIMES
        lw = 1.0 if is_main else 0.3
        alpha = 0.5 if is_main else 0.12
        color = color_main if is_main else color_other
        ax.plot(t_frac, r2, color=color, lw=lw, alpha=alpha, zorder=2 if is_main else 1)
        r2_interp = np.interp(t_norm, t_frac, r2)
        # Exclude diverged experiments (final R² < -2) from aggregate statistics
        if r2_interp[-1] < -2:
            n_diverged += 1
            continue
        all_r2_interp.append(np.clip(r2_interp, 0, None))

    n_used = len(all_r2_interp)
    r2_stack = np.array(all_r2_interp)
    median = np.median(r2_stack, axis=0)
    q25 = np.percentile(r2_stack, 25, axis=0)
    q75 = np.percentile(r2_stack, 75, axis=0)

    ax.fill_between(t_norm, q25, q75, color=color_agg, alpha=0.25, zorder=3)
    ax.plot(t_norm, median, color=color_agg, lw=2.5, zorder=4, label="Median")

    ax.set_xlabel("Fraction of prediction horizon")
    title = f"{model_name} ({len(experiments)} experiments"
    if n_diverged:
        title += f", {n_diverged} diverged"
    title += ")"
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 1.05)
    ax.grid(True, alpha=0.3)


if __name__ == "__main__":
    print("Loading MVAR curves...")
    mvar_exps = load_r2_curves("mvar")
    print(f"  Loaded {len(mvar_exps)} experiments")

    print("Loading LSTM curves...")
    lstm_exps = load_r2_curves("lstm")
    print(f"  Loaded {len(lstm_exps)} experiments")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)

    # MVAR in blue tones
    plot_panel(ax1, mvar_exps, "MVAR", "#1f77b4", "#aec7e8", "#2c7fb8")
    ax1.set_ylabel(r"$R^2_{\mathrm{reconstructed}}(t)$")

    # LSTM in red tones
    plot_panel(ax2, lstm_exps, "LSTM", "#d62728", "#ff9896", "#e31a1c")

    # Shared legend
    custom = [
        Line2D([0], [0], color="#2c7fb8", lw=2.5, label="MVAR median"),
        Line2D([0], [0], color="#e31a1c", lw=2.5, label="LSTM median"),
        Line2D([0], [0], color="#1f77b4", lw=1.0, alpha=0.5, label="Main-text regimes"),
        Line2D([0], [0], color="grey", lw=0.4, alpha=0.3, label="Other regimes"),
    ]
    fig.legend(handles=custom, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(r"$R^2(t)$ Degradation — MVAR vs LSTM", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    out = Path("Thesis_Figures/appF_r2_degradation_combined.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")
