"""Generate density snapshot comparison figures for the thesis.

Produces 3-row (Truth, MVAR, LSTM) × 3-column (t₁, t₂, t₃) panels.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
})

REGIMES = {
    "gas": {
        "path": "oscar_output/systematics/NDYN04_gas_thesis_final/test/test_000",
        "lstm_path": "oscar_output/systematics/NDYN04_gas_preproc_sqrt_none/test/test_000",
        "label": "NDYN04\\_gas",
        "times_s": [0.6, 25.0, 75.0],
        "out": "Thesis_Figures/ch7_snapshots_gas.pdf",
    },
    "blackhole": {
        "path": "oscar_output/systematics/NDYN05_blackhole_thesis_final/test/test_000",
        "label": "NDYN05\\_blackhole",
        "times_s": [0.6, 25.0, 75.0],
        "out": "Thesis_Figures/appE_snapshots_blackhole.pdf",
    },
    "supernova": {
        "path": "oscar_output/systematics/NDYN06_supernova_thesis_final/test/test_000",
        "label": "NDYN06\\_supernova",
        "times_s": [0.4, 1.0, 2.0],  # explosion disperses by t≈5; R²>0.94 here
        "out": "Thesis_Figures/appE_snapshots_supernova.pdf",
    },
}


def load_density(base: str, kind: str):
    """Load density array and times."""
    fname = {
        "true": "density_true.npz",
        "mvar": "density_pred_mvar.npz",
        "lstm": "density_pred_lstm.npz",
    }[kind]
    d = np.load(Path(base) / fname)
    return d["rho"], d["times"], d.get("xgrid", None), d.get("ygrid", None)


def load_trajectory(base: str):
    """Load particle trajectory if available."""
    p = Path(base) / "trajectory.npz"
    if not p.exists():
        return None, None
    d = np.load(p)
    return d["traj"], d["times"]  # traj: (T, N, 2)


def find_idx(times, target):
    return int(np.argmin(np.abs(times - target)))


def make_figure(regime_key: str):
    cfg = REGIMES[regime_key]
    base = cfg["path"]
    targets = cfg["times_s"]

    rho_true, t_true, xgrid, ygrid = load_density(base, "true")
    rho_mvar, t_mvar, _, _ = load_density(base, "mvar")
    lstm_base = cfg.get("lstm_path", base)
    rho_lstm, t_lstm, _, _ = load_density(lstm_base, "lstm")

    # Load particle trajectory for ground-truth overlay
    traj, t_traj = load_trajectory(base)

    ncols = len(targets)
    fig, axes = plt.subplots(3, ncols, figsize=(4.0 * ncols, 10.5))

    # Global colour limits across all panels
    vmin = 0.0
    vmax_candidates = []
    for t_s in targets:
        i_true = find_idx(t_true, t_s)
        i_mvar = find_idx(t_mvar, t_s)
        i_lstm = find_idx(t_lstm, t_s)
        vmax_candidates.extend([
            rho_true[i_true].max(),
            rho_mvar[i_mvar].max(),
            rho_lstm[i_lstm].max(),
        ])
    vmax = max(vmax_candidates)

    # Grid extent for imshow (pixel edges)
    nx = rho_true.shape[2]
    ny = rho_true.shape[1]
    if xgrid is not None and ygrid is not None:
        dx = xgrid[1] - xgrid[0]
        dy = ygrid[1] - ygrid[0]
        extent = [xgrid[0] - dx/2, xgrid[-1] + dx/2,
                  ygrid[0] - dy/2, ygrid[-1] + dy/2]
    else:
        extent = None

    row_labels = ["Ground truth", "MVAR", "LSTM"]
    data_sets = [
        (rho_true, t_true),
        (rho_mvar, t_mvar),
        (rho_lstm, t_lstm),
    ]

    im = None
    for col, t_s in enumerate(targets):
        for row, (rho, t_arr) in enumerate(data_sets):
            idx = find_idx(t_arr, t_s)
            im = axes[row, col].imshow(
                rho[idx],
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                origin="lower",
                aspect="equal",
                extent=extent,
            )
            # Overlay particles on ground truth row
            if row == 0 and traj is not None:
                tidx = find_idx(t_traj, t_s)
                px = traj[tidx, :, 0]
                py = traj[tidx, :, 1]
                axes[row, col].scatter(
                    px, py, s=4, c="red", edgecolors="none",
                    alpha=0.7, zorder=5,
                )
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(f"$t = {t_s}$ s", fontsize=11)
            if col == 0:
                axes[row, col].set_ylabel(row_labels[row], fontsize=11)

    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, label=r"$\rho$")
    fig.subplots_adjust(left=0.06, right=0.90, top=0.95, bottom=0.02, wspace=0.08, hspace=0.08)

    out = Path(cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


if __name__ == "__main__":
    for key in REGIMES:
        print(f"Generating {key}...")
        make_figure(key)
    print("Done.")
