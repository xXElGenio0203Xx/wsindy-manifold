"""Generate companion-matrix eigenvalue spectra plot for the thesis.

Plots eigenvalues in the complex plane with the unit circle as reference.
One panel per regime.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

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

REGIMES = [
    ("gas",           "oscar_output/systematics/NDYN04_gas_thesis_final/MVAR/mvar_model.npz"),
    ("blackhole",     "oscar_output/systematics/NDYN05_blackhole_thesis_final/MVAR/mvar_model.npz"),
    ("supernova",     "oscar_output/systematics/NDYN06_supernova_thesis_final/MVAR/mvar_model.npz"),
    ("pure\\_vicsek", "oscar_output/systematics/NDYN08_pure_vicsek_thesis_final/MVAR/mvar_model.npz"),
    ("gas\\_VS",      "oscar_output/systematics/NDYN04_gas_VS_thesis_final/MVAR/mvar_model.npz"),
    ("blackhole\\_VS","oscar_output/systematics/NDYN05_blackhole_VS_thesis_final/MVAR/mvar_model.npz"),
    ("supernova\\_VS","oscar_output/systematics/NDYN06_supernova_VS_thesis_final/MVAR/mvar_model.npz"),
]


def build_companion(path):
    d = np.load(path)
    A_top = d["A_companion"]
    r = int(d["r"])
    p = int(d["p"])
    n = r * p
    Ac = np.zeros((n, n))
    Ac[:r, :] = A_top
    if p > 1:
        Ac[r:, :-r] = np.eye(r * (p - 1))
    eigs = np.linalg.eigvals(Ac)
    rho_before = float(d["rho_before"])
    rho_after = float(d["rho_after"])
    return eigs, rho_before, rho_after


def main():
    n = len(REGIMES)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    theta = np.linspace(0, 2 * np.pi, 200)
    for i, (label, path) in enumerate(REGIMES):
        ax = axes[i]
        eigs, rho_b, rho_a = build_companion(path)
        # Unit circle
        ax.plot(np.cos(theta), np.sin(theta), "k--", lw=0.8, alpha=0.5)
        # Classify eigenvalues: outside unit circle = "clipped"
        inside = np.abs(eigs) <= 1.0 + 1e-10
        outside = ~inside
        ax.plot(eigs[inside].real, eigs[inside].imag, "o", ms=2.5,
                color="C0", alpha=0.7, label="stable")
        if outside.any():
            ax.plot(eigs[outside].real, eigs[outside].imag, "o", ms=3.5,
                    color="C3", alpha=0.8, markerfacecolor="none",
                    markeredgewidth=1.0, label="unstable")
        ax.set_title(label, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axhline(0, color="gray", lw=0.3)
        ax.axvline(0, color="gray", lw=0.3)
        rho_str = f"$\\rho={rho_b:.3f}$"
        n_unstable = int(outside.sum())
        if n_unstable > 0:
            rho_str += f"\n({n_unstable} clipped)"
        ax.text(0.02, 0.98, rho_str, transform=ax.transAxes,
                fontsize=7, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Re")
        if i % ncols == 0:
            ax.set_ylabel("Im")

    # Hide unused axes
    for j in range(len(REGIMES), len(axes)):
        axes[j].set_visible(False)

    fig.subplots_adjust(hspace=0.35, wspace=0.35)
    out = Path("Thesis_Figures/appE_eigenvalue_spectra.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
