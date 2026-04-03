"""
Task 5.3 — Noise robustness testing for WSINDy.

Adds controlled extrinsic noise ε to a known PDE field at multiple
noise ratios and reports TPR (true positive rate) and coefficient
error as a function of ε.

Uses the exact heat equation u_t = D Δu as ground truth.

Usage
-----
    python -m wsindy.noise_robustness --output noise_robustness.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .grid import GridSpec
from .test_functions import make_separable_psi
from .system import build_weak_system, make_query_indices, default_t_margin
from .fit import wsindy_fit_regression
from .library import default_library


# ── Exact heat equation data ─────────────────────────────────────────

TRUE_TERMS = {"lap:u": 0.1}
ALL_POSSIBLE = None  # set after library is built


def _exact_heat(X, Y, t, D=0.1):
    return (
        1.0
        + np.exp(-D * 2 * t) * np.cos(X) * np.cos(Y)
        + 0.8 * np.exp(-D * 4 * t) * np.cos(2 * X)
        + 0.5 * np.exp(-D * 16 * t) * np.cos(4 * X)
        + 0.6 * np.exp(-D * 9 * t) * np.cos(3 * Y)
    )


def _generate(nx=64, ny=64, T_steps=100, dt=0.05, D=0.1):
    Lx = Ly = 2 * np.pi
    dx, dy = Lx / nx, Ly / ny
    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    grid = GridSpec(dt=dt, dx=dx, dy=dy)
    U = np.zeros((T_steps, nx, ny))
    for i in range(T_steps):
        U[i] = _exact_heat(X, Y, i * dt, D)
    return U, grid


def _add_noise(U, noise_ratio, rng=None):
    """Add Gaussian noise with std = noise_ratio × std(U)."""
    if rng is None:
        rng = np.random.default_rng(42)
    sigma = noise_ratio * np.std(U)
    return U + rng.normal(0, sigma, U.shape)


def _tpr_and_error(model, true_terms=TRUE_TERMS):
    """Compute TPR and max coefficient error."""
    true_set = set(true_terms.keys())
    discovered = set(model.active_terms)

    tp = len(true_set & discovered)
    fn = len(true_set - discovered)
    fp = len(discovered - true_set)
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(len(model.col_names) - len(true_set), 1)

    coeff_err = 0.0
    for name, w_true in true_terms.items():
        if name in model.col_names:
            idx = model.col_names.index(name)
            coeff_err = max(coeff_err, abs(model.w[idx] - w_true))
        else:
            coeff_err = max(coeff_err, abs(w_true))

    return {
        "tpr": tpr,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "coeff_error": coeff_err,
        "n_active": model.n_active,
        "active_terms": model.active_terms,
    }


def noise_sweep(
    noise_ratios=(0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.316, 1.0),
    n_trials: int = 5,
    nx: int = 64,
    ny: int = 64,
    T_steps: int = 100,
    dt: float = 0.05,
    p=(3, 5, 5),
    ell=(5, 5, 5),
):
    """Run WSINDy at many noise levels, averaging over trials."""
    U_clean, grid = _generate(nx, ny, T_steps, dt)
    library_terms = default_library()

    psi = make_separable_psi(
        grid,
        ellt=ell[0], ellx=ell[1], elly=ell[2],
        pt=p[0], px=p[1], py=p[2],
    )
    t_margin = default_t_margin(psi)
    qi = make_query_indices(
        T_steps, nx, ny,
        stride_t=2, stride_x=2, stride_y=2,
        t_margin=t_margin,
    )

    results = []
    for eps in noise_ratios:
        trial_metrics = []
        for trial in range(n_trials):
            rng = np.random.default_rng(1000 * trial + 42)
            U = _add_noise(U_clean, eps, rng) if eps > 0 else U_clean.copy()
            b, G, col_names = build_weak_system(U, grid, psi, library_terms, qi)
            model = wsindy_fit_regression(b, G, col_names)
            metrics = _tpr_and_error(model)
            trial_metrics.append(metrics)

        avg_tpr = np.mean([m["tpr"] for m in trial_metrics])
        avg_err = np.mean([m["coeff_error"] for m in trial_metrics])
        avg_fp = np.mean([m["fp"] for m in trial_metrics])
        avg_act = np.mean([m["n_active"] for m in trial_metrics])

        results.append({
            "noise_ratio": eps,
            "tpr_mean": float(avg_tpr),
            "coeff_error_mean": float(avg_err),
            "fp_mean": float(avg_fp),
            "n_active_mean": float(avg_act),
            "trials": trial_metrics,
        })
        print(
            f"  ε={eps:.3e}  TPR={avg_tpr:.3f}  "
            f"coeff_err={avg_err:.4e}  FP={avg_fp:.1f}  "
            f"active={avg_act:.1f}"
        )

    return results


def plot_noise_robustness(results, outfile="noise_robustness.png"):
    """Generate noise robustness diagnostic plot."""
    eps_vals = [r["noise_ratio"] for r in results]
    tpr_vals = [r["tpr_mean"] for r in results]
    err_vals = [r["coeff_error_mean"] for r in results]
    fp_vals = [r["fp_mean"] for r in results]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))

    # Use log scale for x but handle 0 noise
    eps_plot = [max(e, 1e-4) for e in eps_vals]

    ax1.semilogx(eps_plot, tpr_vals, "o-", lw=2)
    ax1.set_xlabel(r"Noise ratio $\varepsilon$")
    ax1.set_ylabel("TPR (True Positive Rate)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("Support recovery")
    ax1.grid(True, ls=":", alpha=0.5)

    ax2.loglog(eps_plot, [max(e, 1e-12) for e in err_vals], "s-", lw=2, color="C1")
    ax2.set_xlabel(r"Noise ratio $\varepsilon$")
    ax2.set_ylabel(r"$\max_j |w_j - w_j^*|$")
    ax2.set_title("Coefficient error")
    ax2.grid(True, which="both", ls=":", alpha=0.5)

    ax3.semilogx(eps_plot, fp_vals, "^-", lw=2, color="C2")
    ax3.set_xlabel(r"Noise ratio $\varepsilon$")
    ax3.set_ylabel("Mean false positives")
    ax3.set_title("Spurious term count")
    ax3.grid(True, ls=":", alpha=0.5)

    fig.suptitle("WSINDy Noise Robustness (Heat eq.)", fontsize=13)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outfile}")


def run(outfile="noise_robustness.png"):
    """Run the full noise robustness study."""
    print("WSINDy Noise Robustness Study")
    print("=" * 50)
    results = noise_sweep()
    plot_noise_robustness(results, outfile)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WSINDy noise robustness study")
    parser.add_argument("--output", default="noise_robustness.png")
    args = parser.parse_args()
    run(args.output)
