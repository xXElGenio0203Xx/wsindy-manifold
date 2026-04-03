"""
Task 5.2 — Convergence verification for WSINDy.

Verifies two convergence rates from the theory (Messenger & Bortz 2021):
  1.  Coefficient error vs Δt  →  expected O(Δt^η), η related to test-fn order.
  2.  Coefficient error vs nx  →  expected O(Δx^η) via refinement.

Generates exact heat-equation data u_t = D Δu at multiple resolutions,
runs WSINDy, and produces log-log convergence plots.

Usage
-----
    python -m wsindy.convergence_study --output convergence.png
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


# ── Exact solution for heat equation u_t = D Δu on [0, 2π]² ─────────

def _exact_heat(X, Y, t, D=0.1):
    return (
        1.0
        + np.exp(-D * 2 * t) * np.cos(X) * np.cos(Y)
        + 0.8 * np.exp(-D * 4 * t) * np.cos(2 * X)
        + 0.5 * np.exp(-D * 16 * t) * np.cos(4 * X)
        + 0.6 * np.exp(-D * 9 * t) * np.cos(3 * Y)
    )


TRUE_COEFF = {"lap:u": 0.1}  # D = 0.1


def _generate(nx, ny, T_steps, dt, D=0.1):
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


def _coeff_error(model, true=TRUE_COEFF):
    """Max absolute coefficient error on truly active terms."""
    err = 0.0
    for name, w_true in true.items():
        if name in model.col_names:
            idx = model.col_names.index(name)
            err = max(err, abs(model.w[idx] - w_true))
        else:
            err = max(err, abs(w_true))
    return err


def _run_one(nx, ny, T_steps, dt, p=(3, 5, 5), ell=None):
    """Run WSINDy at a single resolution and return (model, error)."""
    U, grid = _generate(nx, ny, T_steps, dt)
    library_terms = default_library()

    if ell is None:
        ell = (min(5, T_steps // 6), min(5, nx // 6), min(5, ny // 6))

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
    b, G, col_names = build_weak_system(U, grid, psi, library_terms, qi)
    model = wsindy_fit_regression(b, G, col_names)
    return model, _coeff_error(model)


# ── Convergence sweeps ────────────────────────────────────────────────

def dt_convergence(
    dts=(0.2, 0.1, 0.05, 0.025, 0.0125),
    nx=64,
    ny=64,
    T_phys=5.0,
):
    """Sweep over Δt at fixed spatial resolution."""
    errors = []
    for dt in dts:
        T_steps = max(int(T_phys / dt), 30)
        _, err = _run_one(nx, ny, T_steps, dt)
        errors.append(err)
        print(f"  Δt={dt:.4f}  T_steps={T_steps}  error={err:.4e}")
    return np.array(dts), np.array(errors)


def dx_convergence(
    nxs=(16, 24, 32, 48, 64, 96),
    dt=0.05,
    T_steps=100,
):
    """Sweep over grid resolution at fixed Δt."""
    dxs = []
    errors = []
    for nx in nxs:
        ny = nx
        dx = 2 * np.pi / nx
        _, err = _run_one(nx, ny, T_steps, dt)
        dxs.append(dx)
        errors.append(err)
        print(f"  nx={nx}  Δx={dx:.4f}  error={err:.4e}")
    return np.array(dxs), np.array(errors)


def _fit_loglog_slope(x, y):
    """Fit slope of log-log relationship, ignoring zeros."""
    mask = (y > 0) & (x > 0)
    if mask.sum() < 2:
        return float("nan")
    coeffs = np.polyfit(np.log10(x[mask]), np.log10(y[mask]), 1)
    return coeffs[0]


def plot_convergence(
    dt_result,
    dx_result,
    outfile="convergence.png",
):
    """Generate log-log convergence plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    dts, dt_errs = dt_result
    slope_dt = _fit_loglog_slope(dts, dt_errs)
    ax1.loglog(dts, dt_errs, "o-", lw=2, label=f"WSINDy (slope={slope_dt:.2f})")
    ax1.set_xlabel(r"$\Delta t$")
    ax1.set_ylabel(r"$\max_j |w_j - w_j^*|$")
    ax1.set_title(r"Coefficient error vs $\Delta t$")
    ax1.legend()
    ax1.grid(True, which="both", ls=":", alpha=0.5)

    dxs, dx_errs = dx_result
    slope_dx = _fit_loglog_slope(dxs, dx_errs)
    ax2.loglog(dxs, dx_errs, "s-", lw=2, color="C1", label=f"WSINDy (slope={slope_dx:.2f})")
    ax2.set_xlabel(r"$\Delta x$")
    ax2.set_ylabel(r"$\max_j |w_j - w_j^*|$")
    ax2.set_title(r"Coefficient error vs $\Delta x$")
    ax2.legend()
    ax2.grid(True, which="both", ls=":", alpha=0.5)

    fig.suptitle("WSINDy Convergence Verification (Heat eq.)", fontsize=13)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outfile}")
    return {"slope_dt": slope_dt, "slope_dx": slope_dx}


def run(outfile="convergence.png"):
    """Run the full convergence study."""
    print("WSINDy Convergence Verification")
    print("=" * 50)
    print("\n1. Δt convergence (fixed nx=64):")
    dt_result = dt_convergence()
    print("\n2. Δx convergence (fixed dt=0.05):")
    dx_result = dx_convergence()
    result = plot_convergence(dt_result, dx_result, outfile)
    print(f"\n  Fitted slopes: Δt → {result['slope_dt']:.2f}, Δx → {result['slope_dx']:.2f}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WSINDy convergence study")
    parser.add_argument("--output", default="convergence.png")
    args = parser.parse_args()
    run(args.output)
