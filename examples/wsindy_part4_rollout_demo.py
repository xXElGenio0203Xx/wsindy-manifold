#!/usr/bin/env python
"""
WSINDy Part 4 Demo — PDE Discovery + Forecasting
==================================================

Two experiments:

  **Experiment 1 – Integrator verification.**
      Construct a WSINDyModel with the *exact* heat-equation coefficient
      (D = 0.1) and forecast via ETDRK4.  Since N(u) = 0 for a pure
      heat equation, ETDRK4 reduces to exact exponential propagation
      → machine-precision R² and mass conservation.

  **Experiment 2 – Full discovery + forecast pipeline.**
      Run WSINDy (Parts 2+3) on synthetic heat-equation data, then
      forecast from U(t=0).  The discovered coefficients may not be
      perfect (multi-collinearity between ``I:u`` and ``lap:u``), but
      the end-to-end workflow is demonstrated.

No plotting; metrics printed to stdout and saved to ``.npz``.
"""

from __future__ import annotations

import os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.wsindy.grid import GridSpec
from src.wsindy.model import WSINDyModel
from src.wsindy.test_functions import make_separable_psi
from src.wsindy.system import build_weak_system, make_query_indices, default_t_margin
from src.wsindy.fit import wsindy_fit_regression
from src.wsindy.forecast import wsindy_forecast
from src.wsindy.eval import rollout_metrics
from src.wsindy.integrators import split_linear_nonlinear
from src.wsindy.rhs import mass

# ── constants ───────────────────────────────────────────────────────
Lx = Ly = 2.0 * np.pi
nx = ny = 64
dx_g = Lx / nx
dy_g = Ly / ny
D_TRUE = 0.1
dt = 0.05


def _exact_heat(X, Y, t, D):
    """Multi-mode exact solution u_t = D Δu (periodic, with offset)."""
    return (
        1.0                                              # constant background
        + np.exp(-D * 2 * t) * np.cos(X) * np.cos(Y)    # k²=2
        + 0.8 * np.exp(-D * 4 * t) * np.cos(2 * X)      # k²=4
        + 0.5 * np.exp(-D * 16 * t) * np.cos(4 * X)     # k²=16
        + 0.6 * np.exp(-D * 9 * t) * np.cos(3 * Y)      # k²=9
    )


def _make_exact_model():
    """Construct WSINDyModel matching the true PDE u_t = 0.1 Δu."""
    return WSINDyModel(
        col_names=["lap:u"],
        w=np.array([D_TRUE]),
        active=np.array([True]),
        best_lambda=0.0,
        col_scale=np.ones(1),
        diagnostics={},
    )


# ═══════════════════════════════════════════════════════════════════
#  Experiment 1 — integrator accuracy with known model
# ═══════════════════════════════════════════════════════════════════

def experiment_1(X, Y, U_all, grid):
    sep = "═" * 60
    print(f"\n{sep}")
    print("  Experiment 1: ETDRK4 integrator accuracy (known model)")
    print(f"{sep}")

    model = _make_exact_model()
    n_steps = 60

    t0 = time.perf_counter()
    U_pred = wsindy_forecast(U_all[0], model, grid, n_steps=n_steps)
    elapsed = time.perf_counter() - t0

    U_true = U_all[: n_steps + 1]
    met = rollout_metrics(U_true, U_pred, grid)

    print(f"  Forecast: {n_steps} steps via ETDRK4 in {elapsed:.3f} s")
    print(f"  mean R²          = {met['r2_mean']:.10f}")
    print(f"  R²(t=end)        = {met['r2_t'][-1]:.10f}")
    print(f"  max rel L2       = {met['rel_l2_t'].max():.2e}")
    print(f"  max |mass drift| = {np.max(np.abs(met['mass_drift'])):.2e}")

    ok = met["r2_mean"] > 1.0 - 1e-10
    tag = "✓" if ok else "✗"
    print(f"  {tag}  R² ≈ 1.0 (exact linear propagation)")
    return met, ok


# ═══════════════════════════════════════════════════════════════════
#  Experiment 2 — full WSINDy discovery + forecast
# ═══════════════════════════════════════════════════════════════════

def experiment_2(X, Y, U_data, grid):
    sep = "═" * 60
    print(f"\n{sep}")
    print("  Experiment 2: WSINDy discovery → forecast")
    print(f"{sep}")

    T_data = U_data.shape[0]
    psi_bundle = make_separable_psi(
        grid, ellt=4, ellx=5, elly=5, pt=3, px=3, py=3,
    )

    library_terms = [
        ("I", "1"),
        ("I", "u"),
        ("dx", "u"),
        ("dy", "u"),
        ("lap", "u"),
        ("I", "u2"),
        ("I", "u3"),
    ]

    t_margin = default_t_margin(psi_bundle)
    query_idx = make_query_indices(
        T_data, nx, ny,
        t_margin=t_margin,
        stride_t=2, stride_x=2, stride_y=2,
    )

    b, G, col_names = build_weak_system(
        U_data, grid, psi_bundle, library_terms, query_idx,
    )
    print(f"  Weak system: b {b.shape}, G {G.shape}")

    lambdas = np.logspace(-3, 1, 50)
    model = wsindy_fit_regression(b, G, col_names, lambdas=lambdas)

    print(f"\n  Discovered (λ* = {model.best_lambda:.2e}):")
    print(model.summary())

    true_terms = {"lap:u": D_TRUE}
    print(f"  True model: {true_terms}")

    lap_idx = col_names.index("lap:u") if "lap:u" in col_names else -1
    if lap_idx >= 0 and model.active[lap_idx]:
        coeff = model.w[lap_idx]
        print(f"  ✓ lap:u selected, coeff = {coeff:.4f} (true = {D_TRUE})")
    else:
        print("  ✗ lap:u NOT selected")

    # Forecast
    n_forecast = 40
    lin, nonlin = split_linear_nonlinear(model)
    method = "etdrk4" if lin else "rk4"
    print(f"\n  Integrator: {method}")
    print(f"    Linear:    {lin}")
    print(f"    Nonlinear: {nonlin}")

    t0 = time.perf_counter()
    U_pred = wsindy_forecast(
        U_data[0], model, grid, n_steps=n_forecast, method=method,
    )
    elapsed = time.perf_counter() - t0

    t_fwd = np.arange(n_forecast + 1) * dt
    U_true_fwd = np.zeros((n_forecast + 1, nx, ny))
    for i, t in enumerate(t_fwd):
        U_true_fwd[i] = _exact_heat(X, Y, t, D_TRUE)

    met = rollout_metrics(U_true_fwd, U_pred, grid)

    print(f"  Forecasted {n_forecast} steps in {elapsed:.3f} s")
    print(f"  mean R²          = {met['r2_mean']:.6f}")
    print(f"  R²(t=0)          = {met['r2_t'][0]:.6f}")
    print(f"  R²(t=end)        = {met['r2_t'][-1]:.6f}")
    print(f"  rel L2(t=end)    = {met['rel_l2_t'][-1]:.4e}")
    print(f"  max |mass drift| = {np.max(np.abs(met['mass_drift'])):.2e}")

    ok = np.all(np.isfinite(U_pred)) and met["r2_t"][0] == 1.0
    tag = "✓" if ok else "✗"
    print(f"  {tag}  forecast finite, R²(0)=1.0")
    return met, model, ok


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    banner = "=" * 60
    print(f"{banner}")
    print("  WSINDy Part 4 Demo — Heat Equation Discovery + Forecast")
    print(f"{banner}")

    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    grid = GridSpec(dt=dt, dx=dx_g, dy=dy_g)

    T_total = 100
    t_arr = np.arange(T_total) * dt
    U_all = np.zeros((T_total, nx, ny))
    for i, t in enumerate(t_arr):
        U_all[i] = _exact_heat(X, Y, t, D_TRUE)

    print(f"  Data: T={T_total}, nx={nx}, ny={ny}, dt={dt}, D={D_TRUE}")

    met1, ok1 = experiment_1(X, Y, U_all, grid)
    met2, model, ok2 = experiment_2(X, Y, U_all, grid)

    # Save artefacts
    out_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "wsindy_part4_rollout.npz")
    np.savez_compressed(
        out_path,
        exp1_r2_t=met1["r2_t"],
        exp1_rel_l2_t=met1["rel_l2_t"],
        exp2_r2_t=met2["r2_t"],
        exp2_rel_l2_t=met2["rel_l2_t"],
        exp2_mass_true=met2["mass_true"],
        exp2_mass_pred=met2["mass_pred"],
        exp2_mass_drift=met2["mass_drift"],
        discovered_w=model.w,
        discovered_active=model.active,
        discovered_col_names=np.array(model.col_names),
    )
    print(f"\n  Saved: {out_path}")

    tag = "✓" if (ok1 and ok2) else "⚠"
    print(f"\n  {tag}  Demo complete — "
          f"{'all checks passed' if (ok1 and ok2) else 'see notes above'}.")
    print(f"{banner}")


if __name__ == "__main__":
    main()
