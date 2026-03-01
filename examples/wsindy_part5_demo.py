#!/usr/bin/env python
"""
WSINDy Part 5 Demo — Automated Model Selection
=================================================

Generates exact heat-equation data (u_t = D Δu) on a periodic 64×64 grid,
then sweeps test-function half-widths ℓ to find the configuration that
best recovers the true PDE.

Demonstrates:
  1. ``default_ell_grid`` for automatic ℓ proposals
  2. ``wsindy_model_selection`` full sweep
  3. Diagnostics table, Pareto frontier, and best-model summary
  4. ETDRK4 forecast from best selected model → rollout R²

Saves the diagnostics table and metrics to ``.npz``.
"""

from __future__ import annotations

import os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.wsindy.grid import GridSpec
from src.wsindy.select import default_ell_grid, wsindy_model_selection
from src.wsindy.forecast import wsindy_forecast
from src.wsindy.eval import rollout_metrics
from src.wsindy.integrators import split_linear_nonlinear

# ── constants ───────────────────────────────────────────────────────
Lx = Ly = 2.0 * np.pi
nx = ny = 64
dx_g = Lx / nx
dy_g = Ly / ny
D_TRUE = 0.1
dt = 0.05
T_DATA = 80
T_FORECAST = 40


def _exact_heat(X, Y, t, D):
    return (
        1.0
        + np.exp(-D * 2 * t) * np.cos(X) * np.cos(Y)
        + 0.8 * np.exp(-D * 4 * t) * np.cos(2 * X)
        + 0.5 * np.exp(-D * 16 * t) * np.cos(4 * X)
        + 0.6 * np.exp(-D * 9 * t) * np.cos(3 * Y)
    )


def main():
    banner = "=" * 60
    sep = "─" * 60
    print(f"{banner}")
    print("  WSINDy Part 5 Demo — Model Selection across ℓ")
    print(f"{banner}\n")

    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    grid = GridSpec(dt=dt, dx=dx_g, dy=dy_g)

    # ── 1. Generate data ───────────────────────────────────────────
    U = np.zeros((T_DATA, nx, ny))
    for i in range(T_DATA):
        U[i] = _exact_heat(X, Y, i * dt, D_TRUE)
    print(f"  Data: T={T_DATA}, nx={nx}, ny={ny}, dt={dt}, D={D_TRUE}")

    # ── 2. Library ─────────────────────────────────────────────────
    library_terms = [
        ("I", "1"),
        ("I", "u"),
        ("dx", "u"),
        ("dy", "u"),
        ("lap", "u"),
        ("I", "u2"),
        ("I", "u3"),
    ]

    # ── 3. ℓ sweep grid ───────────────────────────────────────────
    ell_grid = default_ell_grid(T_DATA, nx, ny, n_points=6)
    # Also add some manually-chosen small ℓ values
    ell_grid = [(2, 3, 3), (3, 4, 4), (3, 5, 5), (4, 6, 6),
                (5, 7, 7), (6, 8, 8)] + ell_grid
    # Deduplicate
    ell_grid = list(dict.fromkeys(ell_grid))

    print(f"  ℓ sweep: {len(ell_grid)} configurations")
    for e in ell_grid:
        print(f"    ({e[0]}, {e[1]}, {e[2]})")

    # ── 4. Run model selection ─────────────────────────────────────
    print(f"\n{sep}")
    print("  Running model selection sweep...")
    print(f"{sep}")

    t0 = time.perf_counter()
    result = wsindy_model_selection(
        U, grid, library_terms, ell_grid,
        p=(2, 2, 2),
        lambdas=np.logspace(-3, 1, 40),
        alpha=0.1,
        beta=0.01,
        verbose=True,
    )
    total_time = time.perf_counter() - t0

    # ── 5. Results ─────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  Results")
    print(f"{sep}")
    print(f"  Total time: {total_time:.2f} s")
    print(f"  Trials completed: {len(result.trials)}")
    print(f"  Pareto-optimal: {len(result.pareto)}")

    print(f"\n  Top-5 ranking:")
    print(result.summary(top_k=5))

    # ── 6. Pareto frontier ─────────────────────────────────────────
    print(f"\n  Pareto frontier (loss vs sparsity):")
    for t in result.pareto:
        tag = " ★" if t is result.best else ""
        print(f"    ℓ=({t.ell[0]},{t.ell[1]},{t.ell[2]})  "
              f"active={t.n_active}  loss={t.normalised_loss:.4e}{tag}")

    # ── 7. Best model details ──────────────────────────────────────
    best = result.best
    model = best.model
    print(f"\n{sep}")
    print(f"  Best model: ℓ=({best.ell[0]},{best.ell[1]},{best.ell[2]})")
    print(f"{sep}")
    print(model.summary())

    true_D = D_TRUE
    if "lap:u" in model.active_terms:
        idx = model.col_names.index("lap:u")
        coeff = model.w[idx]
        print(f"\n  ✓ lap:u selected, coeff = {coeff:.6f} (true = {true_D})")
        print(f"    relative error = {abs(coeff - true_D)/true_D:.2%}")
    else:
        print("\n  ✗ lap:u NOT selected")

    # ── 8. Forecast with best model ────────────────────────────────
    print(f"\n{sep}")
    print("  Forecasting with best model...")
    print(f"{sep}")

    lin, nonlin = split_linear_nonlinear(model)
    method = "etdrk4" if lin else "rk4"
    print(f"  Method: {method}")

    U_pred = wsindy_forecast(U[0], model, grid, n_steps=T_FORECAST)

    U_true_fwd = np.zeros((T_FORECAST + 1, nx, ny))
    for i in range(T_FORECAST + 1):
        U_true_fwd[i] = _exact_heat(X, Y, i * dt, D_TRUE)

    met = rollout_metrics(U_true_fwd, U_pred, grid)

    print(f"  Rollout ({T_FORECAST} steps):")
    print(f"    mean R²          = {met['r2_mean']:.6f}")
    print(f"    R²(t=0)          = {met['r2_t'][0]:.6f}")
    print(f"    R²(t=end)        = {met['r2_t'][-1]:.6f}")
    print(f"    max rel L2       = {met['rel_l2_t'].max():.4e}")
    print(f"    max |mass drift| = {np.max(np.abs(met['mass_drift'])):.2e}")

    # ── 9. Save ────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "wsindy_part5_selection.npz")

    table = result.table()
    np.savez_compressed(
        out_path,
        # Diagnostics table (as structured arrays)
        ell_t=np.array([r["ellt"] for r in table]),
        ell_x=np.array([r["ellx"] for r in table]),
        ell_y=np.array([r["elly"] for r in table]),
        n_active=np.array([r["n_active"] for r in table]),
        normalised_loss=np.array([r["normalised_loss"] for r in table]),
        r2_weak=np.array([r["r2_weak"] for r in table]),
        score=np.array([r["score"] for r in table]),
        # Best model
        best_ell=np.array(best.ell),
        best_w=model.w,
        best_active=model.active,
        best_col_names=np.array(model.col_names),
        # Rollout
        r2_t=met["r2_t"],
        rel_l2_t=met["rel_l2_t"],
        mass_drift=met["mass_drift"],
    )
    print(f"\n  Saved: {out_path}")
    print(f"\n{banner}")
    print("  Demo complete.")
    print(f"{banner}")


if __name__ == "__main__":
    main()
