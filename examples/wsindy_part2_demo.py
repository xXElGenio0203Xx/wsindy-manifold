#!/usr/bin/env python3
"""
WSINDy Part 2 Demo — Build (b, G) on synthetic advecting Gaussian data.

Creates a drifting Gaussian blob on a periodic 2-D domain, builds the
test-function bundle, assembles the weak linear system, and prints
diagnostic information.

Usage:
    python examples/wsindy_part2_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wsindy import GridSpec, make_separable_psi
from wsindy.system import (
    build_weak_system,
    default_t_margin,
    make_query_indices,
)


def create_advecting_gaussian(
    T: int,
    nx: int,
    ny: int,
    Lx: float,
    Ly: float,
    dt: float,
    vx: float = 0.5,
    vy: float = 0.3,
    sigma: float = 2.0,
) -> np.ndarray:
    """Gaussian blob drifting at (vx, vy) on a periodic [0,Lx)×[0,Ly) domain."""
    dx = Lx / nx
    dy = Ly / ny
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    xx, yy = np.meshgrid(x, y, indexing="ij")

    x0, y0 = Lx / 2, Ly / 2
    U = np.empty((T, nx, ny), dtype=np.float64)

    for it in range(T):
        t = it * dt
        ddx = xx - x0 - vx * t
        ddy = yy - y0 - vy * t
        # Periodic wrapping via nearest-image convention
        ddx = ddx - Lx * np.round(ddx / Lx)
        ddy = ddy - Ly * np.round(ddy / Ly)
        U[it] = np.exp(-(ddx**2 + ddy**2) / (2.0 * sigma**2))

    return U


def main() -> None:
    # ── grid & data ─────────────────────────────────────────────────────
    T, nx, ny = 40, 32, 32
    Lx, Ly = 25.0, 25.0
    dt = 0.04
    dx, dy = Lx / nx, Ly / ny

    print("=" * 60)
    print("  WSINDy Part 2 Demo — Advecting Gaussian")
    print("=" * 60)
    print(f"  Grid:  T={T}, nx={nx}, ny={ny}")
    print(f"  Δ:     dt={dt}, dx={dx:.4f}, dy={dy:.4f}")

    U = create_advecting_gaussian(T, nx, ny, Lx, Ly, dt, vx=0.5, vy=0.3)
    print(f"  U:     shape={U.shape}, range=[{U.min():.4f}, {U.max():.4f}]")

    # ── test function ───────────────────────────────────────────────────
    grid = GridSpec(dt=dt, dx=dx, dy=dy,
                    periodic_space=True, periodic_time=False)

    ellx, elly, ellt = 4, 4, 3
    px, py, pt = 6, 6, 6
    psi_bundle = make_separable_psi(grid, ellx, elly, ellt, px, py, pt)
    ks = psi_bundle["psi"].shape
    print(f"\n  ψ kernel shape: {ks}")
    print(f"  ell=(t={ellt}, x={ellx}, y={elly}), p=(t={pt}, x={px}, y={py})")

    # ── query points ────────────────────────────────────────────────────
    margin = default_t_margin(psi_bundle)
    stride_t, stride_x, stride_y = 2, 2, 2
    qidx = make_query_indices(T, nx, ny,
                              stride_t=stride_t,
                              stride_x=stride_x,
                              stride_y=stride_y,
                              t_margin=margin)
    print(f"\n  Query points: K={qidx.shape[0]}  "
          f"(margin={margin}, strides=({stride_t},{stride_x},{stride_y}))")
    print(f"  t range: [{qidx[:, 0].min()}, {qidx[:, 0].max()}]")

    # ── library terms ───────────────────────────────────────────────────
    library_terms = [
        ("I",   "u"),    # ψ * u
        ("dx",  "u"),    # ψ_x * u    (advection x)
        ("dy",  "u"),    # ψ_y * u    (advection y)
        ("lap", "u"),    # (ψ_xx + ψ_yy) * u  (diffusion)
        ("I",   "u2"),   # ψ * u²     (nonlinear source)
    ]

    # ── build system ────────────────────────────────────────────────────
    b, G, col_names = build_weak_system(U, grid, psi_bundle,
                                         library_terms, qidx)

    print(f"\n  b shape: {b.shape}")
    print(f"  G shape: {G.shape}")
    print(f"  Columns: {col_names}")

    # ── diagnostics ─────────────────────────────────────────────────────
    print(f"\n  ‖b‖₂     = {np.linalg.norm(b):.6e}")
    for m, name in enumerate(col_names):
        print(f"  ‖G[:,{m}]‖ = {np.linalg.norm(G[:, m]):.6e}  ({name})")

    cond = np.linalg.cond(G)
    print(f"\n  cond(G)  = {cond:.4e}")
    print(f"  rank(G)  = {np.linalg.matrix_rank(G)} / {G.shape[1]}")

    # Quick least-squares preview (not MSTLS — just for sanity)
    w_ls, res, _, _ = np.linalg.lstsq(G, b, rcond=None)
    print(f"\n  Least-squares coefficients (for reference only):")
    for name, coeff in zip(col_names, w_ls):
        print(f"    {name:>8s} : {coeff:+.6e}")
    rel_res = np.linalg.norm(G @ w_ls - b) / np.linalg.norm(b)
    print(f"  Relative residual: {rel_res:.4e}")

    print(f"\n{'=' * 60}")
    print("  Demo complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
