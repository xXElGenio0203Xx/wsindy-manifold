#!/usr/bin/env python3
"""
WSINDy Part 3 Demo — Sparse regression on synthetic PDE data.

Two experiments:
  1. Heat equation  u_t = D Δu  on periodic domain.
     Library includes extra decoy terms — MSTLS should select only ``lap:u``
     with coefficient ≈ D.

  2. Advection-diffusion  u_t = -vx u_x - vy u_y + D Δu.
     Should select ``dx:u``, ``dy:u``, ``lap:u``.

Usage:
    python examples/wsindy_part3_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wsindy import GridSpec, make_separable_psi
from wsindy.system import build_weak_system, default_t_margin, make_query_indices
from wsindy.fit import wsindy_fit_regression


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic data generators
# ═══════════════════════════════════════════════════════════════════════════

def heat_equation_data(
    T: int,
    nx: int,
    ny: int,
    Lx: float,
    Ly: float,
    dt: float,
    D: float,
    seed: int = 0,
) -> np.ndarray:
    """Generate u_t = D Δu via explicit Euler on periodic domain.

    Initial condition: sum of random Fourier modes.
    """
    dx = Lx / nx
    dy = Ly / ny

    rng = np.random.default_rng(seed)
    # Smooth random IC via low-frequency Fourier modes
    U0 = np.zeros((nx, ny))
    for _ in range(6):
        kx = rng.integers(1, 4)
        ky = rng.integers(1, 4)
        amp = rng.uniform(0.5, 2.0)
        phase = rng.uniform(0, 2 * np.pi)
        x = np.arange(nx) * dx
        y = np.arange(ny) * dy
        xx, yy = np.meshgrid(x, y, indexing="ij")
        U0 += amp * np.cos(2 * np.pi * kx * xx / Lx + phase) * \
              np.cos(2 * np.pi * ky * yy / Ly + phase)

    U = np.empty((T, nx, ny))
    U[0] = U0

    rx = D * dt / dx**2
    ry = D * dt / dy**2
    assert rx + ry < 0.5, f"Explicit Euler unstable: rx={rx:.4f}, ry={ry:.4f}"

    for t in range(1, T):
        u = U[t - 1]
        lap = (
            (np.roll(u, -1, axis=0) - 2 * u + np.roll(u, 1, axis=0)) / dx**2
            + (np.roll(u, -1, axis=1) - 2 * u + np.roll(u, 1, axis=1)) / dy**2
        )
        U[t] = u + dt * D * lap

    return U


def advection_diffusion_data(
    T: int,
    nx: int,
    ny: int,
    Lx: float,
    Ly: float,
    dt: float,
    vx: float,
    vy: float,
    D: float,
    seed: int = 0,
) -> np.ndarray:
    """Generate u_t = -vx u_x - vy u_y + D Δu via spectral method."""
    dx = Lx / nx
    dy = Ly / ny

    rng = np.random.default_rng(seed)
    # Smooth IC
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    xx, yy = np.meshgrid(x, y, indexing="ij")
    U0 = np.exp(-((xx - Lx / 2)**2 + (yy - Ly / 2)**2) / (2 * 3.0**2))

    # Wavenumbers
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="ij")

    U = np.empty((T, nx, ny))
    U[0] = U0
    u_hat = np.fft.fft2(U0)

    # Semi-implicit: diffusion exact, advection Euler
    for t in range(1, T):
        # Advection in Fourier: -vx i kx û - vy i ky û
        adv = -1j * (vx * KX + vy * KY) * u_hat
        # Diffusion decay factor
        decay = np.exp(-D * (KX**2 + KY**2) * dt)
        u_hat = (u_hat + dt * adv) * decay
        U[t] = np.real(np.fft.ifft2(u_hat))

    return U


# ═══════════════════════════════════════════════════════════════════════════
# Demo runner
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(
    name: str,
    U: np.ndarray,
    grid: GridSpec,
    library_terms: list,
    true_terms: dict,
) -> None:
    """Build weak system, fit regression, print results."""
    print(f"\n{'═' * 60}")
    print(f"  {name}")
    print(f"{'═' * 60}")

    T, nx, ny = U.shape
    print(f"  Data: T={T}, nx={nx}, ny={ny}")

    # Test function
    ellx, elly, ellt = 5, 5, 4
    px, py, pt = 6, 6, 6
    psi = make_separable_psi(grid, ellx, elly, ellt, px, py, pt)
    print(f"  ψ kernel: {psi['psi'].shape}")

    # Query points
    margin = default_t_margin(psi)
    qidx = make_query_indices(T, nx, ny, stride_t=2, stride_x=2, stride_y=2,
                              t_margin=margin)
    print(f"  Query points: {qidx.shape[0]}")

    # Build weak system
    b, G, col_names = build_weak_system(U, grid, psi, library_terms, qidx)
    print(f"  b: {b.shape}, G: {G.shape}")

    # Fit
    model = wsindy_fit_regression(b, G, col_names)

    # Print
    print(f"\n  {model.summary()}")
    print(f"\n  True model: {true_terms}")

    # Check discovered terms
    discovered = {n: model.w[i] for i, n in enumerate(col_names) if model.active[i]}
    print(f"  Discovered:  {discovered}")

    return model


def main() -> None:
    print("=" * 60)
    print("  WSINDy Part 3 Demo — Sparse PDE Discovery")
    print("=" * 60)

    Lx, Ly = 25.0, 25.0
    nx, ny = 32, 32
    dx, dy = Lx / nx, Ly / ny
    dt = 0.04
    grid = GridSpec(dt=dt, dx=dx, dy=dy, periodic_space=True, periodic_time=False)

    # ── Experiment 1: Heat equation ─────────────────────────────────────
    D_true = 0.5
    U_heat = heat_equation_data(T=60, nx=nx, ny=ny, Lx=Lx, Ly=Ly, dt=dt,
                                 D=D_true, seed=0)

    library_heat = [
        ("I",   "u"),    # decoy
        ("dx",  "u"),    # decoy
        ("dy",  "u"),    # decoy
        ("lap", "u"),    # ← true term
        ("I",   "u2"),   # decoy
        ("I",   "u3"),   # decoy
    ]

    model_heat = run_experiment(
        "Experiment 1: Heat Equation  u_t = D Δu",
        U_heat, grid, library_heat,
        true_terms={"lap:u": D_true},
    )

    # Verify
    assert model_heat.active[3], "lap:u should be active"
    print(f"\n  ✓ lap:u selected with coeff = {model_heat.w[3]:.4f} (true = {D_true})")

    # ── Experiment 2: Advection-diffusion ───────────────────────────────
    vx_true, vy_true, D_true2 = 2.0, 1.0, 0.3
    U_ad = advection_diffusion_data(T=60, nx=nx, ny=ny, Lx=Lx, Ly=Ly, dt=dt,
                                     vx=vx_true, vy=vy_true, D=D_true2, seed=1)

    library_ad = [
        ("I",   "1"),    # constant (decoy)
        ("I",   "u"),    # decoy
        ("dx",  "u"),    # ← advection x
        ("dy",  "u"),    # ← advection y
        ("dxx", "u"),    # decoy (redundant with lap)
        ("dyy", "u"),    # decoy (redundant with lap)
        ("lap", "u"),    # ← diffusion
        ("I",   "u2"),   # decoy
        ("I",   "u3"),   # decoy
    ]

    model_ad = run_experiment(
        "Experiment 2: Advection-Diffusion  u_t = -vx u_x - vy u_y + D Δu",
        U_ad, grid, library_ad,
        true_terms={"dx:u": -vx_true, "dy:u": -vy_true, "lap:u": D_true2},
    )

    # The weak form integrates by parts, so signs may differ from the
    # strong-form PDE.  What matters: dx:u and dy:u and lap:u are active.
    print(f"\n  Active terms: {model_ad.active_terms}")
    for term in ["dx:u", "dy:u", "lap:u"]:
        idx = model_ad.col_names.index(term)
        assert model_ad.active[idx], f"{term} should be active"
    print(f"  ✓ All three physics terms selected")

    # ── Deterministic sanity test ───────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("  Sanity Test: exact b = G @ w_true recovery")
    print(f"{'═' * 60}")
    rng = np.random.default_rng(12)
    K, M = 300, 10
    G_test = rng.standard_normal((K, M))
    w_test = np.zeros(M)
    w_test[2] = 1.5
    w_test[7] = -3.0
    b_test = G_test @ w_test

    col_names_test = [f"t{i}" for i in range(M)]
    model_test = wsindy_fit_regression(b_test, G_test, col_names_test)

    assert model_test.n_active == 2, f"Expected 2, got {model_test.n_active}"
    assert model_test.active[2] and model_test.active[7]
    np.testing.assert_allclose(model_test.w[2], 1.5, atol=1e-10)
    np.testing.assert_allclose(model_test.w[7], -3.0, atol=1e-10)
    print(f"  ✓ Recovered support [2, 7] with exact coefficients")
    print(f"  ✓ R² = {model_test.diagnostics['r2']:.10f}")

    print(f"\n{'=' * 60}")
    print("  Demo complete — all checks passed.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
