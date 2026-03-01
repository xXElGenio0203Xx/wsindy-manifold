#!/usr/bin/env python
"""
WSINDy Part 4 — Unit Tests
===========================

Tests for spectral derivatives, strong-form RHS, linear/nonlinear
splitting, RK4, ETDRK4, forecast dispatch, and rollout metrics.
"""

from __future__ import annotations

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.wsindy.grid import GridSpec
from src.wsindy.model import WSINDyModel
from src.wsindy.operators import (
    fft_wavenumbers,
    grad_spectral,
    dxx_spectral,
    dyy_spectral,
    laplacian_spectral,
)
from src.wsindy.rhs import (
    apply_operator_pointwise,
    eval_feature_pointwise,
    mass,
    parse_term,
    wsindy_rhs,
)
from src.wsindy.integrators import (
    build_L_hat,
    etdrk4_integrate,
    rk4_integrate,
    rk4_step,
    split_linear_nonlinear,
    _can_use_etdrk4,
)
from src.wsindy.forecast import wsindy_forecast
from src.wsindy.eval import r2_per_snapshot, relative_l2, rollout_metrics

Lx = Ly = 2 * np.pi
nx = ny = 64
dx = Lx / nx
dy = Ly / ny


def _grid(dt: float = 0.01) -> GridSpec:
    return GridSpec(dt=dt, dx=dx, dy=dy)


def _xy():
    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    return np.meshgrid(x, y, indexing="ij")


def _make_model(
    col_names: list[str],
    w: list[float],
    active: list[bool] | None = None,
) -> WSINDyModel:
    w_arr = np.array(w, dtype=np.float64)
    if active is None:
        active_arr = w_arr != 0.0
    else:
        active_arr = np.array(active)
    return WSINDyModel(
        col_names=col_names,
        w=w_arr,
        active=active_arr,
        best_lambda=0.0,
        col_scale=np.ones(len(col_names)),
        diagnostics={},
    )


# ════════════════════════════════════════════════════════════════════
#  1. Spectral operators
# ════════════════════════════════════════════════════════════════════

def test_fft_wavenumbers():
    k = fft_wavenumbers(8, 1.0)
    expected = 2.0 * np.pi * np.fft.fftfreq(8, d=1.0)
    assert np.allclose(k, expected), f"wavenumbers mismatch"
    # k[0] should be 0
    assert k[0] == 0.0


def test_grad_spectral():
    X, Y = _xy()
    u = np.cos(X) * np.cos(Y)
    ux, uy = grad_spectral(u, dx, dy)
    ux_exact = -np.sin(X) * np.cos(Y)
    uy_exact = -np.cos(X) * np.sin(Y)
    assert np.allclose(ux, ux_exact, atol=1e-12), (
        f"ux max err = {np.max(np.abs(ux - ux_exact)):.2e}"
    )
    assert np.allclose(uy, uy_exact, atol=1e-12), (
        f"uy max err = {np.max(np.abs(uy - uy_exact)):.2e}"
    )


def test_laplacian_spectral():
    X, Y = _xy()
    u = np.cos(X) * np.cos(Y)
    lap_u = laplacian_spectral(u, dx, dy)
    # Δ(cos x cos y) = -cos x cos y + (-cos x cos y) = -2 cos x cos y
    lap_exact = -2.0 * np.cos(X) * np.cos(Y)
    assert np.allclose(lap_u, lap_exact, atol=1e-12), (
        f"lap max err = {np.max(np.abs(lap_u - lap_exact)):.2e}"
    )


def test_dxx_dyy_spectral():
    X, Y = _xy()
    u = np.cos(2 * X) * np.cos(3 * Y)
    uxx = dxx_spectral(u, dx, dy)
    uyy = dyy_spectral(u, dx, dy)
    uxx_exact = -4.0 * np.cos(2 * X) * np.cos(3 * Y)
    uyy_exact = -9.0 * np.cos(2 * X) * np.cos(3 * Y)
    assert np.allclose(uxx, uxx_exact, atol=1e-11), (
        f"uxx max err = {np.max(np.abs(uxx - uxx_exact)):.2e}"
    )
    assert np.allclose(uyy, uyy_exact, atol=1e-11), (
        f"uyy max err = {np.max(np.abs(uyy - uyy_exact)):.2e}"
    )


def test_lap_equals_dxx_plus_dyy():
    X, Y = _xy()
    u = np.sin(3 * X) * np.cos(2 * Y) + np.cos(X)
    lap = laplacian_spectral(u, dx, dy)
    uxx = dxx_spectral(u, dx, dy)
    uyy = dyy_spectral(u, dx, dy)
    assert np.allclose(lap, uxx + uyy, atol=1e-12)


# ════════════════════════════════════════════════════════════════════
#  2. RHS building blocks
# ════════════════════════════════════════════════════════════════════

def test_eval_feature_pointwise():
    u = np.array([[2.0, 3.0], [4.0, 5.0]])
    assert np.allclose(eval_feature_pointwise(u, "1"), np.ones_like(u))
    assert np.allclose(eval_feature_pointwise(u, "u"), u)
    assert np.allclose(eval_feature_pointwise(u, "u2"), u ** 2)
    assert np.allclose(eval_feature_pointwise(u, "u3"), u ** 3)


def test_parse_term():
    assert parse_term("lap:u") == ("lap", "u")
    assert parse_term("I:u2") == ("I", "u2")
    assert parse_term("dx:u") == ("dx", "u")


def test_wsindy_rhs_heat():
    """RHS for u_t = 0.1 Δu should equal 0.1 * laplacian_spectral(u)."""
    grid = _grid()
    model = _make_model(["lap:u"], [0.1])
    X, Y = _xy()
    u = np.cos(X) * np.cos(Y) + 1.0
    rhs = wsindy_rhs(u, model, grid)
    expected = 0.1 * laplacian_spectral(u, dx, dy)
    assert np.allclose(rhs, expected, atol=1e-14)


def test_mass():
    grid = _grid()
    u = np.ones((nx, ny))
    m = mass(u, grid)
    expected = nx * ny * dx * dy  # = (2π)²
    assert abs(m - expected) < 1e-10, f"mass = {m}, expected = {expected}"


# ════════════════════════════════════════════════════════════════════
#  3. Linear/nonlinear splitting
# ════════════════════════════════════════════════════════════════════

def test_split_linear_nonlinear():
    model = _make_model(
        ["I:1", "I:u", "lap:u", "I:u2", "dx:u3"],
        [1.0, -0.5, 0.1, 2.0, 0.3],
        [True, True, True, True, True],
    )
    lin, nonlin = split_linear_nonlinear(model)
    assert set(lin.keys()) == {"I:u", "lap:u"}
    assert set(nonlin.keys()) == {"I:1", "I:u2", "dx:u3"}
    assert lin["lap:u"] == 0.1
    assert nonlin["I:u2"] == 2.0


def test_can_use_etdrk4():
    assert _can_use_etdrk4({"lap:u": 0.1}) is True
    assert _can_use_etdrk4({"lap:u": 0.1, "I:u": -0.5}) is True
    assert _can_use_etdrk4({"dx:u": 1.0, "lap:u": 0.1}) is True
    assert _can_use_etdrk4({}) is False  # no linear terms → RK4


# ════════════════════════════════════════════════════════════════════
#  4. build_L_hat
# ════════════════════════════════════════════════════════════════════

def test_build_L_hat_laplacian():
    grid = _grid()
    L = build_L_hat({"lap:u": 0.1}, nx, ny, grid)
    kx = fft_wavenumbers(nx, dx)
    ky = fft_wavenumbers(ny, dy)
    KX = kx[:, None] * np.ones((1, ny))
    KY = np.ones((nx, 1)) * ky[None, :]
    expected = 0.1 * (-(KX ** 2 + KY ** 2))
    assert np.allclose(L, expected, atol=1e-14)
    # L should be real (purely dissipative)
    assert np.allclose(L.imag, 0.0, atol=1e-14)


def test_build_L_hat_advection():
    grid = _grid()
    L = build_L_hat({"dx:u": -2.0, "dy:u": -1.0}, nx, ny, grid)
    kx = fft_wavenumbers(nx, dx)
    ky = fft_wavenumbers(ny, dy)
    KX = kx[:, None] * np.ones((1, ny))
    KY = np.ones((nx, 1)) * ky[None, :]
    expected = -2.0 * 1j * KX + -1.0 * 1j * KY
    assert np.allclose(L, expected, atol=1e-14)
    # Purely imaginary (no dissipation)
    assert np.allclose(L.real, 0.0, atol=1e-14)


# ════════════════════════════════════════════════════════════════════
#  5. RK4 step
# ════════════════════════════════════════════════════════════════════

def test_rk4_step_exponential_decay():
    """du/dt = -u  →  u(dt) = exp(-dt)."""
    u = np.array([[1.0]])
    dt = 0.01
    u1 = rk4_step(u, dt, lambda u: -u)
    exact = np.exp(-dt)
    assert abs(u1[0, 0] - exact) < 1e-12


# ════════════════════════════════════════════════════════════════════
#  6. ETDRK4 — pure diffusion (exact)
# ════════════════════════════════════════════════════════════════════

def test_etdrk4_pure_diffusion():
    """For u_t = D Δu the ETDRK4 linear propagator is exact (N=0)."""
    D = 0.1
    dt = 0.05
    n_steps = 40
    grid = _grid(dt=dt)
    model = _make_model(["lap:u"], [D])

    X, Y = _xy()
    # Two-mode initial condition
    u0 = 2.0 + np.cos(X) * np.cos(Y) + 0.5 * np.cos(2 * X)

    traj = etdrk4_integrate(u0, n_steps, dt, grid, model)

    # Compare every snapshot with exact solution
    for k in range(n_steps + 1):
        t = k * dt
        u_exact = (
            2.0
            + np.exp(-D * 2 * t) * np.cos(X) * np.cos(Y)
            + 0.5 * np.exp(-D * 4 * t) * np.cos(2 * X)
        )
        err = np.max(np.abs(traj[k] - u_exact))
        assert err < 1e-11, (
            f"ETDRK4 pure diffusion: step {k}, max err = {err:.2e}"
        )


def test_etdrk4_mass_conservation():
    """Mass must be conserved for periodic diffusion (zero-mode is constant)."""
    D = 0.1
    dt = 0.05
    grid = _grid(dt=dt)
    model = _make_model(["lap:u"], [D])
    X, Y = _xy()
    u0 = 2.0 + np.cos(X) * np.cos(Y)
    traj = etdrk4_integrate(u0, 20, dt, grid, model)
    m0 = mass(u0, grid)
    for k in range(1, 21):
        mk = mass(traj[k], grid)
        assert abs(mk - m0) < 1e-10, (
            f"Mass drift at step {k}: {abs(mk - m0):.2e}"
        )


# ════════════════════════════════════════════════════════════════════
#  7. Forecast dispatch
# ════════════════════════════════════════════════════════════════════

def test_forecast_auto_selects_etdrk4():
    """auto mode should pick ETDRK4 when linear terms exist."""
    grid = _grid(dt=0.05)
    model = _make_model(["lap:u", "I:u2"], [0.1, -0.01])
    X, Y = _xy()
    u0 = 1.0 + 0.3 * np.cos(X)
    traj = wsindy_forecast(u0, model, grid, n_steps=5, method="auto")
    assert traj.shape == (6, nx, ny)
    # Should not explode
    assert np.all(np.isfinite(traj))


def test_forecast_rk4_fallback():
    """When no linear-in-u terms, auto falls back to RK4."""
    grid = _grid(dt=0.01)
    # Only nonlinear terms (u² growth)
    model = _make_model(["I:u2"], [0.01])
    u0 = np.ones((nx, ny)) * 0.5
    traj = wsindy_forecast(u0, model, grid, n_steps=3, method="auto")
    assert traj.shape == (4, nx, ny)
    assert np.all(np.isfinite(traj))


# ════════════════════════════════════════════════════════════════════
#  8. Rollout metrics
# ════════════════════════════════════════════════════════════════════

def test_r2_per_snapshot():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert abs(r2_per_snapshot(a, a) - 1.0) < 1e-15


def test_relative_l2():
    a = np.ones((4, 4))
    b = 1.1 * np.ones((4, 4))
    rl = relative_l2(a, b)
    expected = 0.1 / 1.0  # ||a-b||/||a|| = 0.1*4 / 1.0*4 = 0.1
    assert abs(rl - expected) < 1e-14


def test_rollout_metrics_perfect():
    grid = _grid()
    T = 10
    U = np.random.default_rng(42).standard_normal((T, nx, ny))
    m = rollout_metrics(U, U.copy(), grid)
    assert abs(m["r2_mean"] - 1.0) < 1e-14
    assert np.allclose(m["rel_l2_t"], 0.0, atol=1e-14)
    # Perfect forecast ⇒ mass curves match exactly
    assert np.allclose(m["mass_pred"], m["mass_true"], atol=1e-14)


# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    banner = "=" * 60
    print(f"{banner}\n  WSINDy Part 4 — Unit Tests\n{banner}\n")

    tests = [
        v for k, v in sorted(globals().items()) if k.startswith("test_")
    ]
    n_pass = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"  ✓ {name}")
            n_pass += 1
        except Exception as exc:
            print(f"  ✗ {name}: {exc}")

    print(f"\n{banner}")
    if n_pass == len(tests):
        print(f"  ALL {n_pass} TESTS PASSED")
    else:
        print(f"  {n_pass}/{len(tests)} passed")
    print(banner)
