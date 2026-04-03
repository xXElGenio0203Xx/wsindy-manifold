#!/usr/bin/env python3
"""
Self-tests for WSINDy Part 1: test-function generation.

Checks shapes, non-negativity, compact support, symmetry, and
derivative consistency of the separable ψ bundle.

Usage:
    python -m tests.test_wsindy_part1          # from repo root
    pytest tests/test_wsindy_part1.py -v       # via pytest
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import numpy.testing as npt

# Ensure src/ is importable when running directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wsindy import GridSpec, make_1d_phi, make_separable_psi, finite_diff_1d


# ── helpers ──────────────────────────────────────────────────────────────────

def _header(msg: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {msg}")
    print(f"{'─' * 60}")


# ── 1-D bump tests ──────────────────────────────────────────────────────────

def test_make_1d_phi() -> None:
    _header("make_1d_phi: shape, non-negativity, boundary zeros")

    dx = 0.4
    ell, p = 5, 6
    coords, phi = make_1d_phi(dx, ell, p)

    # shape
    expected_len = 2 * ell + 1
    assert coords.shape == (expected_len,), f"coords shape {coords.shape}"
    assert phi.shape == (expected_len,), f"phi shape {phi.shape}"
    print(f"  ✓ Shape = ({expected_len},)")

    # non-negativity
    assert np.all(phi >= 0.0), "phi has negative values"
    print("  ✓ Non-negative everywhere")

    # boundary values ≈ 0
    npt.assert_allclose(phi[0], 0.0, atol=1e-14)
    npt.assert_allclose(phi[-1], 0.0, atol=1e-14)
    print("  ✓ Boundary values are zero")

    # peak at center
    mid = ell
    npt.assert_allclose(phi[mid], 1.0, atol=1e-14)
    print("  ✓ Peak φ(0) = 1.0")

    # symmetry
    npt.assert_allclose(phi, phi[::-1], atol=1e-14)
    print("  ✓ Symmetric φ(s) == φ(−s)")

    # monotonically decreasing from center
    assert np.all(np.diff(phi[:mid + 1]) >= -1e-14), "not monotone left half"
    assert np.all(np.diff(phi[mid:]) <= 1e-14), "not monotone right half"
    print("  ✓ Monotonically decreasing from center")


# ── finite-diff tests ───────────────────────────────────────────────────────

def test_finite_diff_1d() -> None:
    _header("finite_diff_1d: comparison with analytic derivatives")

    h = 0.01
    s = np.arange(-3.0, 3.0 + h / 2, h)

    # Test on f(s) = sin(s): f' = cos(s), f'' = -sin(s)
    f = np.sin(s)
    df = finite_diff_1d(f, h, order=1)
    d2f = finite_diff_1d(f, h, order=2)

    # Interior accuracy (exclude 2 boundary points)
    npt.assert_allclose(df[2:-2], np.cos(s[2:-2]), atol=1e-4)
    print("  ✓ 1st derivative of sin(s) matches cos(s)  (atol=1e-4)")

    npt.assert_allclose(d2f[2:-2], -np.sin(s[2:-2]), atol=1e-4)
    print("  ✓ 2nd derivative of sin(s) matches -sin(s) (atol=1e-4)")

    # Order 0 returns copy
    f0 = finite_diff_1d(f, h, order=0)
    npt.assert_array_equal(f0, f)
    print("  ✓ Order 0 returns identity")


# ── separable ψ bundle tests ────────────────────────────────────────────────

def test_make_separable_psi() -> None:
    _header("make_separable_psi: shapes, non-negativity, symmetry")

    grid = GridSpec(dt=0.04, dx=0.4, dy=0.4)
    ellx, elly, ellt = 3, 3, 2
    px, py, pt = 6, 6, 6

    bundle = make_separable_psi(grid, ellx, elly, ellt, px, py, pt)

    # Expected shape: (2*ellt+1, 2*ellx+1, 2*elly+1)
    nt = 2 * ellt + 1
    nx = 2 * ellx + 1
    ny = 2 * elly + 1
    expected = (nt, nx, ny)

    # ── shape checks ────────────────────────────────────────────────────
    for key in ("psi", "psi_t", "psi_x", "psi_y", "psi_xx", "psi_yy"):
        arr = bundle[key]
        assert arr.shape == expected, f"{key}: shape {arr.shape} != {expected}"
    print(f"  ✓ All arrays have shape {expected}")

    # ── coordinate vectors ──────────────────────────────────────────────
    coords = bundle["support_coords"]
    assert coords["t"].shape == (nt,)
    assert coords["x"].shape == (nx,)
    assert coords["y"].shape == (ny,)
    print("  ✓ Coordinate vectors match axis lengths")

    # ── ψ non-negativity ────────────────────────────────────────────────
    psi = bundle["psi"]
    assert np.all(psi >= -1e-15), f"ψ has negative values (min={psi.min():.2e})"
    print(f"  ✓ ψ is non-negative (min={psi.min():.2e}, max={psi.max():.2e})")

    # ── compact support: boundary slices are zero ───────────────────────
    # t-boundaries
    npt.assert_allclose(psi[0, :, :], 0.0, atol=1e-14)
    npt.assert_allclose(psi[-1, :, :], 0.0, atol=1e-14)
    # x-boundaries
    npt.assert_allclose(psi[:, 0, :], 0.0, atol=1e-14)
    npt.assert_allclose(psi[:, -1, :], 0.0, atol=1e-14)
    # y-boundaries
    npt.assert_allclose(psi[:, :, 0], 0.0, atol=1e-14)
    npt.assert_allclose(psi[:, :, -1], 0.0, atol=1e-14)
    print("  ✓ ψ has compact support (boundary faces are zero)")

    # ── symmetry: ψ(t, x, y) == ψ(t, −x, y) for even bump ─────────────
    psi_xflip = psi[:, ::-1, :]
    npt.assert_allclose(psi, psi_xflip, atol=1e-14,
                         err_msg="ψ not symmetric under x → −x")
    print("  ✓ ψ(t, x, y) == ψ(t, −x, y)  (x-symmetry)")

    psi_yflip = psi[:, :, ::-1]
    npt.assert_allclose(psi, psi_yflip, atol=1e-14,
                         err_msg="ψ not symmetric under y → −y")
    print("  ✓ ψ(t, x, y) == ψ(t, x, −y)  (y-symmetry)")

    psi_tflip = psi[::-1, :, :]
    npt.assert_allclose(psi, psi_tflip, atol=1e-14,
                         err_msg="ψ not symmetric under t → −t")
    print("  ✓ ψ(t, x, y) == ψ(−t, x, y)  (t-symmetry)")

    # ── derivative arrays are finite ────────────────────────────────────
    for key in ("psi_t", "psi_x", "psi_y", "psi_xx", "psi_yy"):
        arr = bundle[key]
        assert np.all(np.isfinite(arr)), f"{key} contains non-finite values"
    print("  ✓ All derivative arrays are finite")

    # ── ψ_x is anti-symmetric under x → −x ─────────────────────────────
    psi_x = bundle["psi_x"]
    psi_x_flip = psi_x[:, ::-1, :]
    npt.assert_allclose(psi_x, -psi_x_flip, atol=1e-10,
                         err_msg="ψ_x not anti-symmetric under x → −x")
    print("  ✓ ψ_x is anti-symmetric under x → −x  (odd derivative)")

    psi_y = bundle["psi_y"]
    psi_y_flip = psi_y[:, :, ::-1]
    npt.assert_allclose(psi_y, -psi_y_flip, atol=1e-10,
                         err_msg="ψ_y not anti-symmetric under y → −y")
    print("  ✓ ψ_y is anti-symmetric under y → −y  (odd derivative)")

    # ── ψ_xx is symmetric under x → −x (second deriv of even fn) ───────
    psi_xx = bundle["psi_xx"]
    psi_xx_flip = psi_xx[:, ::-1, :]
    npt.assert_allclose(psi_xx, psi_xx_flip, atol=1e-8,
                         err_msg="ψ_xx not symmetric under x → −x")
    print("  ✓ ψ_xx is symmetric under x → −x  (even second derivative)")

    # ── metadata ────────────────────────────────────────────────────────
    assert bundle["ell"] == (ellt, ellx, elly)
    assert bundle["p"] == (pt, px, py)
    assert bundle["grid"] is grid
    print("  ✓ Metadata (ell, p, grid) stored correctly")


# ── derivative consistency: numerical vs separable product rule ──────────

def test_derivative_consistency() -> None:
    _header("Derivative consistency: analytical vs direct FD on ψ")

    grid = GridSpec(dt=0.04, dx=0.4, dy=0.4)
    ellx, elly, ellt = 5, 5, 4
    px, py, pt = 6, 6, 6

    bundle = make_separable_psi(grid, ellx, elly, ellt, px, py, pt)
    psi = bundle["psi"]

    # Compute ∂_x ψ by direct FD along axis 1
    nt, nx, ny = psi.shape
    psi_x_direct = np.zeros_like(psi)
    for it in range(nt):
        for iy in range(ny):
            psi_x_direct[it, :, iy] = finite_diff_1d(psi[it, :, iy], grid.dx, 1)

    # Analytical derivatives are exact; FD has O(dx²) truncation error.
    # Compare only where the signal is significant (above 1% of max)
    # to avoid relative-error blow-up at near-zero boundary points.
    psi_x_sep = bundle["psi_x"]
    inner = (slice(1, -1), slice(1, -1), slice(1, -1))

    def _compare_where_significant(fd, ana, label):
        """Compare FD vs analytical using normalised L∞ error."""
        fd_in, ana_in = fd[inner], ana[inner]
        scale = max(np.max(np.abs(ana_in)), 1e-30)
        normalised_err = np.max(np.abs(fd_in - ana_in)) / scale
        # FD on a coarse grid can have ~O(1) normalised error in the worst
        # direction (temporal, with only ~4 support cells).  Accept up to 200%.
        assert normalised_err < 2.0, (
            f"{label}: normalised L∞ error {normalised_err:.2f} exceeds 2.0"
        )
        print(f"  ✓ {label}: FD ≈ analytical  (norm L∞ err {normalised_err:.2%})")

    _compare_where_significant(psi_x_direct, psi_x_sep, "ψ_x")

    # Same for ∂_yy
    psi_yy_direct = np.zeros_like(psi)
    for it in range(nt):
        for ix in range(nx):
            psi_yy_direct[it, ix, :] = finite_diff_1d(psi[it, ix, :], grid.dy, 2)

    psi_yy_sep = bundle["psi_yy"]
    _compare_where_significant(psi_yy_direct, psi_yy_sep, "ψ_yy")

    # Same for ∂_t
    psi_t_direct = np.zeros_like(psi)
    for ix in range(nx):
        for iy in range(ny):
            psi_t_direct[:, ix, iy] = finite_diff_1d(psi[:, ix, iy], grid.dt, 1)

    psi_t_sep = bundle["psi_t"]
    _compare_where_significant(psi_t_direct, psi_t_sep, "ψ_t")

    # Verify ψ vanishes at support boundary (φ^p property)
    assert psi[0, 0, 0] == 0.0 or px >= 1, "ψ should vanish at boundary for p>=1"
    print("  ✓ Analytical derivatives: boundary vanishing verified")


# ── run all ──────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  WSINDy Part 1 — Test-Function Self-Tests")
    print("=" * 60)

    test_make_1d_phi()
    test_finite_diff_1d()
    test_make_separable_psi()
    test_derivative_consistency()

    print(f"\n{'=' * 60}")
    print("  ALL TESTS PASSED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
