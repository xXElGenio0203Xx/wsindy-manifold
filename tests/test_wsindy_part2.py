#!/usr/bin/env python3
"""
Tests for WSINDy Part 2: FFT convolution and weak-system assembly.

Verifies:
  1. fft_convolve3d_same matches a slow 6-nested-loop reference.
  2. Query-index generation respects margins and strides.
  3. eval_feature / get_kernel return correct values.
  4. build_weak_system produces correctly shaped outputs.

Usage:
    python tests/test_wsindy_part2.py
    pytest tests/test_wsindy_part2.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import numpy.testing as npt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wsindy import GridSpec, make_separable_psi
from wsindy.system import (
    build_weak_system,
    default_t_margin,
    eval_feature,
    fft_convolve3d_same,
    get_kernel,
    make_query_indices,
)


# ═══════════════════════════════════════════════════════════════════════════
# Slow reference convolution (trivially correct, O(T·nx·ny·kt·kx·ky))
# ═══════════════════════════════════════════════════════════════════════════

def _slow_convolve3d_same(
    data: np.ndarray,
    kernel: np.ndarray,
    periodic: tuple[bool, bool, bool],
) -> np.ndarray:
    """Brute-force discrete convolution with centred kernel.

    result[i] = Σ_j kernel[j] · data[i − j + center]
    """
    T, nx, ny = data.shape
    kt, kx, ky = kernel.shape
    ct, cx, cy = kt // 2, kx // 2, ky // 2
    result = np.zeros((T, nx, ny), dtype=np.float64)

    for it in range(T):
        for ix in range(nx):
            for iy in range(ny):
                s = 0.0
                for jt in range(kt):
                    for jx in range(kx):
                        for jy in range(ky):
                            st = it - jt + ct
                            sx = ix - jx + cx
                            sy = iy - jy + cy
                            # time
                            if periodic[0]:
                                st = st % T
                            elif st < 0 or st >= T:
                                continue
                            # x
                            if periodic[1]:
                                sx = sx % nx
                            elif sx < 0 or sx >= nx:
                                continue
                            # y
                            if periodic[2]:
                                sy = sy % ny
                            elif sy < 0 or sy >= ny:
                                continue
                            s += kernel[jt, jx, jy] * data[st, sx, sy]
                result[it, ix, iy] = s
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _header(msg: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {msg}")
    print(f"{'─' * 60}")


def _random_data_kernel(
    T: int = 6, nx: int = 5, ny: int = 4,
    kt: int = 3, kx: int = 3, ky: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((T, nx, ny))
    kernel = rng.standard_normal((kt, kx, ky))
    return data, kernel


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_conv_fully_periodic() -> None:
    _header("fft_convolve3d_same: fully periodic")
    data, kernel = _random_data_kernel()
    periodic = (True, True, True)

    fast = fft_convolve3d_same(data, kernel, periodic)
    slow = _slow_convolve3d_same(data, kernel, periodic)

    assert fast.shape == data.shape, f"shape {fast.shape} != {data.shape}"
    npt.assert_allclose(fast, slow, atol=1e-12,
                        err_msg="Fully periodic: FFT ≠ slow reference")
    print(f"  ✓ Fully periodic matches slow reference  (max Δ = "
          f"{np.max(np.abs(fast - slow)):.2e})")


def test_conv_periodic_space_nonperiodic_time() -> None:
    _header("fft_convolve3d_same: periodic space, non-periodic time")
    data, kernel = _random_data_kernel()
    periodic = (False, True, True)

    fast = fft_convolve3d_same(data, kernel, periodic)
    slow = _slow_convolve3d_same(data, kernel, periodic)

    assert fast.shape == data.shape
    npt.assert_allclose(fast, slow, atol=1e-12,
                        err_msg="Mixed periodic: FFT ≠ slow reference")
    print(f"  ✓ Periodic-space / non-periodic-time matches  (max Δ = "
          f"{np.max(np.abs(fast - slow)):.2e})")


def test_conv_fully_nonperiodic() -> None:
    _header("fft_convolve3d_same: fully non-periodic")
    data, kernel = _random_data_kernel()
    periodic = (False, False, False)

    fast = fft_convolve3d_same(data, kernel, periodic)
    slow = _slow_convolve3d_same(data, kernel, periodic)

    assert fast.shape == data.shape
    npt.assert_allclose(fast, slow, atol=1e-12,
                        err_msg="Fully non-periodic: FFT ≠ slow reference")
    print(f"  ✓ Fully non-periodic matches  (max Δ = "
          f"{np.max(np.abs(fast - slow)):.2e})")


def test_conv_identity_kernel() -> None:
    _header("fft_convolve3d_same: identity (delta) kernel")
    rng = np.random.default_rng(99)
    data = rng.standard_normal((8, 6, 6))
    # delta kernel: 1 at center, 0 elsewhere
    kernel = np.zeros((3, 3, 3))
    kernel[1, 1, 1] = 1.0

    for periodic in [(True, True, True), (False, True, True)]:
        result = fft_convolve3d_same(data, kernel, periodic)
        npt.assert_allclose(result, data, atol=1e-14,
                            err_msg=f"Delta kernel failed for periodic={periodic}")

    print("  ✓ Delta kernel returns input unchanged (periodic & mixed)")


def test_conv_larger_kernel() -> None:
    _header("fft_convolve3d_same: larger kernel (5×5×5)")
    data, _ = _random_data_kernel(T=10, nx=8, ny=7)
    rng = np.random.default_rng(7)
    kernel = rng.standard_normal((5, 5, 5))
    periodic = (False, True, True)

    fast = fft_convolve3d_same(data, kernel, periodic)
    slow = _slow_convolve3d_same(data, kernel, periodic)

    npt.assert_allclose(fast, slow, atol=1e-11,
                        err_msg="Larger kernel: FFT ≠ slow")
    print(f"  ✓ 5×5×5 kernel matches  (max Δ = "
          f"{np.max(np.abs(fast - slow)):.2e})")


# ── query indices ────────────────────────────────────────────────────────

def test_query_indices() -> None:
    _header("make_query_indices: shapes and margins")
    T, nx, ny = 20, 16, 16

    # No margin, stride 1
    idx = make_query_indices(T, nx, ny)
    assert idx.shape == (T * nx * ny, 3)
    print(f"  ✓ Full grid: {idx.shape[0]} points")

    # With margin and strides
    idx2 = make_query_indices(T, nx, ny, stride_t=2, stride_x=4, stride_y=4,
                              t_margin=3)
    nt_expected = len(range(3, T - 3, 2))
    nx_expected = len(range(0, nx, 4))
    ny_expected = len(range(0, ny, 4))
    assert idx2.shape == (nt_expected * nx_expected * ny_expected, 3), \
        f"shape {idx2.shape}"
    assert idx2[:, 0].min() >= 3
    assert idx2[:, 0].max() <= T - 4  # last valid with margin=3
    print(f"  ✓ Strided + margin: {idx2.shape[0]} points, "
          f"t ∈ [{idx2[:, 0].min()}, {idx2[:, 0].max()}]")


def test_default_margin() -> None:
    _header("default_t_margin")
    bundle = {"ell": (4, 3, 3)}
    assert default_t_margin(bundle) == 4
    print("  ✓ default_t_margin returns ellt = 4")


# ── features ─────────────────────────────────────────────────────────────

def test_eval_feature() -> None:
    _header("eval_feature")
    rng = np.random.default_rng(0)
    U = rng.standard_normal((5, 4, 3))

    npt.assert_allclose(eval_feature(U, "1"), np.ones_like(U))
    npt.assert_allclose(eval_feature(U, "u"), U)
    npt.assert_allclose(eval_feature(U, "u2"), U ** 2)
    npt.assert_allclose(eval_feature(U, "u3"), U ** 3)
    print("  ✓ All features (1, u, u², u³) correct")


# ── kernel extraction ────────────────────────────────────────────────────

def test_get_kernel() -> None:
    _header("get_kernel")
    grid = GridSpec(dt=0.1, dx=0.5, dy=0.5)
    bundle = make_separable_psi(grid, ellx=2, elly=2, ellt=2,
                                px=4, py=4, pt=4)

    for op, key in [("I", "psi"), ("dx", "psi_x"), ("dxx", "psi_xx"),
                    ("dy", "psi_y"), ("dyy", "psi_yy")]:
        npt.assert_array_equal(get_kernel(bundle, op), bundle[key])

    lap = get_kernel(bundle, "lap")
    npt.assert_allclose(lap, bundle["psi_xx"] + bundle["psi_yy"])
    print("  ✓ All operators (I, dx, dy, dxx, dyy, lap) correct")


# ── build_weak_system shapes ─────────────────────────────────────────────

def test_build_weak_system_shapes() -> None:
    _header("build_weak_system: shapes and column names")
    grid = GridSpec(dt=0.1, dx=0.5, dy=0.5)
    T, nx, ny = 20, 12, 12
    rng = np.random.default_rng(1)
    U = rng.standard_normal((T, nx, ny))

    bundle = make_separable_psi(grid, ellx=2, elly=2, ellt=2,
                                px=4, py=4, pt=4)

    terms = [("I", "u"), ("dxx", "u"), ("dyy", "u"), ("lap", "u"),
             ("I", "u2"), ("I", "u3")]
    margin = default_t_margin(bundle)
    qidx = make_query_indices(T, nx, ny, stride_t=2, stride_x=2,
                              stride_y=2, t_margin=margin)

    b, G, names = build_weak_system(U, grid, bundle, terms, qidx)

    K = qidx.shape[0]
    M = len(terms)
    assert b.shape == (K,), f"b shape {b.shape} != ({K},)"
    assert G.shape == (K, M), f"G shape {G.shape} != ({K}, {M})"
    assert len(names) == M
    assert names == ["I:u", "dxx:u", "dyy:u", "lap:u", "I:u2", "I:u3"]
    print(f"  ✓ b shape {b.shape}, G shape {G.shape}")
    print(f"  ✓ Column names: {names}")

    # G columns for lap:u should equal dxx:u + dyy:u
    npt.assert_allclose(G[:, 3], G[:, 1] + G[:, 2], atol=1e-12,
                        err_msg="lap:u ≠ dxx:u + dyy:u")
    print("  ✓ G[:, lap:u] == G[:, dxx:u] + G[:, dyy:u]")


# ═══════════════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("  WSINDy Part 2 — Weak System Tests")
    print("=" * 60)

    test_conv_fully_periodic()
    test_conv_periodic_space_nonperiodic_time()
    test_conv_fully_nonperiodic()
    test_conv_identity_kernel()
    test_conv_larger_kernel()
    test_query_indices()
    test_default_margin()
    test_eval_feature()
    test_get_kernel()
    test_build_weak_system_shapes()

    print(f"\n{'=' * 60}")
    print("  ALL TESTS PASSED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
