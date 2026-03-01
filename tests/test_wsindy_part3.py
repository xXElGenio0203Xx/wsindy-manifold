#!/usr/bin/env python3
"""
Tests for WSINDy Part 3: preconditioning, MSTLS, and fit pipeline.

Verifies:
  1. precondition_columns produces unit-norm columns and correct unscaling.
  2. solve_ls matches np.linalg.lstsq.
  3. MSTLS recovers exact sparse support from noiseless data.
  4. MSTLS recovers correct support with modest noise.
  5. wsindy_fit_regression end-to-end produces valid WSINDyModel.
  6. Metrics (R², relative_l2) are correct for known data.

Usage:
    python tests/test_wsindy_part3.py
    pytest tests/test_wsindy_part3.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import numpy.testing as npt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wsindy.regression import mstls, precondition_columns, solve_ls
from wsindy.metrics import r2_score, wsindy_fit_metrics
from wsindy.fit import wsindy_fit_regression


def _header(msg: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {msg}")
    print(f"{'─' * 60}")


# ═══════════════════════════════════════════════════════════════════════════
# Preconditioning
# ═══════════════════════════════════════════════════════════════════════════

def test_precondition_columns() -> None:
    _header("precondition_columns: unit norm + unscaling")
    rng = np.random.default_rng(0)
    G = rng.standard_normal((50, 5))
    # Scale columns wildly
    G[:, 0] *= 1e4
    G[:, 3] *= 1e-3

    Gs, info = precondition_columns(G)
    col_scale = info["col_scale"]

    # Each column of Gs should have unit norm
    for j in range(5):
        npt.assert_allclose(np.linalg.norm(Gs[:, j]), 1.0, atol=1e-14)
    print("  ✓ All columns have unit ℓ₂ norm")

    # Unscaling: Gs * col_scale == G
    G_recovered = Gs * col_scale[np.newaxis, :]
    npt.assert_allclose(G_recovered, G, atol=1e-12)
    print("  ✓ G == Gs × col_scale  (round-trip)")

    # Unscaling coefficients:  G @ w = Gs @ w_s  →  w = w_s / col_scale
    w_true = rng.standard_normal(5)
    b = G @ w_true
    w_s = solve_ls(Gs, b)
    w_unscaled = w_s / col_scale
    npt.assert_allclose(w_unscaled, w_true, atol=1e-10)
    print("  ✓ w_unscaled = w_scaled / col_scale recovers true w")


# ═══════════════════════════════════════════════════════════════════════════
# Stable LS
# ═══════════════════════════════════════════════════════════════════════════

def test_solve_ls() -> None:
    _header("solve_ls: matches lstsq")
    rng = np.random.default_rng(1)
    G = rng.standard_normal((30, 4))
    w_true = np.array([1.0, 0.0, -2.5, 0.3])
    b = G @ w_true + rng.standard_normal(30) * 0.01

    w = solve_ls(G, b)
    npt.assert_allclose(w, w_true, atol=0.05)
    print(f"  ✓ Recovered w ≈ true w  (max err = {np.max(np.abs(w - w_true)):.4f})")


# ═══════════════════════════════════════════════════════════════════════════
# MSTLS — exact sparse recovery (noiseless)
# ═══════════════════════════════════════════════════════════════════════════

def test_mstls_exact_recovery() -> None:
    _header("MSTLS: exact sparse recovery (noiseless)")
    rng = np.random.default_rng(42)
    K, M = 200, 8
    G = rng.standard_normal((K, M))

    # True model: only columns 1 and 5 active
    w_true = np.zeros(M)
    w_true[1] = 2.0
    w_true[5] = -1.5
    b = G @ w_true  # exact, no noise

    # Precondition
    Gs, info = precondition_columns(G)
    col_scale = info["col_scale"]

    # Scale true w into preconditioned space for reference
    lambdas = np.logspace(-4, 0, 30)
    result = mstls(Gs, b, lambdas, max_iter=25)

    w_out = result["w"]
    active = result["active"]

    # Should recover exactly columns 1 and 5
    assert active[1], "Column 1 should be active"
    assert active[5], "Column 5 should be active"
    assert int(np.sum(active)) == 2, f"Expected 2 active, got {np.sum(active)}"
    print(f"  ✓ Recovered support: {np.where(active)[0].tolist()} == [1, 5]")

    # Unscale and check values
    w_phys = np.zeros(M)
    w_phys[active] = w_out[active] / col_scale[active]
    npt.assert_allclose(w_phys[1], 2.0, atol=1e-10)
    npt.assert_allclose(w_phys[5], -1.5, atol=1e-10)
    print(f"  ✓ Coefficients: w[1]={w_phys[1]:.6f}, w[5]={w_phys[5]:.6f}")

    # Residual should be ~0
    res = np.linalg.norm(b - G @ w_phys)
    assert res < 1e-10, f"Residual {res:.2e} too large"
    print(f"  ✓ Residual = {res:.2e}")


# ═══════════════════════════════════════════════════════════════════════════
# MSTLS — noisy recovery
# ═══════════════════════════════════════════════════════════════════════════

def test_mstls_noisy_recovery() -> None:
    _header("MSTLS: noisy recovery (5% noise)")
    rng = np.random.default_rng(7)
    K, M = 300, 6
    G = rng.standard_normal((K, M))

    w_true = np.zeros(M)
    w_true[0] = 3.0
    w_true[4] = -2.0
    b_clean = G @ w_true
    noise = rng.standard_normal(K) * 0.05 * np.linalg.norm(b_clean) / np.sqrt(K)
    b = b_clean + noise

    Gs, info = precondition_columns(G)
    result = mstls(Gs, b, np.logspace(-4, 0, 30))

    active = result["active"]
    # Should still pick columns 0 and 4
    assert active[0] and active[4], (
        f"Expected [0,4] active, got {np.where(active)[0].tolist()}"
    )
    # Might have small spurious terms, but support should include 0 and 4
    print(f"  ✓ Active set: {np.where(active)[0].tolist()} (expected ⊇ [0, 4])")

    # Unscale
    w_phys = np.zeros(M)
    w_phys[active] = result["w"][active] / info["col_scale"][active]
    print(f"  ✓ w[0]={w_phys[0]:.4f} (true 3.0), w[4]={w_phys[4]:.4f} (true -2.0)")
    npt.assert_allclose(w_phys[0], 3.0, atol=0.15)
    npt.assert_allclose(w_phys[4], -2.0, atol=0.15)


# ═══════════════════════════════════════════════════════════════════════════
# wsindy_fit_regression end-to-end
# ═══════════════════════════════════════════════════════════════════════════

def test_fit_regression_e2e() -> None:
    _header("wsindy_fit_regression: end-to-end")
    rng = np.random.default_rng(99)
    K, M = 200, 5
    G = rng.standard_normal((K, M))
    # Scale columns differently
    G[:, 0] *= 10.0
    G[:, 3] *= 0.01

    w_true = np.zeros(M)
    w_true[0] = 0.5
    w_true[3] = -100.0   # small column, large coeff
    b = G @ w_true

    col_names = [f"term_{i}" for i in range(M)]
    model = wsindy_fit_regression(b, G, col_names)

    # Check types
    assert hasattr(model, "w")
    assert hasattr(model, "active")
    assert hasattr(model, "col_scale")
    assert hasattr(model, "diagnostics")
    print(f"  ✓ WSINDyModel created, {model.n_active} active terms")
    print(f"  ✓ Active terms: {model.active_terms}")

    # Should recover columns 0 and 3
    assert model.active[0], "Column 0 should be active"
    assert model.active[3], "Column 3 should be active"

    # Physical coefficients
    npt.assert_allclose(model.w[0], 0.5, atol=1e-8)
    npt.assert_allclose(model.w[3], -100.0, atol=1e-6)
    print(f"  ✓ w[0]={model.w[0]:.6f} (true 0.5)")
    print(f"  ✓ w[3]={model.w[3]:.6f} (true -100.0)")

    # R² should be ~1
    r2 = model.diagnostics["r2"]
    assert r2 > 0.999, f"R² = {r2:.6f}, expected > 0.999"
    print(f"  ✓ R² = {r2:.6f}")

    # Summary should not crash
    s = model.summary()
    assert "WSINDyModel" in s
    print(f"  ✓ model.summary() works")


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def test_r2_score() -> None:
    _header("r2_score: known values")
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # Perfect prediction
    assert r2_score(y, y) == 1.0
    print("  ✓ R²(y, y) = 1.0")

    # Mean prediction → R² = 0
    y_mean = np.full_like(y, np.mean(y))
    npt.assert_allclose(r2_score(y, y_mean), 0.0, atol=1e-14)
    print("  ✓ R²(y, mean) = 0.0")

    # Worse than mean → R² < 0
    y_bad = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert r2_score(y, y_bad) < 0
    print(f"  ✓ R²(y, y_reversed) = {r2_score(y, y_bad):.4f} < 0")


def test_wsindy_fit_metrics() -> None:
    _header("wsindy_fit_metrics: consistency")
    rng = np.random.default_rng(3)
    G = rng.standard_normal((50, 4))
    w = np.array([1.0, 0.0, -0.5, 0.0])
    b = G @ w

    met = wsindy_fit_metrics(b, G, w)
    npt.assert_allclose(met["residual_norm"], 0.0, atol=1e-12)
    npt.assert_allclose(met["r2"], 1.0, atol=1e-12)
    npt.assert_allclose(met["relative_l2"], 0.0, atol=1e-12)
    assert met["sparsity"] == 2
    print("  ✓ Perfect fit: residual=0, R²=1, sparsity=2")


# ═══════════════════════════════════════════════════════════════════════════
# Dominant-balance thresholding spot check
# ═══════════════════════════════════════════════════════════════════════════

def test_dominant_balance_metric() -> None:
    _header("Dominant-balance: col_contrib = |w_i|·‖G_i‖₂/‖b‖₂")
    from wsindy.regression import _dominant_balance_mask

    K = 100
    G = np.zeros((K, 3))
    G[:, 0] = 2.0   # ‖G_0‖ = 2√K
    G[:, 1] = 0.5   # ‖G_1‖ = 0.5√K
    G[:, 2] = 1.0   # ‖G_2‖ = √K

    w = np.array([1.0, 0.01, 0.5])
    b_norm = 10.0

    # Manual: col_contrib[0] = 1.0 * 2*sqrt(100) / 10 = 2.0
    #         col_contrib[1] = 0.01 * 0.5*sqrt(100) / 10 = 0.005
    #         col_contrib[2] = 0.5 * sqrt(100) / 10 = 0.5
    sqrtK = np.sqrt(K)
    expected = np.array([
        1.0 * 2.0 * sqrtK / b_norm,
        0.01 * 0.5 * sqrtK / b_norm,
        0.5 * 1.0 * sqrtK / b_norm,
    ])
    print(f"  Expected col_contrib: {expected}")

    # With lambda=0.1: range [0.1, 10]
    #   col 0: 2.0 ∈ [0.1, 10] → keep
    #   col 1: 0.005 < 0.1 → drop
    #   col 2: 0.5 ∈ [0.1, 10] → keep
    mask = _dominant_balance_mask(w, G, b_norm, lambda_hat=0.1)
    assert mask[0] == True, f"col 0 should be kept (contrib={expected[0]:.3f})"
    assert mask[1] == False, f"col 1 should be dropped (contrib={expected[1]:.5f})"
    assert mask[2] == True, f"col 2 should be kept (contrib={expected[2]:.3f})"
    print(f"  ✓ mask = {mask.tolist()} == [True, False, True]")
    print(f"  ✓ Dominant-balance uses |w_i|·‖G_i‖₂/‖b‖₂  (not just |w_i|)")


# ═══════════════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("  WSINDy Part 3 — Regression Tests")
    print("=" * 60)

    test_precondition_columns()
    test_solve_ls()
    test_mstls_exact_recovery()
    test_mstls_noisy_recovery()
    test_fit_regression_e2e()
    test_r2_score()
    test_wsindy_fit_metrics()
    test_dominant_balance_metric()

    print(f"\n{'=' * 60}")
    print("  ALL TESTS PASSED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
