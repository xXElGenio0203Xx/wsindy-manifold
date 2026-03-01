#!/usr/bin/env python
"""
WSINDy Part 5 — Model Selection Unit Tests
============================================

Tests for composite scoring, Pareto frontier, default ℓ grid,
TrialResult / SelectionResult containers, and the full
``wsindy_model_selection`` sweep on synthetic data.
"""

from __future__ import annotations

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.wsindy.grid import GridSpec
from src.wsindy.model import WSINDyModel
from src.wsindy.select import (
    SelectionResult,
    TrialResult,
    _composite_score,
    _pareto_frontier,
    default_ell_grid,
    wsindy_model_selection,
)


# ── helpers ─────────────────────────────────────────────────────────

Lx = Ly = 2.0 * np.pi
nx = ny = 32
dx = Lx / nx
dy = Ly / ny
dt = 0.05


def _grid():
    return GridSpec(dt=dt, dx=dx, dy=dy)


def _make_dummy_trial(
    ell, score, n_active, nloss, *, model=None,
) -> TrialResult:
    if model is None:
        model = WSINDyModel(
            col_names=["lap:u"],
            w=np.array([0.1]),
            active=np.array([True]),
            best_lambda=0.01,
            col_scale=np.ones(1),
            diagnostics={"r2": 0.99, "normalised_loss": nloss},
        )
    return TrialResult(
        ell=ell,
        p=(2, 2, 2),
        stride=(2, 2, 2),
        model=model,
        n_query=1000,
        normalised_loss=nloss,
        r2_weak=0.99,
        n_active=n_active,
        best_lambda=0.01,
        condition_number=10.0,
        composite_score=score,
        elapsed_s=0.1,
    )


def _heat_data(T_total=60):
    """Pure heat equation u_t = 0.1 Δu on [0,2π]²."""
    D = 0.1
    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    U = np.zeros((T_total, nx, ny))
    for i in range(T_total):
        t = i * dt
        U[i] = (
            1.0
            + np.exp(-D * 2 * t) * np.cos(X) * np.cos(Y)
            + 0.8 * np.exp(-D * 4 * t) * np.cos(2 * X)
            + 0.5 * np.exp(-D * 16 * t) * np.cos(4 * X)
            + 0.6 * np.exp(-D * 9 * t) * np.cos(3 * Y)
        )
    return U


# ═══════════════════════════════════════════════════════════════════
#  1. Composite score
# ═══════════════════════════════════════════════════════════════════

def test_composite_score_baseline():
    """Pure fit (loss=0, 0 active, good conditioning) → score = 0."""
    s = _composite_score(0.0, 0, 6, 1.0, alpha=0.1, beta=0.01,
                         cond_threshold=1e8)
    assert abs(s) < 1e-15, f"Expected 0, got {s}"


def test_composite_score_complexity_penalty():
    """Activating all M terms adds α·1.0 = 0.1."""
    s = _composite_score(0.0, 6, 6, 1.0, alpha=0.1, beta=0.01,
                         cond_threshold=1e8)
    assert abs(s - 0.1) < 1e-12, f"Expected 0.1, got {s}"


def test_composite_score_cond_penalty():
    """High condition number triggers β·(log10(κ) - log10(κ₀)) penalty."""
    s = _composite_score(0.0, 0, 6, 1e12, alpha=0.0, beta=0.01,
                         cond_threshold=1e8)
    # penalty = 0.01 * (12 - 8) = 0.04
    assert abs(s - 0.04) < 1e-12, f"Expected 0.04, got {s}"


def test_composite_score_additive():
    """All three components combine additively."""
    loss = 0.05
    n_active, M = 3, 6
    kappa = 1e10
    alpha, beta, cond_t = 0.1, 0.01, 1e8
    s = _composite_score(loss, n_active, M, kappa, alpha, beta, cond_t)
    expected = 0.05 + 0.1 * (3 / 6) + 0.01 * (10 - 8)
    assert abs(s - expected) < 1e-12


# ═══════════════════════════════════════════════════════════════════
#  2. Pareto frontier
# ═══════════════════════════════════════════════════════════════════

def test_pareto_trivial():
    """Single trial → it IS the frontier."""
    t = _make_dummy_trial((3, 4, 4), 0.1, 1, 0.01)
    front = _pareto_frontier([t])
    assert len(front) == 1


def test_pareto_dominance():
    """A dominates B (lower loss, same sparsity) → only A on frontier."""
    a = _make_dummy_trial((3, 4, 4), 0.05, 2, 0.01)
    b = _make_dummy_trial((5, 5, 5), 0.10, 2, 0.05)
    front = _pareto_frontier([a, b])
    assert len(front) == 1
    assert front[0] is a


def test_pareto_tradeoff():
    """Neither dominates → both on frontier."""
    a = _make_dummy_trial((3, 4, 4), 0.05, 3, 0.01)  # lower loss, more terms
    b = _make_dummy_trial((5, 5, 5), 0.10, 1, 0.05)  # higher loss, fewer terms
    front = _pareto_frontier([a, b])
    assert len(front) == 2


# ═══════════════════════════════════════════════════════════════════
#  3. TrialResult container
# ═══════════════════════════════════════════════════════════════════

def test_trial_row_dict():
    t = _make_dummy_trial((3, 4, 5), 0.1, 1, 0.01)
    d = t.row_dict()
    assert d["ellt"] == 3
    assert d["ellx"] == 4
    assert d["elly"] == 5
    assert d["n_active"] == 1
    assert "active_terms" in d
    assert "active_coeffs" in d


# ═══════════════════════════════════════════════════════════════════
#  4. SelectionResult container
# ═══════════════════════════════════════════════════════════════════

def test_selection_result_summary():
    t1 = _make_dummy_trial((3, 4, 4), 0.05, 1, 0.01)
    t2 = _make_dummy_trial((5, 5, 5), 0.10, 2, 0.05)
    sr = SelectionResult(trials=[t1, t2], best=t1, pareto=[t1])
    s = sr.summary()
    assert "★" in s
    assert "ℓ=(3,4,4)" in s
    tab = sr.table()
    assert len(tab) == 2


# ═══════════════════════════════════════════════════════════════════
#  5. default_ell_grid
# ═══════════════════════════════════════════════════════════════════

def test_default_ell_grid_shape():
    grid = default_ell_grid(80, 64, 64, n_points=5)
    assert len(grid) >= 2
    for ell in grid:
        assert len(ell) == 3
        assert all(e >= 2 for e in ell)


def test_default_ell_grid_bounds():
    grid = default_ell_grid(80, 64, 64, n_points=5)
    for ellt, ellx, elly in grid:
        assert ellt <= 80 // 4
        assert ellx <= 64 // 4
        assert elly <= 64 // 4


# ═══════════════════════════════════════════════════════════════════
#  6. Full sweep on synthetic heat data
# ═══════════════════════════════════════════════════════════════════

def test_model_selection_runs():
    """End-to-end sweep with 3 ℓ values — must return a valid result."""
    U = _heat_data(T_total=60)
    grid = _grid()
    library_terms = [
        ("I", "u"),
        ("lap", "u"),
        ("I", "u2"),
    ]
    ell_grid = [(2, 3, 3), (3, 4, 4), (4, 5, 5)]

    result = wsindy_model_selection(
        U, grid, library_terms, ell_grid,
        lambdas=np.logspace(-3, 1, 15),
        verbose=False,
    )

    assert isinstance(result, SelectionResult)
    assert len(result.trials) >= 2  # at least 2 should succeed
    assert result.best is not None
    assert len(result.pareto) >= 1


def test_model_selection_best_has_laplacian():
    """The best model for heat data should include lap:u."""
    U = _heat_data(T_total=60)
    grid = _grid()
    library_terms = [
        ("I", "1"),
        ("I", "u"),
        ("lap", "u"),
        ("I", "u2"),
    ]
    ell_grid = [(2, 3, 3), (3, 4, 4), (4, 6, 6), (5, 7, 7)]

    result = wsindy_model_selection(
        U, grid, library_terms, ell_grid,
        lambdas=np.logspace(-3, 1, 20),
        verbose=False,
    )

    best = result.best_model
    assert "lap:u" in best.active_terms, (
        f"Expected 'lap:u' in active terms, got {best.active_terms}"
    )


def test_model_selection_stride_sweep():
    """Sweeping strides produces more trials than just ℓ sweep."""
    U = _heat_data(T_total=60)
    grid = _grid()
    library_terms = [("I", "u"), ("lap", "u")]
    ell_grid = [(3, 4, 4)]

    result_no_stride = wsindy_model_selection(
        U, grid, library_terms, ell_grid,
        lambdas=np.logspace(-2, 0, 10),
        verbose=False,
    )
    result_with_stride = wsindy_model_selection(
        U, grid, library_terms, ell_grid,
        stride_grid=[(1, 1, 1), (2, 2, 2), (3, 3, 3)],
        lambdas=np.logspace(-2, 0, 10),
        verbose=False,
    )

    assert len(result_with_stride.trials) >= len(result_no_stride.trials)


# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    banner = "=" * 60
    print(f"{banner}\n  WSINDy Part 5 — Model Selection Tests\n{banner}\n")

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
