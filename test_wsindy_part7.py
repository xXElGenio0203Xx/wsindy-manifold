"""
Tests for WSINDy Part 7: Uncertainty & Stability
=================================================

Covers:
  - bootstrap_wsindy  : coefficient distributions, CI, inclusion probability
  - stability_selection : frequency tables, robust/fragile classification
"""

import numpy as np
import pytest

from src.wsindy.grid import GridSpec
from src.wsindy.test_functions import make_separable_psi
from src.wsindy.system import build_weak_system, default_t_margin, make_query_indices
from src.wsindy.fit import wsindy_fit_regression
from src.wsindy.uncertainty import BootstrapResult, bootstrap_wsindy
from src.wsindy.stability import StabilityResult, stability_selection


# ── helpers ─────────────────────────────────────────────────────────

def _make_heat_data(T=60, nx=32, ny=32, nu=0.05, dt=0.05, dx=0.5, dy=0.5):
    """Generate heat-equation data u_t = nu * Lap(u) for testing."""
    grid = GridSpec(dt=dt, dx=dx, dy=dy)
    kx = 2 * np.pi * np.fft.fftfreq(nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, dy)
    ksq = kx[:, None] ** 2 + ky[None, :] ** 2

    rng = np.random.default_rng(42)
    u0 = rng.standard_normal((nx, ny))
    u0_hat = np.fft.fft2(u0)

    U = np.empty((T, nx, ny))
    for t in range(T):
        U[t] = np.fft.ifft2(u0_hat * np.exp(-nu * ksq * t * dt)).real
    return U, grid


def _build_system(U, grid, ell=(3, 3, 3), p=(2, 2, 2)):
    """Build weak system with standard small library."""
    T, nx, ny = U.shape
    psi_bundle = make_separable_psi(grid, ell[0], ell[1], ell[2], p[0], p[1], p[2])
    t_margin = default_t_margin(psi_bundle)
    query_idx = make_query_indices(T, nx, ny, 2, 2, 2, t_margin)
    library_terms = [("I", "u"), ("I", "u2"), ("lap", "u")]
    b, G, col_names = build_weak_system(U, grid, psi_bundle, library_terms, query_idx)
    return b, G, col_names


# ════════════════════════════════════════════════════════════════════
#  Bootstrap UQ
# ════════════════════════════════════════════════════════════════════

class TestBootstrapWsindy:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.U, self.grid = _make_heat_data()
        self.b, self.G, self.col_names = _build_system(self.U, self.grid)

    def test_returns_bootstrap_result(self):
        res = bootstrap_wsindy(
            self.b, self.G, self.col_names, B=10,
            rng=np.random.default_rng(0),
        )
        assert isinstance(res, BootstrapResult)
        assert res.B == 10

    def test_coeff_samples_shape(self):
        res = bootstrap_wsindy(
            self.b, self.G, self.col_names, B=15,
            rng=np.random.default_rng(0),
        )
        assert res.coeff_samples.shape == (15, 3)

    def test_active_counts_bounded(self):
        B = 20
        res = bootstrap_wsindy(
            self.b, self.G, self.col_names, B=B,
            rng=np.random.default_rng(1),
        )
        assert np.all(res.active_counts >= 0)
        assert np.all(res.active_counts <= B)

    def test_inclusion_probability_range(self):
        res = bootstrap_wsindy(
            self.b, self.G, self.col_names, B=20,
            rng=np.random.default_rng(2),
        )
        p = res.inclusion_probability
        assert np.all(p >= 0.0)
        assert np.all(p <= 1.0)

    def test_lap_u_frequently_selected(self):
        """The Laplacian term should be selected in most replicates."""
        res = bootstrap_wsindy(
            self.b, self.G, self.col_names, B=30,
            rng=np.random.default_rng(3),
        )
        lap_idx = self.col_names.index("lap:u")
        assert res.inclusion_probability[lap_idx] > 0.5

    def test_confidence_interval_shape(self):
        res = bootstrap_wsindy(
            self.b, self.G, self.col_names, B=20,
            rng=np.random.default_rng(4),
        )
        ci = res.confidence_interval(alpha=0.05)
        assert ci.shape == (3, 2)
        # lo <= hi for all terms
        assert np.all(ci[:, 0] <= ci[:, 1])

    def test_coeff_mean_and_std(self):
        res = bootstrap_wsindy(
            self.b, self.G, self.col_names, B=20,
            rng=np.random.default_rng(5),
        )
        assert res.coeff_mean.shape == (3,)
        assert res.coeff_std.shape == (3,)
        assert np.all(res.coeff_std >= 0)

    def test_summary_string(self):
        res = bootstrap_wsindy(
            self.b, self.G, self.col_names, B=10,
            rng=np.random.default_rng(6),
        )
        s = res.summary()
        assert "Bootstrap UQ" in s
        assert "lap:u" in s

    def test_subsample_frac(self):
        """Subsample fraction < 1 should still work."""
        res = bootstrap_wsindy(
            self.b, self.G, self.col_names, B=10,
            rng=np.random.default_rng(7),
            subsample_frac=0.5,
        )
        assert res.coeff_samples.shape[0] == 10

    def test_base_model_attached(self):
        res = bootstrap_wsindy(
            self.b, self.G, self.col_names, B=5,
            rng=np.random.default_rng(8),
        )
        assert res.base_model is not None
        assert res.base_model.n_active >= 1


# ════════════════════════════════════════════════════════════════════
#  Stability selection
# ════════════════════════════════════════════════════════════════════

class TestStabilitySelection:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.U, self.grid = _make_heat_data()
        self.library_terms = [("I", "u"), ("I", "u2"), ("lap", "u")]

    def test_returns_stability_result(self):
        res = stability_selection(
            self.U, self.grid, self.library_terms,
            ell_grid=[(3, 3, 3)],
            threshold=0.5,
        )
        assert isinstance(res, StabilityResult)

    def test_freq_shape(self):
        res = stability_selection(
            self.U, self.grid, self.library_terms,
            ell_grid=[(3, 3, 3), (4, 4, 4)],
        )
        assert res.freq.shape == (3,)
        # freq_matrix has one row per config
        assert res.freq_matrix.shape[0] == 2

    def test_freq_bounded(self):
        res = stability_selection(
            self.U, self.grid, self.library_terms,
            ell_grid=[(3, 3, 3), (5, 5, 5)],
        )
        assert np.all(res.freq >= 0)
        assert np.all(res.freq <= 1)

    def test_with_bootstrap_replicates(self):
        res = stability_selection(
            self.U, self.grid, self.library_terms,
            ell_grid=[(3, 3, 3)],
            n_bootstrap=5,
            rng=np.random.default_rng(10),
        )
        # 1 full-data + 5 bootstrap = 6 rows
        assert res.freq_matrix.shape[0] == 6

    def test_robust_and_fragile(self):
        res = stability_selection(
            self.U, self.grid, self.library_terms,
            ell_grid=[(3, 3, 3), (4, 4, 4), (5, 5, 5)],
            threshold=0.5,
        )
        # At least one term should be robust
        assert len(res.robust_terms) >= 1
        # The Laplacian should be robust
        assert "lap:u" in res.robust_terms

    def test_summary_string(self):
        res = stability_selection(
            self.U, self.grid, self.library_terms,
            ell_grid=[(3, 3, 3)],
        )
        s = res.summary()
        assert "Stability selection" in s

    def test_config_labels(self):
        res = stability_selection(
            self.U, self.grid, self.library_terms,
            ell_grid=[(3, 3, 3), (4, 4, 4)],
            n_bootstrap=2,
            rng=np.random.default_rng(11),
        )
        # 2 full-data + 2*2 bootstrap = 6
        assert len(res.config_labels) == 6
        assert any("ell=(3,3,3)" in l for l in res.config_labels)
        assert any("boot" in l for l in res.config_labels)
