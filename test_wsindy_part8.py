"""
Tests for WSINDy Part 8: Composable Library Builder
====================================================

Covers:
  - LibraryBuilder   : polynomial, nonlinear diffusion, conservative,
                        gradient nonlinearity, custom terms, deduplication
  - library_from_config : YAML dict → library_terms
  - default_library   : quick helper
  - Feature registration & patching
  - Integration with build_weak_system
"""

import numpy as np
import pytest

from src.wsindy.library import (
    LibraryBuilder,
    clear_registrations,
    default_library,
    get_registered_features,
    library_from_config,
    patch_feature_registries,
    register_feature,
    register_operator,
)
from src.wsindy.grid import GridSpec
from src.wsindy.test_functions import make_separable_psi
from src.wsindy.system import build_weak_system, default_t_margin, make_query_indices


@pytest.fixture(autouse=True)
def _cleanup_registrations():
    """Reset custom registrations after every test."""
    yield
    clear_registrations()


# ════════════════════════════════════════════════════════════════════
#  LibraryBuilder basics
# ════════════════════════════════════════════════════════════════════

class TestLibraryBuilderPolynomials:
    def test_default_polynomials(self):
        terms = LibraryBuilder().add_polynomials(max_power=3).build()
        # Should include I:1, and then u, u2, u3 × all 6 operators
        assert ("I", "1") in terms
        assert ("lap", "u") in terms
        assert ("dxx", "u3") in terms
        # Constant should not appear with non-identity ops
        assert ("dx", "1") not in terms

    def test_no_constant(self):
        terms = (
            LibraryBuilder()
            .add_polynomials(max_power=2, include_constant=False)
            .build()
        )
        assert ("I", "1") not in terms
        assert ("I", "u") in terms

    def test_subset_operators(self):
        terms = (
            LibraryBuilder()
            .add_polynomials(max_power=2, operators=["I", "lap"])
            .build()
        )
        assert ("I", "u") in terms
        assert ("lap", "u") in terms
        assert ("dx", "u") not in terms

    def test_deduplication(self):
        """Adding the same term twice should not create duplicates."""
        builder = LibraryBuilder()
        builder.add_polynomials(max_power=1, operators=["I", "lap"])
        builder.add_polynomials(max_power=1, operators=["I", "lap"])
        terms = builder.build()
        unique = set(terms)
        assert len(terms) == len(unique)


class TestLibraryBuilderAdvanced:
    def test_nonlinear_diffusion(self):
        terms = (
            LibraryBuilder()
            .add_nonlinear_diffusion(powers=[2, 3])
            .build()
        )
        assert ("lap", "u2") in terms
        assert ("lap", "u3") in terms

    def test_conservative(self):
        terms = (
            LibraryBuilder()
            .add_conservative(features=["u2"], directions=["dx", "dy"])
            .build()
        )
        assert ("dx", "u2") in terms
        assert ("dy", "u2") in terms

    def test_gradient_nonlinearity(self):
        terms = (
            LibraryBuilder()
            .add_gradient_nonlinearity()
            .build()
        )
        assert ("I", "grad_u_sq") in terms
        # Should be registered
        feats = get_registered_features()
        assert "grad_u_sq" in feats

    def test_custom_term(self):
        terms = (
            LibraryBuilder()
            .add_term("lap", "u2")
            .add_term("I", "u")
            .build()
        )
        assert terms == [("lap", "u2"), ("I", "u")]

    def test_add_terms_batch(self):
        terms = (
            LibraryBuilder()
            .add_terms([("I", "u"), ("lap", "u"), ("dx", "u2")])
            .build()
        )
        assert len(terms) == 3

    def test_len_and_repr(self):
        builder = LibraryBuilder().add_polynomials(max_power=2, operators=["I", "lap"])
        assert len(builder) > 0
        assert "LibraryBuilder" in repr(builder)


class TestLibraryBuilderChaining:
    def test_full_chain(self):
        terms = (
            LibraryBuilder()
            .add_polynomials(max_power=2, operators=["I", "lap"])
            .add_nonlinear_diffusion(powers=[2])
            .add_conservative(features=["u2"], directions=["dx"])
            .add_gradient_nonlinearity()
            .build()
        )
        assert ("I", "1") in terms
        assert ("lap", "u2") in terms
        assert ("dx", "u2") in terms
        assert ("I", "grad_u_sq") in terms


# ════════════════════════════════════════════════════════════════════
#  YAML config interface
# ════════════════════════════════════════════════════════════════════

class TestLibraryFromConfig:
    def test_basic_config(self):
        cfg = {
            "max_poly": 2,
            "operators": ["I", "lap"],
        }
        terms = library_from_config(cfg)
        assert ("I", "1") in terms
        assert ("I", "u") in terms
        assert ("lap", "u") in terms
        assert ("lap", "u2") in terms

    def test_nonlinear_diffusion_config(self):
        cfg = {
            "max_poly": 1,
            "operators": ["I"],
            "nonlinear_diffusion": {"enabled": True, "powers": [2, 3]},
        }
        terms = library_from_config(cfg)
        assert ("lap", "u2") in terms
        assert ("lap", "u3") in terms

    def test_conservative_config(self):
        cfg = {
            "max_poly": 1,
            "operators": ["I"],
            "conservative": {
                "enabled": True,
                "terms": ["dx:u2", "dy:u2"],
            },
        }
        terms = library_from_config(cfg)
        assert ("dx", "u2") in terms
        assert ("dy", "u2") in terms

    def test_gradient_nonlinearity_config(self):
        cfg = {
            "max_poly": 1,
            "operators": ["I"],
            "gradient_nonlinearity": {"enabled": True},
        }
        terms = library_from_config(cfg)
        assert ("I", "grad_u_sq") in terms

    def test_custom_terms_config(self):
        cfg = {
            "max_poly": 1,
            "operators": ["I"],
            "custom_terms": ["dxx:u3", ["dyy", "u2"]],
        }
        terms = library_from_config(cfg)
        assert ("dxx", "u3") in terms
        assert ("dyy", "u2") in terms

    def test_disabled_sections_ignored(self):
        cfg = {
            "max_poly": 1,
            "operators": ["I"],
            "nonlinear_diffusion": {"enabled": False, "powers": [2]},
            "gradient_nonlinearity": {"enabled": False},
        }
        terms = library_from_config(cfg)
        assert ("lap", "u2") not in terms
        assert ("I", "grad_u_sq") not in terms

    def test_empty_config_uses_defaults(self):
        terms = library_from_config({})
        # Default: max_poly=3, all standard operators
        assert ("I", "1") in terms
        assert ("lap", "u3") in terms


# ════════════════════════════════════════════════════════════════════
#  default_library
# ════════════════════════════════════════════════════════════════════

class TestDefaultLibrary:
    def test_default(self):
        terms = default_library()
        # Should have I:1, plus u/u2/u3 × {I, dx, dy, lap}
        assert ("I", "1") in terms
        assert ("lap", "u") in terms
        assert ("dx", "u3") in terms

    def test_custom_operators(self):
        terms = default_library(max_poly=1, operators=["I"])
        assert len(terms) == 2  # I:1 and I:u

    def test_no_dxx_in_default(self):
        """Default library uses I, dx, dy, lap — not dxx, dyy separately."""
        terms = default_library()
        assert ("dxx", "u") not in terms


# ════════════════════════════════════════════════════════════════════
#  Feature registration
# ════════════════════════════════════════════════════════════════════

class TestFeatureRegistration:
    def test_register_and_retrieve(self):
        register_feature("u4", lambda U: U ** 4)
        feats = get_registered_features()
        assert "u4" in feats

    def test_clear(self):
        register_feature("u5", lambda U: U ** 5)
        clear_registrations()
        assert "u5" not in get_registered_features()

    def test_register_operator(self):
        register_operator("dxy", "psi_xy")
        ops = get_registered_features()  # different registry but check no error
        from src.wsindy.library import get_registered_operators
        assert "dxy" in get_registered_operators()


# ════════════════════════════════════════════════════════════════════
#  Integration with build_weak_system
# ════════════════════════════════════════════════════════════════════

class TestLibraryIntegration:
    def test_builder_output_compatible(self):
        """Library terms from builder should work with build_weak_system."""
        grid = GridSpec(dt=0.05, dx=0.5, dy=0.5)
        T, nx, ny = 40, 16, 16
        rng = np.random.default_rng(42)
        U = rng.standard_normal((T, nx, ny))

        terms = default_library(max_poly=2, operators=["I", "lap"])
        psi_bundle = make_separable_psi(grid, 3, 3, 3, 2, 2, 2)
        t_margin = default_t_margin(psi_bundle)
        query_idx = make_query_indices(T, nx, ny, 2, 2, 2, t_margin)

        b, G, col_names = build_weak_system(U, grid, psi_bundle, terms, query_idx)
        assert b.shape[0] == G.shape[0]
        assert G.shape[1] == len(terms)
        assert len(col_names) == len(terms)

    def test_gradient_feature_evaluates(self):
        """The grad_u_sq feature should produce finite values."""
        LibraryBuilder().add_gradient_nonlinearity()
        patch_feature_registries()

        from src.wsindy.system import eval_feature
        rng = np.random.default_rng(99)
        U = rng.standard_normal((10, 16, 16))
        F = eval_feature(U, "grad_u_sq")
        assert F.shape == U.shape
        assert np.all(np.isfinite(F))
        assert np.all(F >= 0)  # |∇u|² ≥ 0

    def test_higher_poly_evaluates(self):
        """u^5 feature should be auto-registered and evaluable."""
        _ = LibraryBuilder().add_polynomials(max_power=5, operators=["I"]).build()
        patch_feature_registries()

        from src.wsindy.system import eval_feature
        U = np.ones((5, 8, 8)) * 2.0
        F = eval_feature(U, "u5")
        np.testing.assert_allclose(F, 32.0)  # 2^5
