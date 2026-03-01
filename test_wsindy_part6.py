"""
Tests for WSINDy Part 6: Interpretability — pretty.py
=====================================================

Covers:
  - to_text   : plain-text PDE rendering
  - to_latex  : LaTeX PDE rendering
  - group_terms : physical-category grouping
  - Edge cases: empty model, single-term, composite operators
"""

import numpy as np
import pytest

from src.wsindy.model import WSINDyModel
from src.wsindy.pretty import (
    _parse_term,
    _render_term_latex,
    _render_term_text,
    group_terms,
    to_latex,
    to_text,
)


# ── helpers ─────────────────────────────────────────────────────────

def _make_model(col_names, w, active=None):
    """Quick helper to build a WSINDyModel for testing."""
    w = np.asarray(w, dtype=np.float64)
    M = len(col_names)
    if active is None:
        active = w != 0
    return WSINDyModel(
        col_names=col_names,
        w=w,
        active=np.asarray(active, dtype=bool),
        best_lambda=1e-3,
        col_scale=np.ones(M),
        diagnostics={"r2": 0.99},
    )


# ════════════════════════════════════════════════════════════════════
#  Term parsing
# ════════════════════════════════════════════════════════════════════

class TestTermParsing:
    def test_simple_term(self):
        assert _parse_term("lap:u") == ("lap", "u")

    def test_identity_constant(self):
        assert _parse_term("I:1") == ("I", "1")

    def test_composite_operator(self):
        op, feat = _parse_term("lap/I:u2")
        assert op == "lap/I"
        assert feat == "u2"

    def test_bad_term_raises(self):
        with pytest.raises(ValueError):
            _parse_term("nocolon")


# ════════════════════════════════════════════════════════════════════
#  Term rendering
# ════════════════════════════════════════════════════════════════════

class TestTermRendering:
    def test_latex_lap_u(self):
        s = _render_term_latex("lap", "u")
        assert "\\Delta" in s
        assert "u" in s

    def test_latex_identity_u2(self):
        s = _render_term_latex("I", "u2")
        assert "u^2" in s
        # Identity should not add operator decoration
        assert "\\partial" not in s

    def test_text_dx_u3(self):
        s = _render_term_text("dx", "u3")
        assert "d_x" in s
        assert "u^3" in s

    def test_text_identity_1(self):
        s = _render_term_text("I", "1")
        assert s == "1"

    def test_composite_latex(self):
        s = _render_term_latex("lap/I", "u2")
        assert "\\Delta" in s

    def test_composite_text(self):
        s = _render_term_text("lap/I", "u2")
        assert "Lap" in s


# ════════════════════════════════════════════════════════════════════
#  to_text
# ════════════════════════════════════════════════════════════════════

class TestToText:
    def test_heat_equation(self):
        model = _make_model(["I:u", "lap:u"], [0.0, 1.0])
        txt = to_text(model)
        assert "u_t" in txt
        assert "Lap(u)" in txt

    def test_two_active_terms(self):
        model = _make_model(
            ["I:u", "lap:u", "I:u2"],
            [0.0, 0.5, -0.3],
        )
        txt = to_text(model)
        assert "Lap(u)" in txt
        assert "u^2" in txt
        # coefficient signs
        assert "+" in txt or "-" in txt

    def test_empty_model(self):
        model = _make_model(["I:u", "lap:u"], [0.0, 0.0])
        txt = to_text(model)
        assert txt == "u_t = 0"

    def test_custom_lhs(self):
        model = _make_model(["lap:u"], [1.0])
        txt = to_text(model, lhs="rho_t")
        assert txt.startswith("rho_t =")

    def test_single_term(self):
        model = _make_model(["lap:u"], [2.0])
        txt = to_text(model)
        assert "Lap(u)" in txt
        assert "2.0000e+00" in txt


# ════════════════════════════════════════════════════════════════════
#  to_latex
# ════════════════════════════════════════════════════════════════════

class TestToLatex:
    def test_heat_equation(self):
        model = _make_model(["lap:u"], [1.0])
        ltx = to_latex(model)
        assert "\\Delta" in ltx
        assert "u_t" in ltx

    def test_display_mode(self):
        model = _make_model(["lap:u"], [1.0])
        ltx = to_latex(model, display=True)
        assert "\\[" in ltx
        assert "\\]" in ltx

    def test_empty_model_latex(self):
        model = _make_model(["I:u"], [0.0])
        ltx = to_latex(model)
        assert "= 0" in ltx

    def test_precision(self):
        model = _make_model(["I:u2"], [-1.23456789])
        ltx2 = to_latex(model, precision=2)
        assert "1.23e" in ltx2
        ltx6 = to_latex(model, precision=6)
        assert "1.234568e" in ltx6


# ════════════════════════════════════════════════════════════════════
#  group_terms
# ════════════════════════════════════════════════════════════════════

class TestGroupTerms:
    def test_mixed_model(self):
        model = _make_model(
            ["I:1", "I:u", "I:u2", "dx:u", "dy:u", "lap:u", "dxx:u"],
            [0.1, -0.5, 0.3, 1.0, -1.0, 2.0, 0.5],
        )
        groups = group_terms(model)
        assert "source" in groups
        assert "reaction" in groups
        assert "advection" in groups
        assert "diffusion" in groups

        # Check counts
        assert len(groups["source"]) == 1  # I:1
        assert len(groups["reaction"]) == 2  # I:u, I:u2
        assert len(groups["advection"]) == 2  # dx:u, dy:u
        assert len(groups["diffusion"]) == 2  # lap:u, dxx:u

    def test_empty_model_groups(self):
        model = _make_model(["I:u", "lap:u"], [0.0, 0.0])
        groups = group_terms(model)
        assert len(groups) == 0

    def test_diffusion_only(self):
        model = _make_model(["lap:u"], [1.0])
        groups = group_terms(model)
        assert "diffusion" in groups
        assert "reaction" not in groups

    def test_coefficients_preserved(self):
        model = _make_model(["lap:u"], [3.14])
        groups = group_terms(model)
        assert abs(groups["diffusion"][0]["coeff"] - 3.14) < 1e-10

    def test_inactive_terms_excluded(self):
        model = _make_model(
            ["I:u", "lap:u"],
            [1.0, 2.0],
            active=[False, True],
        )
        groups = group_terms(model)
        # I:u should NOT appear
        all_terms = [e["term"] for entries in groups.values() for e in entries]
        assert "I:u" not in all_terms
        assert "lap:u" in all_terms
