"""
Human-readable and LaTeX rendering of discovered WSINDy models.

Provides three main functions:

* :func:`to_text` — plain-text PDE string.
* :func:`to_latex` — LaTeX equation string.
* :func:`group_terms` — classify active terms into physical categories
  (advection, diffusion, reaction, source).

Term naming convention: ``"op:feature"`` where

* **op** ∈ ``{I, dx, dy, dxx, dyy, lap}`` — spatial operator
* **feature** ∈ ``{1, u, u2, u3, …}`` — nonlinear feature of *u*

Composite (Part 8) terms use slash-delimited operators, e.g.
``"lap/I:u2"`` means :math:`\\Delta(u^2)`, and ``"dx/dx+dy/dy:u2"`` means
:math:`\\partial_x(u^2_x) + \\partial_y(u^2_y) = \\nabla\\cdot(u^2\\nabla u)`.
These are parsed recursively.

Multi-field WSINDy also uses opaque symbolic term names such as ``"div_p"``
or ``"rho_dx_Phi"``. Those are rendered via dedicated lookup tables instead of
the scalar ``"op:feature"`` parser.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .model import WSINDyModel


# ═══════════════════════════════════════════════════════════════════
#  Rendering tables
# ═══════════════════════════════════════════════════════════════════

# Operator → LaTeX string (wraps its argument)
_OP_LATEX: Dict[str, str] = {
    "I":   "{arg}",
    "dx":  "\\partial_x {arg}",
    "dy":  "\\partial_y {arg}",
    "dxx": "\\partial_{{xx}} {arg}",
    "dyy": "\\partial_{{yy}} {arg}",
    "lap": "\\Delta {arg}",
}

# Operator → plain-text string
_OP_TEXT: Dict[str, str] = {
    "I":   "{arg}",
    "dx":  "d_x({arg})",
    "dy":  "d_y({arg})",
    "dxx": "d_xx({arg})",
    "dyy": "d_yy({arg})",
    "lap": "Lap({arg})",
}

# Feature → LaTeX
_FEAT_LATEX: Dict[str, str] = {
    "1":  "1",
    "u":  "u",
    "u2": "u^2",
    "u3": "u^3",
}

# Feature → plain-text
_FEAT_TEXT: Dict[str, str] = {
    "1":  "1",
    "u":  "u",
    "u2": "u^2",
    "u3": "u^3",
}

# Multi-field WSINDy term names do not follow the scalar "op:feature" schema.
# Keep these in one place so future multifield exports stay robust even if the
# symbolic names evolve independently from the scalar library encoder.
_OPAQUE_TERM_LATEX: Dict[str, str] = {
    "div_p": r"\nabla\cdot \mathbf{p}",
    "lap_rho": r"\Delta \rho",
    "div_rho_gradPhi": r"\nabla\cdot(\rho\nabla\Phi)",
    "lap_rho2": r"\Delta(\rho^2)",
    "lap_rho3": r"\Delta(\rho^3)",
    "lap_p_sq": r"\Delta(|\mathbf{p}|^2)",
    "px": r"p_x",
    "py": r"p_y",
    "p_sq_px": r"|\mathbf{p}|^2 p_x",
    "p_sq_py": r"|\mathbf{p}|^2 p_y",
    "dx_rho": r"\partial_x \rho",
    "dy_rho": r"\partial_y \rho",
    "dx_rho2": r"\partial_x(\rho^2)",
    "dy_rho2": r"\partial_y(\rho^2)",
    "lap_px": r"\Delta p_x",
    "lap_py": r"\Delta p_y",
    "bilap_px": r"\Delta^2 p_x",
    "bilap_py": r"\Delta^2 p_y",
    "rho_dx_Phi": r"\rho\,\partial_x\Phi",
    "rho_dy_Phi": r"\rho\,\partial_y\Phi",
    "p_dot_grad_px": r"(\mathbf{p}\cdot\nabla)p_x",
    "p_dot_grad_py": r"(\mathbf{p}\cdot\nabla)p_y",
}

_OPAQUE_TERM_TEXT: Dict[str, str] = {
    "div_p": "div(p)",
    "lap_rho": "Lap(rho)",
    "div_rho_gradPhi": "div(rho grad(Phi))",
    "lap_rho2": "Lap(rho^2)",
    "lap_rho3": "Lap(rho^3)",
    "lap_p_sq": "Lap(|p|^2)",
    "px": "p_x",
    "py": "p_y",
    "p_sq_px": "|p|^2 p_x",
    "p_sq_py": "|p|^2 p_y",
    "dx_rho": "d_x(rho)",
    "dy_rho": "d_y(rho)",
    "dx_rho2": "d_x(rho^2)",
    "dy_rho2": "d_y(rho^2)",
    "lap_px": "Lap(p_x)",
    "lap_py": "Lap(p_y)",
    "bilap_px": "Bilap(p_x)",
    "bilap_py": "Bilap(p_y)",
    "rho_dx_Phi": "rho d_x(Phi)",
    "rho_dy_Phi": "rho d_y(Phi)",
    "p_dot_grad_px": "p_dot_grad(p_x)",
    "p_dot_grad_py": "p_dot_grad(p_y)",
}

_OPAQUE_TERM_CATEGORY: Dict[str, str] = {
    "div_p": "advection",
    "lap_rho": "diffusion",
    "div_rho_gradPhi": "other",
    "lap_rho2": "diffusion",
    "lap_rho3": "diffusion",
    "lap_p_sq": "diffusion",
    "px": "reaction",
    "py": "reaction",
    "p_sq_px": "reaction",
    "p_sq_py": "reaction",
    "dx_rho": "advection",
    "dy_rho": "advection",
    "dx_rho2": "advection",
    "dy_rho2": "advection",
    "lap_px": "diffusion",
    "lap_py": "diffusion",
    "bilap_px": "diffusion",
    "bilap_py": "diffusion",
    "rho_dx_Phi": "other",
    "rho_dy_Phi": "other",
    "p_dot_grad_px": "advection",
    "p_dot_grad_py": "advection",
}

# ── Physical grouping categories ───────────────────────────────────

_ADVECTION_OPS = {"dx", "dy"}
_DIFFUSION_OPS = {"dxx", "dyy", "lap"}
_IDENTITY_OP = {"I"}


def _classify_op(op: str) -> str:
    """Classify a single op token into a physical category."""
    if op in _ADVECTION_OPS:
        return "advection"
    if op in _DIFFUSION_OPS:
        return "diffusion"
    if op in _IDENTITY_OP:
        return "reaction"
    return "other"


# ═══════════════════════════════════════════════════════════════════
#  Term rendering
# ═══════════════════════════════════════════════════════════════════

def _render_term_latex(op: str, feat: str) -> str:
    """Render one ``op:feat`` pair as LaTeX math (no sign/coeff)."""
    if op == "opaque":
        return _OPAQUE_TERM_LATEX.get(feat, feat)
    feat_str = _FEAT_LATEX.get(feat, feat)
    if "/" in op or "+" in op:
        # composite operator from Part 8 library
        return _render_composite_latex(op, feat_str)
    op_tmpl = _OP_LATEX.get(op, f"\\text{{{op}}}({{}})".replace("{}", "{arg}"))
    return op_tmpl.format(arg=feat_str)


def _render_composite_latex(op_chain: str, feat_str: str) -> str:
    """Handle composite operators like ``lap/I`` or ``dx/dx+dy/dy``."""
    # Split additive parts first (for conservative div forms)
    parts = op_chain.split("+")
    rendered_parts: List[str] = []
    for part in parts:
        tokens = part.split("/")
        inner = feat_str
        for tok in reversed(tokens):
            tmpl = _OP_LATEX.get(tok, f"\\text{{{tok}}}({{arg}})")
            inner = tmpl.format(arg=inner)
        rendered_parts.append(inner)
    return " + ".join(rendered_parts)


def _render_term_text(op: str, feat: str) -> str:
    """Render one ``op:feat`` pair as plain text (no sign/coeff)."""
    if op == "opaque":
        return _OPAQUE_TERM_TEXT.get(feat, feat)
    feat_str = _FEAT_TEXT.get(feat, feat)
    if "/" in op or "+" in op:
        return _render_composite_text(op, feat_str)
    op_tmpl = _OP_TEXT.get(op, f"{op}({{arg}})")
    return op_tmpl.format(arg=feat_str)


def _render_composite_text(op_chain: str, feat_str: str) -> str:
    """Plain-text composite (e.g. ``lap/I:u2`` → ``Lap(u^2)``)."""
    parts = op_chain.split("+")
    rendered: List[str] = []
    for part in parts:
        tokens = part.split("/")
        inner = feat_str
        for tok in reversed(tokens):
            tmpl = _OP_TEXT.get(tok, f"{tok}({{arg}})")
            inner = tmpl.format(arg=inner)
        rendered.append(inner)
    return " + ".join(rendered)


def _parse_term(term: str) -> Tuple[str, str]:
    """Parse either ``'op:feature'`` or an opaque symbolic term name."""
    if not term:
        raise ValueError("Cannot parse empty term")
    idx = term.rfind(":")
    if idx < 0:
        return "opaque", term
    return term[:idx], term[idx + 1:]


def _classify_term(op: str, feat: str) -> str:
    """Classify parsed terms, including multifield opaque symbols."""
    if op == "opaque":
        return _OPAQUE_TERM_CATEGORY.get(feat, "other")
    base_op = op.split("/")[0].split("+")[0]
    return _classify_op(base_op)


# ═══════════════════════════════════════════════════════════════════
#  Coefficient formatting
# ═══════════════════════════════════════════════════════════════════

def _format_coeff(val: float, precision: int, first: bool) -> str:
    """Format coefficient with sign for accumulation.

    ``first=True`` suppresses leading ``+``.
    """
    if first:
        return f"{val:.{precision}e}"
    if val >= 0:
        return f" + {val:.{precision}e}"
    return f" - {abs(val):.{precision}e}"


def _format_coeff_latex(val: float, precision: int, first: bool) -> str:
    """LaTeX coefficient string."""
    if first:
        return f"{val:.{precision}e}"
    if val >= 0:
        return f" + {val:.{precision}e}"
    return f" - {abs(val):.{precision}e}"


# ═══════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════

def to_text(
    model: WSINDyModel,
    precision: int = 4,
    lhs: str = "u_t",
) -> str:
    """Render discovered PDE as plain-text string.

    Parameters
    ----------
    model : WSINDyModel
    precision : int
        Significant digits for coefficients.
    lhs : str
        Left-hand side symbol.

    Returns
    -------
    str
        E.g. ``"u_t = +1.0000e+00 Lap(u) - 5.0000e-01 u^2"``
    """
    parts: List[str] = []
    order = np.argsort(-np.abs(model.w))
    first = True
    for i in order:
        if not model.active[i]:
            continue
        op, feat = _parse_term(model.col_names[i])
        coeff_str = _format_coeff(model.w[i], precision, first)
        term_str = _render_term_text(op, feat)
        parts.append(f"{coeff_str} {term_str}")
        first = False
    if not parts:
        return f"{lhs} = 0"
    return f"{lhs} = {' '.join(parts)}" if parts else f"{lhs} = 0"


def to_latex(
    model: WSINDyModel,
    precision: int = 4,
    lhs: str = "u_t",
    display: bool = False,
) -> str:
    r"""Render discovered PDE as a LaTeX equation.

    Parameters
    ----------
    model : WSINDyModel
    precision : int
        Significant digits for coefficients.
    lhs : str
        Left-hand side LaTeX symbol.
    display : bool
        If ``True``, wraps in ``\\[ ... \\]``.

    Returns
    -------
    str
        LaTeX equation string.
    """
    parts: List[str] = []
    order = np.argsort(-np.abs(model.w))
    first = True
    for i in order:
        if not model.active[i]:
            continue
        op, feat = _parse_term(model.col_names[i])
        coeff_str = _format_coeff_latex(model.w[i], precision, first)
        term_str = _render_term_latex(op, feat)
        parts.append(f"{coeff_str} \\, {term_str}")
        first = False
    if not parts:
        body = f"{lhs} = 0"
    else:
        body = f"{lhs} = {''.join(parts)}"
    if display:
        return f"\\[\n{body}\n\\]"
    return body


def group_terms(
    model: WSINDyModel,
) -> Dict[str, List[Dict[str, Any]]]:
    """Group active terms by physical category.

    Categories
    ----------
    * **advection** — first-order spatial derivatives (``dx``, ``dy``)
    * **diffusion** — second-order / Laplacian (``dxx``, ``dyy``, ``lap``)
    * **reaction** — identity operator (``I``) applied to nonlinear features
    * **source** — identity operator applied to constant (``I:1``)
    * **other** — anything not classified above

    Returns
    -------
    dict
        ``{category: [{"term": str, "op": str, "feat": str,
        "coeff": float}, ...]}``
    """
    groups: Dict[str, List[Dict[str, Any]]] = OrderedDict(
        source=[], reaction=[], advection=[], diffusion=[], other=[],
    )

    for i, name in enumerate(model.col_names):
        if not model.active[i]:
            continue
        op, feat = _parse_term(name)
        entry = {
            "term": name,
            "op": op,
            "feat": feat,
            "coeff": float(model.w[i]),
        }

        # Source is a special case of reaction
        cat = _classify_term(op, feat)
        if cat == "reaction" and feat == "1":
            groups["source"].append(entry)
        elif cat == "reaction":
            groups["reaction"].append(entry)
        else:
            groups.setdefault(cat, []).append(entry)

    # Remove empty categories
    return OrderedDict(
        (k, v) for k, v in groups.items() if v
    )
