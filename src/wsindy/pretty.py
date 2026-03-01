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
    """Parse ``'op:feature'`` handling composite ops (colon only in last segment)."""
    idx = term.rfind(":")
    if idx < 0:
        raise ValueError(f"Cannot parse term '{term}'; expected 'op:feature'")
    return term[:idx], term[idx + 1:]


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
        base_op = op.split("/")[0].split("+")[0]  # first token
        cat = _classify_op(base_op)
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
