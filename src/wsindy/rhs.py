"""
Strong-form RHS from a discovered WSINDy model.

Converts the identified PDE terms (e.g. ``"lap:u"``, ``"I:u2"``) into
pointwise operators evaluated on a periodic grid using FFT spectral
derivatives from :mod:`operators`.

The model stores ``col_names`` of the form ``"op:feature"``; this module
maps each pair to concrete numpy operations so the PDE can be
time-stepped.
"""

from __future__ import annotations

import numpy as np

from .grid import GridSpec
from .model import WSINDyModel
from .operators import (
    grad_spectral,
    dxx_spectral,
    dyy_spectral,
    laplacian_spectral,
)


# ── pointwise feature evaluation ───────────────────────────────────

def eval_feature_pointwise(u: np.ndarray, name: str) -> np.ndarray:
    """Evaluate a scalar feature of *u* point-wise.

    Supported names: ``"1"``, ``"u"``, ``"u2"``, ``"u3"``.
    """
    if name == "1":
        return np.ones_like(u)
    if name == "u":
        return u.copy()
    if name == "u2":
        return u * u
    if name == "u3":
        return u * u * u
    raise ValueError(f"Unknown feature '{name}'")


# ── spatial operator application ───────────────────────────────────

def apply_operator_pointwise(
    f: np.ndarray,
    op: str,
    grid: GridSpec,
) -> np.ndarray:
    """Apply a spatial operator to field *f* on a periodic grid.

    Parameters
    ----------
    f : ndarray, shape (nx, ny)
    op : ``"I"`` | ``"dx"`` | ``"dy"`` | ``"dxx"`` | ``"dyy"`` | ``"lap"``
    grid : GridSpec (uses *dx*, *dy*)

    Returns
    -------
    ndarray, shape (nx, ny)
    """
    if op == "I":
        return f
    if op == "dx":
        return grad_spectral(f, grid.dx, grid.dy)[0]
    if op == "dy":
        return grad_spectral(f, grid.dx, grid.dy)[1]
    if op == "dxx":
        return dxx_spectral(f, grid.dx, grid.dy)
    if op == "dyy":
        return dyy_spectral(f, grid.dx, grid.dy)
    if op == "lap":
        return laplacian_spectral(f, grid.dx, grid.dy)
    raise ValueError(f"Unknown operator '{op}'")


# ── term parser ────────────────────────────────────────────────────

def parse_term(term: str) -> tuple[str, str]:
    """Parse ``'op:feature'`` into ``(op, feature)``."""
    parts = term.split(":")
    if len(parts) != 2:
        raise ValueError(
            f"Cannot parse term '{term}'; expected 'op:feature'"
        )
    return parts[0], parts[1]


# ── full RHS evaluation ───────────────────────────────────────────

def wsindy_rhs(
    u: np.ndarray,
    model: WSINDyModel,
    grid: GridSpec,
) -> np.ndarray:
    """Evaluate the discovered PDE right-hand side at a single snapshot.

    .. math::
        u_t = \\sum_{m\\,\\text{active}} w_m \\, D_m f_m(u)

    Parameters
    ----------
    u : ndarray, shape (nx, ny)
    model : WSINDyModel
    grid : GridSpec

    Returns
    -------
    du_dt : ndarray, shape (nx, ny)
    """
    rhs = np.zeros_like(u)
    for i, name in enumerate(model.col_names):
        if not model.active[i]:
            continue
        op, feat = parse_term(name)
        f = eval_feature_pointwise(u, feat)
        rhs += model.w[i] * apply_operator_pointwise(f, op, grid)
    return rhs


# ── mass diagnostic ────────────────────────────────────────────────

def mass(u: np.ndarray, grid: GridSpec) -> float:
    r"""Total mass :math:`\int\!\!\int u\,dx\,dy \approx \sum u_{ij}\,dx\,dy`."""
    return float(u.sum() * grid.dx * grid.dy)
