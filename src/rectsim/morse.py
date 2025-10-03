"""Morse potential forces for the D'Orsogna model.

This module implements a Morse-like pair potential and its radial force.

The pair potential used (up to an additive constant) is::

    U(r) = C_r exp(-r / l_r) - C_a exp(-r / l_a)

The radial force magnitude returned by :func:`_morse_pair_force` is the
negative derivative of the potential:

    f(r) = -dU/dr = C_r / l_r * exp(-r / l_r) - C_a / l_a * exp(-r / l_a)

The vector force applied to particle i by j is central and given by::

    F_i = f(r) * (x_i - x_j) / r
    F_j = -F_i

Positive f(r) corresponds to repulsion; negative f(r) to attraction.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .domain import CellList, build_cells, iter_neighbors

try:  # Optional acceleration
    from numba import njit  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    njit = None  # type: ignore

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - optional dependency
    cKDTree = None  # type: ignore

ArrayLike = np.ndarray


def _morse_pair_force(r: float, Cr: float, Ca: float, lr: float, la: float) -> float:
    """Return the scalar Morse force magnitude for a particle pair."""

    r = max(r, 1e-6)
    return Cr / lr * np.exp(-r / lr) - Ca / la * np.exp(-r / la)


def morse_force_pairs(
    x: ArrayLike,
    Cr: float,
    Ca: float,
    lr: float,
    la: float,
    Lx: float,
    Ly: float,
    bc: str,
    rcut: float,
    cell_list: Optional[CellList] = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """Compute social forces using a linked-cell neighbor list."""

    n = x.shape[0]
    fx = np.zeros(n, dtype=float)
    fy = np.zeros(n, dtype=float)

    local_cells = cell_list or build_cells(x, Lx, Ly, rcut, bc)

    for i, j, dx, dy, rij in iter_neighbors(x, local_cells, Lx, Ly, rcut, bc):
        fmag = _morse_pair_force(rij, Cr, Ca, lr, la)
        scale = -fmag / rij
        fx_ij = scale * dx
        fy_ij = scale * dy
        fx[i] += fx_ij
        fy[i] += fy_ij
        fx[j] -= fx_ij
        fy[j] -= fy_ij

    return fx, fy


def morse_force_ckdtree(
    x: ArrayLike,
    Cr: float,
    Ca: float,
    lr: float,
    la: float,
    Lx: float,
    Ly: float,
    bc: str,
    rcut: float,
) -> Tuple[ArrayLike, ArrayLike]:
    """Compute social forces using :class:`scipy.spatial.cKDTree`."""

    if cKDTree is None:  # pragma: no cover - optional dependency
        raise ImportError("scipy is required for cKDTree-based forces")

    n = x.shape[0]
    tree = cKDTree(x, boxsize=(Lx, Ly) if bc == "periodic" else None)
    pairs = tree.query_pairs(rcut, output_type="ndarray")

    fx = np.zeros(n, dtype=float)
    fy = np.zeros(n, dtype=float)

    for i, j in pairs:
        dx = x[j, 0] - x[i, 0]
        dy = x[j, 1] - x[i, 1]
        if bc == "periodic":
            dx -= Lx * np.round(dx / Lx)
            dy -= Ly * np.round(dy / Ly)
        rij = np.hypot(dx, dy)
        if rij <= 0 or rij > rcut:
            continue
        fmag = _morse_pair_force(rij, Cr, Ca, lr, la)
        scale = -fmag / rij
        fx_ij = scale * dx
        fy_ij = scale * dy
        fx[i] += fx_ij
        fy[i] += fy_ij
        fx[j] -= fx_ij
        fy[j] -= fy_ij

    return fx, fy


__all__ = [
    "morse_force_pairs",
    "morse_force_ckdtree",
]
