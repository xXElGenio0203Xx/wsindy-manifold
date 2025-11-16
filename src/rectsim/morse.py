"""Morse potential forces for the D'Orsogna model.

Implements the attractive/repulsive forces described by D'Orsogna et al.
(2006), using the Morse potential parameters ``Cr``, ``Ca``, ``lr`` and ``la``.
The linked-cell implementation mirrors that used in Bhaskar & Ziegelmeier
(2019) for efficient neighbourhood evaluation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import warnings

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


def minimal_image_displacement(dx: float, L: float) -> float:
    """Apply minimal image convention for periodic boundaries.
    
    Returns the shortest displacement between two points on a periodic domain.
    The result is in the range [-L/2, L/2).
    
    Parameters
    ----------
    dx : float
        Raw displacement (can be any value)
    L : float
        Domain length
        
    Returns
    -------
    float
        Minimal image displacement in [-L/2, L/2)
    """
    return dx - L * np.round(dx / L)


def _morse_pair_force(r: float, Cr: float, Ca: float, lr: float, la: float) -> float:
    """Return the scalar Morse force magnitude for a particle pair."""

    r = max(r, 1e-6)
    return Cr / lr * np.exp(-r / lr) - Ca / la * np.exp(-r / la)


def morse_force(
    x: ArrayLike,
    Lx: float,
    Ly: float,
    bc: str,
    Cr: float,
    Ca: float,
    lr: float,
    la: float,
    rcut: float,
    cell_list: Optional[CellList] = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """Compute Morse interaction forces on all particles.

    Parameters
    ----------
    x : ndarray, shape (N, 2)
        Particle positions.
    Lx, Ly : float
        Domain lengths.
    bc : {"periodic", "reflecting"}
        Boundary condition flag.
    Cr, Ca : float
        Repulsive and attractive Morse strengths.
    lr, la : float
        Repulsive and attractive length scales.
    rcut : float
        Interaction cut-off radius used for the linked-cell search.
    cell_list : CellList, optional
        Pre-built linked-cell structure. When ``None`` a fresh structure is
        constructed for the supplied positions.

    Returns
    -------
    (Fx, Fy) : tuple of ndarray
        Cartesian force components for each particle. Newton's third law is
        enforced explicitly so ``Fx.sum()`` and ``Fy.sum()`` are numerically
        close to zero (modulo floating-point error).
    """

    n = x.shape[0]
    fx = np.zeros(n, dtype=float)
    fy = np.zeros(n, dtype=float)

    if rcut < 3.0 * max(lr, la):
        warnings.warn(
            "rcut smaller than 3*max(lr, la); Morse force accuracy may degrade",
            RuntimeWarning,
            stacklevel=2,
        )

    local_cells = cell_list or build_cells(x, Lx, Ly, rcut, bc)

    for i, j, dx, dy, rij in iter_neighbors(x, local_cells, Lx, Ly, rcut, bc):
        if rij <= 1e-12:
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
    """Compute interaction forces using a linked-cell neighbour list.

    The function iterates over particle pairs yielded by
    :func:`rectsim.domain.iter_neighbors` which returns only pairs with
    separation less than or equal to ``rcut``. This avoids constructing
    full NxN matrices and keeps memory cost linear in N for sparse
    interactions.
    """

    return morse_force(
        x,
        Lx,
        Ly,
        bc,
        Cr,
        Ca,
        lr,
        la,
        rcut,
        cell_list=cell_list,
    )


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
    "minimal_image_displacement",
    "morse_force",
    "morse_force_pairs",
    "morse_force_ckdtree",
]
