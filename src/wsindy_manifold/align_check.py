"""Utilities to cross-check Vicsek-style alignment implementations."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from rectsim.domain import pair_displacements
from rectsim.dynamics import apply_alignment_step

Array = np.ndarray


def neighbor_finder_ball(
    x: Array,
    Lx: float,
    Ly: float,
    bc: str,
    rcut: float,
) -> List[np.ndarray]:
    """Return neighbour indices within ``rcut`` using the current boundary condition."""

    _, _, rij, _ = pair_displacements(x, Lx, Ly, bc)
    neighbours: List[np.ndarray] = []
    idx_all = np.arange(x.shape[0])
    for i in range(x.shape[0]):
        mask = (rij[i] <= rcut) & (idx_all != i)
        neighbours.append(idx_all[mask])
    return neighbours


def mean_heading_unit(p_neighbors: Array) -> Optional[Array]:
    """Return the unit-normalised mean heading vector or ``None`` when undefined."""

    if p_neighbors.size == 0:
        return None
    mean_vec = np.sum(p_neighbors, axis=0)
    norm = float(np.linalg.norm(mean_vec))
    if norm < 1e-12:
        return None
    return mean_vec / norm


def gold_alignment_step(
    x: Array,
    p: Array,
    Lx: float,
    Ly: float,
    bc: str,
    mu_r: float,
    lV: float,
    Dtheta: float,
    dt: float,
    neighbor_finder: Callable[[Array, float, float, str, float], Sequence[np.ndarray]],
    rng: Optional[np.random.Generator] = None,
) -> Array:
    """AIM-1 Eq. 6 Vicsek update implemented in vector form."""

    if rng is None:
        rng = np.random.default_rng()

    neighbours = neighbor_finder(x, Lx, Ly, bc, lV)
    noise_scale = np.sqrt(max(0.0, 2.0 * Dtheta * dt))

    p_new = np.empty_like(p)
    for i in range(p.shape[0]):
        idx = neighbours[i]
        p_bar = mean_heading_unit(p[idx]) if len(idx) else None
        drift = np.zeros(2, dtype=float)
        if p_bar is not None and mu_r > 0.0:
            drift = mu_r * (p_bar - p[i])
        noise = noise_scale * rng.normal(size=2)
        p_tmp = p[i] + drift * dt + noise
        norm = float(np.linalg.norm(p_tmp))
        if norm < 1e-12:
            p_new[i] = p[i]
        else:
            p_new[i] = p_tmp / norm
    return p_new


def step_yours_vs_gold(
    x: Array,
    p: Array,
    Lx: float,
    Ly: float,
    bc: str,
    mu_r: float,
    lV: float,
    Dtheta: float,
    dt: float,
    neighbor_finder: Callable[[Array, float, float, str, float], Sequence[np.ndarray]] = neighbor_finder_ball,
    seed: Optional[int] = None,
) -> Tuple[Array, Array]:
    """Return a pair ``(p_yours, p_gold)`` for a single alignment step."""

    rng = np.random.default_rng(seed)
    rng_gold = np.random.default_rng(seed)
    p_yours = apply_alignment_step(
        x,
        p,
        Lx,
        Ly,
        bc,
        mu_r,
        lV,
        Dtheta,
        dt,
        neighbor_finder=neighbor_finder,
        rng=rng,
    )
    p_gold = gold_alignment_step(
        x,
        p,
        Lx,
        Ly,
        bc,
        mu_r,
        lV,
        Dtheta,
        dt,
        neighbor_finder=neighbor_finder,
        rng=rng_gold,
    )
    return p_yours, p_gold


def order_parameter(p: Array) -> float:
    """Polarisation order parameter ψ = ||⟨p_i⟩|| ∈ [0, 1]."""

    return float(np.linalg.norm(np.mean(p, axis=0)))


def angle_diff_mean(p1: Array, p2: Array) -> float:
    """Mean absolute angular difference between two heading fields in radians."""

    dots = np.clip(np.sum(p1 * p2, axis=1), -1.0, 1.0)
    return float(np.mean(np.abs(np.arccos(dots))))


__all__ = [
    "neighbor_finder_ball",
    "mean_heading_unit",
    "gold_alignment_step",
    "step_yours_vs_gold",
    "order_parameter",
    "angle_diff_mean",
]
