"""Proper orthogonal decomposition utilities for density fields."""

from __future__ import annotations

from typing import Dict

import numpy as np

Array = np.ndarray


def fit_pod(Rho: Array, energy_keep: float = 0.99, dx: float | None = None, dy: float | None = None) -> Dict[str, Array]:
    """Fit a POD basis to centered density snapshots.

    Parameters
    ----------
    Rho
        Snapshot matrix of shape ``(T, nc)``.
    energy_keep
        Fraction of singular-value energy to retain.
    dx, dy
        Optional grid spacings stored alongside the model for mass checks.

    Returns
    -------
    dict
        Contains the truncated basis, mean field, singular values, cumulative
        energy ratios, and grid spacings when provided.
    """

    if Rho.ndim != 2:
        raise ValueError("Rho must have shape (T, nc)")
    if not (0.0 < energy_keep <= 1.0):
        raise ValueError("energy_keep must lie in (0, 1]")

    Rho = np.asarray(Rho, dtype=float)
    mean = Rho.mean(axis=0)
    Xc = Rho - mean
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    energy = s**2
    total_energy = float(energy.sum())
    if total_energy == 0.0:
        cum_energy = np.ones_like(energy)
        d = 1
    else:
        cum_energy = np.cumsum(energy) / total_energy
        d = int(np.searchsorted(cum_energy, energy_keep) + 1)
    Ud = Vt[:d].T if d > 0 else np.empty((Rho.shape[1], 0), dtype=float)

    model: Dict[str, Array] = {
        "Ud": Ud,
        "mean": mean,
        "singular_values": s,
        "energy_ratio": cum_energy,
    }
    if dx is not None:
        model["dx"] = np.array(dx, dtype=float)
    if dy is not None:
        model["dy"] = np.array(dy, dtype=float)
    return model


def restrict_pod(Rho_t: Array, model: Dict[str, Array]) -> Array:
    """Project a density snapshot into latent coordinates."""

    Ud = model["Ud"]
    mean = model["mean"]
    return Ud.T @ (Rho_t - mean)


def lift_pod(y: Array, model: Dict[str, Array]) -> Array:
    """Lift latent coordinates back to a physical density snapshot."""

    Ud = model["Ud"]
    mean = model["mean"]
    rho_hat = Ud @ y + mean
    np.clip(rho_hat, 0.0, None, out=rho_hat)
    dx = float(model.get("dx", 1.0))
    dy = float(model.get("dy", 1.0))
    mass = float(np.sum(rho_hat) * dx * dy)
    if mass <= 0:
        raise RuntimeError("Lifted density has non-positive mass")
    rho_hat /= mass
    return rho_hat


def restrict_movie(Rho: Array, model: Dict[str, Array]) -> Array:
    """Vectorised restriction of a movie of density snapshots."""

    Ud = model["Ud"]
    mean = model["mean"]
    return (Rho - mean) @ Ud


__all__ = ["fit_pod", "restrict_pod", "lift_pod", "restrict_movie"]
