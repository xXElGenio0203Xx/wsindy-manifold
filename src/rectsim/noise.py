"""Centralized noise utilities for Vicsek models.

This module provides consistent angular noise generation across all model backends,
ensuring that uniform and Gaussian noise have matching statistical properties when
desired.
"""

from __future__ import annotations

import numpy as np


def angle_noise(
    rng: np.random.Generator,
    kind: str,
    eta: float,
    size,
    *,
    match_variance: bool = True,
) -> np.ndarray:
    """Generate angular noise for Vicsek-type models.
    
    Parameters
    ----------
    rng : np.random.Generator
        Random number generator instance.
    kind : {"uniform", "gaussian"}
        Type of noise distribution.
    eta : float
        Noise amplitude parameter in radians.
        - For uniform: noise is drawn from [-η/2, +η/2]
        - For gaussian: if match_variance=True, σ = η/√12 to match uniform variance
    size : int or tuple
        Shape of output array, typically (N,) for N particles.
    match_variance : bool, optional
        If True and kind="gaussian", set σ = η/√12 so the variance matches
        the uniform distribution. If False, use σ = η directly.
        
    Returns
    -------
    noise : np.ndarray
        Array of angular perturbations with the specified shape.
        
    Notes
    -----
    The original Vicsek model uses uniform noise in [-η/2, +η/2] where η ∈ [0,π].
    The variance of this distribution is Var[U(-η/2, η/2)] = η²/12.
    
    For Gaussian noise with equivalent variance: σ = η/√12 ≈ 0.289·η
    
    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> # Uniform noise as in original Vicsek model
    >>> phi = angle_noise(rng, "uniform", eta=0.5, size=100)
    >>> # Gaussian with matching variance
    >>> phi = angle_noise(rng, "gaussian", eta=0.5, size=100, match_variance=True)
    """
    if kind == "uniform":
        return rng.uniform(-eta / 2, +eta / 2, size=size)
    elif kind == "gaussian":
        sigma = eta / np.sqrt(12) if match_variance else eta
        return rng.normal(0.0, sigma, size=size)
    else:
        raise ValueError(f"Unknown noise kind: '{kind}'. Expected 'uniform' or 'gaussian'.")


def noise_variance(kind: str, eta: float, match_variance: bool = True) -> float:
    """Compute the variance of angle_noise for given parameters.
    
    Parameters
    ----------
    kind : {"uniform", "gaussian"}
        Type of noise distribution.
    eta : float
        Noise amplitude parameter in radians.
    match_variance : bool
        Whether variance matching is enabled for Gaussian noise.
        
    Returns
    -------
    variance : float
        Theoretical variance of the noise distribution.
    """
    if kind == "uniform":
        return eta**2 / 12
    elif kind == "gaussian":
        sigma = eta / np.sqrt(12) if match_variance else eta
        return sigma**2
    else:
        raise ValueError(f"Unknown noise kind: '{kind}'")


__all__ = ["angle_noise", "noise_variance"]
