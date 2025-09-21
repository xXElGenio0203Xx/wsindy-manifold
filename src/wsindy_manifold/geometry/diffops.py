"""
Discrete differential operators on manifolds (chart/mesh backends).
"""
from __future__ import annotations
import numpy as np

def grad(f_vals: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Approximate ∇_M f via chart-wise finite differences (placeholder)."""
    # TODO: replace with proper differential operators (cotangent Laplacian or AD in charts)
    return np.gradient(f_vals, axis=0)

def laplace_beltrami(f_vals: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Approximate Δ_M f (placeholder)."""
    g = np.gradient(np.gradient(f_vals, axis=0), axis=0)
    return g
