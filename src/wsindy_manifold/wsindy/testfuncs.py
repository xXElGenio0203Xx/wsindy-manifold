"""
Compactly supported test functions (e.g., smooth bumps / Bernstein-like).
"""
from __future__ import annotations
import numpy as np

def bump(center: float, width: float):
    """1D C^∞ bump on [center-width, center+width] (product-separable for dD)."""
    def phi(x: np.ndarray) -> np.ndarray:
        z = (x - center) / width
        out = np.zeros_like(z)
        mask = np.abs(z) < 1.0
        out[mask] = np.exp(-1.0 / (1.0 - z[mask] ** 2))
        return out
    return phi
