"""
Quadrature on charts or meshes for weak integrals.
"""
from __future__ import annotations
import numpy as np

def integrate(values: np.ndarray, weights: np.ndarray) -> float:
    """Simple weighted sum as placeholder quadrature."""
    return float(np.tensordot(values, weights, axes=values.ndim-1))
