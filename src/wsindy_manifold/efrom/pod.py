"""
POD/SVD compression utilities.
"""
from __future__ import annotations
import numpy as np

def pod(X: np.ndarray, r: int):
    """Return leading r modes: X ≈ Φ Σ V^T (Φ = left singular vectors)."""
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return U[:, :r], S[:r], Vt[:r, :]
