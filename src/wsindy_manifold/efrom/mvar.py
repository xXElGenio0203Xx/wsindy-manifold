"""
Simple MVAR(p) fit using least squares (demo-grade).
"""
from __future__ import annotations
import numpy as np

def mvar_fit(R: np.ndarray, p: int):
    """R: (T × r) latent time series; returns {A_j} and intercept c."""
    T, r = R.shape
    Y = R[p:]
    X = []
    for j in range(1, p+1):
        X.append(R[p-j:T-j])
    X = np.hstack(X)
    # Add intercept
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    coeff, *_ = np.linalg.lstsq(X, Y, rcond=None)
    A = [coeff[j*r:(j+1)*r].T for j in range(p)]
    c = coeff[-1]
    return A, c
