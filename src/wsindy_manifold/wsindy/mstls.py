"""
Multi-Step Thresholded Least Squares (MSTLS) — sparse regression loop.
"""
from __future__ import annotations
import numpy as np

def mstls(A: np.ndarray, b: np.ndarray, lam: float = 1e-6, tau_schedule=(1e-1, 5e-2, 1e-2)):
    """Return selected support and coefficients using ridge + hard-thresholding refits.
    Args:
        A: (n × p) design matrix
        b: (n,) target vector
        lam: ridge regularization
        tau_schedule: monotonically decreasing thresholds
    """
    # Standardize columns
    col_norms = np.linalg.norm(A, axis=0) + 1e-12
    Astd = A / col_norms

    S = np.arange(A.shape[1])
    for tau in tau_schedule:
        # Ridge on current support
        As = Astd[:, S]
        theta = np.linalg.solve(As.T @ As + lam * np.eye(len(S)), As.T @ b)
        keep = np.where(np.abs(theta) >= tau)[0]
        S = S[keep] if keep.size > 0 else S  # avoid empty set

    # Final refit on selected support (unstandardized back)
    As = A[:, S]
    theta_S = np.linalg.lstsq(As, b, rcond=None)[0]
    theta_full = np.zeros(A.shape[1])
    theta_full[S] = theta_S
    return S, theta_full
