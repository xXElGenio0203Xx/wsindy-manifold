"""
Scoring metrics for WSINDy model fits.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from numpy.typing import NDArray


def r2_score(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
) -> float:
    """Coefficient of determination R².

    Parameters
    ----------
    y_true, y_pred : 1-D ndarrays of the same length.

    Returns
    -------
    r2 : float
        1 − SS_res / SS_tot.  Returns 0.0 if SS_tot ≈ 0.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-30:
        return 0.0
    return 1.0 - ss_res / ss_tot


def wsindy_fit_metrics(
    b: NDArray[np.floating],
    G: NDArray[np.floating],
    w: NDArray[np.floating],
) -> Dict[str, float]:
    """Compute fit-quality metrics for a WSINDy model.

    Parameters
    ----------
    b : ndarray, shape ``(K,)``
        Weak-form LHS vector.
    G : ndarray, shape ``(K, M)``
        Weak-form RHS matrix.
    w : ndarray, shape ``(M,)``
        Coefficient vector (may contain zeros for inactive terms).

    Returns
    -------
    metrics : dict
        ``"residual_norm"`` — ‖b − Gw‖₂
        ``"r2"`` — R² score of Gw vs b
        ``"relative_l2"`` — ‖b − Gw‖₂ / ‖b‖₂  (inf if ‖b‖≈0)
        ``"sparsity"`` — number of active (non-zero) coefficients
    """
    b = np.asarray(b, dtype=np.float64).ravel()
    G = np.asarray(G, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).ravel()

    residual = b - G @ w
    res_norm = float(np.linalg.norm(residual))
    b_norm = float(np.linalg.norm(b))

    return {
        "residual_norm": res_norm,
        "r2": r2_score(b, G @ w),
        "relative_l2": res_norm / b_norm if b_norm > 1e-30 else float("inf"),
        "sparsity": int(np.count_nonzero(w)),
    }
