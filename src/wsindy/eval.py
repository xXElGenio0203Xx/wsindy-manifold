"""
Rollout evaluation metrics for WSINDy forecasts.

Compares a predicted trajectory ``U_pred`` against ground truth ``U_true``
and returns per-time and aggregate diagnostics.
"""

from __future__ import annotations

import numpy as np

from .grid import GridSpec
from .rhs import mass


# ── per-snapshot helpers ────────────────────────────────────────────

def r2_per_snapshot(u_true: np.ndarray, u_pred: np.ndarray) -> float:
    """R² between two 2-D snapshots (both flattened to vectors)."""
    a = u_true.ravel()
    b = u_pred.ravel()
    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((a - a.mean()) ** 2)
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else -np.inf
    return float(1.0 - ss_res / ss_tot)


def relative_l2(u_true: np.ndarray, u_pred: np.ndarray) -> float:
    r"""Relative L2 error :math:`\|u_{\rm true} - u_{\rm pred}\|_2 / \|u_{\rm true}\|_2`."""
    denom = np.linalg.norm(u_true.ravel())
    if denom == 0.0:
        return 0.0 if np.linalg.norm(u_pred.ravel()) == 0.0 else np.inf
    return float(np.linalg.norm((u_true - u_pred).ravel()) / denom)


# ── full rollout metrics ───────────────────────────────────────────

def rollout_metrics(
    U_true: np.ndarray,
    U_pred: np.ndarray,
    grid: GridSpec,
) -> dict:
    """Compute per-time and aggregate forecast diagnostics.

    Parameters
    ----------
    U_true : ndarray (T, nx, ny) – ground truth
    U_pred : ndarray (T, nx, ny) – predicted trajectory
    grid : GridSpec

    Returns
    -------
    dict with keys:

    =========== ======================================================
    key         description
    =========== ======================================================
    r2_t        ndarray (T,)  per-time R²
    rel_l2_t    ndarray (T,)  per-time relative L2 error
    r2_mean     float         mean R² over [0, T)
    mass_true   ndarray (T,)  true mass curve
    mass_pred   ndarray (T,)  predicted mass curve
    mass_drift  ndarray (T,)  relative mass drift from t = 0
    =========== ======================================================
    """
    T = min(U_true.shape[0], U_pred.shape[0])

    r2_t = np.array(
        [r2_per_snapshot(U_true[t], U_pred[t]) for t in range(T)]
    )
    rel_l2_t = np.array(
        [relative_l2(U_true[t], U_pred[t]) for t in range(T)]
    )

    mass_true = np.array([mass(U_true[t], grid) for t in range(T)])
    mass_pred = np.array([mass(U_pred[t], grid) for t in range(T)])

    m0 = abs(mass_pred[0]) if mass_pred[0] != 0.0 else 1.0
    mass_drift = (mass_pred - mass_pred[0]) / m0

    return {
        "r2_t": r2_t,
        "rel_l2_t": rel_l2_t,
        "r2_mean": float(r2_t.mean()),
        "mass_true": mass_true,
        "mass_pred": mass_pred,
        "mass_drift": mass_drift,
    }
