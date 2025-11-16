"""Order parameters and diagnostics for collective motion simulations.

Implements the polarization, angular momentum and nearest-neighbour
statistics used in Bhaskar & Ziegelmeier (2019) alongside generic error
metrics (RMSE, R², mean relative error, tolerance horizon) for EF-ROM
evaluation.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

try:  # Optional dependency for diagnostics tables
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

from .domain import pair_displacements

ArrayLike = np.ndarray

def rmse(y_true: ArrayLike, y_pred: ArrayLike, axis=None) -> float | ArrayLike:
    """Root-mean-square error between ``y_true`` and ``y_pred``."""

    diff = np.asarray(y_true) - np.asarray(y_pred)
    return np.sqrt(np.mean(diff**2, axis=axis))


def r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Coefficient of determination (R²)."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 1.0
    return 1.0 - ss_res / ss_tot


def mean_relative_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    eps: float = 1e-12,
    axis=None,
) -> float | ArrayLike:
    """Mean relative error with epsilon safeguard."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rel = np.abs(y_true - y_pred) / (np.abs(y_true) + eps)
    return np.mean(rel, axis=axis)


def tolerance_horizon(rel_err_t: ArrayLike, tol: float = 0.10) -> int:
    """Return first index where relative error exceeds ``tol`` (else length)."""

    rel_err_t = np.asarray(rel_err_t)
    exceeding = np.argwhere(rel_err_t >= tol)
    return int(exceeding[0, 0]) if exceeding.size else int(len(rel_err_t))



def polarization(v: ArrayLike) -> float:
    """Compute the polarization order parameter."""

    total = np.linalg.norm(np.sum(v, axis=0))
    denom = np.sum(np.linalg.norm(v, axis=1))
    return 0.0 if denom == 0 else float(total / denom)


def _center_of_mass(x: ArrayLike) -> ArrayLike:
    """Return the mean position of all agents."""

    return np.mean(x, axis=0)


def angular_momentum(x: ArrayLike, v: ArrayLike) -> float:
    """Compute normalized angular momentum relative to the center of mass."""

    com = _center_of_mass(x)
    rel = x - com
    cross = rel[:, 0] * v[:, 1] - rel[:, 1] * v[:, 0]
    numerator = np.linalg.norm(np.sum(cross))
    denom = np.sum(np.linalg.norm(rel, axis=1) * np.linalg.norm(v, axis=1))
    return 0.0 if denom == 0 else float(numerator / denom)


def abs_angular_momentum(x: ArrayLike, v: ArrayLike) -> float:
    """Compute the absolute angular momentum order parameter."""

    com = _center_of_mass(x)
    rel = x - com
    cross = np.abs(rel[:, 0] * v[:, 1] - rel[:, 1] * v[:, 0])
    denom = np.sum(np.linalg.norm(rel, axis=1) * np.linalg.norm(v, axis=1))
    return 0.0 if denom == 0 else float(np.sum(cross) / denom)


def dnn(x: ArrayLike, Lx: float, Ly: float, bc: str) -> float:
    """Mean nearest-neighbor distance."""

    _, _, rij, _ = pair_displacements(x, Lx, Ly, bc)
    np.fill_diagonal(rij, np.inf)
    nearest = np.min(rij, axis=1)
    return float(np.mean(nearest))


def compute_timeseries(
    traj: ArrayLike,
    vel: ArrayLike,
    times: Iterable[float],
    Lx: float,
    Ly: float,
    bc: str,
) -> "pd.DataFrame":
    """Compute diagnostic time series for a trajectory."""

    if pd is None:  # pragma: no cover - optional dependency
        raise ImportError("pandas is required to compute time series")

    records = []
    for frame, time_point in enumerate(times):
        x = traj[frame]
        v = vel[frame]
        records.append(
            {
                "time": time_point,
                "polarization": polarization(v),
                "angular_momentum": angular_momentum(x, v),
                "abs_angular_momentum": abs_angular_momentum(x, v),
                "dnn": dnn(x, Lx, Ly, bc),
            }
        )
    return pd.DataFrame.from_records(records)


__all__ = [
    "polarization",
    "angular_momentum",
    "abs_angular_momentum",
    "dnn",
    "compute_timeseries",
    "rmse",
    "r2",
    "mean_relative_error",
    "tolerance_horizon",
]
