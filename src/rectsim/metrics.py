"""Order parameters and diagnostics for collective motion simulations."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

try:  # Optional dependency for diagnostics tables
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

from .domain import pair_displacements

ArrayLike = np.ndarray


def polarization(v: ArrayLike) -> float:
    """Compute the polarization order parameter."""

    total = np.linalg.norm(np.sum(v, axis=0))
    denom = np.sum(np.linalg.norm(v, axis=1))
    return 0.0 if denom == 0 else float(total / denom)


def _center_of_mass(x: ArrayLike) -> ArrayLike:
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
]
