"""Metric utilities for assessing latent heatmap forecasts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

Array = np.ndarray

try:  # Optional structural similarity
    from skimage.metrics import structural_similarity as _ssim
except Exception:  # pragma: no cover - optional dependency
    _ssim = None


def frame_metrics(
    r_true: Array,
    r_pred: Array,
    dx: float,
    dy: float,
    grid_shape: Tuple[int, int] | None = None,
    ssim_data_range: float | None = None,
) -> Dict[str, float | None]:
    """Compute error metrics for a single pair of density frames."""

    r_true = np.asarray(r_true, dtype=float)
    r_pred = np.asarray(r_pred, dtype=float)
    if r_true.shape != r_pred.shape:
        raise ValueError("Frame arrays must have identical shapes")

    delta = r_pred - r_true
    eps = 1e-12

    mass_true = float(r_true.sum() * dx * dy)
    mass_pred = float(r_pred.sum() * dx * dy)
    l2 = float(np.linalg.norm(delta))
    mse = float(np.mean(delta**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(delta)))
    mre = float(np.mean(np.abs(delta) / (np.abs(r_true) + eps)))

    centered = r_true - r_true.mean()
    denom = float(np.sum(centered**2))
    num = float(np.sum(delta**2))
    if denom < eps:
        r2 = 1.0 if num < eps else 0.0
    else:
        r2 = float(1.0 - num / denom)

    ssim_val: float | None = None
    if _ssim is not None and grid_shape is not None:
        ny, nx = grid_shape
        frame_true = r_true.reshape(ny, nx)
        frame_pred = r_pred.reshape(ny, nx)
        data_range = ssim_data_range
        if data_range is None:
            data_range = float(max(frame_true.max(), frame_pred.max()) - min(frame_true.min(), frame_pred.min()))
            if data_range <= eps:
                data_range = 1.0
        try:
            ssim_val = float(_ssim(frame_true, frame_pred, data_range=data_range))
        except Exception:  # pragma: no cover - ssim may fail for degenerate frames
            ssim_val = None

    return {
        "mass_true": mass_true,
        "mass_pred": mass_pred,
        "l2": l2,
        "rmse": rmse,
        "mae": mae,
        "mre": mre,
        "r2": r2,
        "ssim": ssim_val,
    }


def series_metrics(
    R_true: Array,
    R_pred: Array,
    times: Array | None,
    dx: float,
    dy: float,
    grid_shape: Tuple[int, int] | None = None,
) -> Dict[str, object]:
    """Evaluate metrics for an entire sequence of heatmaps."""

    R_true = np.asarray(R_true, dtype=float)
    R_pred = np.asarray(R_pred, dtype=float)
    if R_true.shape != R_pred.shape:
        raise ValueError("True and predicted arrays must have identical shapes")

    T = R_true.shape[0]
    if times is not None and times.shape[0] != T:
        raise ValueError("Times array length must match the number of frames")

    per_frame: List[Dict[str, float | None]] = []
    for i in range(T):
        per_frame.append(frame_metrics(R_true[i], R_pred[i], dx, dy, grid_shape=grid_shape))

    aggregate: Dict[str, float] = {}
    keys = per_frame[0].keys() if per_frame else []
    for key in keys:
        values = [entry[key] for entry in per_frame if entry[key] is not None]
        if not values:
            continue
        aggregate[f"{key}_mean"] = float(np.mean(values))
        aggregate[f"{key}_median"] = float(np.median(values))

    t_tol = np.nan
    if per_frame:
        mre_series = [entry["mre"] for entry in per_frame if entry["mre"] is not None]
        if mre_series:
            threshold_indices = [idx for idx, val in enumerate(mre_series) if val >= 0.10]
            if threshold_indices:
                first_idx = threshold_indices[0]
                if times is not None:
                    t_tol = float(times[first_idx])
                else:
                    t_tol = float(first_idx)
            else:
                t_tol = float(times[-1]) if times is not None else float(T - 1)

    return {"per_frame": per_frame, "aggregate": aggregate, "t_tol": t_tol}


def save_metrics_json_csv(out_dir: str | Path, metrics: Dict[str, object]) -> Tuple[str, str]:
    """Persist metrics as JSON and CSV files."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / "metrics.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    csv_path = out_path / "metrics.csv"
    per_frame = metrics.get("per_frame", [])
    if isinstance(per_frame, list) and per_frame:
        fieldnames = sorted({key for frame in per_frame for key in frame.keys()})
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for frame in per_frame:
                writer.writerow(frame)
    else:
        csv_path.touch()

    return str(json_path), str(csv_path)


__all__ = ["frame_metrics", "series_metrics", "save_metrics_json_csv"]
