"""Latent equation-free modeling utilities for rectangular density data."""

from .kde import (
    make_grid,
    kde_gaussian,
    trajectories_to_density_movie,
    minimal_image_dxdy,
    pair_dxdy_to_grid,
)
from .pod import fit_pod, restrict_pod, lift_pod, restrict_movie
from .mvar import (
    build_lagged,
    fit_mvar,
    forecast_step,
    rollout,
    save_mvar,
    load_mvar,
)
from .flow import train_from_trajectories, train_from_heatmap_npz, forecast
from .anim import ensure_writer, animate_heatmap_movie, animate_side_by_side
from .metrics import frame_metrics, series_metrics, save_metrics_json_csv

__all__ = [
    "make_grid",
    "kde_gaussian",
    "trajectories_to_density_movie",
    "minimal_image_dxdy",
    "pair_dxdy_to_grid",
    "fit_pod",
    "restrict_pod",
    "lift_pod",
    "restrict_movie",
    "build_lagged",
    "fit_mvar",
    "forecast_step",
    "rollout",
    "save_mvar",
    "load_mvar",
    "train_from_trajectories",
    "train_from_heatmap_npz",
    "forecast",
    "ensure_writer",
    "animate_heatmap_movie",
    "animate_side_by_side",
    "frame_metrics",
    "series_metrics",
    "save_metrics_json_csv",
]
