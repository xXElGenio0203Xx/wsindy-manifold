"""
Restriction: trajectories → density movies via (geodesic) KDE.
"""
from __future__ import annotations
import numpy as np
from .geodesic_kde import geodesic_kde

def trajectories_to_density(trajectories: list[np.ndarray], grid: np.ndarray, bandwidth: float):
    """Compute density frames from particle trajectories.
    trajectories: list over times, each element is (N_i × d) points on M (coords).
    grid: (G × d) evaluation coordinates.
    """
    frames = []
    for pts in trajectories:
        frames.append(geodesic_kde(pts, grid, bandwidth))
    return np.stack(frames, axis=0)
