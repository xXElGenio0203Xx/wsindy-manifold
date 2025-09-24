"""Density estimation and movie generation for trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import imageio
import numpy as np
from scipy.ndimage import gaussian_filter

ArrayLike = np.ndarray


def hist2d_movie(
    traj: ArrayLike,
    times: Iterable[float],
    Lx: float,
    Ly: float,
    nx: int,
    ny: int,
    bandwidth: float,
    bc: str,
    out_dir: str | Path | None = None,
    animate: bool = False,
    fps: int = 24,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Compute a Gaussian-smoothed density movie and optionally save artifacts."""

    out_path = Path(out_dir) if out_dir is not None else None
    x_edges = np.linspace(0.0, Lx, nx + 1)
    y_edges = np.linspace(0.0, Ly, ny + 1)
    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]

    grids = []
    mode = "wrap" if bc == "periodic" else "nearest"

    for frame in range(traj.shape[0]):
        hist, _, _ = np.histogram2d(
            traj[frame, :, 0],
            traj[frame, :, 1],
            bins=[x_edges, y_edges],
            range=[[0.0, Lx], [0.0, Ly]],
        )
        density = hist / (dx * dy)
        if bandwidth > 0:
            density = gaussian_filter(density, sigma=bandwidth, mode=mode)
        grids.append(density)

    rho = np.stack(grids, axis=0)
    xgrid = 0.5 * (x_edges[:-1] + x_edges[1:])
    ygrid = 0.5 * (y_edges[:-1] + y_edges[1:])
    times_arr = np.array(list(times))

    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)
        np.savez(out_path / "density.npz", rho=rho, xgrid=xgrid, ygrid=ygrid, times=times_arr)

        if animate:
            movie_path = out_path / "density_anim.mp4"
            with imageio.get_writer(movie_path, fps=fps) as writer:
                for frame in rho:
                    frame_norm = frame - frame.min()
                    if frame_norm.max() > 0:
                        frame_norm /= frame_norm.max()
                    image = (255 * frame_norm).astype(np.uint8)
                    writer.append_data(image)

    return rho, xgrid, ygrid


__all__ = ["hist2d_movie"]
