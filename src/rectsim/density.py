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
            # First attempt: use imageio's ffmpeg writer directly. If that fails,
            # fall back to writing PNG frames and assembling with the ffmpeg
            # binary provided by imageio-ffmpeg (if available).
            try:
                w = imageio.get_writer(movie_path, fps=fps, format="FFMPEG")
                with w as writer:
                    for frame in rho:
                        frame_norm = frame - frame.min()
                        if frame_norm.max() > 0:
                            frame_norm /= frame_norm.max()
                        image = (255 * frame_norm).astype(np.uint8)
                        writer.append_data(image)
            except Exception:
                # fallback: write PNG frames then call ffmpeg binary
                try:
                    import imageio_ffmpeg as iio_ff
                    import subprocess

                    frames_dir = out_path / "frames"
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    for i, frame in enumerate(rho):
                        frame_norm = frame - frame.min()
                        if frame_norm.max() > 0:
                            frame_norm /= frame_norm.max()
                        image = (255 * frame_norm).astype(np.uint8)
                        frame_path = frames_dir / f"frame_{i:04d}.png"
                        imageio.imwrite(frame_path, image)

                    ffmpeg_exe = iio_ff.get_ffmpeg_exe()
                    cmd = [
                        ffmpeg_exe,
                        "-y",
                        "-framerate",
                        str(fps),
                        "-i",
                        str(frames_dir / "frame_%04d.png"),
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        str(movie_path),
                    ]
                    subprocess.run(cmd, check=True)
                except Exception as exc:
                    raise RuntimeError(
                        "Failed to create animation: ensure 'imageio-ffmpeg' is installed "
                        "and that ffmpeg can run in this environment."
                    ) from exc

    return rho, xgrid, ygrid


__all__ = ["hist2d_movie"]
