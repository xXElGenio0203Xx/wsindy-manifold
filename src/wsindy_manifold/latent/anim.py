"""Animation helpers for latent heatmap pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

Array = np.ndarray


def ensure_writer() -> str:
    """Return the preferred available matplotlib animation writer."""

    candidates = ["ffmpeg", "pillow"]
    for name in candidates:
        try:
            if animation.writers.is_available(name):
                return name
        except (RuntimeError, AttributeError):
            continue
    return candidates[-1]


def _grid_shape_from_meta(dx: float, dy: float, Lx: float, Ly: float, nc: int) -> Tuple[int, int]:
    nx = max(1, int(round(Lx / dx)))
    ny = max(1, int(round(Ly / dy)))
    if nx * ny != nc:
        raise ValueError("Grid shape inferred from meta does not match data length")
    return nx, ny


def _reshape_frames(Rho: Array, nx: int, ny: int) -> Array:
    return Rho.reshape(Rho.shape[0], ny, nx)


def _resolve_limits(Rho: Array, vmin: float | None, vmax: float | None) -> Tuple[float, float]:
    if vmin is not None and vmax is not None:
        return vmin, vmax
    flat = Rho.ravel()
    if vmin is None:
        vmin = float(np.percentile(flat, 1))
    if vmax is None:
        vmax = float(np.percentile(flat, 99))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12
    return vmin, vmax


def animate_heatmap_movie(
    Rho: Array,
    Xc: Array,
    dx: float,
    dy: float,
    Lx: float,
    Ly: float,
    out_path: str,
    fps: int = 20,
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str = "",
) -> str:
    """Create an animation for a sequence of flattened density fields."""

    Rho = np.asarray(Rho, dtype=float)
    if Xc.shape[0] != Rho.shape[1]:
        raise ValueError("Xc and Rho dimensions are inconsistent")
    nc = Rho.shape[1]
    nx, ny = _grid_shape_from_meta(dx, dy, Lx, Ly, nc)
    frames = _reshape_frames(Rho, nx, ny)
    vmin, vmax = _resolve_limits(frames, vmin, vmax)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(
        frames[0],
        extent=(0, Lx, 0, Ly),
        origin="lower",
        cmap=cmap,
        animated=True,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax)

    def update(frame_index: int):
        im.set_array(frames[frame_index])
        return (im,)

    interval = 1000.0 / max(fps, 1)
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frames.shape[0],
        interval=interval,
        blit=True,
    )

    writer_name = ensure_writer()
    out_path_obj = Path(out_path)
    if writer_name == "ffmpeg" and out_path_obj.suffix.lower() != ".mp4":
        out_path_obj = out_path_obj.with_suffix(".mp4")
    elif writer_name != "ffmpeg" and out_path_obj.suffix.lower() != ".gif":
        out_path_obj = out_path_obj.with_suffix(".gif")

    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path_obj, writer=writer_name, fps=fps)
    plt.close(fig)
    return str(out_path_obj)


def animate_side_by_side(
    R_true: Array,
    R_pred: Array,
    Xc: Array,
    dx: float,
    dy: float,
    Lx: float,
    Ly: float,
    out_path: str,
    fps: int = 20,
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    titles: Tuple[str, str] = ("True", "Pred"),
    include_error: bool = True,
) -> str:
    """Create a side-by-side animation comparing true and predicted fields."""

    R_true = np.asarray(R_true, dtype=float)
    R_pred = np.asarray(R_pred, dtype=float)
    if R_true.shape != R_pred.shape:
        raise ValueError("True and predicted arrays must share the same shape")

    if Xc.shape[0] != R_true.shape[1]:
        raise ValueError("Xc and heatmap dimensions are inconsistent")
    nc = R_true.shape[1]
    nx, ny = _grid_shape_from_meta(dx, dy, Lx, Ly, nc)
    true_frames = _reshape_frames(R_true, nx, ny)
    pred_frames = _reshape_frames(R_pred, nx, ny)
    combined = np.concatenate([true_frames, pred_frames], axis=0)
    vmin, vmax = _resolve_limits(combined, vmin, vmax)

    ncols = 3 if include_error else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), sharex=True, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.ravel()

    ims = []
    ims.append(
        axes[0].imshow(
            true_frames[0],
            extent=(0, Lx, 0, Ly),
            origin="lower",
            cmap=cmap,
            animated=True,
            vmin=vmin,
            vmax=vmax,
        )
    )
    axes[0].set_title(titles[0])

    ims.append(
        axes[1].imshow(
            pred_frames[0],
            extent=(0, Lx, 0, Ly),
            origin="lower",
            cmap=cmap,
            animated=True,
            vmin=vmin,
            vmax=vmax,
        )
    )
    axes[1].set_title(titles[1])

    if include_error:
        diff0 = np.abs(pred_frames[0] - true_frames[0])
        ims.append(
            axes[2].imshow(
                diff0,
                extent=(0, Lx, 0, Ly),
                origin="lower",
                cmap="inferno",
                animated=True,
            )
        )
        axes[2].set_title("|Δ|")

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.colorbar(ims[0], ax=axes[:2], shrink=0.9)
    if include_error:
        fig.colorbar(ims[2], ax=axes[2], shrink=0.9)

    def update(frame_index: int):
        ims[0].set_array(true_frames[frame_index])
        ims[1].set_array(pred_frames[frame_index])
        if include_error:
            ims[2].set_array(np.abs(pred_frames[frame_index] - true_frames[frame_index]))
        return tuple(ims)

    interval = 1000.0 / max(fps, 1)
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=true_frames.shape[0],
        interval=interval,
        blit=True,
    )

    writer_name = ensure_writer()
    out_path_obj = Path(out_path)
    if writer_name == "ffmpeg" and out_path_obj.suffix.lower() != ".mp4":
        out_path_obj = out_path_obj.with_suffix(".mp4")
    elif writer_name != "ffmpeg" and out_path_obj.suffix.lower() != ".gif":
        out_path_obj = out_path_obj.with_suffix(".gif")

    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path_obj, writer=writer_name, fps=fps)
    plt.close(fig)
    return str(out_path_obj)


__all__ = ["ensure_writer", "animate_heatmap_movie", "animate_side_by_side"]
