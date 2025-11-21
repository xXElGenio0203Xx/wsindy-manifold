"""
I/O utilities for reproducible simulation and EF-ROM pipelines.

Provides helpers for:
- Creating run directories with timestamped names
- Saving manifests with git commits and metadata
- Saving arrays, CSVs, and videos with standardized naming
"""
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

Array = np.ndarray


def get_git_commit() -> str:
    """Get short git commit hash, or 'unknown' if not in git repo."""
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
        return commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


def create_run_id(seed: int = 0) -> str:
    """
    Create standardized run ID: YYYYMMDD-HHMMSS_<gitshort>_<seed>
    
    Args:
        seed: Random seed for this run
        
    Returns:
        run_id: Unique run identifier
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    git_short = get_git_commit()
    return f"{timestamp}_{git_short}_{seed}"


def create_run_dir(root: str, sim_name: str, seed: int = 0) -> Path:
    """
    Create run directory: <root>/<sim_name>__<run_id>/
    
    Args:
        root: Root directory (e.g., 'simulations')
        sim_name: Simulation name slug (e.g., 'vicsek_discrete_eta0p30')
        seed: Random seed
        
    Returns:
        run_dir: Path to created directory
    """
    run_id = create_run_id(seed)
    run_dir = Path(root) / f"{sim_name}__{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_manifest(
    root: Path,
    sim_name: str,
    config_path: str,
    simulator: str,
    seed: int = 0,
    notes: str = "",
    **extra
) -> None:
    """
    Save manifest.json with run metadata.
    
    Args:
        root: Run directory
        sim_name: Simulation name
        config_path: Path to config file used
        simulator: Simulator identifier
        seed: Random seed
        notes: Optional notes
        **extra: Additional metadata fields
    """
    manifest = {
        "run_id": root.name.split("__", 1)[1] if "__" in root.name else "unknown",
        "sim_name": str(sim_name),
        "datetime": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "seed": int(seed),
        "config_path": str(config_path),
        "simulator": str(simulator),
        "notes": str(notes),
    }
    
    # Convert extra kwargs to JSON-serializable types
    for key, value in extra.items():
        if isinstance(value, (np.integer, np.int64, np.int32)):
            manifest[key] = int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32)):
            manifest[key] = float(value)
        elif isinstance(value, (bool, int, float, str)) or value is None:
            manifest[key] = value
        else:
            manifest[key] = str(value)
    
    manifest_path = root / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Saved manifest: {manifest_path}")


def save_arrays(path: Path, **named_arrays) -> None:
    """
    Save multiple numpy arrays to .npy files.
    
    Args:
        path: Directory to save arrays
        **named_arrays: name=array pairs (e.g., traj=traj, vel=vel)
    """
    path.mkdir(parents=True, exist_ok=True)
    
    for name, array in named_arrays.items():
        np.save(path / f"{name}.npy", array)
    
    print(f"Saved {len(named_arrays)} arrays to {path}/")


def save_csv(path: Path, df: pd.DataFrame, name: str) -> None:
    """
    Save DataFrame to CSV.
    
    Args:
        path: Directory to save CSV
        df: DataFrame to save
        name: Filename (without .csv extension)
    """
    path.mkdir(parents=True, exist_ok=True)
    csv_path = path / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")


def save_video(
    path: Path,
    frames: Array,
    fps: int,
    name: str,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None
) -> None:
    """
    Save 2D heatmap frames as MP4 video.
    
    Args:
        path: Directory to save video
        frames: Array of shape (T, ny, nx)
        fps: Frames per second
        name: Filename (without .mp4 extension)
        cmap: Colormap name
        vmin: Min value for colormap (auto if None)
        vmax: Max value for colormap (auto if None)
        title: Video title
    """
    path.mkdir(parents=True, exist_ok=True)
    video_path = path / f"{name}.mp4"
    
    if vmin is None:
        vmin = frames.min()
    if vmax is None:
        vmax = frames.max()
    
    T, ny, nx = frames.shape
    
    # Subsample if too many frames
    max_frames = 500
    if T > max_frames:
        indices = np.linspace(0, T - 1, max_frames, dtype=int)
        frames = frames[indices]
        T = max_frames
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Initial frame
    im = ax.imshow(frames[0].T, origin='lower', cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect='auto')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Density', fontsize=11)
    
    # Time text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       color='white', fontsize=11, va='top',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    
    # Write video
    writer = FFMpegWriter(fps=fps, bitrate=2000)
    
    with writer.saving(fig, video_path, dpi=100):
        for t in range(T):
            im.set_data(frames[t].T)
            time_text.set_text(f'Frame {t+1}/{T}')
            writer.grab_frame()
    
    plt.close(fig)
    print(f"Saved video: {video_path}")


def side_by_side_video(
    path: Path,
    left_frames: Array,
    right_frames: Array,
    lower_strip_timeseries: Optional[Array] = None,
    name: str = "comparison",
    fps: int = 20,
    cmap: str = "viridis",
    titles: tuple[str, str] = ("Ground Truth", "Prediction")
) -> None:
    """
    Create side-by-side comparison video with optional error timeseries below.
    
    Args:
        path: Directory to save video
        left_frames: Left panel frames (T, ny, nx)
        right_frames: Right panel frames (T, ny, nx)
        lower_strip_timeseries: Optional timeseries (T,) to plot below
        name: Filename (without .mp4 extension)
        fps: Frames per second
        cmap: Colormap name
        titles: (left_title, right_title)
    """
    path.mkdir(parents=True, exist_ok=True)
    video_path = path / f"{name}.mp4"
    
    T = len(left_frames)
    
    # Subsample if needed
    max_frames = 500
    if T > max_frames:
        indices = np.linspace(0, T - 1, max_frames, dtype=int)
        left_frames = left_frames[indices]
        right_frames = right_frames[indices]
        if lower_strip_timeseries is not None:
            lower_strip_timeseries = lower_strip_timeseries[indices]
        T = max_frames
    
    # Shared colormap limits
    vmin = min(left_frames.min(), right_frames.min())
    vmax = max(left_frames.max(), right_frames.max())
    
    # Create figure
    if lower_strip_timeseries is not None:
        fig = plt.figure(figsize=(14, 7))
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.3, wspace=0.3)
        ax_left = fig.add_subplot(gs[0, 0])
        ax_right = fig.add_subplot(gs[0, 1])
        ax_ts = fig.add_subplot(gs[1, :])
    else:
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))
        ax_ts = None
    
    # Left panel
    im_left = ax_left.imshow(left_frames[0].T, origin='lower', cmap=cmap,
                             vmin=vmin, vmax=vmax, aspect='auto')
    ax_left.set_title(titles[0], fontsize=13, fontweight='bold')
    ax_left.set_xlabel('x', fontsize=11)
    ax_left.set_ylabel('y', fontsize=11)
    plt.colorbar(im_left, ax=ax_left, fraction=0.046, pad=0.04)
    
    # Right panel
    im_right = ax_right.imshow(right_frames[0].T, origin='lower', cmap=cmap,
                               vmin=vmin, vmax=vmax, aspect='auto')
    ax_right.set_title(titles[1], fontsize=13, fontweight='bold')
    ax_right.set_xlabel('x', fontsize=11)
    ax_right.set_ylabel('y', fontsize=11)
    plt.colorbar(im_right, ax=ax_right, fraction=0.046, pad=0.04)
    
    # Timeseries panel
    if ax_ts is not None and lower_strip_timeseries is not None:
        time_steps = np.arange(T)
        line, = ax_ts.plot(time_steps, lower_strip_timeseries, 'b-', linewidth=2)
        marker, = ax_ts.plot([0], [lower_strip_timeseries[0]], 'ro', markersize=8)
        ax_ts.set_xlabel('Time Step', fontsize=11)
        ax_ts.set_ylabel('Error', fontsize=11)
        ax_ts.set_title('Relative L² Error Over Time', fontsize=12)
        ax_ts.grid(True, alpha=0.3)
        ax_ts.set_xlim([0, T-1])
        ax_ts.set_ylim([0, lower_strip_timeseries.max() * 1.1])
    
    plt.tight_layout()
    
    # Write video
    writer = FFMpegWriter(fps=fps, bitrate=3000)
    
    with writer.saving(fig, video_path, dpi=100):
        for t in range(T):
            im_left.set_data(left_frames[t].T)
            im_right.set_data(right_frames[t].T)
            
            if ax_ts is not None and lower_strip_timeseries is not None:
                marker.set_data([t], [lower_strip_timeseries[t]])
            
            writer.grab_frame()
    
    plt.close(fig)
    print(f"Saved comparison video: {video_path}")


def create_latest_symlink(target_dir: Path, root: Path, link_name: str) -> None:
    """
    Create 'latest' symlink to most recent run.
    
    Args:
        target_dir: Directory to point to
        root: Parent directory containing symlink
        link_name: Symlink name (e.g., 'vicsek__latest')
    """
    symlink_path = root / link_name
    
    # Remove old symlink if exists
    if symlink_path.is_symlink() or symlink_path.exists():
        symlink_path.unlink()
    
    # Create new symlink
    symlink_path.symlink_to(target_dir.name)
    print(f"Created symlink: {symlink_path} -> {target_dir.name}")
