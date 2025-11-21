"""Video generation utilities for ROM/MVAR evaluation.

This module provides tools to create comparison videos of ground truth vs
predicted density fields for ROM/MVAR evaluation.

Author: Maria
Date: November 2025
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
import imageio

# Use non-interactive backend
mpl.use('Agg')


def make_truth_vs_pred_density_video(
    density_true: np.ndarray,
    density_pred: np.ndarray,
    out_path: Path,
    fps: int = 20,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    times: Optional[np.ndarray] = None,
) -> None:
    """Create side-by-side video of true vs predicted density.
    
    Parameters
    ----------
    density_true : np.ndarray
        Ground truth density, shape (T, Ny, Nx).
    density_pred : np.ndarray
        Predicted density, shape (T, Ny, Nx).
    out_path : Path
        Output MP4 file path.
    fps : int, default=20
        Frames per second.
    cmap : str, default="viridis"
        Colormap name.
    vmin, vmax : Optional[float]
        Color scale limits. If None, auto-compute from both fields.
    title : Optional[str]
        Video title (displayed at top).
    times : Optional[np.ndarray]
        Time points for display, shape (T,).
        
    Notes
    -----
    Uses imageio to write MP4 directly without requiring ffmpeg on PATH.
    """
    T, Ny, Nx = density_true.shape
    assert density_pred.shape == density_true.shape, \
        f"Shape mismatch: {density_pred.shape} vs {density_true.shape}"
    
    # Auto-compute color scale
    if vmin is None:
        vmin = min(density_true.min(), density_pred.min())
    if vmax is None:
        vmax = max(density_true.max(), density_pred.max())
    
    if times is None:
        times = np.arange(T)
    
    # Setup figure with side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    # Initial frames
    im_true = axes[0].imshow(
        density_true[0],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin='lower',
        aspect='equal',
    )
    axes[0].set_title("Ground Truth", fontsize=12)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    
    im_pred = axes[1].imshow(
        density_pred[0],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin='lower',
        aspect='equal',
    )
    axes[1].set_title("ROM/MVAR Prediction", fontsize=12)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    
    # Add colorbar
    cbar = fig.colorbar(im_true, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label("Density", fontsize=11)
    
    # Time text
    time_text = fig.text(0.5, 0.92, f"Time: {times[0]:.2f}", ha='center', fontsize=11)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Render frames to memory
    print(f"  Rendering {T} frames...")
    frames = []
    
    for t in range(T):
        im_true.set_data(density_true[t])
        im_pred.set_data(density_pred[t])
        time_text.set_text(f"Time: {times[t]:.2f}")
        
        # Render to RGB array
        fig.canvas.draw()
        # Use buffer_rgba and convert to RGB
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frame = buf[:, :, :3]  # Drop alpha channel
        frames.append(frame)
        
        if (t + 1) % 50 == 0:
            print(f"    {t+1}/{T} frames rendered")
    
    plt.close(fig)
    
    # Write video
    print(f"  Writing video to {out_path}...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    imageio.mimsave(
        out_path,
        frames,
        fps=fps,
        codec='libx264',
        quality=8,
        pixelformat='yuv420p',
    )
    
    file_size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  ✓ Video created: {file_size_mb:.2f} MB")


def make_best_run_videos(
    best_runs: Dict[str, Any],
    model: Any,  # PODMVARModel
    samples_dict: Dict[Tuple[str, str], Any],  # SimulationSample
    out_root: Path,
    train_frac: float = 0.8,
    fps: int = 20,
) -> None:
    """Generate truth vs prediction videos for best runs per IC type.
    
    Parameters
    ----------
    best_runs : Dict[str, SimulationMetrics]
        Best simulation metrics per IC type.
    model : PODMVARModel
        Trained ROM/MVAR model.
    samples_dict : Dict[tuple, SimulationSample]
        Simulation samples keyed by (ic_type, name).
    out_root : Path
        Output root directory.
    train_frac : float, default=0.8
        Fraction for forecast split.
    fps : int, default=20
        Video frames per second.
    """
    from rectsim.rom_eval_pipeline import predict_single_simulation
    
    print("Generating truth vs prediction videos for best runs...")
    print()
    
    for ic_type, metrics in best_runs.items():
        key = (ic_type, metrics.name)
        
        if key not in samples_dict:
            print(f"  Warning: No sample found for {ic_type}/{metrics.name}")
            continue
        
        sample = samples_dict[key]
        
        print(f"  {ic_type}/{metrics.name} (R²={metrics.r2:.4f})...")
        
        try:
            # Run prediction
            _, preds = predict_single_simulation(
                model,
                sample,
                train_frac=train_frac,
                tol=0.1,
                return_predictions=True,
            )
            
            density_true = preds["density_true"]
            density_pred = preds["density_pred"]
            times = preds["times"]
            
            # Generate video
            out_path = out_root / ic_type / "best_truth_vs_pred.mp4"
            
            make_truth_vs_pred_density_video(
                density_true,
                density_pred,
                out_path,
                fps=fps,
                title=f"Best Run: {ic_type} (R²={metrics.r2:.4f})",
                times=times,
            )
            
            print()
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            print()


def make_density_snapshot_comparison(
    density_true: np.ndarray,
    density_pred: np.ndarray,
    time_indices: list[int],
    times: Optional[np.ndarray],
    out_path: Path,
    title: Optional[str] = None,
    cmap: str = "viridis",
) -> None:
    """Create multi-panel comparison at specific time snapshots.
    
    Parameters
    ----------
    density_true : np.ndarray
        Ground truth, shape (T, Ny, Nx).
    density_pred : np.ndarray
        Prediction, shape (T, Ny, Nx).
    time_indices : list[int]
        List of time indices to plot (e.g., [0, T//4, T//2, 3*T//4, T-1]).
    times : Optional[np.ndarray]
        Time values for labels.
    out_path : Path
        Output PNG path.
    title : Optional[str]
        Figure title.
    cmap : str, default="viridis"
        Colormap.
        
    Notes
    -----
    Creates a grid with 2 rows (true, pred) and N columns (time snapshots).
    """
    n_snapshots = len(time_indices)
    
    # Auto-compute color scale
    vmin = min(density_true.min(), density_pred.min())
    vmax = max(density_true.max(), density_pred.max())
    
    if times is None:
        times = np.arange(density_true.shape[0])
    
    fig, axes = plt.subplots(2, n_snapshots, figsize=(4 * n_snapshots, 8))
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Ensure axes is 2D
    if n_snapshots == 1:
        axes = axes.reshape(2, 1)
    
    for col, t_idx in enumerate(time_indices):
        # True
        im = axes[0, col].imshow(
            density_true[t_idx],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin='lower',
            aspect='equal',
        )
        axes[0, col].set_title(f"True (t={times[t_idx]:.2f})", fontsize=10)
        axes[0, col].axis('off')
        
        # Predicted
        axes[1, col].imshow(
            density_pred[t_idx],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin='lower',
            aspect='equal',
        )
        axes[1, col].set_title(f"Pred (t={times[t_idx]:.2f})", fontsize=10)
        axes[1, col].axis('off')
    
    # Add colorbar
    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04, label="Density")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
