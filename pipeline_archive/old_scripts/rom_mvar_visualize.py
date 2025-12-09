#!/usr/bin/env python3
"""ROM/MVAR visualization: post-hoc video and plot generation.

This script reads saved NPZ/CSV data from a completed evaluation and generates:
- Density comparison videos (true vs predicted)
- Error-over-time plots with tolerance horizon τ marked
- Dashboard PNGs

NO simulation computation - pure visualization from disk.
Run this locally after rsync from Oscar.

Usage:
    # Visualize all test ICs
    python scripts/rom_mvar_visualize.py --experiment my_rom_experiment

    # Visualize specific ICs
    python scripts/rom_mvar_visualize.py \\
        --experiment my_rom_experiment \\
        --ic_ids 0 1 2

    # Skip videos (only plots)
    python scripts/rom_mvar_visualize.py \\
        --experiment my_rom_experiment \\
        --no_videos

Author: Maria
Date: November 2025
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec


def create_density_video(
    density_true: np.ndarray,
    density_pred: np.ndarray,
    times: np.ndarray,
    output_path: Path,
    Lx: float = 20.0,
    Ly: float = 20.0,
    fps: int = 10,
) -> None:
    """Create side-by-side comparison video of true vs predicted density.
    
    Parameters
    ----------
    density_true : ndarray, shape (T, nx, ny)
        True density evolution.
    density_pred : ndarray, shape (T, nx, ny)
        Predicted density evolution.
    times : ndarray, shape (T,)
        Time values.
    output_path : Path
        Output video path.
    Lx : float
        Domain width.
    Ly : float
        Domain height.
    fps : int
        Frames per second.
    """
    T, nx, ny = density_true.shape
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Set up colorbars with same scale (use percentiles for better contrast)
    vmin = min(np.percentile(density_true, 1), np.percentile(density_pred, 1))
    vmax = max(np.percentile(density_true, 99), np.percentile(density_pred, 99))
    
    # Use proper spatial extent and consistent colormap (magma like in existing code)
    im_true = axes[0].imshow(
        density_true[0],
        extent=(0, Lx, 0, Ly),
        origin="lower",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        animated=True,
    )
    axes[0].set_title("True Density", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")
    plt.colorbar(im_true, ax=axes[0], label="Density")
    
    im_pred = axes[1].imshow(
        density_pred[0],
        extent=(0, Lx, 0, Ly),
        origin="lower",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        animated=True,
    )
    axes[1].set_title("ROM/MVAR Forecast", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")
    plt.colorbar(im_pred, ax=axes[1], label="Density")
    
    time_text = fig.text(0.5, 0.95, f"t = {times[0]:.2f}", ha="center", fontsize=12)
    
    def update(frame):
        im_true.set_array(density_true[frame])
        im_pred.set_array(density_pred[frame])
        time_text.set_text(f"t = {times[frame]:.2f}")
        return im_true, im_pred, time_text
    
    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False)
    
    # Save as GIF (more portable than mp4)
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close()


def create_error_dashboard(
    metrics_df: pd.DataFrame,
    summary: dict,
    output_path: Path,
) -> None:
    """Create comprehensive error dashboard plot.
    
    Parameters
    ----------
    metrics_df : DataFrame
        Timeseries metrics.
    summary : dict
        Summary metrics including tolerance horizon.
    output_path : Path
        Output PNG path.
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    tau = summary["tolerance_horizon"]
    tol = summary["error_tolerance"]
    metric_name = summary["error_metric_for_tau"]
    
    # R² over time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(metrics_df["t"], metrics_df["r2"], linewidth=2)
    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax1.set_ylabel("R²", fontsize=12)
    ax1.set_title("Coefficient of Determination", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-0.1, 1.1])
    
    # RMSE over time with tolerance
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(metrics_df["t"], metrics_df["rmse"], linewidth=2, label="RMSE")
    if metric_name == "rmse":
        ax2.axhline(tol, color='r', linestyle='--', linewidth=2, label=f'Tolerance = {tol}')
        ax2.axvline(tau, color='g', linestyle=':', linewidth=2, label=f'τ = {tau}')
    ax2.set_ylabel("RMSE", fontsize=12)
    ax2.set_title("Root Mean Squared Error", fontsize=13, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Relative L² error
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(metrics_df["t"], metrics_df["e2"], linewidth=2, color='C2')
    if metric_name == "e2":
        ax3.axhline(tol, color='r', linestyle='--', linewidth=2)
        ax3.axvline(tau, color='g', linestyle=':', linewidth=2)
    ax3.set_ylabel("Relative L² Error", fontsize=12)
    ax3.set_title("L² Norm Error", fontsize=13, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    
    # Relative L∞ error
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(metrics_df["t"], metrics_df["einf"], linewidth=2, color='C3')
    if metric_name == "einf":
        ax4.axhline(tol, color='r', linestyle='--', linewidth=2)
        ax4.axvline(tau, color='g', linestyle=':', linewidth=2)
    ax4.set_ylabel("Relative L∞ Error", fontsize=12)
    ax4.set_title("L∞ Norm Error", fontsize=13, fontweight="bold")
    ax4.grid(True, alpha=0.3)
    
    # Mass error
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(metrics_df["t"], metrics_df["mass_error"], linewidth=2, color='C4')
    if metric_name == "mass_error":
        ax5.axhline(tol, color='r', linestyle='--', linewidth=2)
        ax5.axvline(tau, color='g', linestyle=':', linewidth=2)
    ax5.set_xlabel("Time Step", fontsize=12)
    ax5.set_ylabel("Absolute Mass Error", fontsize=12)
    ax5.set_title("Mass Conservation", fontsize=13, fontweight="bold")
    ax5.grid(True, alpha=0.3)
    
    # Summary statistics
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis("off")
    
    summary_text = f"""
    SUMMARY METRICS
    
    R² (mean):       {summary['r2_mean']:.4f}
    R² (median):     {summary['r2_median']:.4f}
    R² (min):        {summary['r2_min']:.4f}
    
    RMSE (mean):     {summary['rmse_mean']:.4e}
    RMSE (median):   {summary['rmse_median']:.4e}
    RMSE (max):      {summary['rmse_max']:.4e}
    
    Mass Err (mean): {summary['mass_error_mean']:.4e}
    Mass Err (max):  {summary['mass_error_max']:.4e}
    
    Tolerance (τ):   {tau} steps
    Error metric:    {metric_name}
    Threshold:       {tol}
    """
    
    ax6.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualizations for ROM/MVAR evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Name of ROM experiment",
    )
    parser.add_argument(
        "--rom_root",
        type=str,
        default="rom_mvar",
        help="ROM root directory (default: rom_mvar)",
    )
    parser.add_argument(
        "--ic_ids",
        type=int,
        nargs="*",
        help="Specific IC IDs to visualize (default: all)",
    )
    parser.add_argument(
        "--no_videos",
        action="store_true",
        help="Skip video generation (only plots)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Video frame rate (default: 10)",
    )
    
    args = parser.parse_args()
    
    rom_root = Path(args.rom_root)
    exp_dir = rom_root / args.experiment
    test_ics_dir = exp_dir / "test_ics"
    
    if not test_ics_dir.exists():
        print(f"ERROR: Test ICs directory not found: {test_ics_dir}")
        print("Run rom_mvar_eval.py first to generate evaluation data.")
        sys.exit(1)
    
    print("=" * 70)
    print("ROM/MVAR VISUALIZATION")
    print("=" * 70)
    print(f"Experiment: {args.experiment}")
    print(f"Generate videos: {not args.no_videos}")
    print("=" * 70)
    
    # Find all IC directories
    ic_dirs = sorted(test_ics_dir.glob("ic_*"))
    
    if args.ic_ids is not None:
        # Filter to specified ICs
        ic_dirs = [d for d in ic_dirs if int(d.name.split("_")[1]) in args.ic_ids]
    
    if not ic_dirs:
        print("\nNo IC directories found to visualize.")
        sys.exit(1)
    
    print(f"\nFound {len(ic_dirs)} IC(s) to visualize")
    
    # Process each IC
    for ic_dir in ic_dirs:
        ic_name = ic_dir.name
        ic_id = int(ic_name.split("_")[1])
        
        print(f"\n{'='*70}")
        print(f"Processing {ic_name}")
        print(f"{'='*70}")
        
        # Load data
        print("  Loading data...")
        true_data = np.load(ic_dir / "true_density.npz")
        pred_data = np.load(ic_dir / "pred_density.npz")
        metrics_df = pd.read_csv(ic_dir / "metrics_timeseries.csv")
        
        with open(ic_dir / "metrics_summary.json", "r") as f:
            summary = json.load(f)
        
        density_true = true_data["density"]
        density_pred = pred_data["density"]
        times = true_data["times"]
        
        # Get domain size (with defaults if not saved)
        Lx = float(true_data.get("Lx", 20.0))
        Ly = float(true_data.get("Ly", 20.0))
        
        # Align lengths
        T = min(density_true.shape[0], density_pred.shape[0])
        density_true = density_true[:T]
        density_pred = density_pred[:T]
        times = times[:T]
        
        print(f"    Density shape: {density_true.shape}")
        print(f"    Domain: {Lx} x {Ly}")
        print(f"    Time range: {times[0]:.2f} to {times[-1]:.2f}")
        
        # Create videos directory
        videos_dir = ic_dir / "videos"
        videos_dir.mkdir(exist_ok=True)
        
        # Generate dashboard plot
        print("  Generating dashboard plot...")
        create_error_dashboard(
            metrics_df,
            summary,
            videos_dir / "error_dashboard.png",
        )
        print(f"    ✓ Saved to {videos_dir / 'error_dashboard.png'}")
        
        # Generate comparison video
        if not args.no_videos:
            print("  Generating comparison video...")
            try:
                create_density_video(
                    density_true,
                    density_pred,
                    times,
                    videos_dir / "density_comparison.gif",
                    Lx=Lx,
                    Ly=Ly,
                    fps=args.fps,
                )
                print(f"    ✓ Saved to {videos_dir / 'density_comparison.gif'}")
            except Exception as e:
                print(f"    ✗ Video generation failed: {e}")
        
        print(f"\n  ✓ Visualization complete for {ic_name}")
    
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nVisualizations saved to: {test_ics_dir}/*/videos/")
    print()


if __name__ == "__main__":
    main()
