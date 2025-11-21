#!/usr/bin/env python3
"""Generate visualizations only for best-performing ICs from generalization test.

This script:
1. Reads aggregate_stats.json to find best uniform and gaussian ICs
2. Generates density comparison videos for best ICs
3. Generates order parameter plots for best ICs
4. Creates comparison plots (uniform vs gaussian performance)

Usage:
    python scripts/rom_mvar_visualize_best.py --experiment vicsek_morse_test
    
Author: Maria
Date: November 2025
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter


def create_density_video(
    density_true: np.ndarray,
    density_pred: np.ndarray,
    times: np.ndarray,
    Lx: float,
    Ly: float,
    output_path: Path,
    title: str = "",
    fps: int = 10,
) -> None:
    """Create side-by-side density comparison video."""
    T, nx, ny = density_true.shape
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    vmin = min(np.percentile(density_true, 1), np.percentile(density_pred, 1))
    vmax = max(np.percentile(density_true, 99), np.percentile(density_pred, 99))
    
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
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    
    time_text = fig.text(0.5, 0.02, f"t = {times[0]:.2f}", ha="center", fontsize=12)
    
    def update(frame):
        im_true.set_array(density_true[frame])
        im_pred.set_array(density_pred[frame])
        time_text.set_text(f"t = {times[frame]:.2f}")
        return im_true, im_pred, time_text
    
    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False)
    
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close()
    print(f"    ✓ Saved video: {output_path}")


def plot_order_parameters(
    order_metrics: pd.DataFrame,
    output_path: Path,
    title: str = "",
) -> None:
    """Plot order parameter evolution."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    
    # Polarization
    axes[0, 0].plot(order_metrics["t"], order_metrics["polarization"], linewidth=2)
    axes[0, 0].set_ylabel("Polarization", fontsize=11)
    axes[0, 0].set_xlabel("Time", fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1.05])
    
    # Angular momentum
    axes[0, 1].plot(order_metrics["t"], order_metrics["angular_momentum"], linewidth=2, color="orange")
    axes[0, 1].set_ylabel("Angular Momentum", fontsize=11)
    axes[0, 1].set_xlabel("Time", fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Mean speed
    axes[1, 0].plot(order_metrics["t"], order_metrics["speed_mean"], linewidth=2, color="green")
    axes[1, 0].set_ylabel("Mean Speed", fontsize=11)
    axes[1, 0].set_xlabel("Time", fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Speed std
    axes[1, 1].plot(order_metrics["t"], order_metrics["speed_std"], linewidth=2, color="red")
    axes[1, 1].set_ylabel("Speed Std Dev", fontsize=11)
    axes[1, 1].set_xlabel("Time", fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    ✓ Saved order params plot: {output_path}")


def plot_performance_comparison(
    stats: dict,
    cluster_counts: list,
    output_path: Path,
) -> None:
    """Create comparison plot: uniform vs gaussian (by cluster count)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # R² comparison
    uniform_r2 = stats["uniform"]["r2_mean"]
    uniform_r2_std = stats["uniform"]["r2_std"]
    
    gaussian_r2_means = [
        stats["gaussian"][f"{nc}_clusters"]["r2_mean"] for nc in cluster_counts
    ]
    gaussian_r2_stds = [
        stats["gaussian"][f"{nc}_clusters"]["r2_std"] for nc in cluster_counts
    ]
    
    x = np.arange(len(cluster_counts) + 1)
    labels = ["Uniform"] + [f"{nc} clust" for nc in cluster_counts]
    r2_vals = [uniform_r2] + gaussian_r2_means
    r2_errs = [uniform_r2_std] + gaussian_r2_stds
    
    axes[0].bar(x, r2_vals, yerr=r2_errs, capsize=5, alpha=0.7, color=["blue"] + ["orange"]*len(cluster_counts))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].set_ylabel("R² (mean ± std)", fontsize=12)
    axes[0].set_title("Coefficient of Determination", fontsize=13, fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].set_ylim([0, 1.0])
    
    # RMSE comparison
    uniform_rmse = stats["uniform"]["rmse_mean"]
    uniform_rmse_std = stats["uniform"]["rmse_std"]
    
    gaussian_rmse_means = [
        stats["gaussian"][f"{nc}_clusters"]["rmse_mean"] for nc in cluster_counts
    ]
    gaussian_rmse_stds = [
        stats["gaussian"][f"{nc}_clusters"]["rmse_std"] for nc in cluster_counts
    ]
    
    rmse_vals = [uniform_rmse] + gaussian_rmse_means
    rmse_errs = [uniform_rmse_std] + gaussian_rmse_stds
    
    axes[1].bar(x, rmse_vals, yerr=rmse_errs, capsize=5, alpha=0.7, color=["blue"] + ["orange"]*len(cluster_counts))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].set_ylabel("RMSE (mean ± std)", fontsize=12)
    axes[1].set_title("Root Mean Squared Error", fontsize=13, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved comparison plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--rom_root", default="rom_mvar", help="ROM root directory")
    parser.add_argument("--fps", type=int, default=10, help="Video frame rate")
    
    args = parser.parse_args()
    
    rom_root = Path(args.rom_root)
    exp_dir = rom_root / args.experiment
    gen_test_dir = exp_dir / "generalization_test"
    
    if not gen_test_dir.exists():
        print(f"ERROR: Generalization test directory not found: {gen_test_dir}")
        print("Run rom_mvar_generalization_test.py first.")
        sys.exit(1)
    
    # Load aggregate stats
    stats_file = gen_test_dir / "aggregate_stats.json"
    if not stats_file.exists():
        print(f"ERROR: Stats file not found: {stats_file}")
        sys.exit(1)
    
    with open(stats_file, "r") as f:
        stats = json.load(f)
    
    print("=" * 70)
    print("VISUALIZING BEST ICs")
    print("=" * 70)
    
    # Create videos directory
    videos_dir = gen_test_dir / "best_ic_videos"
    videos_dir.mkdir(exist_ok=True)
    
    # Visualize best uniform IC
    print("\nBest Uniform IC:")
    best_uniform_seed = stats["uniform"]["best_ic_seed"]
    uniform_dir = gen_test_dir / f"uniform_ic_{best_uniform_seed:04d}"
    
    if uniform_dir.exists():
        data = np.load(uniform_dir / "densities.npz")
        order_metrics = pd.read_csv(uniform_dir / "order_params.csv")
        
        print(f"  Seed: {best_uniform_seed}")
        print(f"  R²: {stats['uniform']['best_ic_r2']:.4f}")
        
        create_density_video(
            data["density_true"],
            data["density_pred"],
            data["times"],
            float(data["Lx"]),
            float(data["Ly"]),
            videos_dir / f"best_uniform_seed{best_uniform_seed}.gif",
            title=f"Best Uniform IC (seed={best_uniform_seed}, R²={stats['uniform']['best_ic_r2']:.4f})",
            fps=args.fps,
        )
        
        plot_order_parameters(
            order_metrics,
            videos_dir / f"best_uniform_seed{best_uniform_seed}_order_params.png",
            title=f"Best Uniform IC - Order Parameters (seed={best_uniform_seed})",
        )
    
    # Visualize best gaussian IC for each cluster count
    cluster_counts = [int(k.split("_")[0]) for k in stats["gaussian"].keys()]
    
    for nc in cluster_counts:
        print(f"\nBest {nc}-Cluster Gaussian IC:")
        best_seed = stats["gaussian"][f"{nc}_clusters"]["best_ic_seed"]
        best_r2 = stats["gaussian"][f"{nc}_clusters"]["best_ic_r2"]
        gaussian_dir = gen_test_dir / f"gaussian_{nc}clust_ic_{best_seed:04d}"
        
        if gaussian_dir.exists():
            data = np.load(gaussian_dir / "densities.npz")
            order_metrics = pd.read_csv(gaussian_dir / "order_params.csv")
            
            print(f"  Seed: {best_seed}")
            print(f"  R²: {best_r2:.4f}")
            
            create_density_video(
                data["density_true"],
                data["density_pred"],
                data["times"],
                float(data["Lx"]),
                float(data["Ly"]),
                videos_dir / f"best_gaussian_{nc}clust_seed{best_seed}.gif",
                title=f"Best {nc}-Cluster Gaussian IC (seed={best_seed}, R²={best_r2:.4f})",
                fps=args.fps,
            )
            
            plot_order_parameters(
                order_metrics,
                videos_dir / f"best_gaussian_{nc}clust_seed{best_seed}_order_params.png",
                title=f"Best {nc}-Cluster Gaussian IC - Order Parameters (seed={best_seed})",
            )
    
    # Create comparison plot
    print("\nGenerating performance comparison plot...")
    plot_performance_comparison(
        stats,
        cluster_counts,
        videos_dir / "performance_comparison.png",
    )
    
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nAll visualizations saved to: {videos_dir}")
    print("\nGenerated files:")
    print("  - best_uniform_seed*.gif (density video)")
    print("  - best_uniform_seed*_order_params.png (order parameters)")
    print("  - best_gaussian_*clust_seed*.gif (density videos)")
    print("  - best_gaussian_*clust_seed*_order_params.png (order parameters)")
    print("  - performance_comparison.png (uniform vs gaussian stats)")


if __name__ == "__main__":
    main()
