#!/usr/bin/env python3
"""
Generate KDE Heatmaps from Simulation Trajectories

This example demonstrates how to generate Gaussian kernel density estimation (KDE)
heatmaps from particle trajectory data, similar to the visualization approach used
in Bhaskar & Ziegelmeier (2019).

Usage:
    python examples/generate_kde_heatmaps.py --input trajectories.npz --output kde_output/

Features:
    - Mass-conserving KDE on periodic domains
    - Multiple colormap options (magma, viridis, hot)
    - Automatic bandwidth selection
    - Side-by-side snapshot grids
    - Animation generation
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from wsindy_manifold.latent.kde import trajectories_to_density_movie, make_grid


def generate_snapshot_grid(
    Rho: np.ndarray,
    Lx: float,
    Ly: float,
    nx: int,
    ny: int,
    times: np.ndarray,
    output_path: Path,
    n_snapshots: int = 6,
    cmap: str = "magma",
):
    """Generate a grid of density snapshots at different times.
    
    Parameters
    ----------
    Rho : ndarray, shape (T, nx*ny)
        Density movie from KDE
    Lx, Ly : float
        Domain extents
    nx, ny : int
        Grid dimensions
    times : ndarray
        Time values
    output_path : Path
        Where to save the figure
    n_snapshots : int
        Number of snapshots to show
    cmap : str
        Colormap name
    """
    T = Rho.shape[0]
    frame_indices = np.linspace(0, T - 1, n_snapshots, dtype=int)
    
    fig, axes = plt.subplots(2, n_snapshots // 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Global colorbar limits
    vmin = Rho.min()
    vmax = Rho.max()
    
    for idx, frame_idx in enumerate(frame_indices):
        ax = axes[idx]
        
        # Reshape to 2D for visualization
        density_2d = Rho[frame_idx].reshape(ny, nx)
        
        im = ax.imshow(
            density_2d,
            extent=(0, Lx, 0, Ly),
            origin='lower',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect='equal'
        )
        
        time_val = times[frame_idx] if times is not None else frame_idx
        ax.set_title(f't = {time_val:.2f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10)
        
        # Add colorbar to each subplot
        plt.colorbar(im, ax=ax, label='ρ', fraction=0.046, pad=0.04)
    
    plt.suptitle('KDE Density Snapshots', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved snapshot grid: {output_path}")


def generate_animation(
    Rho: np.ndarray,
    Lx: float,
    Ly: float,
    nx: int,
    ny: int,
    times: np.ndarray,
    output_path: Path,
    fps: int = 10,
    cmap: str = "magma",
):
    """Generate an animated GIF of the density evolution.
    
    Parameters
    ----------
    Rho : ndarray, shape (T, nx*ny)
        Density movie from KDE
    Lx, Ly : float
        Domain extents
    nx, ny : int
        Grid dimensions
    times : ndarray
        Time values
    output_path : Path
        Where to save the animation
    fps : int
        Frames per second
    cmap : str
        Colormap name
    """
    T = Rho.shape[0]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Global colorbar limits for consistency
    vmin = Rho.min()
    vmax = Rho.max()
    
    # Initial frame
    density_2d = Rho[0].reshape(ny, nx)
    im = ax.imshow(
        density_2d,
        extent=(0, Lx, 0, Ly),
        origin='lower',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect='equal',
        animated=True
    )
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('KDE Density Evolution', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, label='ρ')
    
    time_val = times[0] if times is not None else 0
    time_text = ax.text(0.02, 0.98, f't = {time_val:.2f}', 
                       transform=ax.transAxes,
                       fontsize=12, 
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def update(frame):
        density_2d = Rho[frame].reshape(ny, nx)
        im.set_array(density_2d)
        time_val = times[frame] if times is not None else frame
        time_text.set_text(f't = {time_val:.2f}')
        return im, time_text
    
    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=True)
    
    # Save as GIF
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close()
    print(f"  ✓ Saved animation: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate KDE heatmaps from trajectory data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input NPZ file with 'traj' array (T, N, 2)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("kde_output"),
        help="Output directory (default: kde_output/)",
    )
    parser.add_argument(
        "--Lx",
        type=float,
        default=20.0,
        help="Domain width (default: 20.0)",
    )
    parser.add_argument(
        "--Ly",
        type=float,
        default=20.0,
        help="Domain height (default: 20.0)",
    )
    parser.add_argument(
        "--nx",
        type=int,
        default=64,
        help="Grid resolution in x (default: 64)",
    )
    parser.add_argument(
        "--ny",
        type=int,
        default=64,
        help="Grid resolution in y (default: 64)",
    )
    parser.add_argument(
        "--hx",
        type=float,
        default=0.8,
        help="KDE bandwidth in x (default: 0.8)",
    )
    parser.add_argument(
        "--hy",
        type=float,
        default=0.8,
        help="KDE bandwidth in y (default: 0.8)",
    )
    parser.add_argument(
        "--bc",
        type=str,
        default="periodic",
        choices=["periodic", "reflecting"],
        help="Boundary conditions (default: periodic)",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="magma",
        choices=["magma", "viridis", "hot", "plasma", "inferno"],
        help="Colormap (default: magma)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Animation frame rate (default: 10)",
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="Skip animation generation",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with synthetic data",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("KDE HEATMAP GENERATION")
    print("=" * 70)
    
    # Load or generate trajectory data
    if args.demo or args.input is None:
        print("\nRunning in DEMO mode with synthetic data")
        print("Generating sample trajectories...")
        
        # Generate synthetic data
        T, N = 50, 100
        np.random.seed(42)
        
        t = np.linspace(0, 2*np.pi, T)
        traj = np.zeros((T, N, 2))
        
        # Create 3 clusters that rotate
        for i in range(N):
            cluster = i % 3
            center_x = 5 + cluster * 7.5
            center_y = 10.0
            
            angle_offset = (i / N) * 2 * np.pi
            traj[:, i, 0] = center_x + 2*np.cos(t + angle_offset) + np.random.randn(T) * 0.2
            traj[:, i, 1] = center_y + 2*np.sin(t + angle_offset) + np.random.randn(T) * 0.2
        
        # Wrap to domain
        traj[:, :, 0] = traj[:, :, 0] % args.Lx
        traj[:, :, 1] = traj[:, :, 1] % args.Ly
        
        times = t
        
    else:
        print(f"\nLoading trajectories from: {args.input}")
        data = np.load(args.input)
        traj = data['traj']
        times = data.get('times', None)
        
        if times is None:
            times = np.arange(traj.shape[0])
    
    T, N, _ = traj.shape
    print(f"\nTrajectory info:")
    print(f"  Frames (T): {T}")
    print(f"  Particles (N): {N}")
    print(f"  Domain: {args.Lx} x {args.Ly}")
    print(f"  Grid: {args.nx} x {args.ny}")
    print(f"  Bandwidth: hx={args.hx}, hy={args.hy}")
    print(f"  Boundary: {args.bc}")
    
    # Generate KDE density movie
    print(f"\nComputing KDE density movie...")
    Rho, meta = trajectories_to_density_movie(
        X_all=traj,
        Lx=args.Lx,
        Ly=args.Ly,
        nx=args.nx,
        ny=args.ny,
        hx=args.hx,
        hy=args.hy,
        bc=args.bc,
    )
    
    print(f"  ✓ Density shape: {Rho.shape}")
    
    # Check mass conservation
    dx = float(meta['dx'])
    dy = float(meta['dy'])
    masses = Rho.sum(axis=1) * dx * dy
    print(f"\nMass conservation:")
    print(f"  Mean: {masses.mean():.6f}")
    print(f"  Std: {masses.std():.2e}")
    print(f"  Min: {masses.min():.6f}")
    print(f"  Max: {masses.max():.6f}")
    
    # Save density data
    print(f"\nSaving outputs to: {args.output}")
    np.savez_compressed(
        args.output / "kde_density.npz",
        Rho=Rho,
        times=times,
        **meta
    )
    print(f"  ✓ Saved: kde_density.npz")
    
    # Generate visualizations
    print(f"\nGenerating visualizations (colormap: {args.cmap})...")
    
    # Snapshot grid
    generate_snapshot_grid(
        Rho, args.Lx, args.Ly, args.nx, args.ny, times,
        args.output / f"kde_snapshots_{args.cmap}.png",
        n_snapshots=6,
        cmap=args.cmap,
    )
    
    # Animation
    if not args.no_animation:
        generate_animation(
            Rho, args.Lx, args.Ly, args.nx, args.ny, times,
            args.output / f"kde_animation_{args.cmap}.gif",
            fps=args.fps,
            cmap=args.cmap,
        )
    
    print("\n" + "=" * 70)
    print("✓ COMPLETE")
    print("=" * 70)
    print(f"\nOutput files in: {args.output}")
    print(f"  - kde_density.npz (density data + metadata)")
    print(f"  - kde_snapshots_{args.cmap}.png (snapshot grid)")
    if not args.no_animation:
        print(f"  - kde_animation_{args.cmap}.gif (animation)")
    print()


if __name__ == "__main__":
    main()
