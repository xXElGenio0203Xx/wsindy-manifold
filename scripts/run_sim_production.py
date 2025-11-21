#!/usr/bin/env python3
"""
⚠️  DEPRECATED: Production simulation runner with standardized output schema.

WARNING: This script uses legacy wsindy_manifold modules.
USE INSTEAD: 'rectsim-single' or scripts/rom_mvar_*.py

Generates outputs in simulations/<sim_name>__<run_id>/ with:
- manifest.json
- arrays/ (traj.npy, vel.npy, times.npy)
- csv/ (order_params.csv)
- density/ (kde.npz, kde_meta.json, grid.npy)
- videos/ (traj_quiver.mp4, kde_heatmap.mp4)
- plots/ (order_params_panel.png, snapshot_grid.png)
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rectsim import simulate_backend
from rectsim.unified_config import apply_defaults, validate_config, convert_to_legacy_format

# Import new modules
from rectsim.io_outputs import (
    create_traj_animation,
    create_density_animation
)
from rectsim.standard_metrics import (
    compute_all_metrics,
    compute_metrics_series
)
from rectsim.density import compute_density_grid

# Legacy I/O helpers - inline simplified versions
def create_run_dir(root: str, sim_name: str, seed: int = 0):
    """Create run directory with timestamp."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(root) / f"{sim_name}__{timestamp}_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def derive_sim_name(config: dict) -> str:
    """
    Derive descriptive simulation name from config.
    
    Format: <physics>_<boundary>_N<particles>_<key_params>
    Example: morse_alignment_reflect_N200_Ca10p0_eta0p3
             vicsek_only_periodic_N100_eta0p5
             morse_only_reflect_N150_Ca5p0
    """
    # Check if in outputs section (custom name)
    if 'outputs' in config and 'run_name' in config['outputs']:
        custom_name = config['outputs']['run_name']
        if custom_name:
            return custom_name
    
    # Otherwise derive from config
    parts = []
    
    # Physics type
    forces_enabled = config['dynamics']['forces']['enabled']
    alignment_enabled = config['dynamics']['alignment']['enabled']
    
    if forces_enabled and alignment_enabled:
        parts.append("morse_alignment")
    elif forces_enabled:
        parts.append("morse_only")
    elif alignment_enabled:
        parts.append("vicsek_only")
    else:
        parts.append("noise_only")
    
    # Boundary conditions
    bc = config['simulation']['bc']
    if bc == 'reflecting':
        parts.append("reflect")
    elif bc == 'periodic':
        parts.append("periodic")
    else:
        parts.append(bc)
    
    # Particle count
    N = config['simulation']['N']
    parts.append(f"N{N}")
    
    # Key parameters
    if forces_enabled:
        Ca = config['dynamics']['forces']['Ca']
        parts.append(f"Ca{Ca:.1f}".replace('.', 'p'))
    
    eta = config['dynamics']['noise']['eta']
    parts.append(f"eta{eta:.2f}".replace('.', 'p'))
    
    return "_".join(parts)


def plot_order_params_panel(
    df: pd.DataFrame,
    save_path: Path,
    include_nematic: bool = False
):
    """
    Create 3-panel stacked plot of order parameters.
    
    Args:
        df: DataFrame with columns 't', 'phi', 'mean_speed', 'speed_std', 'nematic' (optional)
        save_path: Output path
        include_nematic: Whether to plot nematic order
    """
    n_panels = 3 if include_nematic else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3*n_panels), sharex=True)
    
    if n_panels == 2:
        axes = list(axes)
    
    # Panel 1: Polarization
    ax = axes[0]
    ax.plot(df['t'], df['phi'], 'b-', linewidth=2)
    ax.set_ylabel('Polarization Φ', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Annotate final median
    median_phi = df['phi'].iloc[-len(df)//4:].median()
    ax.axhline(median_phi, color='r', linestyle='--', alpha=0.5,
               label=f'Final median: {median_phi:.3f}')
    ax.legend(loc='best', fontsize=10)
    
    # Panel 2: Speeds
    ax = axes[1]
    ax.plot(df['t'], df['mean_speed'], 'g-', linewidth=2, label='Mean speed')
    ax.plot(df['t'], df['speed_std'], 'orange', linewidth=2, linestyle='--',
            label='Speed std')
    ax.set_ylabel('Speed', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # Panel 3: Nematic (if available)
    if include_nematic and 'nematic' in df.columns:
        ax = axes[2]
        ax.plot(df['t'], df['nematic'], 'm-', linewidth=2)
        ax.set_ylabel('Nematic Order', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
    
    axes[-1].set_xlabel('Time', fontsize=12, fontweight='bold')
    
    plt.suptitle('Order Parameters Over Time', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved {save_path}")


def plot_snapshot_grid(
    traj: np.ndarray,
    densities: np.ndarray,
    times: np.ndarray,
    domain_bounds: tuple,
    save_path: Path,
    n_snapshots: int = 6
):
    """
    Create grid of trajectory + density snapshots at different times.
    
    Args:
        traj: Positions (T, N, 2)
        densities: Density fields (T, ny, nx)
        times: Time array (T,)
        domain_bounds: (xmin, xmax, ymin, ymax)
        save_path: Output path
        n_snapshots: Number of snapshots to show
    """
    T = len(times)
    indices = np.linspace(0, T-1, n_snapshots, dtype=int)
    
    fig, axes = plt.subplots(2, n_snapshots, figsize=(3*n_snapshots, 6))
    
    xmin, xmax, ymin, ymax = domain_bounds
    
    for col, t_idx in enumerate(indices):
        # Top row: trajectory
        ax = axes[0, col]
        ax.scatter(traj[t_idx, :, 0], traj[t_idx, :, 1], s=10, c='blue', alpha=0.6)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_aspect('equal')
        ax.set_title(f't = {times[t_idx]:.1f}', fontsize=10)
        
        if col == 0:
            ax.set_ylabel('Trajectory', fontsize=11, fontweight='bold')
        
        # Bottom row: density
        ax = axes[1, col]
        im = ax.imshow(densities[t_idx].T, origin='lower', cmap='viridis',
                      extent=[xmin, xmax, ymin, ymax], aspect='auto')
        ax.set_aspect('equal')
        
        if col == 0:
            ax.set_ylabel('Density', fontsize=11, fontweight='bold')
        
        # Colorbar for last column
        if col == n_snapshots - 1:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('ρ', fontsize=10)
    
    plt.suptitle('Trajectory and Density Snapshots', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved {save_path}")


def plot_order_params_panel(
    df: pd.DataFrame,
    save_path: Path,
    include_nematic: bool = True
):
    """Create multi-panel plot of order parameters over time."""


def main():
    parser = argparse.ArgumentParser(
        description='Run simulation with production output schema'
    )
    parser.add_argument('config', type=Path, help='YAML configuration file')
    parser.add_argument('--sim-name', type=str, default=None,
                       help='Override simulation name (auto-derived if not provided)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Override random seed')
    parser.add_argument('--out-root', type=Path, default=Path('simulations'),
                       help='Output root directory (default: simulations/)')
    parser.add_argument('--no-videos', action='store_true',
                       help='Skip video generation')
    
    args = parser.parse_args()
    
    # Load and validate config
    print(f"Loading configuration: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config = apply_defaults(config)
    
    errors = validate_config(config)
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"  • {error}")
        sys.exit(1)
    
    # Apply overrides
    if args.seed is not None:
        config['integration']['seed'] = args.seed
    
    seed = config['integration']['seed']
    
    # Derive simulation name
    if args.sim_name:
        sim_name = args.sim_name
    else:
        sim_name = derive_sim_name(config)
    
    print(f"\nSimulation: {sim_name}")
    print(f"Seed: {seed}")
    
    # Create run directory
    run_dir = create_run_dir(root=str(args.out_root), sim_name=sim_name, seed=seed)
    print(f"Output: {run_dir}\n")
    
    # Convert to legacy format and run simulation
    model_type = config['model']['type']
    legacy_config = convert_to_legacy_format(config, model_type)
    legacy_config['initial_distribution'] = config['particles']['initial_distribution']
    
    print("="*80)
    print("RUNNING SIMULATION")
    print("="*80)
    
    t_start = time.time()
    rng = np.random.default_rng(seed)
    result = simulate_backend(legacy_config, rng)
    t_sim = time.time() - t_start
    
    print(f"\n✓ Simulation complete ({t_sim:.2f}s)")
    print(f"  • Frames: {len(result['times'])}")
    print(f"  • Particles: {result['traj'].shape[1]}")
    
    # Extract data
    times = result['times']
    traj = result['traj']
    vel = result['vel']
    T, N, _ = traj.shape
    
    # Domain
    Lx, Ly = config['domain']['Lx'], config['domain']['Ly']
    domain_bounds = (0, Lx, 0, Ly)
    bc = config['domain']['bc']
    
    print("\n" + "="*80)
    print("GENERATING OUTPUTS")
    print("="*80)
    
    # 1. Save manifest
    print("\n[1/8] Manifest...")
    save_manifest(
        root=run_dir,
        sim_name=sim_name,
        config_path=str(args.config),
        simulator=f"{model_type}_vicsek" if model_type == "discrete" else f"{model_type}_dorsogna",
        seed=seed,
        code_version="1.0.0",
        N=N,
        T_total=config['integration']['T'],
        T_frames=T,
        dt=config['integration']['dt']
    )
    
    # Copy config
    shutil.copy(args.config, run_dir / "config.yaml")
    print(f"✓ Copied config.yaml")
    
    # 2. Save arrays
    print("\n[2/8] Arrays...")
    arrays_dir = run_dir / "arrays"
    save_arrays(
        arrays_dir,
        traj=traj,
        vel=vel,
        times=times
    )
    
    # 3. Compute and save order parameters
    print("\n[3/8] Order parameters...")
    order_params_list = []
    
    for t_idx, t in enumerate(times):
        params = compute_order_params(vel[t_idx], include_nematic=True)
        params['t'] = t
        order_params_list.append(params)
    
    df_order = pd.DataFrame(order_params_list)
    df_order = df_order[['t', 'phi', 'mean_speed', 'speed_std', 'nematic']]  # Reorder columns
    
    csv_dir = run_dir / "csv"
    save_csv(csv_dir, df_order, "order_params")
    
    # 4. Compute and save density
    print("\n[4/8] Density (KDE)...")
    nx = config['outputs']['density_resolution']
    ny = nx
    bandwidth = 1.5  # Default bandwidth in grid units
    
    t_kde_start = time.time()
    densities, kde_meta = kde_density_movie(
        traj, Lx, Ly, nx, ny, bandwidth, bc=bc
    )
    t_kde = time.time() - t_kde_start
    
    print(f"✓ KDE complete ({t_kde:.2f}s)")
    print(f"  • Grid: {nx}×{ny}")
    print(f"  • Bandwidth: {bandwidth}")
    
    density_dir = run_dir / "density"
    density_dir.mkdir(exist_ok=True)
    
    # Save density as NPZ
    np.savez_compressed(density_dir / "kde.npz", rho=densities)
    print(f"✓ Saved {density_dir}/kde.npz")
    
    # Save metadata
    with open(density_dir / "kde_meta.json", 'w') as f:
        json.dump(kde_meta, f, indent=2)
    print(f"✓ Saved {density_dir}/kde_meta.json")
    
    # Save grid
    xx, yy = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), indexing='ij')
    np.save(density_dir / "grid.npy", np.stack([xx, yy], axis=-1))
    print(f"✓ Saved {density_dir}/grid.npy")
    
    # Check mass conservation
    print()
    mass_stats = check_mass_conservation(densities, rtol=5e-3, verbose=True)
    
    # 5. Generate videos
    videos_dir = run_dir / "videos"
    
    if not args.no_videos:
        print("\n[5/8] Videos...")
        videos_dir.mkdir(exist_ok=True)
        
        # Trajectory animation with velocity arrows  
        create_traj_animation(
            times, traj, vel, domain_bounds,
            videos_dir / "traj_animation.mp4",
            fps=20, arrow_scale=1.0, arrow_mode='speed',
            show_arrows=True
        )
        
        # Density heatmap animation from positions
        # Let the animation auto-compute vmax from its own KDE
        create_density_animation(
            times, traj, domain_bounds, nx,
            videos_dir / "density_animation.mp4",
            fps=20, vmin=None, vmax=None,  # Auto-compute color scale
            bandwidth_mode="manual", manual_H=(bandwidth, bandwidth),
            periodic_x=(bc=='periodic')
        )
    else:
        print("\n[5/8] Videos... SKIPPED")
    
    # 6. Order parameters panel
    print("\n[6/8] Order parameters panel...")
    plots_dir = run_dir / "plots"
    try:
        plot_order_params_panel(
            df_order,
            plots_dir / "order_params_panel.png",
            include_nematic=True
        )
    except Exception as e:
        print(f"✗ Error creating order params panel: {e}")
        import traceback
        traceback.print_exc()
    
    # 7. Snapshot grid
    print("\n[7/8] Snapshot grid...")
    plot_snapshot_grid(
        traj, densities, times, domain_bounds,
        plots_dir / "snapshot_grid.png",
        n_snapshots=6
    )
    
    # 8. Timing metadata
    print("\n[8/8] Timing metadata...")
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    timings = {
        'simulation_time_s': t_sim,
        'kde_time_s': t_kde,
        'total_steps': int(config['integration']['T'] / config['integration']['dt']),
        'frames_saved': T,
        'fps_simulation': (config['integration']['T'] / config['integration']['dt']) / t_sim
    }
    
    with open(logs_dir / "timings.json", 'w') as f:
        json.dump(timings, f, indent=2)
    print(f"✓ Saved {logs_dir}/timings.json")
    
    # Create latest symlink
    print("\n[Symlink] Creating latest link...")
    create_latest_symlink(run_dir, args.out_root, f"{sim_name}__latest")
    
    # Final summary
    print("\n" + "="*80)
    print("✓ SIMULATION COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {run_dir}")
    print(f"\nStructure:")
    print(f"  manifest.json          - Run metadata")
    print(f"  config.yaml            - Configuration used")
    print(f"  arrays/")
    print(f"    traj.npy             - Positions (T={T}, N={N}, 2)")
    print(f"    vel.npy              - Velocities (T={T}, N={N}, 2)")
    print(f"    times.npy            - Time points (T={T},)")
    print(f"  csv/")
    print(f"    order_params.csv     - Φ, speeds, nematic")
    print(f"  density/")
    print(f"    kde.npz              - Density movie ({T}×{nx}×{ny})")
    print(f"    kde_meta.json        - KDE metadata")
    print(f"    grid.npy             - Grid coordinates")
    
    if not args.no_videos:
        print(f"  videos/")
        print(f"    traj_quiver.mp4      - Trajectories + arrows")
        print(f"    kde_heatmap.mp4      - Density evolution")
    
    print(f"  plots/")
    print(f"    order_params_panel.png  - Time series")
    print(f"    snapshot_grid.png       - Spatial snapshots")
    print(f"  logs/")
    print(f"    timings.json         - Performance metrics")
    
    print(f"\nFinal order parameters (t={times[-1]:.1f}):")
    print(f"  Φ = {df_order['phi'].iloc[-1]:.4f}")
    print(f"  v̄ = {df_order['mean_speed'].iloc[-1]:.4f}")
    print(f"  σᵥ = {df_order['speed_std'].iloc[-1]:.4f}")
    print(f"  Q = {df_order['nematic'].iloc[-1]:.4f}")
    
    print(f"\nMass conservation:")
    print(f"  Max drift: {mass_stats['max_drift']*100:.3f}%")
    print(f"  Status: {'✓ PASS' if mass_stats['within_tolerance'] else '✗ FAIL'}")
    
    print(f"\nSymlink: {args.out_root}/{sim_name}__latest -> {run_dir.name}")
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
