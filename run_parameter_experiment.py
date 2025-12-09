#!/usr/bin/env python3
"""
Parameter Experimentation Pipeline
====================================

Quick visualization pipeline for testing simulation parameters.
No ROM training, no MVAR - just simulate and visualize to see if parameters are interesting.

Features:
---------
- Run 1 simulation per IC type to preview behavior
- Generate trajectory videos, density videos, and order parameters
- Save to experiments/ folder for easy review
- Perfect for testing: N particles, domain size, bandwidth, Morse parameters, noise, etc.

Usage:
------
python run_parameter_experiment.py --config configs/my_experiment.yaml --experiment_name test_params

Output Structure:
-----------------
experiments/
└── test_params/
    ├── config_used.yaml
    ├── summary.json
    └── simulations/
        ├── gaussian_cluster/
        │   ├── trajectory.mp4
        │   ├── density.mp4
        │   ├── order_parameters.png
        │   ├── trajectory.npz
        │   └── density.npz
        ├── uniform/
        ├── ring/
        └── two_clusters/

Author: Maria
Date: December 2025
"""

import numpy as np
from pathlib import Path
import json
import time
import argparse
import yaml
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import pipeline modules
from rectsim.config_loader import load_config
from rectsim.ic_generator import generate_training_configs
from rectsim.vicsek_discrete import simulate_backend
from rectsim.legacy_functions import (
    kde_density_movie,
    trajectory_video,
    side_by_side_video,
    polarization,
    mean_speed,
    nematic_order
)
from rectsim.metrics import angular_momentum


def simulate_single_preview(config, base_config, output_dir, density_nx, density_ny, density_bandwidth):
    """
    Run a single simulation and create all visualizations.
    
    Parameters
    ----------
    config : dict
        IC configuration with distribution, ic_params, label
    base_config : dict
        Base simulation configuration
    output_dir : Path
        Output directory for this IC type
    density_nx, density_ny : int
        Density grid resolution
    density_bandwidth : float
        KDE bandwidth
    
    Returns
    -------
    dict
        Metadata for this simulation
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    distribution = config['distribution']
    ic_params = config['ic_params']
    label = config['label']
    
    # Set up configuration
    sim_config = base_config.copy()
    sim_config["seed"] = config['run_id'] + 42
    sim_config["initial_distribution"] = distribution
    sim_config["ic_params"] = ic_params
    
    print(f"\n{'='*60}")
    print(f"Simulating: {distribution}")
    print(f"Label: {label}")
    print(f"{'='*60}")
    
    # Run simulation
    rng = np.random.default_rng(sim_config["seed"])
    result = simulate_backend(sim_config, rng)
    
    # Extract data
    times = result["times"]
    traj = result["traj"]
    vel = result["vel"]
    
    # Save trajectory
    np.savez_compressed(
        output_dir / "trajectory.npz",
        traj=traj,
        vel=vel,
        times=times
    )
    
    print(f"✓ Simulation complete: T={len(times)} steps")
    
    # Compute density
    print("Computing density field...")
    rho, meta = kde_density_movie(
        traj,
        Lx=sim_config["sim"]["Lx"],
        Ly=sim_config["sim"]["Ly"],
        nx=density_nx,
        ny=density_ny,
        bandwidth=density_bandwidth,
        bc=sim_config["sim"].get("bc", "periodic")
    )
    
    # Create grids
    xgrid = np.linspace(0, sim_config["sim"]["Lx"], density_nx, endpoint=False) + sim_config["sim"]["Lx"]/(2*density_nx)
    ygrid = np.linspace(0, sim_config["sim"]["Ly"], density_ny, endpoint=False) + sim_config["sim"]["Ly"]/(2*density_ny)
    
    # Save density
    np.savez_compressed(
        output_dir / "density.npz",
        rho=rho,
        xgrid=xgrid,
        ygrid=ygrid,
        times=times
    )
    
    print("✓ Density computed")
    
    # Create trajectory video
    print("Creating trajectory video...")
    trajectory_video(
        traj=traj,
        times=times,
        output_path=str(output_dir / "trajectory.mp4"),
        Lx=sim_config["sim"]["Lx"],
        Ly=sim_config["sim"]["Ly"],
        title=f"{distribution} - Trajectories",
        fps=10
    )
    print("✓ Trajectory video saved")
    
    # Create density video
    print("Creating density video...")
    side_by_side_video(
        rho_true=rho,
        rho_pred=rho,  # Just show truth twice for preview
        times=times,
        xgrid=xgrid,
        ygrid=ygrid,
        output_path=str(output_dir / "density.mp4"),
        title=f"{distribution} - Density",
        fps=10,
        mode="single"  # Only show truth
    )
    print("✓ Density video saved")
    
    # Compute order parameters
    print("Computing order parameters...")
    T = len(times)
    
    pol = np.array([polarization(vel[t]) for t in range(T)])
    nem = np.array([nematic_order(vel[t]) for t in range(T)])
    spd = np.array([mean_speed(vel[t]) for t in range(T)])
    ang_mom = np.array([angular_momentum(traj[t], vel[t], 
                                         sim_config["sim"]["Lx"], 
                                         sim_config["sim"]["Ly"]) 
                       for t in range(T)])
    
    # Plot order parameters
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(times, pol, linewidth=2, color='#2E86AB')
    axes[0, 0].set_xlabel('Time (s)', fontsize=12)
    axes[0, 0].set_ylabel('Polarization', fontsize=12)
    axes[0, 0].set_title('Polarization Order', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1.05])
    
    axes[0, 1].plot(times, nem, linewidth=2, color='#A23B72')
    axes[0, 1].set_xlabel('Time (s)', fontsize=12)
    axes[0, 1].set_ylabel('Nematic Order', fontsize=12)
    axes[0, 1].set_title('Nematic Order', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.05])
    
    axes[1, 0].plot(times, spd, linewidth=2, color='#F18F01')
    axes[1, 0].set_xlabel('Time (s)', fontsize=12)
    axes[1, 0].set_ylabel('Mean Speed', fontsize=12)
    axes[1, 0].set_title('Mean Speed', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(times, ang_mom, linewidth=2, color='#C73E1D')
    axes[1, 1].set_xlabel('Time (s)', fontsize=12)
    axes[1, 1].set_ylabel('Angular Momentum', fontsize=12)
    axes[1, 1].set_title('Angular Momentum', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{distribution} - Order Parameters', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / "order_parameters.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    print("✓ Order parameters plotted")
    
    # Compute summary statistics
    metadata = {
        "distribution": distribution,
        "label": label,
        "ic_params": ic_params,
        "seed": sim_config["seed"],
        "T_steps": T,
        "T_seconds": float(times[-1]),
        "final_polarization": float(pol[-1]),
        "mean_polarization": float(pol.mean()),
        "final_nematic": float(nem[-1]),
        "mean_nematic": float(nem.mean()),
        "mean_speed": float(spd.mean()),
        "mean_angular_momentum": float(ang_mom.mean())
    }
    
    # Save metadata
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description='Parameter Experimentation Pipeline')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name')
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("="*80)
    print("PARAMETER EXPERIMENTATION PIPELINE")
    print("="*80)
    print(f"\nExperiment: {args.experiment_name}")
    print(f"Config: {args.config}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nPurpose: Quick preview of simulation parameters")
    print("  → 1 simulation per IC type")
    print("  → Full visualizations (trajectories, density, order parameters)")
    print("  → No ROM/MVAR training")
    
    # Load configuration
    (BASE_CONFIG, DENSITY_NX, DENSITY_NY, DENSITY_BANDWIDTH,
     train_ic_config, test_ic_config, test_sim_config, rom_config, eval_config) = load_config(args.config)
    
    OUTPUT_DIR = Path(f"experiments/{args.experiment_name}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save config for reference
    shutil.copy(args.config, OUTPUT_DIR / "config_used.yaml")
    
    print(f"\nConfiguration:")
    print(f"   N particles: {BASE_CONFIG['sim']['N']}")
    print(f"   T duration: {BASE_CONFIG['sim']['T']}s")
    print(f"   dt: {BASE_CONFIG['sim']['dt']}s")
    print(f"   Domain: {BASE_CONFIG['sim']['Lx']}×{BASE_CONFIG['sim']['Ly']}")
    print(f"   Density resolution: {DENSITY_NX}×{DENSITY_NY}")
    print(f"   KDE bandwidth: {DENSITY_BANDWIDTH}")
    
    # Check for noise and forces
    if 'noise' in BASE_CONFIG and 'eta' in BASE_CONFIG['noise']:
        print(f"   Noise (η): {BASE_CONFIG['noise']['eta']}")
    if BASE_CONFIG.get('forces', {}).get('enabled', False):
        print(f"   Forces: ENABLED (Morse potential)")
    
    # Generate training configs (we'll pick one per IC type)
    print(f"\n{'='*80}")
    print("Generating IC Preview Simulations")
    print("="*80)
    
    all_configs = generate_training_configs(train_ic_config, BASE_CONFIG)
    
    # Group by distribution type and pick first of each
    ic_types = {}
    for cfg in all_configs:
        dist = cfg['distribution']
        if dist not in ic_types:
            ic_types[dist] = cfg
    
    preview_configs = list(ic_types.values())
    
    print(f"\nIC types to preview: {len(preview_configs)}")
    for dist in ic_types.keys():
        print(f"   • {dist}")
    
    # Create simulations directory
    SIM_DIR = OUTPUT_DIR / "simulations"
    SIM_DIR.mkdir(exist_ok=True)
    
    # Run preview simulations
    print(f"\n{'='*80}")
    print("Running Preview Simulations")
    print("="*80)
    
    all_metadata = []
    
    for cfg in preview_configs:
        dist = cfg['distribution']
        output_dir = SIM_DIR / dist
        
        metadata = simulate_single_preview(
            config=cfg,
            base_config=BASE_CONFIG,
            output_dir=output_dir,
            density_nx=DENSITY_NX,
            density_ny=DENSITY_NY,
            density_bandwidth=DENSITY_BANDWIDTH
        )
        
        all_metadata.append(metadata)
    
    # Save summary
    total_time = time.time() - start_time
    
    summary = {
        'experiment_name': args.experiment_name,
        'config': args.config,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_ic_types': len(preview_configs),
        'ic_types': list(ic_types.keys()),
        'parameters': {
            'N': int(BASE_CONFIG['sim']['N']),
            'T': float(BASE_CONFIG['sim']['T']),
            'dt': float(BASE_CONFIG['sim']['dt']),
            'Lx': float(BASE_CONFIG['sim']['Lx']),
            'Ly': float(BASE_CONFIG['sim']['Ly']),
            'density_nx': int(DENSITY_NX),
            'density_ny': int(DENSITY_NY),
            'bandwidth': float(DENSITY_BANDWIDTH)
        },
        'simulations': all_metadata,
        'total_time_minutes': total_time / 60
    }
    
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Final summary
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\nTotal time: {total_time/60:.1f}m")
    print(f"Output directory: {OUTPUT_DIR}")
    
    print(f"\n📁 Generated visualizations:")
    for dist in ic_types.keys():
        print(f"   {dist}/")
        print(f"      ├── trajectory.mp4       (particle trajectories)")
        print(f"      ├── density.mp4          (density field)")
        print(f"      ├── order_parameters.png (collective behavior)")
        print(f"      └── metadata.json        (summary statistics)")
    
    print(f"\n💡 Review your simulations in: {OUTPUT_DIR}/simulations/")
    print(f"   If parameters look good, run full pipeline with this config!")
    
    print("\n✅ Parameter experimentation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
