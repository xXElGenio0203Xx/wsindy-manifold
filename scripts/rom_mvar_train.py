#!/usr/bin/env python3
"""ROM/MVAR training pipeline: augmented ensemble → global POD → MVAR.

This script:
1. Generates M_train simulations with varied ICs but identical dynamics
2. Computes a single global POD basis from all training snapshots
3. Projects all training runs to latent space
4. Fits one MVAR model on concatenated latent data
5. Saves model artifacts (POD basis, MVAR params, train summary)

Usage:
    # Train with default settings
    python scripts/rom_mvar_train.py --config configs/rom_train.yaml

    # Override training parameters
    python scripts/rom_mvar_train.py \\
        --config configs/rom_train.yaml \\
        rom.num_train_ics=20 \\
        rom.mvar_order=6 \\
        rom.energy_threshold=0.99

Author: Maria
Date: November 2025
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from rectsim.cli import _run_single
from rectsim.config import load_config
from rectsim.density import compute_density_grid
from rectsim.rom_mvar import (
    ROMTrainConfig,
    compute_global_pod,
    fit_mvar,
    project_to_pod,
    save_mvar_model,
    save_pod_model,
    save_train_summary,
    setup_rom_directories,
)


def load_density_from_simulation(sim_result: dict, nx: int, ny: int) -> np.ndarray:
    """Extract density time series from simulation result.
    
    Parameters
    ----------
    sim_result : dict
        Result from _run_single containing 'results' with traj, times, etc.
    nx, ny : int
        Grid resolution.
    
    Returns
    -------
    ndarray, shape (T, nx * ny)
        Flattened density snapshots.
    """
    results = sim_result["results"]
    cfg = sim_result["config"]
    traj = results["traj"]  # (T, N, 2)
    Lx = cfg["sim"]["Lx"]
    Ly = cfg["sim"]["Ly"]
    bc = cfg["sim"]["bc"]
    
    T = traj.shape[0]
    density_movie = np.zeros((T, nx, ny))
    
    for t in range(T):
        rho, _, _ = compute_density_grid(
            traj[t], nx, ny, Lx, Ly, bandwidth=0.5, bc=bc
        )
        density_movie[t] = rho
    
    # Flatten spatial dimensions
    return density_movie.reshape(T, -1)


def main():
    parser = argparse.ArgumentParser(
        description="Train ROM/MVAR model on augmented ensemble",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file with simulation and ROM settings",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in key=value format",
    )
    
    args = parser.parse_args()
    
    # Parse overrides
    override_pairs = []
    for arg in args.overrides:
        if "=" not in arg:
            parser.error(f"Override must be key=value format, got: {arg}")
        key, value = arg.split("=", 1)
        # Try to parse as JSON for type conversion
        import json
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass  # Keep as string
        override_pairs.append((key, value))
    
    # Load config
    cfg = load_config(args.config, overrides=override_pairs)
    
    # Extract ROM training config
    rom_cfg_dict = cfg.get("rom", {})
    rom_cfg = ROMTrainConfig(
        experiment_name=rom_cfg_dict.get("experiment_name", "default_rom"),
        num_train_ics=rom_cfg_dict.get("num_train_ics", 10),
        train_seeds=rom_cfg_dict.get("train_seeds"),
        mvar_order=rom_cfg_dict.get("mvar_order", 4),
        ridge=rom_cfg_dict.get("ridge", 1e-6),
        energy_threshold=rom_cfg_dict.get("energy_threshold", 0.995),
        latent_dim=rom_cfg_dict.get("latent_dim"),
        train_frac=rom_cfg_dict.get("train_frac", 1.0),
        rom_root=Path(rom_cfg_dict.get("rom_root", "rom_mvar")),
    )
    
    print("=" * 70)
    print("ROM/MVAR TRAINING PIPELINE")
    print("=" * 70)
    print(f"Experiment: {rom_cfg.experiment_name}")
    print(f"Training ICs: {rom_cfg.num_train_ics}")
    print(f"MVAR order: {rom_cfg.mvar_order}")
    print(f"Ridge: {rom_cfg.ridge}")
    print(f"Energy threshold: {rom_cfg.energy_threshold}")
    print("=" * 70)
    
    # Setup directories
    dirs = setup_rom_directories(rom_cfg.experiment_name, rom_cfg.rom_root)
    print(f"\n✓ Created output directories in {dirs['base']}\n")
    
    # Extract grid info
    grid_cfg = cfg.get("outputs", {}).get("grid_density", {})
    nx = grid_cfg.get("nx", 128)
    ny = grid_cfg.get("ny", 128)
    Lx = cfg["sim"]["Lx"]
    Ly = cfg["sim"]["Ly"]
    
    grid_info = {"nx": nx, "ny": ny, "Lx": Lx, "Ly": Ly}
    
    # ========================================================================
    # Step 1: Generate training simulations with varied ICs
    # ========================================================================
    print("STEP 1: Generating training simulations")
    print("-" * 70)
    
    all_density_snapshots = []
    all_latent_trajectories = []  # Will populate after POD
    
    for i, seed in enumerate(rom_cfg.train_seeds):
        print(f"\n[{i+1}/{rom_cfg.num_train_ics}] Running training IC with seed={seed}")
        
        # Prepare config for this IC
        train_cfg = deepcopy(cfg)
        train_cfg["seed"] = seed
        # Disable videos and plots during training
        train_cfg["outputs"]["animate_traj"] = False
        train_cfg["outputs"]["animate_density"] = False
        train_cfg["outputs"]["plot_order_params"] = False
        
        # Run simulation
        sim_result = _run_single(train_cfg, ic_id=None, enable_videos=False, enable_order_plots=False)
        
        # Extract density time series
        density_movie = load_density_from_simulation(sim_result, nx, ny)
        all_density_snapshots.append(density_movie)
        
        print(f"  ✓ Generated {density_movie.shape[0]} density snapshots")
    
    # Stack all snapshots for POD
    stacked_snapshots = np.vstack(all_density_snapshots)
    print(f"\n✓ Total snapshots for POD: {stacked_snapshots.shape[0]}")
    
    # ========================================================================
    # Step 2: Compute global POD basis
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Computing global POD basis")
    print("-" * 70)
    
    pod_result = compute_global_pod(
        stacked_snapshots,
        energy_threshold=rom_cfg.energy_threshold,
        latent_dim=rom_cfg.latent_dim,
    )
    
    print(f"\n✓ POD basis computed")
    print(f"  Latent dimension: {pod_result['latent_dim']}")
    print(f"  Energy captured: {pod_result['energy'][pod_result['latent_dim']-1]:.4f}")
    print(f"  Compression ratio: {(nx*ny) / pod_result['latent_dim']:.1f}x")
    
    # Save POD model
    save_pod_model(
        dirs["model"],
        pod_result["mean_mode"],
        pod_result["pod_modes"],
        pod_result["singular_values"],
        pod_result["energy"],
        grid_info,
    )
    print(f"  Saved to {dirs['model'] / 'pod_basis.npz'}")
    
    # ========================================================================
    # Step 3: Project all training runs to latent space
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Projecting training data to latent space")
    print("-" * 70)
    
    for i, density_movie in enumerate(all_density_snapshots):
        latent_traj = project_to_pod(
            density_movie,
            pod_result["mean_mode"],
            pod_result["pod_modes"],
        )
        all_latent_trajectories.append(latent_traj)
        print(f"  [{i+1}/{rom_cfg.num_train_ics}] Projected to shape {latent_traj.shape}")
    
    print(f"\n✓ All training data projected to latent space")
    
    # ========================================================================
    # Step 4: Fit MVAR model
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Fitting MVAR model")
    print("-" * 70)
    
    # Apply train_frac if needed
    if rom_cfg.train_frac < 1.0:
        print(f"\nUsing {rom_cfg.train_frac*100:.1f}% of time steps for training")
        truncated_trajs = []
        for Z in all_latent_trajectories:
            T_train = int(Z.shape[0] * rom_cfg.train_frac)
            truncated_trajs.append(Z[:T_train])
        train_latents = truncated_trajs
    else:
        train_latents = all_latent_trajectories
    
    mvar_result = fit_mvar(
        train_latents,
        order=rom_cfg.mvar_order,
        ridge=rom_cfg.ridge,
    )
    
    print(f"\n✓ MVAR model fitted")
    print(f"  Order: {mvar_result['order']}")
    print(f"  Latent dimension: {mvar_result['latent_dim']}")
    print(f"  Total parameters: {mvar_result['latent_dim'] * (1 + mvar_result['order'] * mvar_result['latent_dim'])}")
    
    # Save MVAR model
    save_mvar_model(
        dirs["model"],
        mvar_result["A0"],
        mvar_result["A_coeffs"],
        mvar_result["order"],
        mvar_result["latent_dim"],
    )
    print(f"  Saved to {dirs['model'] / 'mvar_params.npz'}")
    
    # ========================================================================
    # Step 5: Save training summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Saving training summary")
    print("-" * 70)
    
    train_stats = {
        "num_train_ics": rom_cfg.num_train_ics,
        "train_seeds": rom_cfg.train_seeds,
        "total_snapshots": stacked_snapshots.shape[0],
        "snapshots_per_ic": [Z.shape[0] for Z in all_latent_trajectories],
    }
    
    save_train_summary(
        dirs["model"],
        rom_cfg,
        pod_result,
        mvar_result,
        train_stats,
    )
    print(f"\n✓ Training summary saved to {dirs['model'] / 'train_summary.json'}")
    
    # ========================================================================
    # Done
    # ========================================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nModel artifacts saved to: {dirs['model']}")
    print("\nNext steps:")
    print(f"  1. Evaluate on unseen ICs:")
    print(f"     python scripts/rom_mvar_eval.py --experiment {rom_cfg.experiment_name} --config {args.config}")
    print(f"  2. Generate visualizations (after evaluation):")
    print(f"     python scripts/rom_mvar_visualize.py --experiment {rom_cfg.experiment_name}")
    print()


if __name__ == "__main__":
    main()
