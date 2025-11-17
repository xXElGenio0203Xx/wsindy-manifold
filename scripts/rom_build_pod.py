#!/usr/bin/env python3
"""Build global POD basis from training runs.

This script implements Stage 2 of the ROM pipeline:
1. Load density movies from training simulation runs
2. Compute global POD basis across all training data
3. Save POD basis and project all runs to latent space

Usage
-----
python scripts/rom_build_pod.py \
    --experiment_name my_experiment \
    --sim_root simulations/my_experiment/runs \
    --train_runs 0 1 2 3 4 \
    --test_runs 5 6 \
    --energy_threshold 0.995

Oscar workflow:
1. Generate ensemble: rectsim ensemble --config CONFIG
2. Build POD: python scripts/rom_build_pod.py --experiment_name EXP --train_runs ...
3. Train MVAR: python scripts/rom_train_mvar.py --experiment_name EXP

Output structure:
    rom/<experiment_name>/
    ├── pod/
    │   ├── basis.npz           # POD modes, singular values, energy
    │   ├── pod_energy.png      # Scree plot
    │   └── pod_info.json       # Metadata
    └── latent/
        ├── run_0000_latent.npz # Latent trajectories for each run
        └── ...

Author: Maria
Date: November 2025
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rectsim.mvar import (
    build_global_snapshot_matrix,
    compute_pod,
    load_density_movies,
    plot_pod_energy,
    project_to_pod,
)
from rectsim.rom_eval import ROMConfig, setup_rom_directories, split_runs_train_test


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build global POD basis from training runs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Unique experiment identifier",
    )
    parser.add_argument(
        "--sim_root",
        type=str,
        required=True,
        help="Root directory containing simulation run folders",
    )
    parser.add_argument(
        "--rom_root",
        type=str,
        default="rom",
        help="Root directory for ROM outputs",
    )
    parser.add_argument(
        "--train_runs",
        type=int,
        nargs="+",
        required=True,
        help="Indices of training runs (space-separated)",
    )
    parser.add_argument(
        "--test_runs",
        type=int,
        nargs="+",
        default=None,
        help="Indices of test runs (space-separated, optional)",
    )
    parser.add_argument(
        "--energy_threshold",
        type=float,
        default=0.995,
        help="POD energy threshold for mode selection",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=None,
        help="Fixed number of POD modes (overrides energy_threshold)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="run_*/density.npz",
        help="Glob pattern to match density.npz files",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup configuration
    config = ROMConfig(
        experiment_name=args.experiment_name,
        train_runs=args.train_runs,
        test_runs=args.test_runs if args.test_runs else [],
        energy_threshold=args.energy_threshold,
        latent_dim=args.latent_dim,
        sim_root=Path(args.sim_root),
        rom_root=Path(args.rom_root),
    )

    print("=" * 80)
    print("ROM Pipeline - Stage 2: Build Global POD Basis")
    print("=" * 80)
    print(f"Experiment:       {config.experiment_name}")
    print(f"Simulation root:  {config.sim_root}")
    print(f"ROM root:         {config.rom_root}")
    print(f"Training runs:    {config.train_runs}")
    if config.test_runs:
        print(f"Test runs:        {config.test_runs}")
    print(f"Energy threshold: {config.energy_threshold}")
    if config.latent_dim:
        print(f"Fixed latent dim: {config.latent_dim}")
    print()

    # Create directory structure
    paths = setup_rom_directories(config)
    print(f"Created directory structure under: {paths['base']}")
    print()

    # Find all run directories
    density_files = list(config.sim_root.rglob(args.pattern))
    if not density_files:
        print(f"ERROR: No files matching pattern '{args.pattern}' in {config.sim_root}")
        sys.exit(1)

    all_run_dirs = sorted([f.parent for f in density_files])
    print(f"Found {len(all_run_dirs)} total runs")

    # Split into train/test
    train_dirs, test_dirs = split_runs_train_test(
        all_run_dirs,
        config.train_runs,
        config.test_runs,
    )

    print(f"Training runs: {len(train_dirs)}")
    if test_dirs:
        print(f"Test runs:     {len(test_dirs)}")
    print()

    # ========================================================================
    # Step 1: Load training density movies
    # ========================================================================
    print("[1/4] Loading training density movies...")

    train_density_dict = load_density_movies(train_dirs)

    if not train_density_dict:
        print("ERROR: No training density movies loaded")
        sys.exit(1)

    print(f"Loaded {len(train_density_dict)} training runs")

    # Get grid info
    first_rho = next(iter(train_density_dict.values()))["rho"]
    T_first, ny, nx = first_rho.shape
    print(f"Grid shape: ({ny}, {nx})")
    print()

    # ========================================================================
    # Step 2: Build global snapshot matrix (training only)
    # ========================================================================
    print("[2/4] Building global snapshot matrix from training data...")

    X_train, run_slices_train, global_mean_flat = build_global_snapshot_matrix(
        train_density_dict, subtract_mean=True
    )

    T_train_total, d = X_train.shape
    print(f"Training snapshot matrix: ({T_train_total}, {d})")
    print()

    # ========================================================================
    # Step 3: Compute global POD basis
    # ========================================================================
    print("[3/4] Computing global POD basis...")

    pod_basis = compute_pod(
        X_train,
        r=config.latent_dim,
        energy_threshold=config.energy_threshold,
    )

    r = pod_basis["r"]
    energy_captured = pod_basis["energy"][r - 1]

    print(f"Number of POD modes: {r}")
    print(f"Energy captured:     {energy_captured:.6f}")
    print(f"Compression ratio:   {d / r:.1f}x")
    print()

    # Save POD basis
    basis_path = paths["pod"] / "basis.npz"
    np.savez(
        basis_path,
        Phi=pod_basis["Phi"],
        S=pod_basis["S"],
        mean=global_mean_flat,
        energy=pod_basis["energy"],
        r=r,
        ny=ny,
        nx=nx,
    )

    # Save metadata
    info = {
        "experiment_name": config.experiment_name,
        "r": int(r),
        "d": int(d),
        "ny": int(ny),
        "nx": int(nx),
        "energy_threshold": float(config.energy_threshold),
        "energy_captured": float(energy_captured),
        "compression_ratio": float(d / r),
        "train_runs": config.train_runs,
        "n_train_runs": len(train_dirs),
        "T_train_total": int(T_train_total),
    }

    with open(paths["pod"] / "pod_info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Plot POD energy
    plot_pod_energy(
        pod_basis["S"],
        paths["pod"] / "pod_energy.png",
        r_mark=r,
        energy_threshold=config.energy_threshold,
    )

    print(f"Saved POD basis to: {paths['pod']}")
    print()

    # ========================================================================
    # Step 4: Project all runs to latent space
    # ========================================================================
    print("[4/4] Projecting all runs to latent space...")

    # Load all density movies (train + test)
    all_density_dict = load_density_movies(all_run_dirs)

    # Project to latent space
    latent_dict = project_to_pod(
        all_density_dict,
        pod_basis["Phi"],
        global_mean_flat,
    )

    # Save latent trajectories
    for run_name, latent_data in latent_dict.items():
        # Extract run index from name (e.g., "run_0003" -> 3)
        run_idx = int(run_name.split("_")[-1])

        out_file = paths["latent"] / f"run_{run_idx:04d}_latent.npz"
        np.savez(
            out_file,
            Y=latent_data["Y"],
            times=latent_data["times"],
            run_name=run_name,
        )

    print(f"Saved {len(latent_dict)} latent trajectories to: {paths['latent']}")
    print()

    # Save configuration
    config_path = paths["base"] / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    print("=" * 80)
    print("POD basis construction complete!")
    print(f"All outputs saved to: {paths['base']}")
    print()
    print("Next steps:")
    print(f"  1. Train MVAR: python scripts/rom_train_mvar.py --experiment_name {config.experiment_name}")
    print(f"  2. Evaluate:   python scripts/rom_evaluate.py --experiment_name {config.experiment_name}")
    print("=" * 80)


if __name__ == "__main__":
    main()
