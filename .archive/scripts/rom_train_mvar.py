#!/usr/bin/env python3
"""Train MVAR model on latent trajectories.

This script implements Stage 3 of the ROM pipeline:
1. Load latent trajectories from POD projection
2. Fit MVAR model on training runs with time-based split
3. Save MVAR model for evaluation

Usage
-----
python scripts/rom_train_mvar.py \
    --experiment_name my_experiment \
    --mvar_order 4 \
    --ridge 1e-6 \
    --train_frac 0.8

Oscar workflow:
1. Generate ensemble: rectsim ensemble --config CONFIG
2. Build POD: python scripts/rom_build_pod.py --experiment_name EXP
3. Train MVAR: python scripts/rom_train_mvar.py --experiment_name EXP
4. Evaluate: python scripts/rom_evaluate.py --experiment_name EXP

Output structure:
    rom/<experiment_name>/mvar/
    ├── mvar_model.npz      # MVAR coefficients
    └── train_info.json     # Training metadata

Author: Maria
Date: November 2025
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rectsim.mvar import fit_mvar_from_runs
from rectsim.rom_eval import ROMConfig, get_forecast_split_indices


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train MVAR model on latent trajectories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Unique experiment identifier",
    )
    parser.add_argument(
        "--rom_root",
        type=str,
        default="rom",
        help="Root directory for ROM outputs",
    )
    parser.add_argument(
        "--mvar_order",
        type=int,
        default=4,
        help="MVAR model order (number of lags)",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-6,
        help="Ridge regularization parameter",
    )
    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.8,
        help="Fraction of each run's time steps for training (0.0-1.0)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    rom_root = Path(args.rom_root)
    exp_dir = rom_root / args.experiment_name

    print("=" * 80)
    print("ROM Pipeline - Stage 3: Train MVAR Model")
    print("=" * 80)
    print(f"Experiment:   {args.experiment_name}")
    print(f"ROM root:     {rom_root}")
    print(f"MVAR order:   {args.mvar_order}")
    print(f"Ridge:        {args.ridge}")
    print(f"Train frac:   {args.train_frac}")
    print()

    # Load configuration
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        print(f"ERROR: Configuration not found: {config_path}")
        print("Run rom_build_pod.py first!")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = ROMConfig.from_dict(json.load(f))

    # Update config with MVAR parameters
    config.mvar_order = args.mvar_order
    config.ridge = args.ridge
    config.train_frac = args.train_frac

    print(f"Training runs: {config.train_runs}")
    print()

    # ========================================================================
    # Step 1: Load latent trajectories for training runs
    # ========================================================================
    print("[1/2] Loading latent trajectories...")

    latent_dir = exp_dir / "latent"
    if not latent_dir.exists():
        print(f"ERROR: Latent directory not found: {latent_dir}")
        print("Run rom_build_pod.py first!")
        sys.exit(1)

    # Load training run latents
    latent_dict = {}
    for run_idx in config.train_runs:
        latent_file = latent_dir / f"run_{run_idx:04d}_latent.npz"

        if not latent_file.exists():
            print(f"WARNING: Latent file not found: {latent_file}")
            continue

        data = np.load(latent_file)
        run_name = str(data["run_name"])

        # Apply train_frac split to get training segment
        Y_full = data["Y"]
        times_full = data["times"]
        T = Y_full.shape[0]

        T_train, _ = get_forecast_split_indices(T, config.train_frac)

        # Use only training portion
        latent_dict[run_name] = {
            "Y": Y_full[:T_train],
            "times": times_full[:T_train],
        }

    if not latent_dict:
        print("ERROR: No latent trajectories loaded")
        sys.exit(1)

    print(f"Loaded {len(latent_dict)} training latent trajectories")

    # Get latent dimension
    first_Y = next(iter(latent_dict.values()))["Y"]
    r = first_Y.shape[1]
    print(f"Latent dimension: {r}")
    print()

    # ========================================================================
    # Step 2: Fit MVAR model
    # ========================================================================
    print("[2/2] Fitting MVAR model...")

    model, train_info = fit_mvar_from_runs(
        latent_dict,
        order=config.mvar_order,
        ridge=config.ridge,
        train_frac=1.0,  # Already split above
    )

    print(f"MVAR fitted:")
    print(f"  Order:           {model.order}")
    print(f"  Latent dim:      {model.latent_dim}")
    print(f"  Training samples: {train_info['total_samples']}")
    print()

    # Save MVAR model
    mvar_dir = exp_dir / "mvar"
    mvar_dir.mkdir(parents=True, exist_ok=True)

    model.save(mvar_dir / "mvar_model.npz")

    # Add configuration to train_info
    train_info.update({
        "experiment_name": config.experiment_name,
        "train_runs": config.train_runs,
        "train_frac": config.train_frac,
        "mvar_order": config.mvar_order,
        "ridge": config.ridge,
    })

    with open(mvar_dir / "train_info.json", "w") as f:
        json.dump(train_info, f, indent=2)

    print(f"Saved MVAR model to: {mvar_dir}")
    print()

    # Update main config
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    print("=" * 80)
    print("MVAR training complete!")
    print(f"Model saved to: {mvar_dir}")
    print()
    print("Next steps:")
    print(f"  Evaluate: python scripts/rom_evaluate.py --experiment_name {config.experiment_name}")
    print("=" * 80)


if __name__ == "__main__":
    main()
