#!/usr/bin/env python3
"""
Generate a single simulation (training or test) for Oscar array job.

This script is called by SLURM array jobs to generate one simulation at a time.
Outputs trajectory NPZ, density NPZ, and optionally order parameters.

Usage:
    # Training simulation
    python oscar_pipeline/generate_single_sim.py \\
        --mode train \\
        --sim_id 0 \\
        --output_dir oscar_outputs/training

    # Test simulation with order parameters
    python oscar_pipeline/generate_single_sim.py \\
        --mode test \\
        --sim_id 0 \\
        --output_dir oscar_outputs/test \\
        --compute_order_params
"""

import argparse
import json
import sys
from pathlib import Path

# Add src and parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from rectsim.vicsek_discrete import simulate_backend
from rectsim.legacy_functions import kde_density_movie, compute_order_params

# Import shared configuration
from oscar_pipeline.config import (
    BASE_CONFIG,
    IC_TYPES,
    N_TRAIN,
    M_TEST,
    DENSITY_NX,
    DENSITY_NY,
    DENSITY_BANDWIDTH,
)


def generate_simulation(sim_id: int, mode: str, output_dir: Path, compute_order: bool = False):
    """
    Generate a single simulation with trajectory and density.
    
    Args:
        sim_id: Simulation ID (0 to N-1)
        mode: 'train' or 'test'
        output_dir: Output directory
        compute_order: Whether to compute order parameters (test only)
    """
    # Determine IC type (cycle through IC_TYPES)
    n_sims = N_TRAIN if mode == "train" else M_TEST
    ic_type = IC_TYPES[sim_id % len(IC_TYPES)]
    
    # Create run directory
    run_name = f"{mode}_{sim_id:03d}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Unique seed for reproducibility
    seed = 1000 + sim_id if mode == "train" else 2000 + sim_id
    
    print(f"="*60)
    print(f"Generating {mode} simulation {sim_id}/{n_sims-1}")
    print(f"IC type: {ic_type}, Seed: {seed}")
    print(f"Output: {run_dir}")
    print(f"="*60)
    
    # Run simulation
    config = BASE_CONFIG.copy()
    config["ic"] = {"kind": ic_type}
    
    # Create RNG
    rng = np.random.default_rng(seed)
    
    out = simulate_backend(config, rng)
    
    # Save trajectory
    traj_path = run_dir / "traj.npz"
    np.savez(
        traj_path,
        positions=out["traj"],
        times=out["times"],
        config=config,
    )
    print(f"✓ Saved trajectory: {traj_path.name}")
    
    # Compute density using KDE
    rho, meta = kde_density_movie(
        out["traj"],
        Lx=config["sim"]["Lx"],
        Ly=config["sim"]["Ly"],
        nx=DENSITY_NX,
        ny=DENSITY_NY,
        bandwidth=DENSITY_BANDWIDTH,
        bc=config["sim"]["bc"],
    )
    
    # Save density
    density_path = run_dir / "density.npz"
    np.savez(
        density_path,
        rho=rho,
        times=out["times"],
        x_edges=np.linspace(0, meta["Lx"], meta["nx"] + 1),
        y_edges=np.linspace(0, meta["Ly"], meta["ny"] + 1),
        extent=meta["extent"],
    )
    print(f"✓ Saved density: {density_path.name}")
    
    # Compute and save order parameters (test only)
    if compute_order and mode == "test":
        # Compute order params for each timestep
        T = out["vel"].shape[0]
        order_params = {
            "phi": [],
            "mean_speed": [],
            "speed_std": []
        }
        for t in range(T):
            params_t = compute_order_params(out["vel"][t])
            order_params["phi"].append(params_t["phi"])
            order_params["mean_speed"].append(params_t["mean_speed"])
            order_params["speed_std"].append(params_t["speed_std"])
        
        # Convert to arrays
        phi_array = np.array(order_params["phi"])
        mean_speed_array = np.array(order_params["mean_speed"])
        speed_std_array = np.array(order_params["speed_std"])
        
        order_path = run_dir / "order_params.npz"
        np.savez(
            order_path,
            phi=phi_array,
            mean_speed=mean_speed_array,
            speed_std=speed_std_array,
            times=out["times"],
        )
        print(f"✓ Saved order parameters: {order_path.name}")
    
    # Save metadata
    metadata = {
        "run_name": run_name,
        "sim_id": sim_id,
        "ic_type": ic_type,
        "seed": seed,
        "mode": mode,
        "config": config,
    }
    
    metadata_path = run_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_path.name}")
    
    print(f"\n✅ Completed {run_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate single simulation for Oscar array job"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "test"],
        help="Simulation mode (train or test)",
    )
    parser.add_argument(
        "--sim_id",
        type=int,
        required=True,
        help="Simulation ID (0 to N-1)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for this simulation",
    )
    parser.add_argument(
        "--compute_order_params",
        action="store_true",
        help="Compute order parameters (test mode only)",
    )
    
    args = parser.parse_args()
    
    # Validate sim_id
    n_sims = N_TRAIN if args.mode == "train" else M_TEST
    if not (0 <= args.sim_id < n_sims):
        raise ValueError(f"sim_id must be in [0, {n_sims-1}] for mode={args.mode}")
    
    # Generate simulation
    generate_simulation(
        sim_id=args.sim_id,
        mode=args.mode,
        output_dir=args.output_dir,
        compute_order=args.compute_order_params,
    )


if __name__ == "__main__":
    main()
