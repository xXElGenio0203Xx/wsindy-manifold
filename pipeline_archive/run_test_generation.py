#!/usr/bin/env python3
"""
Test Generation and Prediction Pipeline
========================================

This script continues from an existing training run that has POD/MVAR models.
It generates test simulations and predictions using the trained models.

Usage:
    python run_test_generation.py --experiment_name vicsek_forces_oscar --n_test 50
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
import time
from multiprocessing import Pool, cpu_count
import argparse

# Import rectsim modules
from rectsim.vicsek_discrete import simulate_backend
from rectsim.legacy_functions import kde_density_movie

# =============================================================================
# LOAD EXISTING CONFIG AND MODELS
# =============================================================================

def load_existing_setup(experiment_name):
    """Load configuration and trained models from existing run."""
    base_dir = Path(f"oscar_output/{experiment_name}")
    config_path = base_dir / "config_used.yaml"
    
    print(f"Loading from: {base_dir}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract parameters
    sim_config = config.get('sim', {})
    base_config = {
        "sim": {
            "N": sim_config.get('N', 100),
            "Lx": sim_config.get('Lx', 20.0),
            "Ly": sim_config.get('Ly', 20.0),
            "bc": sim_config.get('bc', 'periodic'),
            "T": sim_config.get('T', 10.0),
            "dt": sim_config.get('dt', 0.1),
            "save_every": sim_config.get('save_every', 1),
            "neighbor_rebuild": sim_config.get('neighbor_rebuild', 5),
        },
        "model": config.get('model', {}),
        "params": config.get('params', {}),
        "noise": config.get('noise', {}),
        "forces": config.get('forces', {}),
    }
    
    outputs = config.get('outputs', {})
    density_nx = outputs.get('density_resolution', 64)
    density_ny = outputs.get('density_resolution', 64)
    density_bandwidth = outputs.get('density_bandwidth', 2.0)
    
    # Load trained models
    mvar_dir = base_dir / "mvar"
    pod_data = np.load(mvar_dir / "pod_basis.npz")
    mvar_data = np.load(mvar_dir / "mvar_model.npz")
    X_train_mean = np.load(mvar_dir / "X_train_mean.npy")
    
    U = pod_data['U']
    A_matrices = mvar_data['A_matrices']
    p = int(mvar_data['p'])
    train_r2 = float(mvar_data['train_r2'])
    
    print(f"✓ Loaded POD: {U.shape[1]} modes ({pod_data['energy_ratio']:.4f} energy)")
    print(f"✓ Loaded MVAR: lag={p}, R²={train_r2:.4f}")
    
    return base_config, density_nx, density_ny, density_bandwidth, U, A_matrices, p, X_train_mean, base_dir

# =============================================================================
# TEST SIMULATION WORKER
# =============================================================================

def simulate_test_run(args_tuple):
    """Worker function to simulate one test run."""
    i, ic_type, base_config, test_dir, density_nx, density_ny, density_bandwidth = args_tuple
    
    run_name = f"test_{i:03d}"
    run_dir = test_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up config with IC and seed
    config = base_config.copy()
    config["initial_distribution"] = ic_type
    config["seed"] = 2000 + i
    
    rng = np.random.default_rng(2000 + i)
    
    # Run simulation
    result = simulate_backend(config, rng)
    
    # Compute density
    Lx = config["sim"]["Lx"]
    Ly = config["sim"]["Ly"]
    rho, meta = kde_density_movie(
        result["traj"],
        Lx=Lx,
        Ly=Ly,
        nx=density_nx,
        ny=density_ny,
        bandwidth=density_bandwidth,
        bc=config["sim"]["bc"]
    )
    
    xgrid = np.linspace(0, Lx, density_nx, endpoint=False) + Lx/(2*density_nx)
    ygrid = np.linspace(0, Ly, density_ny, endpoint=False) + Ly/(2*density_ny)
    
    # Save trajectory
    np.savez_compressed(
        run_dir / "trajectory.npz",
        traj=result["traj"],
        vel=result["vel"],
        times=result["times"]
    )
    
    # Save density (as density_true.npz for test runs)
    np.savez_compressed(
        run_dir / "density_true.npz",
        rho=rho,
        xgrid=xgrid,
        ygrid=ygrid,
        times=result["times"]
    )
    
    return run_name, ic_type, rho.shape[0]

# =============================================================================
# PREDICTION GENERATION
# =============================================================================

def generate_predictions(test_runs, U, A_matrices, p, X_train_mean, test_dir, pred_dir):
    """Generate ROM-MVAR predictions for test runs."""
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("STEP 4: Generating ROM-MVAR Predictions")
    print("="*80)
    
    all_r2 = []
    
    for run_name, ic_type, T in tqdm(test_runs, desc="Predictions"):
        run_dir = test_dir / run_name
        pred_run_dir = pred_dir / run_name
        pred_run_dir.mkdir(parents=True, exist_ok=True)
        
        # Load true density
        density_data = np.load(run_dir / "density_true.npz")
        rho_true = density_data['rho']
        xgrid = density_data['xgrid']
        ygrid = density_data['ygrid']
        
        T_actual = rho_true.shape[0]
        
        # Flatten spatial dimensions
        rho_flat = rho_true.reshape(T_actual, -1)
        
        # Project to latent space (centered)
        x_latent = (rho_flat - X_train_mean) @ U
        
        # MVAR prediction
        r = U.shape[1]
        x_pred = np.zeros((T_actual, r))
        x_pred[:p] = x_latent[:p]  # Use true initial conditions
        
        for t in range(p, T_actual):
            x_next = np.zeros(r)
            for lag_idx in range(p):
                x_next += A_matrices[lag_idx] @ x_pred[t - lag_idx - 1]
            x_pred[t] = x_next
        
        # Reconstruct density
        rho_pred_flat = x_pred @ U.T + X_train_mean
        rho_pred = rho_pred_flat.reshape(T_actual, xgrid.shape[0], ygrid.shape[0])
        
        # Compute R²
        ss_res = np.sum((rho_true - rho_pred) ** 2)
        ss_tot = np.sum((rho_true - rho_true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else (1.0 if ss_res < 1e-10 else 0.0)
        all_r2.append(r2)
        
        # Save prediction
        np.savez_compressed(
            pred_run_dir / "density_pred.npz",
            rho=rho_pred,
            xgrid=xgrid,
            ygrid=ygrid
        )
        
        # Save true density as well
        np.savez_compressed(
            pred_run_dir / "density_true.npz",
            rho=rho_true,
            xgrid=xgrid,
            ygrid=ygrid
        )
    
    mean_r2 = np.mean(all_r2)
    std_r2 = np.std(all_r2)
    
    print(f"\n✓ Generated {len(test_runs)} predictions")
    print(f"   Mean R²: {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"   Range: [{np.min(all_r2):.4f}, {np.max(all_r2):.4f}]")
    
    return mean_r2, std_r2, all_r2

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate test data and predictions')
    parser.add_argument('--experiment_name', type=str, required=True,
                       help='Name of existing experiment with trained models')
    parser.add_argument('--n_test', type=int, default=50,
                       help='Number of test runs to generate')
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("="*80)
    print("TEST GENERATION AND PREDICTION PIPELINE")
    print("="*80)
    print(f"\nExperiment: {args.experiment_name}")
    print(f"Test runs: {args.n_test}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load existing setup
    print("\n" + "="*80)
    print("Loading Existing Configuration and Models")
    print("="*80)
    
    base_config, density_nx, density_ny, density_bandwidth, U, A_matrices, p, X_train_mean, base_dir = \
        load_existing_setup(args.experiment_name)
    
    # Prepare IC distribution (stratified sampling)
    ic_types = ["uniform", "gaussian_cluster", "ring", "two_clusters"]
    test_ics = []
    for i in range(args.n_test):
        test_ics.append(ic_types[i % len(ic_types)])
    
    print(f"\nIC distribution:")
    for ic_type in ic_types:
        count = test_ics.count(ic_type)
        print(f"  {ic_type}: {count} runs")
    
    # Generate test simulations
    test_dir = base_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("STEP 3: Generating Test Ensemble")
    print("="*80)
    
    n_workers = min(cpu_count(), 16)  # Cap at 16 for Oscar
    print(f"Using {n_workers} parallel workers...")
    
    sim_args = [
        (i, test_ics[i], base_config, test_dir, density_nx, density_ny, density_bandwidth)
        for i in range(args.n_test)
    ]
    
    sim_start = time.time()
    
    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(simulate_test_run, sim_args),
            total=args.n_test,
            desc="Test sims"
        ))
    
    sim_time = time.time() - sim_start
    print(f"\n✓ Generated {args.n_test} test runs")
    print(f"   Time: {sim_time/60:.1f}m")
    
    # Generate predictions
    pred_dir = Path(f"predictions/{args.experiment_name}")
    
    pred_start = time.time()
    mean_r2, std_r2, all_r2 = generate_predictions(
        results, U, A_matrices, p, X_train_mean, test_dir, pred_dir
    )
    pred_time = time.time() - pred_start
    
    # Final summary
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\nTiming:")
    print(f"  Test generation: {sim_time/60:.1f}m")
    print(f"  Predictions: {pred_time:.1f}s")
    print(f"  Total: {total_time/60:.1f}m")
    print(f"\nResults:")
    print(f"  Prediction R²: {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"  Best R²: {np.max(all_r2):.4f}")
    print(f"  Worst R²: {np.min(all_r2):.4f}")
    print(f"\nOutput:")
    print(f"  Test data: {test_dir}")
    print(f"  Predictions: {pred_dir}")
    print(f"\nReady for visualization pipeline!")

if __name__ == "__main__":
    main()
