#!/usr/bin/env python3
"""
Test Gaussian-trained POD+MVAR on Different IC Distributions
=============================================================

Uses existing POD+MVAR models from gaussians_oscar experiment
Tests generalization to uniform and ring initial conditions
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import yaml
import time
import argparse

from rectsim.vicsek_discrete import simulate_backend
from rectsim.legacy_functions import kde_density_movie

def load_config(config_path):
    """Load configuration from YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    sim_config = config.get('sim', {})
    base_config = {
        "sim": sim_config,
        "model": config.get('model', {}),
        "params": config.get('params', {}),
        "noise": config.get('noise', {}),
        "forces": config.get('forces', {}),
    }
    
    outputs = config.get('outputs', {})
    density_nx = outputs.get('density_resolution', 64)
    density_ny = outputs.get('density_resolution', 64)
    density_bandwidth = outputs.get('density_bandwidth', 2.0)
    
    test_ic_config = config.get('test_ic', {})
    rom_config = config.get('rom', {})
    
    return base_config, density_nx, density_ny, density_bandwidth, test_ic_config, rom_config

def main():
    parser = argparse.ArgumentParser(description='Test POD+MVAR on new distributions')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name')
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("="*80)
    print("TESTING POD+MVAR GENERALIZATION TO NEW DISTRIBUTIONS")
    print("="*80)
    print(f"\nExperiment: {args.experiment_name}")
    print(f"Config: {args.config}")
    
    # Load configuration
    BASE_CONFIG, DENSITY_NX, DENSITY_NY, DENSITY_BANDWIDTH, test_ic_config, rom_config = load_config(args.config)
    
    OUTPUT_DIR = Path(f"oscar_output/{args.experiment_name}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load existing POD+MVAR models
    source_experiment = rom_config.get('existing_experiment', 'gaussians_oscar')
    SOURCE_MVAR_DIR = Path(f"oscar_output/{source_experiment}/mvar")
    
    print(f"\n{'='*80}")
    print("STEP 1: Loading Existing POD+MVAR Models")
    print("="*80)
    print(f"Source: {source_experiment}")
    
    if not SOURCE_MVAR_DIR.exists():
        raise FileNotFoundError(f"Cannot find models at {SOURCE_MVAR_DIR}")
    
    pod_data = np.load(SOURCE_MVAR_DIR / "pod_basis.npz")
    U_r = pod_data["U"]
    R_POD = U_r.shape[1]
    energy_captured = pod_data["energy_ratio"]
    
    mvar_data = np.load(SOURCE_MVAR_DIR / "mvar_model.npz")
    A_matrices = mvar_data["A_matrices"]
    P_LAG = A_matrices.shape[0]
    r = A_matrices.shape[1]
    p = P_LAG
    
    X_mean = np.load(SOURCE_MVAR_DIR / "X_train_mean.npy")
    
    print(f"✓ Loaded POD: {R_POD} modes ({energy_captured:.4f} energy)")
    print(f"✓ Loaded MVAR: {r}D latent, lag={p}")
    
    # Generate test simulations
    print(f"\n{'='*80}")
    print("STEP 2: Generating Test Simulations with New Distributions")
    print("="*80)
    
    test_cases = test_ic_config.get('test_cases', [])
    print(f"\nTest cases: {len(test_cases)}")
    
    TEST_DIR = OUTPUT_DIR / "test"
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    test_metadata = []
    
    for i, test_case in enumerate(tqdm(test_cases, desc="Test sims")):
        run_name = f"test_{i:03d}"
        run_dir = TEST_DIR / run_name
        run_dir.mkdir(exist_ok=True, parents=True)
        
        distribution = test_case['distribution']
        label = test_case['label']
        ic_params = test_case.get('ic_params', {})
        
        # Set up config
        config = BASE_CONFIG.copy()
        config["seed"] = 3000 + i
        config["initial_distribution"] = distribution
        config["ic_params"] = ic_params
        
        # Run simulation
        rng = np.random.default_rng(config["seed"])
        result = simulate_backend(config, rng)
        
        # Save trajectories
        np.savez_compressed(
            run_dir / "trajectory.npz",
            traj=result["traj"],
            vel=result["vel"],
            times=result["times"]
        )
        
        # Compute KDE densities
        rho, meta = kde_density_movie(
            result["traj"],
            Lx=config["sim"]["Lx"],
            Ly=config["sim"]["Ly"],
            nx=DENSITY_NX,
            ny=DENSITY_NY,
            bandwidth=DENSITY_BANDWIDTH,
            bc=config["sim"]["bc"]
        )
        
        xgrid = np.linspace(0, config["sim"]["Lx"], DENSITY_NX, endpoint=False) + config["sim"]["Lx"]/(2*DENSITY_NX)
        ygrid = np.linspace(0, config["sim"]["Ly"], DENSITY_NY, endpoint=False) + config["sim"]["Ly"]/(2*DENSITY_NY)
        
        # Save ground truth density
        np.savez_compressed(
            run_dir / "density_true.npz",
            rho=rho,
            xgrid=xgrid,
            ygrid=ygrid,
            times=result["times"]
        )
        
        test_metadata.append({
            "run_id": i,
            "run_name": run_name,
            "distribution": distribution,
            "label": label,
            "ic_params": ic_params,
            "seed": config["seed"],
            "T": len(result["times"])
        })
    
    test_time = time.time() - start_time
    
    # Save metadata
    with open(TEST_DIR / "metadata.json", "w") as f:
        json.dump(test_metadata, f, indent=2)
    
    print(f"\n✓ Generated {len(test_cases)} test runs")
    print(f"   Time: {test_time/60:.1f}m")
    
    # Generate predictions
    print(f"\n{'='*80}")
    print("STEP 3: Generating ROM-MVAR Predictions")
    print("="*80)
    
    PRED_DIR = Path(f"predictions/{args.experiment_name}")
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    
    all_r2 = []
    
    for meta in tqdm(test_metadata, desc="Predictions"):
        run_name = meta["run_name"]
        run_dir = TEST_DIR / run_name
        pred_run_dir = PRED_DIR / run_name
        pred_run_dir.mkdir(parents=True, exist_ok=True)
        
        # Load true density
        density_data = np.load(run_dir / "density_true.npz")
        rho_true = density_data['rho']
        xgrid = density_data['xgrid']
        ygrid = density_data['ygrid']
        
        T_test = rho_true.shape[0]
        
        # Flatten and project to latent space
        rho_flat = rho_true.reshape(T_test, -1)
        x_latent = (rho_flat - X_mean) @ U_r
        
        # MVAR prediction
        x_pred = np.zeros((T_test, r))
        x_pred[:p] = x_latent[:p]  # Use true IC
        
        for t in range(p, T_test):
            x_next = np.zeros(r)
            for lag_idx in range(p):
                x_next += A_matrices[lag_idx] @ x_pred[t - lag_idx - 1]
            x_pred[t] = x_next
        
        # Reconstruct density
        rho_pred_flat = x_pred @ U_r.T + X_mean
        rho_pred = rho_pred_flat.reshape(T_test, ygrid.shape[0], xgrid.shape[0])
        
        # Compute R²
        ss_res = np.sum((rho_true - rho_pred) ** 2)
        ss_tot = np.sum((rho_true - rho_true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else (1.0 if ss_res < 1e-10 else 0.0)
        all_r2.append(r2)
        
        # Save predictions
        np.savez_compressed(
            run_dir / "density_pred.npz",
            rho=rho_pred,
            xgrid=xgrid,
            ygrid=ygrid,
            times=density_data['times']
        )
    
    pred_time = time.time() - start_time - test_time
    mean_r2 = np.mean(all_r2)
    std_r2 = np.std(all_r2)
    
    print(f"\n✓ Generated {len(test_cases)} predictions")
    print(f"   Mean R²: {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"   Range: [{np.min(all_r2):.4f}, {np.max(all_r2):.4f}]")
    print(f"   Time: {pred_time:.1f}s")
    
    # Summary
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\nTiming:")
    print(f"  Test generation: {test_time/60:.1f}m")
    print(f"  Predictions: {pred_time:.1f}s")
    print(f"  Total: {total_time/60:.1f}m")
    
    print(f"\nResults:")
    print(f"  Source POD+MVAR: {source_experiment} (trained on Gaussians)")
    print(f"  Test distributions: {len(set(m['distribution'] for m in test_metadata))} types")
    print(f"  Test runs: {len(test_cases)}")
    print(f"  Prediction R²: {mean_r2:.4f} ± {std_r2:.4f}")
    
    print(f"\nOutput:")
    print(f"  Data: {OUTPUT_DIR}")
    print(f"  Predictions: {PRED_DIR}")
    
    print("\n✅ Ready for visualization!")

if __name__ == "__main__":
    main()
