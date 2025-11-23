#!/usr/bin/env python3
"""
Data Generation Pipeline (Part 1 of 2)
========================================

Heavy Computation Pipeline for Oscar Cluster or Local Execution:
- Generates training ensemble (simulations + densities)
- Trains global POD + MVAR models
- Generates test ensemble (simulations + densities + order parameters)
- Runs MVAR-ROM predictions

All data files are saved for later visualization by the companion pipeline.

Compatible with: Oscar SLURM cluster + local execution
Output: All .npz files, CSVs, and model artifacts needed for visualization
"""

import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import json
import pandas as pd
import time
import argparse
import yaml
from multiprocessing import Pool, cpu_count
import os

# Import rectsim modules
from rectsim.vicsek_discrete import simulate_backend
from rectsim.legacy_functions import (
    kde_density_movie,
    compute_order_params
)

# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def load_config(config_path):
    """Load configuration from YAML file and extract parameters."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract simulation parameters
    sim_config = config.get('sim', {})
    
    # Build BASE_CONFIG in expected format
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
        "model": {
            "speed": config.get('model', {}).get('speed', 1.0)
        },
        "params": config.get('params', {"R": 2.0}),
        "noise": config.get('noise', {"kind": "gaussian", "eta": 0.3}),
        "forces": config.get('forces', {"enabled": False}),
    }
    
    # Extract density parameters if specified
    outputs = config.get('outputs', {})
    density_nx = outputs.get('density_resolution', 64)
    density_ny = outputs.get('density_resolution', 64)
    density_bandwidth = outputs.get('density_bandwidth', 2.0)
    
    # IC types (default to stratified sampling)
    ic_types = ["uniform", "gaussian_cluster", "ring", "two_clusters"]
    
    # POD/MVAR parameters (from config or defaults)
    rom_config = config.get('rom', {})
    target_energy = rom_config.get('pod_energy', 0.995)
    p_lag = rom_config.get('mvar_lag', 4)
    ridge_alpha = rom_config.get('ridge_alpha', 1e-6)
    
    return base_config, ic_types, density_nx, density_ny, density_bandwidth, target_energy, p_lag, ridge_alpha

# =============================================================================
# WORKER FUNCTION FOR PARALLEL SIMULATION
# =============================================================================

def simulate_single_run(args_tuple):
    """Worker function for parallel simulation execution."""
    i, IC_TYPES, BASE_CONFIG, TRAIN_DIR, DENSITY_NX, DENSITY_NY, DENSITY_BANDWIDTH, is_test = args_tuple
    
    if is_test:
        run_name = f"test_{i:03d}"
        run_dir = TRAIN_DIR.parent / "test" / run_name  # TRAIN_DIR is actually base dir
    else:
        run_name = f"train_{i:03d}"
        run_dir = TRAIN_DIR / run_name
    
    run_dir.mkdir(exist_ok=True, parents=True)
    
    ic_type = IC_TYPES[i % len(IC_TYPES)]
    seed = (2000 if is_test else 1000) + i
    
    config = BASE_CONFIG.copy()
    config["initial_distribution"] = ic_type
    config["seed"] = seed
    
    rng = np.random.default_rng(seed)
    result = simulate_backend(config, rng)
    
    # Save trajectories
    np.savez(
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
    
    # Save density with appropriate filename
    density_filename = "density_true.npz" if is_test else "density.npz"
    np.savez(
        run_dir / density_filename,
        rho=rho,
        xgrid=xgrid,
        ygrid=ygrid,
        times=result["times"]
    )
    
    metadata = {
        "run_id": i,
        "run_name": run_name,
        "ic_type": ic_type,
        "seed": seed,
        "T": len(result["times"])
    }
    
    return metadata, rho

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def fit_mvar(Y, p, alpha=1e-6):
    """Fit multivariate autoregressive model with ridge regularization."""
    T, r = Y.shape
    
    # Build design matrix X and target Y_target
    X_list = []
    Y_target_list = []
    
    for t in range(p, T):
        # Lag features: [Y[t-1], Y[t-2], ..., Y[t-p]]
        lags = []
        for lag in range(1, p+1):
            lags.append(Y[t-lag])
        X_list.append(np.concatenate(lags))
        Y_target_list.append(Y[t])
    
    X = np.array(X_list)  # (T-p, r*p)
    Y_target = np.array(Y_target_list)  # (T-p, r)
    
    # Ridge regression: A = (X'X + alpha*I)^-1 X' Y
    XtX = X.T @ X
    XtY = X.T @ Y_target
    A = np.linalg.solve(XtX + alpha * np.eye(X.shape[1]), XtY)  # (r*p, r)
    
    # Reshape to coefficient matrices [A1, A2, ..., Ap]
    A_matrices = []
    for i in range(p):
        A_matrices.append(A[i*r:(i+1)*r, :].T)  # (r, r)
    
    # Compute training R²
    Y_pred = X @ A
    ss_res = np.sum((Y_target - Y_pred)**2)
    ss_tot = np.sum((Y_target - Y_target.mean(axis=0))**2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((Y_target - Y_pred)**2))
    
    return A_matrices, r2, rmse


def mvar_forecast(y_init, A_matrices, T_forecast):
    """Forecast latent state using MVAR model."""
    p = len(A_matrices)
    r = A_matrices[0].shape[0]
    
    # Initialize with history
    y_history = list(y_init)  # Last p steps
    y_forecast = []
    
    for t in range(T_forecast):
        # Predict next step: y[t] = sum(Ai * y[t-i])
        y_next = np.zeros(r)
        for i, A in enumerate(A_matrices):
            y_next += A @ y_history[-(i+1)]
        
        y_forecast.append(y_next)
        y_history.append(y_next)
    
    return np.array(y_forecast)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Data Generation Pipeline (Heavy Computation)")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to YAML configuration file (e.g., configs/strong_clustering.yaml)")
    parser.add_argument("--experiment_name", type=str, default="test_sim",
                       help="Descriptive name for this experiment (creates subfolder in oscar_output)")
    parser.add_argument("--n_train", type=int, default=100,
                       help="Number of training simulations")
    parser.add_argument("--n_test", type=int, default=20,
                       help="Number of test simulations")
    parser.add_argument("--clean", action="store_true",
                       help="Clean experiment directory before starting")
    
    args = parser.parse_args()
    
    # Load configuration from YAML file
    print(f"\n📄 Loading configuration from: {args.config}")
    BASE_CONFIG, IC_TYPES, DENSITY_NX, DENSITY_NY, DENSITY_BANDWIDTH, TARGET_ENERGY, P_LAG, RIDGE_ALPHA = load_config(args.config)
    
    # Setup directories with experiment subfolder
    BASE_DIR = Path("oscar_output")
    OUTPUT_DIR = BASE_DIR / args.experiment_name
    TRAIN_DIR = OUTPUT_DIR / "train"
    TEST_DIR = OUTPUT_DIR / "test"
    MVAR_DIR = OUTPUT_DIR / "mvar"
    
    if args.clean and OUTPUT_DIR.exists():
        print(f"\n🗑️  Cleaning previous experiment: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    
    for d in [TRAIN_DIR, TEST_DIR, MVAR_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Save config file to output directory for reproducibility
    config_copy_path = OUTPUT_DIR / "config_used.yaml"
    shutil.copy(args.config, config_copy_path)
    
    print("="*80)
    print("DATA GENERATION PIPELINE (PART 1/2)")
    print("="*80)
    print(f"\n📄 Config: {args.config}")
    print(f"📁 Experiment: {args.experiment_name}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"\n⚙️  Configuration:")
    print(f"   Training runs: {args.n_train}")
    print(f"   Test runs: {args.n_test}")
    print(f"   IC types: {IC_TYPES}")
    print(f"   Particles: {BASE_CONFIG['sim']['N']}")
    print(f"   Duration: {BASE_CONFIG['sim']['T']}s (dt={BASE_CONFIG['sim']['dt']})")
    print(f"   Density: {DENSITY_NX}×{DENSITY_NY}, bandwidth={DENSITY_BANDWIDTH}")
    print(f"   POD target energy: {TARGET_ENERGY*100:.1f}%, MVAR lag: {P_LAG}")
    
    pipeline_start = time.time()
    
    # =============================================================================
    # STEP 1: TRAINING ENSEMBLE
    # =============================================================================
    
    print("\n" + "="*80)
    print("STEP 1: Generating Training Ensemble")
    print("="*80)
    
    step1_start = time.time()
    train_metadata = []
    train_densities = []
    
    # Determine number of workers
    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count()))
    print(f"\nSimulating {args.n_train} training runs using {n_workers} parallel workers...")
    
    # Prepare arguments for parallel execution
    train_args = [(i, IC_TYPES, BASE_CONFIG, TRAIN_DIR, DENSITY_NX, DENSITY_NY, DENSITY_BANDWIDTH, False) 
                  for i in range(args.n_train)]
    
    # Run simulations in parallel
    with Pool(n_workers) as pool:
        results = list(tqdm(pool.imap(simulate_single_run, train_args), 
                           total=args.n_train, 
                           desc="Training sims"))
    
    # Unpack results
    for metadata, rho in results:
        train_metadata.append(metadata)
        
        # Collect densities for POD
        T, ny, nx = rho.shape
        rho_flat = rho.reshape(T, nx * ny)
        train_densities.append(rho_flat)
    
    # Save training metadata
    with open(TRAIN_DIR / "metadata.json", "w") as f:
        json.dump(train_metadata, f, indent=2)
    
    print(f"\n✓ Generated {args.n_train} training runs")
    step1_time = time.time() - step1_start
    print(f"   Time: {step1_time//60:.0f}m {step1_time%60:.1f}s")
    
    # =============================================================================
    # STEP 2: GLOBAL POD + MVAR TRAINING
    # =============================================================================
    
    print("\n" + "="*80)
    print("STEP 2: Global POD and MVAR Training")
    print("="*80)
    
    step2_start = time.time()
    
    # Concatenate all training densities
    X_train = np.vstack(train_densities)
    print(f"\n✓ Snapshot matrix shape: {X_train.shape}")
    
    # Build index mapping
    index_map = []
    global_idx = 0
    for meta in train_metadata:
        for t in range(meta["T"]):
            index_map.append({
                "global_idx": global_idx,
                "run_id": meta["run_id"],
                "run_name": meta["run_name"],
                "ic_type": meta["ic_type"],
                "local_time": t
            })
            global_idx += 1
    
    pd.DataFrame(index_map).to_csv(TRAIN_DIR / "index_mapping.csv", index=False)
    
    # Compute global POD
    print(f"\nComputing global POD...")
    U_svd, S, Vt = np.linalg.svd(X_train, full_matrices=False)
    
    total_energy = np.sum(S**2)
    cumulative_energy = np.cumsum(S**2)
    cumulative_ratio = cumulative_energy / total_energy
    
    R_POD = np.argmax(cumulative_ratio >= TARGET_ENERGY) + 1
    actual_energy = cumulative_ratio[R_POD - 1]
    
    print(f"✓ R_POD = {R_POD} modes ({actual_energy*100:.2f}% energy)")
    
    # Extract POD basis
    U = Vt[:R_POD, :].T
    singular_values = S[:R_POD]
    
    # Save POD artifacts
    np.savez(
        MVAR_DIR / "pod_basis.npz",
        U=U,
        singular_values=singular_values,
        all_singular_values=S,
        total_energy=total_energy,
        explained_energy=np.sum(singular_values**2),
        energy_ratio=actual_energy,
        cumulative_ratio=cumulative_ratio
    )
    
    # Project to latent space
    Y_train = X_train @ U
    print(f"✓ Latent training data shape: {Y_train.shape}")
    
    # Save latent trajectories
    latent_runs = {}
    for meta in train_metadata:
        run_id = meta["run_id"]
        run_name = meta["run_name"]
        mask = [idx["run_id"] == run_id for idx in index_map]
        y_run = Y_train[mask, :]
        latent_runs[run_name] = y_run
    
    np.savez(MVAR_DIR / "latent_trajectories.npz", **latent_runs)
    
    # Train MVAR
    print(f"\nTraining global MVAR (p={P_LAG})...")
    Y_mvar_list = []
    for meta in train_metadata:
        run_name = meta["run_name"]
        y_run = latent_runs[run_name]
        if len(y_run) > P_LAG:
            Y_mvar_list.append(y_run[P_LAG:])
    
    Y_mvar = np.vstack(Y_mvar_list)
    A_matrices, train_r2, train_rmse = fit_mvar(Y_mvar, P_LAG, RIDGE_ALPHA)
    
    print(f"✓ MVAR trained: {R_POD}D latent, lag={P_LAG}")
    print(f"   Training R²: {train_r2:.4f}")
    print(f"   Training RMSE: {train_rmse:.4f}")
    
    # Save MVAR model
    np.savez(
        MVAR_DIR / "mvar_model.npz",
        A_matrices=A_matrices,
        p=P_LAG,
        r=R_POD,
        alpha=RIDGE_ALPHA,
        train_r2=train_r2,
        train_rmse=train_rmse
    )
    
    # Save X_train mean for R² calculation
    np.save(MVAR_DIR / "X_train_mean.npy", X_train.mean(axis=0))
    
    step2_time = time.time() - step2_start
    print(f"   Time: {step2_time//60:.0f}m {step2_time%60:.1f}s")
    
    # =============================================================================
    # STEP 3: TEST ENSEMBLE
    # =============================================================================
    
    print("\n" + "="*80)
    print("STEP 3: Generating Test Ensemble")
    print("="*80)
    
    step3_start = time.time()
    
    # Distribute test runs evenly
    runs_per_ic = args.n_test // len(IC_TYPES)
    extra_runs = args.n_test % len(IC_TYPES)
    
    test_ic_distribution = {ic: runs_per_ic for ic in IC_TYPES}
    for i in range(extra_runs):
        test_ic_distribution[IC_TYPES[i]] += 1
    
    print(f"\nSimulating {args.n_test} test runs:")
    for ic, count in test_ic_distribution.items():
        print(f"   {ic}: {count} runs")
    
    # Prepare arguments for parallel execution
    test_args = [(i, IC_TYPES, BASE_CONFIG, TRAIN_DIR, DENSITY_NX, DENSITY_NY, DENSITY_BANDWIDTH, True) 
                 for i in range(args.n_test)]
    
    # Run test simulations in parallel
    print(f"\nUsing {n_workers} parallel workers...")
    with Pool(n_workers) as pool:
        test_results = list(tqdm(pool.imap(simulate_single_run, test_args), 
                                total=args.n_test, 
                                desc="Test sims"))
    
    # Unpack test results
    test_metadata = []
    test_run_idx = 0
    
    for metadata, rho_true in test_results:
        test_metadata.append(metadata)
        test_run_idx += 1
    
    # Save test metadata
    with open(TEST_DIR / "metadata.json", "w") as f:
        json.dump(test_metadata, f, indent=2)
    
    print(f"\n✓ Generated {args.n_test} test runs")
    step3_time = time.time() - step3_start
    print(f"   Time: {step3_time//60:.0f}m {step3_time%60:.1f}s")
    
    # =============================================================================
    # STEP 4: ROM PREDICTION
    # =============================================================================
    
    print("\n" + "="*80)
    print("STEP 4: MVAR-ROM Prediction on Test Data")
    print("="*80)
    
    step4_start = time.time()
    
    print(f"\nMaking predictions on {args.n_test} test runs...")
    
    for meta in tqdm(test_metadata, desc="Predictions"):
        run_name = meta["run_name"]
        run_dir = TEST_DIR / run_name
        
        # Load true density
        data = np.load(run_dir / "density_true.npz")
        rho_true = data["rho"]
        times = data["times"]
        T, ny, nx = rho_true.shape
        n_space = nx * ny
        
        # Project to latent space
        rho_flat = rho_true.reshape(T, n_space)
        y_true = rho_flat @ U
        
        # Initialize MVAR
        y_init = y_true[:P_LAG]
        
        # Forecast
        T_forecast = T - P_LAG
        y_pred = mvar_forecast(y_init, A_matrices, T_forecast)
        
        # Combine init + forecast
        y_full = np.vstack([y_init, y_pred])
        
        # Reconstruct density
        rho_pred_flat = y_full @ U.T
        rho_pred = rho_pred_flat.reshape(T, ny, nx)
        
        # Save predicted density
        np.savez(
            run_dir / "density_pred.npz",
            rho=rho_pred,
            xgrid=data["xgrid"],
            ygrid=data["ygrid"],
            times=times
        )
        
        # Save latent trajectories for this test run
        np.savez(
            run_dir / "latent.npz",
            y_true=y_true,
            y_pred=y_full
        )
    
    print(f"\n✓ Completed predictions for {args.n_test} test runs")
    step4_time = time.time() - step4_start
    print(f"   Time: {step4_time//60:.0f}m {step4_time%60:.1f}s")
    
    # =============================================================================
    # SUMMARY
    # =============================================================================
    
    total_time = time.time() - pipeline_start
    
    # Save summary JSON
    summary_data = {
        "experiment_name": args.experiment_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "configuration": {
            "n_train": args.n_train,
            "n_test": args.n_test,
            "N": BASE_CONFIG["sim"]["N"],
            "T": BASE_CONFIG["sim"]["T"],
            "dt": BASE_CONFIG["sim"]["dt"],
            "density_resolution": f"{DENSITY_NX}×{DENSITY_NY}",
            "density_bandwidth": DENSITY_BANDWIDTH,
        },
        "rom_parameters": {
            "pod_modes": int(R_POD),
            "pod_energy_captured": float(actual_energy),
            "pod_energy_threshold": float(TARGET_ENERGY),
            "mvar_lag": int(P_LAG),
            "mvar_train_r2": float(train_r2),
            "ridge_alpha": float(RIDGE_ALPHA),
        },
        "timing": {
            "total_seconds": float(total_time),
            "step1_training_seconds": float(step1_time),
            "step2_pod_mvar_seconds": float(step2_time),
            "step3_testing_seconds": float(step3_time),
            "step4_prediction_seconds": float(step4_time),
        },
        "output_directory": str(OUTPUT_DIR),
    }
    
    summary_path = OUTPUT_DIR / "pipeline_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n✓ Pipeline summary saved: {summary_path}")
    
    print("\n" + "="*80)
    print("DATA GENERATION COMPLETE! 🎉")
    print("="*80)
    
    print(f"\n📂 All data saved to: {OUTPUT_DIR}")
    print(f"\n📊 Summary:")
    print(f"   • Training: {args.n_train} runs")
    print(f"   • Test: {args.n_test} runs")
    print(f"   • POD: {R_POD} modes ({actual_energy*100:.2f}% energy)")
    print(f"   • MVAR: p={P_LAG}, R²={train_r2:.4f}")
    print(f"\n⏱️  Total time: {total_time//60:.0f}m {total_time%60:.1f}s")
    print(f"   Step 1 (Training): {step1_time//60:.0f}m {step1_time%60:.1f}s")
    print(f"   Step 2 (POD+MVAR): {step2_time//60:.0f}m {step2_time%60:.1f}s")
    print(f"   Step 3 (Testing):  {step3_time//60:.0f}m {step3_time%60:.1f}s")
    print(f"   Step 4 (Predict):  {step4_time//60:.0f}m {step4_time%60:.1f}s")
    
    print(f"\n✅ Ready for visualization pipeline!")
    print(f"   Next: python run_visualizations.py --experiment_name {args.experiment_name}")
    print("="*80)


if __name__ == "__main__":
    main()
