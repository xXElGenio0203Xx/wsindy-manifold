#!/usr/bin/env python3
"""
Gaussians Data Generation Pipeline
===================================

Specialized pipeline for Gaussian IC experiments:
- Training: Gaussian ICs with SAME center, VARYING variances
- Testing: Gaussian ICs with DIFFERENT centers, FIXED variance

This tests if ROM-MVAR can:
1. Learn variance dynamics from training
2. Generalize to spatial translations (different centers)
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import pandas as pd
import time
import argparse
import yaml
from multiprocessing import Pool, cpu_count

# Import rectsim modules
from rectsim.vicsek_discrete import simulate_backend
from rectsim.legacy_functions import kde_density_movie

# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def load_config(config_path):
    """Load Gaussians configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract simulation parameters
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
    
    # Extract density parameters
    outputs = config.get('outputs', {})
    density_nx = outputs.get('density_resolution', 64)
    density_ny = outputs.get('density_resolution', 64)
    density_bandwidth = outputs.get('density_bandwidth', 2.0)
    
    # Extract ROM parameters
    rom_config = config.get('rom', {})
    target_energy = rom_config.get('pod_energy', 0.995)
    p_lag = rom_config.get('mvar_lag', 4)
    ridge_alpha = rom_config.get('ridge_alpha', 1e-6)
    
    # Extract custom Gaussian IC parameters
    train_ic_config = config.get('train_ic', {})
    test_ic_config = config.get('test_ic', {})
    
    return (base_config, density_nx, density_ny, density_bandwidth, 
            target_energy, p_lag, ridge_alpha, train_ic_config, test_ic_config)

# =============================================================================
# WORKER FUNCTION FOR PARALLEL SIMULATION
# =============================================================================

def simulate_single_run(args_tuple):
    """Worker function for parallel simulation execution."""
    (i, run_params, BASE_CONFIG, OUTPUT_DIR, DENSITY_NX, DENSITY_NY, 
     DENSITY_BANDWIDTH, is_test) = args_tuple
    
    if is_test:
        run_name = f"test_{i:03d}"
        run_dir = OUTPUT_DIR / "test" / run_name
    else:
        run_name = f"train_{i:03d}"
        run_dir = OUTPUT_DIR / "train" / run_name
    
    run_dir.mkdir(exist_ok=True, parents=True)
    
    # Unpack run-specific parameters
    center_x, center_y, variance, seed = run_params
    
    # Set up config - use gaussian_cluster with custom parameters
    config = BASE_CONFIG.copy()
    config["seed"] = seed
    config["initial_distribution"] = "gaussian_cluster"
    config["ic_params"] = {
        "center": (float(center_x), float(center_y)),
        "sigma": float(np.sqrt(variance))  # variance -> std dev
    }
    
    # Run simulation
    rng = np.random.default_rng(seed)
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
    
    # Save density with appropriate filename
    density_filename = "density_true.npz" if is_test else "density.npz"
    np.savez_compressed(
        run_dir / density_filename,
        rho=rho,
        xgrid=xgrid,
        ygrid=ygrid,
        times=result["times"]
    )
    
    metadata = {
        "run_id": i,
        "run_name": run_name,
        "ic_type": f"gaussian_cx{center_x:.1f}_cy{center_y:.1f}_var{variance:.1f}",
        "center_x": float(center_x),
        "center_y": float(center_y),
        "variance": float(variance),
        "seed": seed,
        "T": len(result["times"])
    }
    
    return metadata

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Gaussians data generation pipeline')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name')
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("="*80)
    print("GAUSSIANS DATA GENERATION PIPELINE")
    print("="*80)
    print(f"\nExperiment: {args.experiment_name}")
    print(f"Config: {args.config}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load configuration
    (BASE_CONFIG, DENSITY_NX, DENSITY_NY, DENSITY_BANDWIDTH, 
     TARGET_ENERGY, P_LAG, RIDGE_ALPHA, train_ic_config, test_ic_config) = load_config(args.config)
    
    OUTPUT_DIR = Path(f"oscar_output/{args.experiment_name}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save config for reference
    import shutil
    shutil.copy(args.config, OUTPUT_DIR / "config_used.yaml")
    
    print(f"\nConfiguration:")
    print(f"   N: {BASE_CONFIG['sim']['N']}")
    print(f"   T: {BASE_CONFIG['sim']['T']}s")
    print(f"   Domain: {BASE_CONFIG['sim']['Lx']}×{BASE_CONFIG['sim']['Ly']}")
    print(f"   Density: {DENSITY_NX}×{DENSITY_NY}")
    
    # =============================================================================
    # STEP 1: Generate Training Data (Varying Variance, Same Center)
    # =============================================================================
    
    print("\n" + "="*80)
    print("STEP 1: Checking for Existing Training Data")
    print("="*80)
    
    # Check if training data already exists
    TRAIN_DIR = OUTPUT_DIR / "train"
    existing_train_runs = []
    if TRAIN_DIR.exists():
        existing_train_runs = sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir() and d.name.startswith("train_")])
        print(f"\nFound {len(existing_train_runs)} existing training runs")
    
    if len(existing_train_runs) >= 190:  # At least most of training data exists
        print("✓ Using existing training data (skipping generation)")
        train_runs = []  # Empty list to skip generation
    else:
        print(f"Generating training data...")
        center_x = train_ic_config.get('center_x', train_ic_config['center'][0])
        center_y = train_ic_config.get('center_y', train_ic_config['center'][1])
        variances = train_ic_config['variances']
        n_samples_per_variance = train_ic_config['n_samples_per_variance']
        
        print(f"\nTraining ICs:")
        print(f"   Fixed center: ({center_x}, {center_y})")
        print(f"   Variances: {variances}")
        print(f"   Samples per variance: {n_samples_per_variance}")
        
        train_runs = []
        train_idx = 0
        for variance in variances:
            for sample in range(n_samples_per_variance):
                seed = 1000 + train_idx
                train_runs.append((center_x, center_y, variance, seed))
                train_idx += 1
        
        n_train = len(train_runs)
        print(f"   Total training runs: {n_train}")
        
        TRAIN_DIR.mkdir(parents=True, exist_ok=True)
        
        n_workers = min(cpu_count(), 16)
        print(f"\nUsing {n_workers} parallel workers...")
        
        train_args = [(i, run_params, BASE_CONFIG, OUTPUT_DIR, DENSITY_NX, DENSITY_NY, DENSITY_BANDWIDTH, False) 
                      for i, run_params in enumerate(train_runs)]
        
        train_start = time.time()
        
        with Pool(n_workers) as pool:
            train_metadata = list(tqdm(
                pool.imap(simulate_single_run, train_args),
                total=n_train,
                desc="Training sims"
            ))
        
        train_time = time.time() - train_start
        
        # Save training metadata
        with open(TRAIN_DIR / "metadata.json", "w") as f:
            json.dump(train_metadata, f, indent=2)
        
        # Save index mapping
        train_df = pd.DataFrame(train_metadata)
        train_df.to_csv(TRAIN_DIR / "index_mapping.csv", index=False)
        
        print(f"\n✓ Generated {n_train} training runs")
        print(f"   Time: {train_time/60:.1f}m")
    
    # =============================================================================
    # STEP 2: Build POD Basis and Train MVAR
    # =============================================================================
    
    print("\n" + "="*80)
    print("STEP 2: Global POD and MVAR Training")
    print("="*80)
    
    # Check if POD/MVAR models already exist
    MVAR_DIR = OUTPUT_DIR / "mvar"
    pod_basis_file = MVAR_DIR / "pod_basis.npz"
    mvar_model_file = MVAR_DIR / "mvar_model.npz"
    
    if pod_basis_file.exists() and mvar_model_file.exists():
        print("\n✓ POD and MVAR models already exist (skipping training)")
        print(f"   Loading from: {MVAR_DIR}")
        
        # Load existing models
        pod_data = np.load(pod_basis_file)
        U_r = pod_data["U"]
        R_POD = U_r.shape[1]
        energy_captured = pod_data["energy_ratio"]
        
        mvar_data = np.load(mvar_model_file)
        A_matrices = mvar_data["A_matrices"]
        P_LAG = A_matrices.shape[0]
        
        X_mean = np.load(MVAR_DIR / "X_train_mean.npy")
        
        print(f"✓ Loaded POD: {R_POD} modes ({energy_captured:.4f} energy)")
        print(f"✓ Loaded MVAR: {R_POD}D latent, lag={P_LAG}")
        
    else:
        # Load training metadata if not already loaded
        if 'train_metadata' not in locals():
            metadata_file = TRAIN_DIR / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    train_metadata = json.load(f)
                print(f"✓ Loaded metadata for {len(train_metadata)} training runs")
            else:
                # Reconstruct metadata from existing directories
                print("Reconstructing metadata from existing training runs...")
                train_metadata = []
                for train_dir in sorted(existing_train_runs):
                    run_name = train_dir.name
                    train_metadata.append({"run_name": run_name})
                print(f"✓ Reconstructed metadata for {len(train_metadata)} runs")
        
        # Load all training densities
        print("\nLoading training densities...")
        all_rho = []
        skipped = 0
        for meta in tqdm(train_metadata, desc="Loading"):
            run_dir = TRAIN_DIR / meta["run_name"]
            density_file = run_dir / "density.npz"
            if not density_file.exists():
                skipped += 1
                continue
            density_data = np.load(density_file)
            rho = density_data["rho"]
            all_rho.append(rho)
        
        if skipped > 0:
            print(f"⚠ Skipped {skipped} incomplete runs (using {len(all_rho)} runs)")
        
        # Stack into snapshot matrix
        all_rho = np.array(all_rho)
        M, T, ny, nx = all_rho.shape
        X_all = all_rho.reshape(M * T, ny * nx)
        
        print(f"\n✓ Snapshot matrix shape: ({M*T}, {ny*nx})")
        
        # Compute POD
        print("\nComputing global POD...")
        X_mean = X_all.mean(axis=0)
        X_centered = X_all - X_mean
        
        U, S, Vt = np.linalg.svd(X_centered.T, full_matrices=False)
        
        # Determine number of modes
        total_energy = np.sum(S**2)
        cumulative_energy = np.cumsum(S**2) / total_energy
        R_POD = np.searchsorted(cumulative_energy, TARGET_ENERGY) + 1
        
        U_r = U[:, :R_POD]
        energy_captured = cumulative_energy[R_POD - 1]
        
        print(f"✓ R_POD = {R_POD} modes ({energy_captured:.4f} energy)")
        
        # Project to latent space
        X_latent = X_centered @ U_r
        print(f"✓ Latent training data shape: ({M*T}, {R_POD})")
        
        # Train MVAR
        print(f"\nTraining global MVAR (p={P_LAG})...")
        from sklearn.linear_model import Ridge
        
        # Build MVAR design matrix
        p = P_LAG
        n = M * T - p
        r = R_POD
        
        Y = X_latent[p:, :]
        Phi = np.zeros((n, p * r))
        for lag in range(p):
            Phi[:, lag*r:(lag+1)*r] = X_latent[p-lag-1:n+p-lag-1, :]
        
        # Fit Ridge regression
        model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
        model.fit(Phi, Y)
        A_flat = model.coef_
        A_matrices = A_flat.reshape(r, p, r).transpose(1, 0, 2)
        
        # Compute training metrics
        Y_pred = Phi @ A_flat.T
        ss_res = np.sum((Y - Y_pred) ** 2)
        ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2)
        train_r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else (1.0 if ss_res < 1e-10 else 0.0)
        train_rmse = np.sqrt(np.mean((Y - Y_pred) ** 2))
        
        print(f"✓ MVAR trained: {r}D latent, lag={p}")
        print(f"   Training R²: {train_r2:.4f}")
        print(f"   Training RMSE: {train_rmse:.4f}")
        
        # Save models
        MVAR_DIR.mkdir(parents=True, exist_ok=True)
        
        np.save(MVAR_DIR / "X_train_mean.npy", X_mean)
        
        np.savez_compressed(
            MVAR_DIR / "pod_basis.npz",
            U=U_r,
            singular_values=S[:R_POD],
            all_singular_values=S,
            total_energy=total_energy,
            explained_energy=cumulative_energy[R_POD-1] * total_energy,
            energy_ratio=energy_captured,
            cumulative_ratio=cumulative_energy
        )
        
        np.savez_compressed(
            MVAR_DIR / "mvar_model.npz",
            A_matrices=A_matrices,
            p=p,
            r=r,
            alpha=RIDGE_ALPHA,
            train_r2=train_r2,
            train_rmse=train_rmse
        )
        
        # Save latent trajectories
        latent_dict = {}
        for i, meta in enumerate(train_metadata):
            run_latent = X_latent[i*T:(i+1)*T, :]
            latent_dict[meta["run_name"]] = run_latent
        
        np.savez_compressed(MVAR_DIR / "latent_trajectories.npz", **latent_dict)
    
    pod_mvar_time = time.time() - train_start - train_time
    print(f"   Time: {pod_mvar_time/60:.1f}m")
    
    # =============================================================================
    # STEP 3: Generate Test Data (Varying Center, Fixed Variance)
    # =============================================================================
    
    print("\n" + "="*80)
    print("STEP 3: Generating Test Ensemble (Varying Center)")
    print("="*80)
    
    test_variance = test_ic_config['variance']
    test_centers = test_ic_config['centers']
    
    print(f"\nTest ICs:")
    print(f"   Fixed variance: {test_variance}")
    print(f"   Centers: {test_centers}")
    
    test_runs = []
    test_idx = 0
    for center in test_centers:
        center_x, center_y = center
        seed = 2000 + test_idx
        test_runs.append((center_x, center_y, test_variance, seed))
        test_idx += 1
    
    n_test = len(test_runs)
    print(f"   Total test runs: {n_test}")
    
    TEST_DIR = OUTPUT_DIR / "test"
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    test_args = [(i, run_params, BASE_CONFIG, OUTPUT_DIR, DENSITY_NX, DENSITY_NY, DENSITY_BANDWIDTH, True) 
                 for i, run_params in enumerate(test_runs)]
    
    test_start = time.time()
    
    with Pool(n_workers) as pool:
        test_metadata = list(tqdm(
            pool.imap(simulate_single_run, test_args),
            total=n_test,
            desc="Test sims"
        ))
    
    test_time = time.time() - test_start
    
    # Save test metadata
    with open(TEST_DIR / "metadata.json", "w") as f:
        json.dump(test_metadata, f, indent=2)
    
    print(f"\n✓ Generated {n_test} test runs")
    print(f"   Time: {test_time/60:.1f}m")
    
    # =============================================================================
    # STEP 4: Generate Predictions
    # =============================================================================
    
    print("\n" + "="*80)
    print("STEP 4: Generating ROM-MVAR Predictions")
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
            pred_run_dir / "density_pred.npz",
            rho=rho_pred,
            xgrid=xgrid,
            ygrid=ygrid
        )
        
        np.savez_compressed(
            pred_run_dir / "density_true.npz",
            rho=rho_true,
            xgrid=xgrid,
            ygrid=ygrid
        )
        
        # Also copy to test directory for visualization pipeline
        np.savez_compressed(
            run_dir / "density_pred.npz",
            rho=rho_pred,
            xgrid=xgrid,
            ygrid=ygrid
        )
    
    pred_time = time.time() - test_start - test_time
    mean_r2 = np.mean(all_r2)
    std_r2 = np.std(all_r2)
    
    print(f"\n✓ Generated {n_test} predictions")
    print(f"   Mean R²: {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"   Range: [{np.min(all_r2):.4f}, {np.max(all_r2):.4f}]")
    print(f"   Time: {pred_time:.1f}s")
    
    # =============================================================================
    # SUMMARY
    # =============================================================================
    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\nTiming:")
    print(f"  Training generation: {train_time/60:.1f}m")
    print(f"  POD + MVAR: {pod_mvar_time/60:.1f}m")
    print(f"  Test generation: {test_time/60:.1f}m")
    print(f"  Predictions: {pred_time:.1f}s")
    print(f"  Total: {total_time/60:.1f}m")
    
    print(f"\nResults:")
    print(f"  Training: {n_train} runs (varying variance)")
    print(f"  POD: {R_POD} modes ({energy_captured:.4f} energy)")
    print(f"  MVAR: R²={train_r2:.4f} (training)")
    print(f"  Test: {n_test} runs (varying center)")
    print(f"  Prediction R²: {mean_r2:.4f} ± {std_r2:.4f}")
    
    print(f"\nOutput:")
    print(f"  Data: {OUTPUT_DIR}")
    print(f"  Predictions: {PRED_DIR}")
    
    print("\n✅ Ready for visualization!")

if __name__ == "__main__":
    main()
