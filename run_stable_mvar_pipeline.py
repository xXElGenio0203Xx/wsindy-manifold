#!/usr/bin/env python3
"""
























































































exit $EXIT_CODEecho "Job ended: $(date)"echo ""fi    echo "Check error log: slurm_logs/stable_mvar_v2_${SLURM_JOB_ID}.err"    echo "========================================"    echo "❌ PIPELINE FAILED"else    echo "  - Lower training R² (~0.95) but stable dynamics"    echo "  - Test R² > 0.5 (hopefully!)"    echo "  - Max eigenvalue < 0.95 (STABLE by design)"    echo "Expected improvements over robust_v1:"    echo ""    echo "  - predictions/stable_mvar_v2/"    echo "  - oscar_output/stable_mvar_v2/"    echo "Results saved to:"    echo ""    echo "========================================"    echo "✅ PIPELINE COMPLETED SUCCESSFULLY"if [ $EXIT_CODE -eq 0 ]; thenecho "========================================"echo ""EXIT_CODE=$?    --experiment_name stable_mvar_v2    --config configs/stable_mvar_v2.yaml \python run_stable_mvar_pipeline.py \echo "========================================"echo "RUNNING PIPELINE"echo "========================================"# Run the pipelineecho ""echo "  - Test runs: 40"echo "  - Training runs: 400 (mixed)"echo "  - Eigenvalue threshold: 0.95"echo "  - Ridge alpha: 0.5 (very strong)"echo "  - MVAR lag: 2 (reduced)"echo "  - POD energy: 99% (reduced)"echo "KEY PARAMETERS:"echo ""echo "Config: configs/stable_mvar_v2.yaml"echo "Experiment: stable_mvar_v2"echo "========================================"echo "CONFIGURATION"echo "========================================"echo ""pip install -q numpy scipy matplotlib pyyaml tqdm scikit-learnecho "Installing dependencies..."# Install dependencies# source ~/miniconda3/bin/activate wsindy# Activate conda if needed (optional)cd ~/wsindy-manifold# Navigate to project directorymodule load gcc/10.2module load python/3.9.0# Load modulesecho ""echo "Job ID: $SLURM_JOB_ID"echo "Node: $(hostname)"echo "Job started: $(date)"# ============================================================================#   4. Same mixed distributions (400 training runs)#   3. Eigenvalue scaling: Scale down if max|λ| > 0.95#   2. Very strong regularization: α=0.5 (10× stronger)#   1. Reduced complexity: POD 99% (vs 99.5%), lag=2 (vs 4)# Key changes from robust_v1:# ============================================================================# Stable MVAR v2: Explicit Stability Enforcement# ============================================================================#SBATCH --partition=batch#SBATCH --mem=64G#SBATCH --cpus-per-task=32#SBATCH --ntasks=1#SBATCH --time=36:00:00#SBATCH --error=slurm_logs/stable_mvar_v2_%j.err#SBATCH --output=slurm_logs/stable_mvar_v2_%j.outRobust ROM-MVAR Pipeline - Mixed Distribution Training
=======================================================

Comprehensive pipeline supporting:
- Multiple IC distributions (Gaussian, uniform, ring, two-cluster)
- Spatial grid coverage
- Strong regularization for stability
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
    
    train_ic_config = config.get('train_ic', {})
    test_ic_config = config.get('test_ic', {})
    rom_config = config.get('rom', {})
    
    return base_config, density_nx, density_ny, density_bandwidth, train_ic_config, test_ic_config, rom_config

def generate_training_configs(train_ic_config, base_config):
    """Generate list of training run configurations."""
    configs = []
    run_id = 0
    
    # Gaussian configurations
    if train_ic_config.get('gaussian', {}).get('enabled', False):
        gauss_cfg = train_ic_config['gaussian']
        positions_x = gauss_cfg['positions_x']
        positions_y = gauss_cfg['positions_y']
        variances = gauss_cfg['variances']
        n_samples = gauss_cfg['n_samples_per_config']
        
        for px in positions_x:
            for py in positions_y:
                for var in variances:
                    for sample in range(n_samples):
                        configs.append({
                            'run_id': run_id,
                            'distribution': 'gaussian_cluster',
                            'ic_params': {
                                'center': (float(px), float(py)),
                                'sigma': float(np.sqrt(var))
                            },
                            'label': f'gauss_x{px:.1f}_y{py:.1f}_var{var:.1f}_s{sample}'
                        })
                        run_id += 1
    
    # Uniform configurations
    if train_ic_config.get('uniform', {}).get('enabled', False):
        n_uniform = train_ic_config['uniform']['n_runs']
        for i in range(n_uniform):
            configs.append({
                'run_id': run_id,
                'distribution': 'uniform',
                'ic_params': {},
                'label': f'uniform_s{i}'
            })
            run_id += 1
    
    # Ring configurations
    if train_ic_config.get('ring', {}).get('enabled', False):
        ring_cfg = train_ic_config['ring']
        radii = ring_cfg['radii']
        widths = ring_cfg['widths']
        n_samples = ring_cfg['n_samples_per_config']
        
        for radius in radii:
            for width in widths:
                for sample in range(n_samples):
                    configs.append({
                        'run_id': run_id,
                        'distribution': 'ring',
                        'ic_params': {
                            'radius': float(radius),
                            'width': float(width)
                        },
                        'label': f'ring_r{radius:.1f}_w{width:.1f}_s{sample}'
                    })
                    run_id += 1
    
    # Two-cluster configurations
    if train_ic_config.get('two_clusters', {}).get('enabled', False):
        cluster_cfg = train_ic_config['two_clusters']
        separations = cluster_cfg['separations']
        sigmas = cluster_cfg['sigmas']
        n_samples = cluster_cfg['n_samples_per_config']
        
        for sep in separations:
            for sigma in sigmas:
                for sample in range(n_samples):
                    configs.append({
                        'run_id': run_id,
                        'distribution': 'two_clusters',
                        'ic_params': {
                            'separation': float(sep),
                            'sigma': float(sigma)
                        },
                        'label': f'twocluster_sep{sep:.1f}_sig{sigma:.1f}_s{sample}'
                    })
                    run_id += 1
    
    return configs

def generate_test_configs(test_ic_config, base_config):
    """Generate list of test run configurations."""
    configs = []
    run_id = 0
    
    # Gaussian test
    if test_ic_config.get('gaussian', {}).get('enabled', False):
        gauss_cfg = test_ic_config['gaussian']
        test_px = gauss_cfg.get('test_positions_x', [])
        test_py = gauss_cfg.get('test_positions_y', [])
        test_var = gauss_cfg.get('test_variances', [])
        
        for px in test_px:
            for py in test_py:
                for var in test_var:
                    configs.append({
                        'run_id': run_id,
                        'distribution': 'gaussian_cluster',
                        'ic_params': {
                            'center': (float(px), float(py)),
                            'sigma': float(np.sqrt(var))
                        },
                        'label': f'gauss_test_x{px:.1f}_y{py:.1f}_var{var:.1f}'
                    })
                    run_id += 1
        
        # Extrapolation tests
        extrap_pos = gauss_cfg.get('extrapolation_positions', [])
        extrap_var = gauss_cfg.get('extrapolation_variance', [])
        for pos in extrap_pos:
            for var in extrap_var:
                configs.append({
                    'run_id': run_id,
                    'distribution': 'gaussian_cluster',
                    'ic_params': {
                        'center': (float(pos[0]), float(pos[1])),
                        'sigma': float(np.sqrt(var))
                    },
                    'label': f'gauss_extrap_x{pos[0]:.1f}_y{pos[1]:.1f}_var{var:.1f}'
                })
                run_id += 1
    
    # Uniform test
    if test_ic_config.get('uniform', {}).get('enabled', False):
        n_uniform = test_ic_config['uniform']['n_runs']
        for i in range(n_uniform):
            configs.append({
                'run_id': run_id,
                'distribution': 'uniform',
                'ic_params': {},
                'label': f'uniform_test_s{i}'
            })
            run_id += 1
    
    # Ring test
    if test_ic_config.get('ring', {}).get('enabled', False):
        ring_cfg = test_ic_config['ring']
        test_radii = ring_cfg.get('test_radii', [])
        test_widths = ring_cfg.get('test_widths', [])
        n_samples = ring_cfg.get('n_samples_per_config', 1)
        
        for radius in test_radii:
            for width in test_widths:
                for sample in range(n_samples):
                    configs.append({
                        'run_id': run_id,
                        'distribution': 'ring',
                        'ic_params': {
                            'radius': float(radius),
                            'width': float(width)
                        },
                        'label': f'ring_test_r{radius:.1f}_w{width:.1f}_s{sample}'
                    })
                    run_id += 1
    
    # Two-cluster test
    if test_ic_config.get('two_clusters', {}).get('enabled', False):
        cluster_cfg = test_ic_config['two_clusters']
        test_sep = cluster_cfg.get('test_separations', [])
        test_sig = cluster_cfg.get('test_sigmas', [])
        
        for sep in test_sep:
            for sig in test_sig:
                configs.append({
                    'run_id': run_id,
                    'distribution': 'two_clusters',
                    'ic_params': {
                        'separation': float(sep),
                        'sigma': float(sig)
                    },
                    'label': f'twocluster_test_sep{sep:.1f}_sig{sig:.1f}'
                })
                run_id += 1
        
        # Extrapolation
        extrap_sep = cluster_cfg.get('extrapolation_separations', [])
        extrap_sig = cluster_cfg.get('extrapolation_sigma', [])
        for sep in extrap_sep:
            for sig in extrap_sig:
                configs.append({
                    'run_id': run_id,
                    'distribution': 'two_clusters',
                    'ic_params': {
                        'separation': float(sep),
                        'sigma': float(sig)
                    },
                    'label': f'twocluster_extrap_sep{sep:.1f}_sig{sig:.1f}'
                })
                run_id += 1
    
    return configs

def simulate_single_run(args_tuple):
    """Worker function for parallel simulation."""
    (run_config, BASE_CONFIG, OUTPUT_DIR, DENSITY_NX, DENSITY_NY, DENSITY_BANDWIDTH, is_test) = args_tuple
    
    run_id = run_config['run_id']
    distribution = run_config['distribution']
    ic_params = run_config['ic_params']
    label = run_config['label']
    
    if is_test:
        run_name = f"test_{run_id:03d}"
        run_dir = OUTPUT_DIR / "test" / run_name
    else:
        run_name = f"train_{run_id:03d}"
        run_dir = OUTPUT_DIR / "train" / run_name
    
    run_dir.mkdir(exist_ok=True, parents=True)
    
    # Set up simulation config
    config = BASE_CONFIG.copy()
    config["seed"] = (2000 if is_test else 1000) + run_id
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
    
    # Save density
    density_filename = "density_true.npz" if is_test else "density.npz"
    np.savez_compressed(
        run_dir / density_filename,
        rho=rho,
        xgrid=xgrid,
        ygrid=ygrid,
        times=result["times"]
    )
    
    metadata = {
        "run_id": run_id,
        "run_name": run_name,
        "distribution": distribution,
        "label": label,
        "ic_params": ic_params,
        "seed": config["seed"],
        "T": len(result["times"])
    }
    
    return metadata

def main():
    parser = argparse.ArgumentParser(description='Robust ROM-MVAR pipeline')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name')
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("="*80)
    print("ROBUST ROM-MVAR PIPELINE - MIXED DISTRIBUTIONS")
    print("="*80)
    print(f"\nExperiment: {args.experiment_name}")
    print(f"Config: {args.config}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load configuration
    (BASE_CONFIG, DENSITY_NX, DENSITY_NY, DENSITY_BANDWIDTH,
     train_ic_config, test_ic_config, rom_config) = load_config(args.config)
    
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
    print(f"   Ridge α: {rom_config.get('ridge_alpha', 1e-6)}")
    
    # Generate training configurations
    print(f"\n{'='*80}")
    print("STEP 1: Generating Training Data (Mixed Distributions)")
    print("="*80)
    
    train_configs = generate_training_configs(train_ic_config, BASE_CONFIG)
    n_train = len(train_configs)
    
    print(f"\nTraining configurations:")
    print(f"   Total runs: {n_train}")
    
    # Count by distribution
    dist_counts = {}
    for cfg in train_configs:
        dist = cfg['distribution']
        dist_counts[dist] = dist_counts.get(dist, 0) + 1
    for dist, count in dist_counts.items():
        print(f"   {dist}: {count} runs")
    
    TRAIN_DIR = OUTPUT_DIR / "train"
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    
    n_workers = min(cpu_count(), 16)
    print(f"\nUsing {n_workers} parallel workers...")
    
    train_args = [(cfg, BASE_CONFIG, OUTPUT_DIR, DENSITY_NX, DENSITY_NY, DENSITY_BANDWIDTH, False)
                  for cfg in train_configs]
    
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
    
    train_df = pd.DataFrame(train_metadata)
    train_df.to_csv(TRAIN_DIR / "index_mapping.csv", index=False)
    
    print(f"\n✓ Generated {n_train} training runs")
    print(f"   Time: {train_time/60:.1f}m")
    
    # POD and MVAR training
    print(f"\n{'='*80}")
    print("STEP 2: Global POD and MVAR Training")
    print("="*80)
    
    # Load all training densities
    print("\nLoading training densities...")
    all_rho = []
    for meta in tqdm(train_metadata, desc="Loading"):
        run_dir = TRAIN_DIR / meta["run_name"]
        density_file = run_dir / "density.npz"
        density_data = np.load(density_file)
        rho = density_data["rho"]
        all_rho.append(rho)
    
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
    TARGET_ENERGY = rom_config.get('pod_energy', 0.995)
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
    P_LAG = rom_config.get('mvar_lag', 4)
    RIDGE_ALPHA = rom_config.get('ridge_alpha', 1e-6)
    
    print(f"\nTraining global MVAR (p={P_LAG}, α={RIDGE_ALPHA})...")
    from sklearn.linear_model import Ridge
    
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
    
    # Check stability
    max_eig = max(np.max(np.abs(np.linalg.eigvals(A_matrices[i]))) for i in range(p))
    
    # Eigenvalue scaling for stability (NEW)
    eigenvalue_threshold = rom_config.get('eigenvalue_threshold', 0.95)
    if max_eig > eigenvalue_threshold:
        scale_factor = eigenvalue_threshold / max_eig
        A_matrices_orig = A_matrices.copy()
        A_matrices = A_matrices * scale_factor
        max_eig_scaled = max(np.max(np.abs(np.linalg.eigvals(A_matrices[i]))) for i in range(p))
        print(f"   ⚠ Max eigenvalue {max_eig:.4f} > {eigenvalue_threshold}")
        print(f"   → Scaled matrices by {scale_factor:.4f}")
        print(f"   → New max eigenvalue: {max_eig_scaled:.4f}")
        max_eig = max_eig_scaled
    
    print(f"✓ MVAR trained: {r}D latent, lag={p}")
    print(f"   Training R²: {train_r2:.4f}")
    print(f"   Training RMSE: {train_rmse:.4f}")
    print(f"   Max |eigenvalue|: {max_eig:.4f}")
    if max_eig < 1.0:
        print(f"   ✓ Model is STABLE (all |λ| < 1)")
    else:
        print(f"   ⚠ Model may be UNSTABLE ({max_eig:.4f} ≥ 1)")
    
    # Save models
    MVAR_DIR = OUTPUT_DIR / "mvar"
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
        train_rmse=train_rmse,
        max_eigenvalue=max_eig
    )
    
    pod_mvar_time = time.time() - train_start - train_time
    print(f"   Time: {pod_mvar_time/60:.1f}m")
    
    # Generate test data
    print(f"\n{'='*80}")
    print("STEP 3: Generating Test Data (Mixed Distributions)")
    print("="*80)
    
    test_configs = generate_test_configs(test_ic_config, BASE_CONFIG)
    n_test = len(test_configs)
    
    print(f"\nTest configurations:")
    print(f"   Total runs: {n_test}")
    
    # Count by distribution
    dist_counts = {}
    for cfg in test_configs:
        dist = cfg['distribution']
        dist_counts[dist] = dist_counts.get(dist, 0) + 1
    for dist, count in dist_counts.items():
        print(f"   {dist}: {count} runs")
    
    TEST_DIR = OUTPUT_DIR / "test"
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    test_args = [(cfg, BASE_CONFIG, OUTPUT_DIR, DENSITY_NX, DENSITY_NY, DENSITY_BANDWIDTH, True)
                 for cfg in test_configs]
    
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
    
    # Generate predictions
    print(f"\n{'='*80}")
    print("STEP 4: Generating ROM-MVAR Predictions")
    print("="*80)
    
    PRED_DIR = Path(f"predictions/{args.experiment_name}")
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    
    all_r2 = []
    
    for meta in tqdm(test_metadata, desc="Predictions"):
        run_name = meta["run_name"]
        run_dir = TEST_DIR / run_name
        
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
    
    pred_time = time.time() - test_start - test_time
    mean_r2 = np.mean(all_r2)
    std_r2 = np.std(all_r2)
    
    print(f"\n✓ Generated {n_test} predictions")
    print(f"   Mean R²: {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"   Range: [{np.min(all_r2):.4f}, {np.max(all_r2):.4f}]")
    print(f"   Time: {pred_time:.1f}s")
    
    # Summary
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\nTiming:")
    print(f"  Training generation: {train_time/60:.1f}m")
    print(f"  POD + MVAR: {pod_mvar_time/60:.1f}m")
    print(f"  Test generation: {test_time/60:.1f}m")
    print(f"  Predictions: {pred_time:.1f}s")
    print(f"  Total: {total_time/60:.1f}m")
    
    print(f"\nResults:")
    print(f"  Training: {n_train} runs (mixed distributions)")
    print(f"  POD: {R_POD} modes ({energy_captured:.4f} energy)")
    print(f"  MVAR: R²={train_r2:.4f}, max|λ|={max_eig:.4f}")
    print(f"  Test: {n_test} runs (mixed distributions)")
    print(f"  Prediction R²: {mean_r2:.4f} ± {std_r2:.4f}")
    
    print(f"\nOutput:")
    print(f"  Data: {OUTPUT_DIR}")
    print(f"  Predictions: {PRED_DIR}")
    
    print("\n✅ Ready for visualization!")

if __name__ == "__main__":
    main()
