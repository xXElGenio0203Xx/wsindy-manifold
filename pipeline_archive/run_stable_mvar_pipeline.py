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
from rectsim.legacy_functions import kde_density_movie, polarization, mean_speed as compute_mean_speed, nematic_order
from rectsim.metrics import angular_momentum

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
    with open(args.config, 'r') as f:
        full_config = yaml.safe_load(f)
    
    (BASE_CONFIG, DENSITY_NX, DENSITY_NY, DENSITY_BANDWIDTH,
     train_ic_config, test_ic_config, rom_config) = load_config(args.config)
    
    # Extract evaluation config if present
    eval_config = full_config.get('evaluation', {})
    
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
    
    # Get ROM subsampling parameter (check both 'rom_subsample' and 'subsample' keys)
    ROM_SUBSAMPLE = rom_config.get('rom_subsample', rom_config.get('subsample', 1))
    dt_micro = BASE_CONFIG['sim']['dt']
    dt_rom = ROM_SUBSAMPLE * dt_micro
    
    print(f"\nROM temporal subsampling:")
    print(f"   Micro dt: {dt_micro}s")
    print(f"   ROM subsample: every {ROM_SUBSAMPLE} frames")
    print(f"   ROM dt: {dt_rom}s")
    
    # Load all training densities
    print("\nLoading training densities...")
    all_rho = []
    for meta in tqdm(train_metadata, desc="Loading"):
        run_dir = TRAIN_DIR / meta["run_name"]
        density_file = run_dir / "density.npz"
        density_data = np.load(density_file)
        rho = density_data["rho"]
        
        # Subsample for ROM
        rho_rom = rho[::ROM_SUBSAMPLE]
        all_rho.append(rho_rom)
    
    # Stack into snapshot matrix
    all_rho = np.array(all_rho)
    M, T_rom, ny, nx = all_rho.shape
    X_all = all_rho.reshape(M * T_rom, ny * nx)
    
    print(f"\n✓ Original frames per run: {rho.shape[0]}")
    print(f"✓ ROM frames per run: {T_rom}")
    print(f"✓ Spatial grid: {ny} × {nx}")
    print(f"✓ Snapshot matrix shape: ({M*T_rom}, {ny*nx})")
    
    # Compute POD
    print("\nComputing global POD...")
    X_mean = X_all.mean(axis=0)
    X_centered = X_all - X_mean
    
    U, S, Vt = np.linalg.svd(X_centered.T, full_matrices=False)
    
    # Determine number of modes
    # Priority: fixed_modes/fixed_d (if specified) > pod_energy (threshold)
    FIXED_D = rom_config.get('fixed_modes', None)  # Check 'fixed_modes' first (standard name)
    if FIXED_D is None:
        FIXED_D = rom_config.get('fixed_d', None)  # Fall back to 'fixed_d' for backward compatibility
    
    TARGET_ENERGY = rom_config.get('pod_energy', 0.995)
    
    total_energy = np.sum(S**2)
    cumulative_energy = np.cumsum(S**2) / total_energy
    
    if FIXED_D is not None:
        # Use fixed dimension (PRIORITY: explicit mode count overrides energy threshold)
        R_POD = min(FIXED_D, len(S))
        energy_captured = cumulative_energy[R_POD - 1]
        print(f"✓ Using FIXED d={R_POD} modes (energy={energy_captured:.4f}, hard cap from config)")
    else:
        # Use energy threshold
        R_POD = np.searchsorted(cumulative_energy, TARGET_ENERGY) + 1
        energy_captured = cumulative_energy[R_POD - 1]
        print(f"✓ R_POD = {R_POD} modes (energy={energy_captured:.4f}, threshold={TARGET_ENERGY})")
    
    U_r = U[:, :R_POD]
    
    # Project to latent space
    X_latent = X_centered @ U_r
    print(f"✓ Latent training data shape: ({M*T_rom}, {R_POD})")
    
    # Train MVAR
    P_LAG = rom_config.get('mvar_lag', 5)  # Default to 5 for MVAR(5)
    RIDGE_ALPHA = rom_config.get('ridge_alpha', 0.05)  # Default to 0.05
    
    print(f"\nTraining MVAR({P_LAG}) in latent space (d={R_POD}, α={RIDGE_ALPHA})...")
    
    w = P_LAG  # lag window
    d = R_POD  # latent dimension
    
    # Build regression dataset (Y_minus, Y_plus) as per Step 4.2
    print(f"   Building regression dataset...")
    X_minus_list = []
    X_plus_list = []
    
    # X_latent has shape (M*T_rom, d), we need to split by runs and build time series
    
    for run_idx in range(M):
        Y_r = X_latent[run_idx * T_rom:(run_idx + 1) * T_rom, :]  # (T_rom, d)
        T_steps = Y_r.shape[0]
        
        for t in range(w, T_steps):
            # Current target y_t
            x_plus = Y_r[t]  # shape (d,)
            # Stack the last w states [y_{t-1}, ..., y_{t-w}]
            x_minus = Y_r[t-w:t][::-1].reshape(-1)  # (w, d) -> reverse -> flatten to (w*d,)
            X_plus_list.append(x_plus)
            X_minus_list.append(x_minus)
    
    # Stack over all runs
    X_plus = np.vstack(X_plus_list)    # shape (N, d)
    X_minus = np.vstack(X_minus_list)  # shape (N, d*w)
    N_samples = X_plus.shape[0]
    
    print(f"   Dataset size: N={N_samples} samples (from {M} runs × ~{T_rom-w} timesteps each)")
    
    # Fit ridge regression (Step 4.3)
    print(f"   Fitting ridge regression with α={RIDGE_ALPHA}...")
    Y_plus = X_plus.T    # (d, N)
    Y_minus = X_minus.T  # (d*w, N)
    
    G = (Y_minus @ Y_minus.T) / N_samples      # (d*w, d*w)
    C = (Y_plus @ Y_minus.T) / N_samples       # (d, d*w)
    
    A = C @ np.linalg.inv(G + RIDGE_ALPHA * np.eye(d * w))   # (d, d*w)
    
    # Reshape A into block matrices A_1,...,A_w (Step 4.4)
    A_blocks = []
    for j in range(w):
        A_j = A[:, j*d:(j+1)*d]    # each A_j has shape (d, d)
        A_blocks.append(A_j)
    
    # Convert to format expected by rest of pipeline: (w, d, d)
    A_matrices = np.stack(A_blocks, axis=0)  # shape (w, d, d)
    
    print(f"   Block matrices A_1,...,A_{w} created, each shape ({d}, {d})")
    for j in range(w):
        norm_j = np.linalg.norm(A_blocks[j])
        print(f"     ||A_{j+1}|| = {norm_j:.4f}")
    
    # Compute training metrics
    Y_pred_train = X_minus @ A.T  # (N, d)
    ss_res = np.sum((X_plus - Y_pred_train) ** 2)
    ss_tot = np.sum((X_plus - X_plus.mean(axis=0)) ** 2)
    train_r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else (1.0 if ss_res < 1e-10 else 0.0)
    train_rmse = np.sqrt(np.mean((X_plus - Y_pred_train) ** 2))
    
    print(f"\n✓ MVAR({w}) trained: {d}D latent, {N_samples} samples")
    print(f"   Training R²: {train_r2:.4f}")
    print(f"   Training RMSE: {train_rmse:.4f}")
    
    # ===================================================================
    # STEP 5: EIGENVALUE SCALING FOR STABILITY (COMPANION MATRIX METHOD)
    # ===================================================================
    print(f"\n{'='*80}")
    print("STEP 5: Eigenvalue Scaling for Stability (Companion Matrix)")
    print("="*80)
    
    # 5.1. Build companion matrix A_comp ∈ R^(dw × dw)
    print(f"\n5.1. Building companion matrix ({d*w} × {d*w})...")
    dw = d * w
    A_comp = np.zeros((dw, dw))
    
    # Top block row: [A_1 A_2 ... A_w]
    A_comp[:d, :] = np.hstack([A_matrices[j] for j in range(w)])  # shape (d, d*w)
    
    # Subdiagonal identity blocks (shift register structure)
    for i in range(1, w):
        A_comp[i*d:(i+1)*d, (i-1)*d:i*d] = np.eye(d)
    
    print(f"   ✓ Companion matrix shape: {A_comp.shape}")
    
    # 5.2. Eigenvalue scaling
    print(f"\n5.2. Computing eigenvalues and scaling...")
    rho_max = rom_config.get('eigenvalue_threshold', None)
    
    eigvals, eigvecs = np.linalg.eig(A_comp)
    rho_before = np.max(np.abs(eigvals))
    print(f"   Companion spectral radius (before): {rho_before:.6f}")
    
    if rho_max is not None and rho_before > rho_max:
        print(f"   ⚠ Spectral radius {rho_before:.6f} > {rho_max}")
        print(f"   → Applying eigenvalue scaling...")
        
        # Global scaling: scale all eigenvalues uniformly
        c = rho_max / rho_before
        eigvals_scaled = c * eigvals
        
        # Reconstruct scaled companion matrix
        A_comp_scaled = eigvecs @ np.diag(eigvals_scaled) @ np.linalg.inv(eigvecs)
        A_comp_scaled = A_comp_scaled.real  # Discard tiny imaginary parts
        
        # Verify new spectral radius
        eigvals_new, _ = np.linalg.eig(A_comp_scaled)
        rho_after = np.max(np.abs(eigvals_new))
        print(f"   → Scaling factor: {c:.6f}")
        print(f"   ✓ Companion spectral radius (after): {rho_after:.6f}")
    else:
        print(f"   ✓ Already stable (ρ = {rho_before:.6f} ≤ {rho_max})")
        A_comp_scaled = A_comp
        rho_after = rho_before
    
    # 5.3. Extract stabilized A_j blocks from top row
    print(f"\n5.3. Extracting stabilized A_j blocks...")
    top_row = A_comp_scaled[:d, :]  # shape (d, d*w)
    
    A_matrices_stable = []
    for j in range(w):
        A_j_stable = top_row[:, j*d:(j+1)*d]  # (d, d)
        A_matrices_stable.append(A_j_stable)
    
    # Update A_matrices with stabilized version
    A_matrices = np.stack(A_matrices_stable, axis=0)  # (w, d, d)
    
    # Print block matrix norms for stabilized version
    print(f"   Stabilized block matrix norms:")
    for j in range(w):
        norm_j = np.linalg.norm(A_matrices[j], 'fro')
        print(f"      ||A_{j+1}|| = {norm_j:.4f}")
    
    print(f"\n✓ MVAR model spectrally stabilized")
    print(f"   Spectral radius: {rho_before:.6f} → {rho_after:.6f}")
    print(f"   Training R²: {train_r2:.4f}")
    if rho_after < 1.0:
        print(f"   ✓ Model is STABLE (ρ = {rho_after:.4f} < 1)")
    else:
        print(f"   ⚠ Model may be UNSTABLE (ρ = {rho_after:.4f} ≥ 1)")
    
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
        A_companion=A_comp_scaled,
        p=w,
        r=d,
        alpha=RIDGE_ALPHA,
        train_r2=train_r2,
        train_rmse=train_rmse,
        rho_before=float(rho_before),
        rho_after=float(rho_after)
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
    
    # Check if test uses different T than training
    TEST_CONFIG = BASE_CONFIG.copy()
    if 'test_T' in test_ic_config and test_ic_config['test_T'] is not None:
        TEST_CONFIG = {**BASE_CONFIG, 'sim': {**BASE_CONFIG['sim'], 'T': test_ic_config['test_T']}}
        print(f"\n⚠️  Using different T for testing: train={BASE_CONFIG['sim']['T']}s, test={test_ic_config['test_T']}s")
    
    test_args = [(cfg, TEST_CONFIG, OUTPUT_DIR, DENSITY_NX, DENSITY_NY, DENSITY_BANDWIDTH, True)
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
    all_r2_latent = []
    all_r2_pod = []
    all_mass_violations = []
    
    for meta in tqdm(test_metadata, desc="Predictions"):
        run_name = meta["run_name"]
        run_dir = TEST_DIR / run_name
        
        # Load true density
        density_data = np.load(run_dir / "density_true.npz")
        rho_true_full = density_data['rho']
        xgrid = density_data['xgrid']
        ygrid = density_data['ygrid']
        times_full = density_data['times']
        
        # Subsample for ROM (same as training)
        rho_true = rho_true_full[::ROM_SUBSAMPLE]
        times = times_full[::ROM_SUBSAMPLE]
        
        T_test = rho_true.shape[0]
        
        # Compute cell area for mass integration
        dx = xgrid[1] - xgrid[0] if len(xgrid) > 1 else 1.0
        dy = ygrid[1] - ygrid[0] if len(ygrid) > 1 else 1.0
        cell_area = dx * dy
        
        # Flatten and project to latent space
        rho_flat = rho_true.reshape(T_test, -1)
        x_latent = (rho_flat - X_mean) @ U_r
        
        # MVAR prediction
        x_pred = np.zeros((T_test, d))
        x_pred[:w] = x_latent[:w]  # Use true IC for first w timesteps
        
        for t in range(w, T_test):
            x_next = np.zeros(d)
            for lag_idx in range(w):
                x_next += A_matrices[lag_idx] @ x_pred[t - lag_idx - 1]
            x_pred[t] = x_next
        
        # Reconstruct density
        rho_pred_flat = x_pred @ U_r.T + X_mean
        rho_pred = rho_pred_flat.reshape(T_test, ygrid.shape[0], xgrid.shape[0])
        
        # ENFORCE MASS CONSERVATION
        # Compute initial mass from true density
        mass_true_initial = np.sum(rho_true[0]) * cell_area
        
        # Rescale each predicted timestep to conserve mass
        for t in range(T_test):
            mass_pred_t = np.sum(rho_pred[t]) * cell_area
            if mass_pred_t > 1e-10:  # Avoid division by zero
                rho_pred[t] *= (mass_true_initial / mass_pred_t)
        
        # Compute mass conservation violation (after correction)
        mass_true_t = np.sum(rho_true, axis=(1, 2)) * cell_area
        mass_pred_t = np.sum(rho_pred, axis=(1, 2)) * cell_area
        rel_mass_error = np.abs(mass_pred_t - mass_true_initial) / mass_true_initial
        max_mass_violation = np.max(rel_mass_error)
        all_mass_violations.append(max_mass_violation)
        
        # =================================================================
        # COMPUTE MULTIPLE R² METRICS
        # =================================================================
        
        # Check if we should evaluate on a subset of timesteps
        eval_start_time = eval_config.get('forecast_start', None)
        eval_end_time = eval_config.get('forecast_end', None)
        
        # Determine evaluation window
        if eval_start_time is not None and eval_end_time is not None:
            dt_rom = BASE_CONFIG['sim']['dt'] * ROM_SUBSAMPLE
            t_start_idx = int(eval_start_time / dt_rom)
            t_end_idx = int(eval_end_time / dt_rom)
            t_start_idx = max(w, t_start_idx)  # Ensure we're past warm-up
            t_end_idx = min(T_test, t_end_idx)
            
            # Extract evaluation window
            rho_true_eval = rho_true[t_start_idx:t_end_idx]
            rho_pred_eval = rho_pred[t_start_idx:t_end_idx]
            x_latent_eval = x_latent[t_start_idx:t_end_idx]
            x_pred_eval = x_pred[t_start_idx:t_end_idx]
            rho_pod_recon_eval = None  # Will compute below
            
            eval_window_str = f" (t={eval_start_time:.1f}-{eval_end_time:.1f}s)"
        else:
            # Use all timesteps after warm-up
            rho_true_eval = rho_true[w:]
            rho_pred_eval = rho_pred[w:]
            x_latent_eval = x_latent[w:]
            x_pred_eval = x_pred[w:]
            rho_pod_recon_eval = None
            
            eval_window_str = f" (t={w*BASE_CONFIG['sim']['dt']*ROM_SUBSAMPLE:.1f}s+)"
        
        # 1. R² in reconstructed (physical) space
        ss_res_phys = np.sum((rho_true_eval - rho_pred_eval) ** 2)
        ss_tot_phys = np.sum((rho_true_eval - rho_true_eval.mean()) ** 2)
        r2_reconstructed = 1 - ss_res_phys / ss_tot_phys if ss_tot_phys > 1e-10 else (1.0 if ss_res_phys < 1e-10 else 0.0)
        all_r2.append(r2_reconstructed)
        
        # 2. R² in latent space (ROM forecasting accuracy)
        ss_res_latent = np.sum((x_latent_eval - x_pred_eval) ** 2)
        ss_tot_latent = np.sum((x_latent_eval - x_latent_eval.mean(axis=0)) ** 2)
        r2_latent = 1 - ss_res_latent / ss_tot_latent if ss_tot_latent > 1e-10 else (1.0 if ss_res_latent < 1e-10 else 0.0)
        
        # 3. POD reconstruction error (how well POD captures true density)
        rho_pod_recon_flat = x_latent @ U_r.T + X_mean
        rho_pod_recon = rho_pod_recon_flat.reshape(T_test, ygrid.shape[0], xgrid.shape[0])
        
        if eval_start_time is not None and eval_end_time is not None:
            rho_pod_recon_eval = rho_pod_recon[t_start_idx:t_end_idx]
        else:
            rho_pod_recon_eval = rho_pod_recon[w:]
        
        ss_res_pod = np.sum((rho_true_eval - rho_pod_recon_eval) ** 2)
        ss_tot_pod = np.sum((rho_true_eval - rho_true_eval.mean()) ** 2)
        r2_pod = 1 - ss_res_pod / ss_tot_pod if ss_tot_pod > 1e-10 else (1.0 if ss_res_pod < 1e-10 else 0.0)
        
        # Relative reconstruction error (normalized RMSE)
        rmse_recon = np.sqrt(np.mean((rho_true_eval - rho_pred_eval) ** 2))
        rmse_pod = np.sqrt(np.mean((rho_true_eval - rho_pod_recon_eval) ** 2))
        rms_true = np.sqrt(np.mean(rho_true_eval ** 2))
        rel_error_recon = rmse_recon / rms_true if rms_true > 1e-10 else 0.0
        rel_error_pod = rmse_pod / rms_true if rms_true > 1e-10 else 0.0
        
        # Store metrics for all runs
        all_r2_latent.append(r2_latent)
        all_r2_pod.append(r2_pod)
        
        # =================================================================
        # COMPUTE TIME-RESOLVED R² (from warm-up to end)
        # =================================================================
        
        # Compute POD reconstruction for full trajectory (needed for time-resolved)
        rho_pod_recon_full = rho_pod_recon  # Already computed above
        
        # Initialize arrays for time-resolved R²
        r2_recon_vs_time = np.zeros(T_test - w)
        r2_latent_vs_time = np.zeros(T_test - w)
        r2_pod_vs_time = np.zeros(T_test - w)
        
        # Compute R² at each timestep (comparing against mean up to that point)
        for t_idx in range(w, T_test):
            t_slice = slice(w, t_idx + 1)
            
            # R² reconstructed (physical space)
            rho_true_slice = rho_true[t_slice]
            rho_pred_slice = rho_pred[t_slice]
            ss_res = np.sum((rho_true_slice - rho_pred_slice) ** 2)
            ss_tot = np.sum((rho_true_slice - rho_true_slice.mean()) ** 2)
            r2_recon_vs_time[t_idx - w] = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
            
            # R² latent (ROM space)
            x_latent_slice = x_latent[t_slice]
            x_pred_slice = x_pred[t_slice]
            ss_res_lat = np.sum((x_latent_slice - x_pred_slice) ** 2)
            ss_tot_lat = np.sum((x_latent_slice - x_latent_slice.mean(axis=0)) ** 2)
            r2_latent_vs_time[t_idx - w] = 1 - ss_res_lat / ss_tot_lat if ss_tot_lat > 1e-10 else 0.0
            
            # R² POD (reconstruction quality)
            rho_pod_slice = rho_pod_recon_full[t_slice]
            ss_res_pod = np.sum((rho_true_slice - rho_pod_slice) ** 2)
            r2_pod_vs_time[t_idx - w] = 1 - ss_res_pod / ss_tot if ss_tot > 1e-10 else 0.0
        
        # Save time-resolved metrics
        dt_rom = BASE_CONFIG['sim']['dt'] * ROM_SUBSAMPLE
        times_eval = np.arange(w, T_test) * dt_rom
        
        r2_time_df = pd.DataFrame({
            'time': times_eval,
            'r2_reconstructed': r2_recon_vs_time,
            'r2_latent': r2_latent_vs_time,
            'r2_pod': r2_pod_vs_time
        })
        r2_time_df.to_csv(run_dir / "r2_vs_time.csv", index=False)
        
        # Plot time-resolved R² (import matplotlib here to avoid issues)
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(times_eval, r2_recon_vs_time, 'b-', linewidth=2, label='R² Reconstructed (Physical Space)')
            ax.plot(times_eval, r2_latent_vs_time, 'r-', linewidth=2, label='R² Latent (ROM Space)')
            ax.plot(times_eval, r2_pod_vs_time, 'g-', linewidth=2, label='R² POD (Basis Quality)')
            ax.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('R² Score', fontsize=12)
            ax.set_title(f'Time-Resolved R² - {run_name}', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([-0.1, 1.05])
            plt.tight_layout()
            plt.savefig(run_dir / "r2_vs_time.png", dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Could not generate R² vs time plot: {e}")
        
        # Compute order parameters from density (spatial std as proxy)
        order_spatial_true = np.array([np.std(rho_true[t]) for t in range(T_test)])
        order_spatial_pred = np.array([np.std(rho_pred[t]) for t in range(T_test)])
        
        # Load trajectory for particle-based order parameters
        traj_data = np.load(run_dir / "trajectory.npz")
        traj = traj_data['traj']  # (T, N, 2)
        vel = traj_data['vel']    # (T, N, 2)
        
        # Compute particle-based order parameters
        phi_true = np.array([polarization(vel[t]) for t in range(T_test)])
        nematic_true = np.array([nematic_order(vel[t]) for t in range(T_test)])
        speed_true = np.array([compute_mean_speed(vel[t]) for t in range(T_test)])
        
        # Compute angular momentum (needs positions and velocities)
        config = BASE_CONFIG.copy()
        Lx, Ly = config['sim']['Lx'], config['sim']['Ly']
        ang_mom_true = np.array([angular_momentum(traj[t], vel[t]) for t in range(T_test)])
        
        # Save comprehensive order parameters
        order_df = pd.DataFrame({
            't': times,
            # Density-based
            'spatial_order_true': order_spatial_true,
            'spatial_order_pred': order_spatial_pred,
            # Particle-based (only from true trajectories)
            'polarization': phi_true,
            'nematic': nematic_true,
            'mean_speed': speed_true,
            'angular_momentum': ang_mom_true,
            # Mass conservation
            'mass_true': mass_true_t,
            'mass_pred': mass_pred_t,
            'mass_error_rel': rel_mass_error
        })
        order_df.to_csv(run_dir / "order_params_density.csv", index=False)
        
        # Save per-run metrics
        metrics_dict = {
            'run_name': run_name,
            'r2_reconstructed': r2_reconstructed,
            'r2_latent': r2_latent,
            'r2_pod': r2_pod,
            'rmse_recon': rmse_recon,
            'rmse_pod': rmse_pod,
            'rel_error_recon': rel_error_recon,
            'rel_error_pod': rel_error_pod,
            'max_mass_violation': max_mass_violation
        }
        with open(run_dir / "metrics_summary.json", 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        # Save predictions
        np.savez_compressed(
            run_dir / "density_pred.npz",
            rho=rho_pred,
            xgrid=xgrid,
            ygrid=ygrid,
            times=times
        )
    
    pred_time = time.time() - test_start - test_time
    mean_r2 = np.mean(all_r2)
    std_r2 = np.std(all_r2)
    mean_r2_latent = np.mean(all_r2_latent)
    std_r2_latent = np.std(all_r2_latent)
    mean_r2_pod = np.mean(all_r2_pod)
    std_r2_pod = np.std(all_r2_pod)
    mean_mass_violation = np.mean(all_mass_violations)
    max_mass_violation_overall = np.max(all_mass_violations)
    
    print(f"\n✓ Generated {n_test} predictions")
    print(f"   R² (reconstructed): {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"     Range: [{np.min(all_r2):.4f}, {np.max(all_r2):.4f}]")
    print(f"   R² (latent space): {mean_r2_latent:.4f} ± {std_r2_latent:.4f}")
    print(f"     Range: [{np.min(all_r2_latent):.4f}, {np.max(all_r2_latent):.4f}]")
    print(f"   R² (POD reconstruction): {mean_r2_pod:.4f} ± {std_r2_pod:.4f}")
    print(f"     Range: [{np.min(all_r2_pod):.4f}, {np.max(all_r2_pod):.4f}]")
    print(f"   Mass conservation:")
    print(f"     Mean violation: {mean_mass_violation*100:.4f}%")
    print(f"     Max violation: {max_mass_violation_overall*100:.4f}%")
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
    print(f"  MVAR: R²={train_r2:.4f}, ρ={rho_after:.4f}")
    print(f"  Test: {n_test} runs (mixed distributions)")
    print(f"  Prediction Metrics:")
    print(f"    R² (reconstructed): {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"    R² (latent space): {mean_r2_latent:.4f} ± {std_r2_latent:.4f}")
    print(f"    R² (POD reconstruction): {mean_r2_pod:.4f} ± {std_r2_pod:.4f}")
    
    print(f"\nOutput:")
    print(f"  Data: {OUTPUT_DIR}")
    print(f"  Predictions: {PRED_DIR}")
    
    # Save comprehensive summary JSON
    summary_data = {
        "experiment": args.experiment_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "configuration": {
            "n_particles": BASE_CONFIG['sim']['N'],
            "domain": {
                "Lx": BASE_CONFIG['sim']['Lx'],
                "Ly": BASE_CONFIG['sim']['Ly']
            },
            "trajectory_length": BASE_CONFIG['sim']['T'],
            "dt": BASE_CONFIG['sim']['dt'],
            "rom_subsample": ROM_SUBSAMPLE,
            "dt_rom": BASE_CONFIG['sim']['dt'] * ROM_SUBSAMPLE
        },
        "training": {
            "n_runs": n_train,
            "n_frames_per_run": len(np.arange(0, BASE_CONFIG['sim']['T'], BASE_CONFIG['sim']['dt'])),
            "n_frames_rom_per_run": len(np.arange(0, BASE_CONFIG['sim']['T'], BASE_CONFIG['sim']['dt'])) // ROM_SUBSAMPLE
        },
        "pod": {
            "n_modes": int(R_POD),
            "energy_captured": float(energy_captured),
            "r2_reconstruction": {
                "mean": float(mean_r2_pod),
                "std": float(std_r2_pod),
                "min": float(np.min(all_r2_pod)),
                "max": float(np.max(all_r2_pod))
            }
        },
        "mvar": {
            "lag_order": int(w),
            "latent_dimension": int(d),
            "n_parameters": int(d * d * w),
            "ridge_alpha": float(RIDGE_ALPHA),
            "training_r2": float(train_r2),
            "spectral_radius_before": float(rho_before),
            "spectral_radius_after": float(rho_after),
            "eigenvalue_threshold": float(rom_config.get('eigenvalue_threshold') or 0.0)
        },
        "testing": {
            "n_runs": int(n_test),
            "r2_reconstructed": {
                "mean": float(mean_r2),
                "std": float(std_r2),
                "min": float(np.min(all_r2)),
                "max": float(np.max(all_r2))
            },
            "r2_latent": {
                "mean": float(mean_r2_latent),
                "std": float(std_r2_latent),
                "min": float(np.min(all_r2_latent)),
                "max": float(np.max(all_r2_latent))
            },
            "r2_pod_reconstruction": {
                "mean": float(mean_r2_pod),
                "std": float(std_r2_pod),
                "min": float(np.min(all_r2_pod)),
                "max": float(np.max(all_r2_pod))
            },
            "mass_conservation": {
                "mean_violation_percent": float(mean_mass_violation * 100),
                "max_violation_percent": float(max_mass_violation_overall * 100)
            }
        },
        "timing": {
            "training_generation_min": float(train_time / 60),
            "pod_mvar_min": float(pod_mvar_time / 60),
            "test_generation_min": float(test_time / 60),
            "predictions_sec": float(pred_time),
            "total_min": float(total_time / 60)
        }
    }
    
    summary_path = OUTPUT_DIR / "pipeline_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"  Summary: {summary_path}")
    print("\n✅ Ready for visualization!")

if __name__ == "__main__":
    main()
