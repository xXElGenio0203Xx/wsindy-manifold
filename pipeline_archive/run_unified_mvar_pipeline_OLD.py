#!/usr/bin/env python3
"""
Unified ROM-MVAR Pipeline
==========================

Comprehensive pipeline supporting ALL experiment types:
- Multiple IC distributions (Gaussian, uniform, ring, two-cluster)
- Custom Gaussian experiments (variance/center variations)
- Flexible ROM configuration (fixed modes or energy threshold)
- Optional stability enforcement (eigenvalue scaling)
- Time-resolved evaluation
- Strong regularization options

This unified pipeline replaces:
- run_stable_mvar_pipeline.py
- run_robust_mvar_pipeline.py
- run_gaussians_pipeline.py
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

# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Flexible loader supporting multiple config formats:
    - Mixed distributions (train_ic/test_ic with multiple types)
    - Custom Gaussian experiments (simple gaussian configs)
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract simulation parameters
    sim_config = config.get('sim', {})
    base_config = {
        "sim": sim_config,
        "model": config.get('model', {}),
        "params": config.get('params', {}),
        "noise": config.get('noise', {}),
        "forces": config.get('forces', {}),
        "alignment": config.get('alignment', {}),
    }
    
    # Extract output/density parameters
    outputs = config.get('outputs', {})
    density_nx = outputs.get('density_resolution', 64)
    density_ny = outputs.get('density_resolution', 64)
    density_bandwidth = outputs.get('density_bandwidth', 2.0)
    
    # Extract ROM parameters
    rom_config = config.get('rom', {})
    
    # Extract IC configurations
    train_ic_config = config.get('train_ic', {})
    test_ic_config = config.get('test_ic', {})
    
    # Extract test simulation config (if separate from test_ic)
    test_sim_config = config.get('test_sim', {})
    
    # Extract evaluation config (optional, for time-resolved analysis)
    eval_config = config.get('evaluation', config.get('eval', {}))  # Support both names
    
    return (base_config, density_nx, density_ny, density_bandwidth,
            train_ic_config, test_ic_config, test_sim_config, rom_config, eval_config)

# =============================================================================
# TRAINING CONFIGURATION GENERATION
# =============================================================================

def generate_training_configs(train_ic_config, base_config):
    """
    Generate list of training run configurations.
    
    Supports:
    - Mixed distributions (gaussian, uniform, ring, two_clusters)
    - Custom Gaussian experiments (simple format)
    """
    configs = []
    run_id = 0
    
    # Check if this is a simple Gaussian experiment (custom format)
    if 'center' in train_ic_config and 'variances' in train_ic_config:
        # Custom Gaussian format (from gaussians pipeline)
        center = train_ic_config['center']
        variances = train_ic_config['variances']
        n_samples = train_ic_config.get('n_samples_per_variance', 1)
        
        for var in variances:
            for sample in range(n_samples):
                configs.append({
                    'run_id': run_id,
                    'distribution': 'gaussian_cluster',
                    'ic_params': {
                        'center': (float(center[0]), float(center[1])),
                        'sigma': float(np.sqrt(var))
                    },
                    'label': f'gauss_center{center[0]:.1f},{center[1]:.1f}_var{var:.1f}_s{sample}'
                })
                run_id += 1
        return configs
    
    # Otherwise, use mixed distributions format
    ic_type = train_ic_config.get('type', 'mixed_comprehensive')
    
    # Gaussian configurations
    if train_ic_config.get('gaussian', {}).get('enabled', False):
        gauss_cfg = train_ic_config['gaussian']
        positions_x = gauss_cfg.get('positions_x', [])
        positions_y = gauss_cfg.get('positions_y', [])
        variances = gauss_cfg.get('variances', [])
        n_samples = gauss_cfg.get('n_samples_per_config', 1)
        
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
        n_uniform = train_ic_config['uniform'].get('n_runs', 0)
        if n_uniform == 0:  # fallback to n_samples if n_runs not specified
            n_uniform = train_ic_config['uniform'].get('n_samples', 0)
        
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
        radii = ring_cfg.get('radii', [])
        widths = ring_cfg.get('widths', [])
        n_samples = ring_cfg.get('n_samples_per_config', 1)
        
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
        separations = cluster_cfg.get('separations', [])
        sigmas = cluster_cfg.get('sigmas', [])
        n_samples = cluster_cfg.get('n_samples_per_config', 1)
        
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
                        'label': f'two_cluster_sep{sep:.1f}_sig{sigma:.1f}_s{sample}'
                    })
                    run_id += 1
    
    return configs

# =============================================================================
# TEST CONFIGURATION GENERATION
# =============================================================================

def generate_test_configs(test_ic_config, base_config):
    """
    Generate list of test run configurations.
    
    Supports:
    - Mixed distributions with interpolation/extrapolation
    - Custom Gaussian experiments (different centers)
    """
    configs = []
    run_id = 0
    
    # Check if this is a custom Gaussian experiment
    if 'centers' in test_ic_config and 'variance' in test_ic_config:
        # Custom Gaussian format (from gaussians pipeline)
        centers = test_ic_config['centers']
        variance = test_ic_config['variance']
        n_samples = test_ic_config.get('n_samples_per_center', 1)
        
        for center in centers:
            for sample in range(n_samples):
                configs.append({
                    'run_id': run_id,
                    'distribution': 'gaussian_cluster',
                    'ic_params': {
                        'center': (float(center[0]), float(center[1])),
                        'sigma': float(np.sqrt(variance))
                    },
                    'label': f'test_gauss_center{center[0]:.1f},{center[1]:.1f}_s{sample}'
                })
                run_id += 1
        return configs
    
    # Otherwise, use mixed distributions format
    
    # Gaussian test configurations
    if test_ic_config.get('gaussian', {}).get('enabled', False):
        gauss_cfg = test_ic_config['gaussian']
        
        # Interpolation tests
        test_positions_x = gauss_cfg.get('test_positions_x', [])
        test_positions_y = gauss_cfg.get('test_positions_y', [])
        test_variances = gauss_cfg.get('test_variances', [])
        
        for px in test_positions_x:
            for py in test_positions_y:
                for var in test_variances:
                    configs.append({
                        'run_id': run_id,
                        'distribution': 'gaussian_cluster',
                        'ic_params': {
                            'center': (float(px), float(py)),
                            'sigma': float(np.sqrt(var))
                        },
                        'label': f'test_gauss_interp_x{px:.1f}_y{py:.1f}_var{var:.1f}'
                    })
                    run_id += 1
        
        # Extrapolation tests
        extrap_positions = gauss_cfg.get('extrapolation_positions', [])
        extrap_variance = gauss_cfg.get('extrapolation_variance', [0.5])[0]
        
        for pos in extrap_positions:
            configs.append({
                'run_id': run_id,
                'distribution': 'gaussian_cluster',
                'ic_params': {
                    'center': (float(pos[0]), float(pos[1])),
                    'sigma': float(np.sqrt(extrap_variance))
                },
                'label': f'test_gauss_extrap_x{pos[0]:.1f}_y{pos[1]:.1f}'
            })
            run_id += 1
    
    # Uniform test configurations
    if test_ic_config.get('uniform', {}).get('enabled', False):
        n_uniform = test_ic_config['uniform'].get('n_runs', 0)
        if n_uniform == 0:
            n_uniform = test_ic_config['uniform'].get('n_samples', 0)
        
        for i in range(n_uniform):
            configs.append({
                'run_id': run_id,
                'distribution': 'uniform',
                'ic_params': {},
                'label': f'test_uniform_s{i}'
            })
            run_id += 1
    
    # Ring test configurations
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
                        'label': f'test_ring_r{radius:.1f}_w{width:.1f}_s{sample}'
                    })
                    run_id += 1
    
    # Two-cluster test configurations
    if test_ic_config.get('two_clusters', {}).get('enabled', False):
        cluster_cfg = test_ic_config['two_clusters']
        
        # Interpolation tests
        test_separations = cluster_cfg.get('test_separations', [])
        test_sigmas = cluster_cfg.get('test_sigmas', [])
        
        for sep in test_separations:
            for sigma in test_sigmas:
                configs.append({
                    'run_id': run_id,
                    'distribution': 'two_clusters',
                    'ic_params': {
                        'separation': float(sep),
                        'sigma': float(sigma)
                    },
                    'label': f'test_two_cluster_interp_sep{sep:.1f}_sig{sigma:.1f}'
                })
                run_id += 1
        
        # Extrapolation tests
        extrap_separations = cluster_cfg.get('extrapolation_separations', [])
        extrap_sigma = cluster_cfg.get('extrapolation_sigma', [1.0])[0]
        
        for sep in extrap_separations:
            configs.append({
                'run_id': run_id,
                'distribution': 'two_clusters',
                'ic_params': {
                    'separation': float(sep),
                    'sigma': float(extrap_sigma)
                },
                'label': f'test_two_cluster_extrap_sep{sep:.1f}'
            })
            run_id += 1
    
    return configs

# =============================================================================
# SIMULATION WORKER
# =============================================================================

def simulate_single_run(args_tuple):
    """Worker function for parallel simulation execution."""
    (cfg, BASE_CONFIG, OUTPUT_DIR, DENSITY_NX, DENSITY_NY, DENSITY_BANDWIDTH, is_test) = args_tuple
    
    run_id = cfg['run_id']
    distribution = cfg['distribution']
    ic_params = cfg['ic_params']
    label = cfg['label']
    
    # Determine output directory
    if is_test:
        run_name = f"test_{run_id:03d}"
        run_dir = OUTPUT_DIR / "test" / run_name
    else:
        run_name = f"train_{run_id:03d}"
        run_dir = OUTPUT_DIR / "train" / run_name
    
    run_dir.mkdir(exist_ok=True, parents=True)
    
    # Set up configuration (keep nested structure for simulate_backend)
    config = BASE_CONFIG.copy()
    config["seed"] = run_id + 1000 * (1 if is_test else 0)
    config["initial_distribution"] = distribution
    config["ic_params"] = ic_params
    
    # Run simulation
    rng = np.random.default_rng(config["seed"])
    result = simulate_backend(config, rng)
    
    # Extract trajectories
    times = result["times"]
    traj = result["traj"]
    vel = result["vel"]
    
    # Save trajectories (REQUIRED for visualization pipeline)
    np.savez_compressed(
        run_dir / "trajectory.npz",
        traj=traj,
        vel=vel,
        times=times
    )
    
    # Compute density movies
    from rectsim.legacy_functions import kde_density_movie as kde_func
    rho, meta = kde_func(
        traj,
        Lx=config["sim"]["Lx"],
        Ly=config["sim"]["Ly"],
        nx=DENSITY_NX,
        ny=DENSITY_NY,
        bandwidth=DENSITY_BANDWIDTH,
        bc=config["sim"].get("bc", "periodic")
    )
    
    # Create spatial grids
    xgrid = np.linspace(0, config["sim"]["Lx"], DENSITY_NX, endpoint=False) + config["sim"]["Lx"]/(2*DENSITY_NX)
    ygrid = np.linspace(0, config["sim"]["Ly"], DENSITY_NY, endpoint=False) + config["sim"]["Ly"]/(2*DENSITY_NY)
    
    # Save density data (with xgrid, ygrid for compatibility)
    # Use different filename for test runs (visualization expects this)
    density_filename = "density_true.npz" if is_test else "density.npz"
    np.savez_compressed(
        run_dir / density_filename,
        rho=rho,
        xgrid=xgrid,
        ygrid=ygrid,
        times=times
    )
    
    # Create metadata
    metadata = {
        "run_id": run_id,
        "run_name": run_name,
        "label": label,
        "distribution": distribution,
        "ic_params": ic_params,
        "seed": config["seed"],
        "T": len(times)
    }
    
    return metadata

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Unified ROM-MVAR pipeline')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name')
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("="*80)
    print("UNIFIED ROM-MVAR PIPELINE")
    print("="*80)
    print(f"\nExperiment: {args.experiment_name}")
    print(f"Config: {args.config}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load configuration
    with open(args.config, 'r') as f:
        full_config = yaml.safe_load(f)
    
    (BASE_CONFIG, DENSITY_NX, DENSITY_NY, DENSITY_BANDWIDTH,
     train_ic_config, test_ic_config, test_sim_config, rom_config, eval_config) = load_config(args.config)
    
    OUTPUT_DIR = Path(f"oscar_output/{args.experiment_name}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save config for reference
    import shutil
    shutil.copy(args.config, OUTPUT_DIR / "config_used.yaml")
    
    print(f"\nConfiguration:")
    print(f"   N: {BASE_CONFIG['sim']['N']}")
    print(f"   T: {BASE_CONFIG['sim']['T']}s")
    print(f"   dt: {BASE_CONFIG['sim']['dt']}s")
    print(f"   Domain: {BASE_CONFIG['sim']['Lx']}×{BASE_CONFIG['sim']['Ly']}")
    print(f"   Density: {DENSITY_NX}×{DENSITY_NY}")
    print(f"   ROM lag: {rom_config.get('mvar_lag', 'auto')}")
    print(f"   Ridge α: {rom_config.get('ridge_alpha', 1e-6)}")
    
    # Generate training configurations
    print(f"\n{'='*80}")
    print("STEP 1: Generating Training Data")
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
    
    # =========================================================================
    # STEP 2: Global POD and MVAR Training
    # =========================================================================
    
    print(f"\n{'='*80}")
    print("STEP 2: Global POD and MVAR Training")
    print("="*80)
    
    # Get ROM subsampling parameter (check both names for compatibility)
    ROM_SUBSAMPLE = rom_config.get('subsample', rom_config.get('rom_subsample', 1))
    
    print(f"\nLoading training density data (subsample={ROM_SUBSAMPLE})...")
    
    # Load all training density data
    X_list = []
    for i in range(n_train):
        run_dir = TRAIN_DIR / f"train_{i:03d}"
        data = np.load(run_dir / "density.npz")
        density = data['rho']  # Changed from 'density' to 'rho' (stable format)
        
        # Subsample in time if requested
        if ROM_SUBSAMPLE > 1:
            density = density[::ROM_SUBSAMPLE]
        
        # Flatten each timestep
        T_sub = density.shape[0]
        X_run = density.reshape(T_sub, -1)
        X_list.append(X_run)
    
    # Stack all data
    X_all = np.vstack(X_list)
    M = n_train
    T_rom = X_list[0].shape[0]
    
    print(f"✓ Loaded data shape: {X_all.shape}")
    print(f"   {M} runs × {T_rom} timesteps × {X_all.shape[1]} spatial dims")
    
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
    RIDGE_ALPHA = rom_config.get('ridge_alpha', 1e-6)
    
    print(f"\nTraining global MVAR (p={P_LAG}, α={RIDGE_ALPHA})...")
    from sklearn.linear_model import Ridge
    
    # Reshape latent data for MVAR
    X_latent_runs = X_latent.reshape(M, T_rom, R_POD)
    
    # Build training matrices
    X_train_list = []
    Y_train_list = []
    
    for m in range(M):
        X_m = X_latent_runs[m]  # Shape: (T_rom, R_POD)
        
        for t in range(P_LAG, T_rom):
            # Feature vector: [x(t-p), ..., x(t-1)]
            x_hist = X_m[t-P_LAG:t].flatten()  # Shape: (P_LAG * R_POD,)
            y_target = X_m[t]  # Shape: (R_POD,)
            
            X_train_list.append(x_hist)
            Y_train_list.append(y_target)
    
    X_train = np.array(X_train_list)
    Y_train = np.array(Y_train_list)
    
    print(f"✓ MVAR training data: X{X_train.shape}, Y{Y_train.shape}")
    
    # Train Ridge regression
    mvar_model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    mvar_model.fit(X_train, Y_train)
    
    # Training R²
    Y_train_pred = mvar_model.predict(X_train)
    ss_res = np.sum((Y_train - Y_train_pred)**2)
    ss_tot = np.sum((Y_train - Y_train.mean(axis=0))**2)
    r2_train = 1 - ss_res / ss_tot
    
    print(f"✓ Training R² = {r2_train:.4f}")
    
    # Optional: Eigenvalue stability check and scaling
    eigenvalue_threshold = rom_config.get('eigenvalue_threshold', None)
    
    if eigenvalue_threshold is not None:
        # Reshape MVAR coefficients to transition matrix form
        A_coef = mvar_model.coef_  # Shape: (R_POD, P_LAG * R_POD)
        
        # For MVAR(p), create companion matrix form
        # This is approximate - full companion form would be larger
        # Here we just check the largest lag coefficient matrix
        A_p = A_coef[:, -R_POD:]  # Last lag coefficients
        
        eigenvalues = np.linalg.eigvals(A_p)
        max_eig = np.max(np.abs(eigenvalues))
        
        print(f"\nStability check:")
        print(f"   Max |eigenvalue| = {max_eig:.4f}")
        
        if max_eig > eigenvalue_threshold:
            scale_factor = eigenvalue_threshold / max_eig
            print(f"   ⚠️  Scaling coefficients by {scale_factor:.4f} to enforce stability")
            mvar_model.coef_ *= scale_factor
            if mvar_model.intercept_ is not None:
                mvar_model.intercept_ *= scale_factor
        else:
            print(f"   ✓ Model is stable (threshold={eigenvalue_threshold})")
    
    # Save ROM artifacts (MATCH STABLE PIPELINE FORMAT)
    MVAR_DIR = OUTPUT_DIR / "mvar"
    MVAR_DIR.mkdir(exist_ok=True)
    
    # Save mean separately (stable pipeline format)
    np.save(MVAR_DIR / "X_train_mean.npy", X_mean)
    
    # Save POD basis with stable pipeline keys
    np.savez_compressed(
        MVAR_DIR / "pod_basis.npz",
        U=U_r,  # Changed from U_r to U
        singular_values=S[:R_POD],
        all_singular_values=S,
        total_energy=total_energy,
        explained_energy=cumulative_energy[R_POD-1] * total_energy,
        energy_ratio=energy_captured,
        cumulative_ratio=cumulative_energy
    )
    
    # Reshape MVAR coefficients to match stable pipeline format
    # Stable pipeline stores A_matrices as (p, d, d) and uses different structure
    # For compatibility, we'll store both formats
    A_matrices = mvar_model.coef_.reshape(R_POD, P_LAG, R_POD).transpose(1, 0, 2)  # (p, d, d)
    
    # Compute RMSE for compatibility
    train_rmse = np.sqrt(np.mean((Y_train - Y_train_pred)**2))
    
    # For eigenvalue info (if scaled)
    if eigenvalue_threshold is not None:
        A_p = mvar_model.coef_[:, -R_POD:]
        eigenvalues = np.linalg.eigvals(A_p)
        rho_after = np.max(np.abs(eigenvalues))
        rho_before = rho_after / scale_factor if 'scale_factor' in locals() else rho_after
    else:
        rho_before = 0.0
        rho_after = 0.0
    
    # Save MVAR model with stable pipeline keys
    np.savez_compressed(
        MVAR_DIR / "mvar_model.npz",
        A_matrices=A_matrices,
        A_companion=mvar_model.coef_,  # Store flat version as companion
        p=P_LAG,  # Changed from p_lag to p
        r=R_POD,  # Changed from R_POD to r
        alpha=RIDGE_ALPHA,  # Changed from ridge_alpha to alpha
        train_r2=r2_train,
        train_rmse=train_rmse,
        rho_before=float(rho_before),
        rho_after=float(rho_after)
    )
    
    print(f"\n✓ ROM artifacts saved to {MVAR_DIR}/")
    
    # =========================================================================
    # STEP 3: Test Data Generation
    # =========================================================================
    
    print(f"\n{'='*80}")
    print("STEP 3: Generating Test Data")
    print("="*80)
    
    test_configs = generate_test_configs(test_ic_config, BASE_CONFIG)
    n_test = len(test_configs)
    
    if n_test == 0:
        print("\n⚠️  No test configurations specified. Skipping test evaluation.")
    else:
        print(f"\nTest configurations:")
        print(f"   Total runs: {n_test}")
        
        # Count by distribution
        test_dist_counts = {}
        for cfg in test_configs:
            dist = cfg['distribution']
            test_dist_counts[dist] = test_dist_counts.get(dist, 0) + 1
        for dist, count in test_dist_counts.items():
            print(f"   {dist}: {count} runs")
        
        # Get test duration (may differ from training)
        # Check test_sim.T first, then test_ic.test_T, then default to training T
        test_T = test_sim_config.get('T', test_ic_config.get('test_T', BASE_CONFIG['sim']['T']))
        
        # Temporarily override T for test runs
        BASE_CONFIG_TEST = BASE_CONFIG.copy()
        BASE_CONFIG_TEST['sim'] = BASE_CONFIG['sim'].copy()
        BASE_CONFIG_TEST['sim']['T'] = test_T
        
        print(f"\nTest duration: {test_T}s (train was {BASE_CONFIG['sim']['T']}s)")
        
        TEST_DIR = OUTPUT_DIR / "test"
        TEST_DIR.mkdir(parents=True, exist_ok=True)
        
        test_args = [(cfg, BASE_CONFIG_TEST, OUTPUT_DIR, DENSITY_NX, DENSITY_NY, DENSITY_BANDWIDTH, True)
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
        
        test_df = pd.DataFrame(test_metadata)
        test_df.to_csv(TEST_DIR / "index_mapping.csv", index=False)
        
        print(f"\n✓ Generated {n_test} test runs")
        print(f"   Time: {test_time/60:.1f}m")
        
        # =====================================================================
        # STEP 4: Test Evaluation
        # =====================================================================
        
        print(f"\n{'='*80}")
        print("STEP 4: ROM-MVAR Test Evaluation")
        print("="*80)
        
        # Determine evaluation mode
        save_time_resolved = eval_config.get('save_time_resolved', False)
        forecast_start = eval_config.get('forecast_start', BASE_CONFIG['sim']['T'])
        forecast_end = eval_config.get('forecast_end', test_T)
        
        if save_time_resolved:
            print(f"\nTime-resolved evaluation enabled:")
            print(f"   Forecast period: t={forecast_start}s to t={forecast_end}s")
        
        print(f"\nEvaluating {n_test} test runs...")
        
        # Create spatial grids for density output (needed for visualization)
        xgrid = np.linspace(0, BASE_CONFIG_TEST['sim']['Lx'], DENSITY_NX, endpoint=False) + BASE_CONFIG_TEST['sim']['Lx']/(2*DENSITY_NX)
        ygrid = np.linspace(0, BASE_CONFIG_TEST['sim']['Ly'], DENSITY_NY, endpoint=False) + BASE_CONFIG_TEST['sim']['Ly']/(2*DENSITY_NY)
        
        # Evaluation loop
        test_results = []
        
        for test_idx in tqdm(range(n_test), desc="Evaluating"):
            test_run_dir = TEST_DIR / f"test_{test_idx:03d}"
            
            # Load test density (now from density_true.npz)
            test_data = np.load(test_run_dir / "density_true.npz")
            test_density = test_data['rho']  # Changed from 'density' to 'rho'
            test_times = test_data['times']
            
            # Subsample if needed
            if ROM_SUBSAMPLE > 1:
                test_density = test_density[::ROM_SUBSAMPLE]
                test_times = test_times[::ROM_SUBSAMPLE]
            
            T_test = test_density.shape[0]
            test_density_flat = test_density.reshape(T_test, -1)
            
            # Project to latent space
            test_centered = test_density_flat - X_mean
            test_latent = test_centered @ U_r
            
            # Determine initial condition window
            T_train = int(BASE_CONFIG['sim']['T'] / BASE_CONFIG['sim']['dt'] / ROM_SUBSAMPLE)
            
            # Use last P_LAG timesteps from training period as IC
            if T_train < P_LAG:
                print(f"⚠️  Warning: Training period ({T_train} steps) < lag ({P_LAG}). Using all available.")
                ic_window = test_latent[:T_train]
            else:
                ic_window = test_latent[T_train-P_LAG:T_train]
            
            # Autoregressive prediction
            pred_latent = []
            current_history = ic_window.copy()
            
            for t in range(T_train, T_test):
                # Prepare feature vector
                x_hist = current_history[-P_LAG:].flatten()
                
                # Predict next step
                y_next = mvar_model.predict(x_hist.reshape(1, -1))[0]
                pred_latent.append(y_next)
                
                # Update history
                current_history = np.vstack([current_history[1:], y_next])
            
            pred_latent = np.array(pred_latent)
            
            # Reconstruct to physical space
            pred_physical = (pred_latent @ U_r.T) + X_mean
            pred_physical = pred_physical.reshape(-1, DENSITY_NX, DENSITY_NY)
            
            # Ground truth (forecasted region)
            true_physical = test_density[T_train:]
            
            # Compute R² metrics
            # 1. Reconstructed (physical space)
            ss_res_phys = np.sum((true_physical.flatten() - pred_physical.flatten())**2)
            ss_tot_phys = np.sum((true_physical.flatten() - true_physical.flatten().mean())**2)
            r2_reconstructed = 1 - ss_res_phys / ss_tot_phys
            
            # 2. Latent (ROM space)
            true_latent = test_latent[T_train:]
            ss_res_lat = np.sum((true_latent.flatten() - pred_latent.flatten())**2)
            ss_tot_lat = np.sum((true_latent.flatten() - true_latent.flatten().mean())**2)
            r2_latent = 1 - ss_res_lat / ss_tot_lat
            
            # 3. POD reconstruction quality (using true latent)
            true_reconstructed = (true_latent @ U_r.T) + X_mean
            true_reconstructed = true_reconstructed.reshape(-1, DENSITY_NX, DENSITY_NY)
            ss_res_pod = np.sum((true_physical.flatten() - true_reconstructed.flatten())**2)
            r2_pod = 1 - ss_res_pod / ss_tot_phys
            
            # Compute RMSE metrics (for compatibility)
            rmse_recon = np.sqrt(np.mean((true_physical.flatten() - pred_physical.flatten())**2))
            rmse_latent = np.sqrt(np.mean((true_latent.flatten() - pred_latent.flatten())**2))
            rmse_pod = np.sqrt(np.mean((true_physical.flatten() - true_reconstructed.flatten())**2))
            
            # Compute relative errors
            rel_error_recon = rmse_recon / (np.mean(np.abs(true_physical.flatten())) + 1e-10)
            rel_error_pod = rmse_pod / (np.mean(np.abs(true_physical.flatten())) + 1e-10)
            
            # Compute mass conservation violation
            true_mass = np.sum(true_physical, axis=(1, 2))
            pred_mass = np.sum(pred_physical, axis=(1, 2))
            mass_violations = np.abs(pred_mass - true_mass) / (true_mass + 1e-10)
            max_mass_violation = np.max(mass_violations)
            
            # Store results
            result = {
                'test_id': test_idx,
                'r2_reconstructed': r2_reconstructed,
                'r2_latent': r2_latent,
                'r2_pod': r2_pod,
                'rmse_recon': rmse_recon,
                'rmse_latent': rmse_latent,
                'rmse_pod': rmse_pod,
                'rel_error_recon': rel_error_recon,
                'rel_error_pod': rel_error_pod,
                'max_mass_violation': max_mass_violation,
                'T_forecast': len(pred_latent)
            }
            
            # Save metrics summary JSON (REQUIRED for visualization pipeline)
            metrics_dict = {
                'r2_recon': float(r2_reconstructed),
                'r2_latent': float(r2_latent),
                'r2_pod': float(r2_pod),
                'rmse_recon': float(rmse_recon),
                'rmse_latent': float(rmse_latent),
                'rmse_pod': float(rmse_pod),
                'rel_error_recon': float(rel_error_recon),
                'rel_error_pod': float(rel_error_pod),
                'max_mass_violation': float(max_mass_violation)
            }
            with open(test_run_dir / "metrics_summary.json", 'w') as f:
                json.dump(metrics_dict, f, indent=2)
            
            # Define forecast times (needed for predictions and optional for time-resolved)
            forecast_times = test_times[T_train:]
            
            # Time-resolved analysis (if requested)
            if save_time_resolved and len(pred_latent) > 0:
                T_forecast = len(pred_latent)
                r2_vs_time = []
                
                for t_idx in range(T_forecast):
                    # R² up to time t
                    true_t = true_physical[:t_idx+1]
                    pred_t = pred_physical[:t_idx+1]
                    
                    ss_res_t = np.sum((true_t - pred_t)**2)
                    ss_tot_t = np.sum((true_t - true_t.mean())**2)
                    r2_t_reconstructed = 1 - ss_res_t / ss_tot_t
                    
                    # Latent R²
                    true_lat_t = true_latent[:t_idx+1]
                    pred_lat_t = pred_latent[:t_idx+1]
                    ss_res_lat_t = np.sum((true_lat_t - pred_lat_t)**2)
                    ss_tot_lat_t = np.sum((true_lat_t - true_lat_t.mean())**2)
                    r2_t_latent = 1 - ss_res_lat_t / ss_tot_lat_t
                    
                    # POD R²
                    true_recon_t = true_reconstructed[:t_idx+1]
                    ss_res_pod_t = np.sum((true_t - true_recon_t)**2)
                    r2_t_pod = 1 - ss_res_pod_t / ss_tot_t
                    
                    r2_vs_time.append({
                        'time': forecast_times[t_idx],
                        'r2_reconstructed': r2_t_reconstructed,
                        'r2_latent': r2_t_latent,
                        'r2_pod': r2_t_pod
                    })
                
                # Save time-resolved data
                r2_df = pd.DataFrame(r2_vs_time)
                r2_df.to_csv(test_run_dir / "r2_vs_time.csv", index=False)
            
            # Save predicted density (REQUIRED for visualization pipeline)
            # Use same format as stable pipeline
            np.savez_compressed(
                test_run_dir / "density_pred.npz",
                rho=pred_physical,
                xgrid=xgrid,
                ygrid=ygrid,
                times=forecast_times
            )
            
            test_results.append(result)
        
        # Save test results
        test_results_df = pd.DataFrame(test_results)
        test_results_df.to_csv(TEST_DIR / "test_results.csv", index=False)
        
        # Summary statistics
        mean_r2_recon = test_results_df['r2_reconstructed'].mean()
        mean_r2_latent = test_results_df['r2_latent'].mean()
        mean_r2_pod = test_results_df['r2_pod'].mean()
        
        print(f"\n{'='*80}")
        print("Test Results Summary")
        print("="*80)
        print(f"Mean R² (reconstructed): {mean_r2_recon:.4f}")
        print(f"Mean R² (latent):        {mean_r2_latent:.4f}")
        print(f"Mean R² (POD):           {mean_r2_pod:.4f}")
        print(f"\nDetailed results: {TEST_DIR}/test_results.csv")
        
        if save_time_resolved:
            print(f"Time-resolved R²: {TEST_DIR}/test_*/r2_vs_time.csv")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\nTotal time: {total_time/60:.1f}m")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nKey files:")
    print(f"   {OUTPUT_DIR}/config_used.yaml")
    print(f"   {OUTPUT_DIR}/train/metadata.json")
    print(f"   {OUTPUT_DIR}/mvar/pod_basis.npz")
    print(f"   {OUTPUT_DIR}/mvar/mvar_model.npz")
    if n_test > 0:
        print(f"   {OUTPUT_DIR}/test/test_results.csv")
    
    # Save final summary
    summary = {
        'experiment_name': args.experiment_name,
        'config': args.config,
        'n_train': n_train,
        'n_test': n_test if n_test > 0 else 0,
        'r_pod': int(R_POD),
        'p_lag': int(P_LAG),
        'ridge_alpha': float(RIDGE_ALPHA),
        'r2_train': float(r2_train),
        'total_time_minutes': total_time / 60
    }
    
    if n_test > 0:
        summary.update({
            'mean_r2_reconstructed': float(mean_r2_recon),
            'mean_r2_latent': float(mean_r2_latent),
            'mean_r2_pod': float(mean_r2_pod)
        })
    
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Pipeline completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()
