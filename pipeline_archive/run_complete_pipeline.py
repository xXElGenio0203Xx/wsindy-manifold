#!/usr/bin/env python3
"""
Complete MVAR-ROM Pipeline with IC Type Stratification
========================================================

This orchestrates the full simulation → density (KDE) → global POD → MVAR → evaluation
pipeline using existing functions and I/O conventions.

Workflow:
1. Training ensemble: N_train simulations with fixed physics, varied ICs
2. Global POD + global MVAR training on concatenated density data
3. Test ensemble: M_test simulations with stratified IC types
4. Evaluation: Metrics by IC type + best run visualizations per IC type
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm
import json
import pandas as pd
from typing import Dict, List, Tuple
import time

# Import rectsim modules
from rectsim.vicsek_discrete import simulate_backend
from rectsim.legacy_functions import (
    kde_density_movie,
    trajectory_video,
    side_by_side_video,
    compute_frame_metrics,
    compute_summary_metrics,
    plot_errors_timeseries,
    compute_order_params
)

print("="*80)
print("COMPLETE MVAR-ROM PIPELINE WITH IC TYPE STRATIFICATION")
print("="*80)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Fixed Vicsek-Morse discrete model configuration
# Same parameters for all runs; only ICs and seeds vary
BASE_CONFIG = {
    "sim": {
        "N": 40,
        "Lx": 15.0,
        "Ly": 15.0,
        "bc": "periodic",
        "T": 2.0,
        "dt": 0.1,
        "save_every": 1,
        "neighbor_rebuild": 5,
    },
    "model": {"speed": 1.0},
    "params": {"R": 2.0},
    "noise": {"kind": "gaussian", "eta": 0.3},
    "forces": {"enabled": False},  # Pure Vicsek for now
}

# IC types to use (will cycle through these)
IC_TYPES = ["uniform", "gaussian_cluster", "ring", "two_clusters"]

# Training/test split
N_TRAIN = 100
M_TEST = 20

# Density estimation parameters (smooth continuous heatmaps)
DENSITY_NX = 64
DENSITY_NY = 64
DENSITY_BANDWIDTH = 2.0

# POD/MVAR parameters
TARGET_ENERGY = 0.995  # 99.5% variance captured
P_LAG = 4
RIDGE_ALPHA = 1e-6

# Output directories
OUTPUT_DIR = Path("outputs/complete_pipeline")
TRAIN_DIR = OUTPUT_DIR / "train"
TEST_DIR = OUTPUT_DIR / "test"
MVAR_DIR = OUTPUT_DIR / "mvar"
BEST_RUNS_DIR = OUTPUT_DIR / "best_runs"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Clean and create directories
if OUTPUT_DIR.exists():
    print(f"\n🗑️  Cleaning previous outputs: {OUTPUT_DIR}")
    shutil.rmtree(OUTPUT_DIR)

for d in [TRAIN_DIR, TEST_DIR, MVAR_DIR, BEST_RUNS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"\n📁 Output directory: {OUTPUT_DIR}")
print(f"\n⚙️  Configuration:")
print(f"   Training runs: {N_TRAIN}")
print(f"   Test runs: {M_TEST}")
print(f"   IC types: {IC_TYPES}")
print(f"   Particles: {BASE_CONFIG['sim']['N']}")
print(f"   Duration: {BASE_CONFIG['sim']['T']}s (dt={BASE_CONFIG['sim']['dt']})")
print(f"   Density: {DENSITY_NX}×{DENSITY_NY}, bandwidth={DENSITY_BANDWIDTH}")
print(f"   POD target energy: {TARGET_ENERGY*100:.1f}%, MVAR lag: {P_LAG}")

# Track timing
pipeline_start_time = time.time()
timing_info = {}

# =============================================================================
# STEP 1: TRAINING ENSEMBLE (Fixed physics, varied ICs)
# =============================================================================

print("\n" + "="*80)
print("STEP 1: Generating Training Ensemble")
print("="*80)

step1_start = time.time()
train_metadata = []  # Track (run_id, ic_type, seed) for each training run
train_densities = []  # Will collect all density movies

print(f"\nSimulating {N_TRAIN} training runs (cycling through {len(IC_TYPES)} IC types)...")

for i in tqdm(range(N_TRAIN), desc="Training sims"):
    run_name = f"train_{i:03d}"
    run_dir = TRAIN_DIR / run_name
    run_dir.mkdir(exist_ok=True)
    
    # Cycle through IC types
    ic_type = IC_TYPES[i % len(IC_TYPES)]
    seed = 1000 + i
    
    # Create config with IC type
    config = BASE_CONFIG.copy()
    config["initial_distribution"] = ic_type
    config["seed"] = seed
    
    # Run simulation
    rng = np.random.default_rng(seed)
    result = simulate_backend(config, rng)
    
    # Save trajectories and velocities
    np.savez(
        run_dir / "trajectory.npz",
        traj=result["traj"],
        vel=result["vel"],
        times=result["times"]
    )
    
    # Compute KDE densities on fixed grid
    rho, meta = kde_density_movie(
        result["traj"],
        Lx=config["sim"]["Lx"],
        Ly=config["sim"]["Ly"],
        nx=DENSITY_NX,
        ny=DENSITY_NY,
        bandwidth=DENSITY_BANDWIDTH,
        bc=config["sim"]["bc"]
    )
    
    # Compute grid centers
    xgrid = np.linspace(0, config["sim"]["Lx"], DENSITY_NX, endpoint=False) + config["sim"]["Lx"]/(2*DENSITY_NX)
    ygrid = np.linspace(0, config["sim"]["Ly"], DENSITY_NY, endpoint=False) + config["sim"]["Ly"]/(2*DENSITY_NY)
    
    # Save density
    np.savez(
        run_dir / "density.npz",
        rho=rho,
        xgrid=xgrid,
        ygrid=ygrid,
        times=result["times"]
    )
    
    # Track metadata
    train_metadata.append({
        "run_id": i,
        "run_name": run_name,
        "ic_type": ic_type,
        "seed": seed,
        "T": len(result["times"])
    })
    
    # Collect densities for POD (flatten spatial dimensions)
    T, ny, nx = rho.shape
    rho_flat = rho.reshape(T, nx * ny)  # (T, n_space)
    train_densities.append(rho_flat)

# Save training metadata
with open(TRAIN_DIR / "metadata.json", "w") as f:
    json.dump(train_metadata, f, indent=2)

print(f"✓ Generated {N_TRAIN} training runs")
print(f"   IC type distribution:")
for ic_type in IC_TYPES:
    count = sum(1 for m in train_metadata if m["ic_type"] == ic_type)
    print(f"     {ic_type}: {count} runs")

timing_info["step1_training_ensemble"] = time.time() - step1_start

# =============================================================================
# STEP 2: GLOBAL POD + GLOBAL MVAR TRAINING
# =============================================================================

print("\n" + "="*80)
print("STEP 2: Global POD and MVAR Training")
print("="*80)

step2_start = time.time()

# Concatenate all training densities
print("\nBuilding global snapshot matrix...")
X_train = np.vstack(train_densities)  # (T_total, n_space)
print(f"✓ Snapshot matrix shape: {X_train.shape}")

# Build index mapping: global_index → (run_id, ic_type, local_time)
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

# Save index mapping
pd.DataFrame(index_map).to_csv(TRAIN_DIR / "index_mapping.csv", index=False)

# Compute global POD with automatic rank selection based on target energy
print(f"\nComputing global POD (target energy={TARGET_ENERGY*100:.1f}%)...")
U_svd, S, Vt = np.linalg.svd(X_train, full_matrices=False)

# Determine number of modes r to achieve target energy
total_energy = np.sum(S**2)
cumulative_energy = np.cumsum(S**2)
cumulative_ratio = cumulative_energy / total_energy

# Find minimum r such that cumulative energy >= target
R_POD = np.argmax(cumulative_ratio >= TARGET_ENERGY) + 1
actual_energy = cumulative_ratio[R_POD - 1]

print(f"✓ Determined R_POD = {R_POD} modes to achieve {actual_energy*100:.2f}% energy (target: {TARGET_ENERGY*100:.1f}%)")

# Extract POD basis
U = Vt[:R_POD, :].T  # (n_space, r)
singular_values = S[:R_POD]
explained_energy = np.sum(singular_values**2)
energy_ratio = explained_energy / total_energy

print(f"✓ POD basis shape: {U.shape}")
print(f"✓ Cumulative energy: {100*energy_ratio:.2f}%")

# Save POD artifacts
np.savez(
    MVAR_DIR / "pod_basis.npz",
    U=U,
    singular_values=singular_values,
    total_energy=total_energy,
    explained_energy=explained_energy,
    energy_ratio=energy_ratio
)

# Plot 1: POD singular values (log scale)
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(range(1, len(S)+1), S, 'o-', linewidth=2, markersize=2, color='steelblue', alpha=0.6)
ax.axvline(R_POD, color='r', linestyle='--', linewidth=2, label=f'Selected r={R_POD} ({actual_energy*100:.2f}% energy)')
ax.set_xlabel('POD Mode Index', fontsize=12)
ax.set_ylabel('Singular Value (log scale)', fontsize=12)
ax.set_title(f'POD Singular Values (target: {TARGET_ENERGY*100:.1f}% energy)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=10)
# Scale x-axis to show at least up to R_POD + 20% margin
x_max = max(R_POD * 1.2, min(500, len(S)))
ax.set_xlim(0, x_max)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "pod_singular_values.png", dpi=200)
plt.close()

# Plot 2: Cumulative energy (percentage) - clearer interpretation
fig, ax = plt.subplots(figsize=(10, 6))
cumulative_energy_pct = 100 * cumulative_ratio
ax.plot(range(1, len(S)+1), cumulative_energy_pct, 'o-', linewidth=2, markersize=2, color='forestgreen', alpha=0.7)
ax.axvline(R_POD, color='r', linestyle='--', linewidth=2, label=f'Selected r={R_POD}')
ax.axhline(90, color='orange', linestyle=':', alpha=0.7, linewidth=2, label='90% energy')
ax.axhline(95, color='purple', linestyle=':', alpha=0.7, linewidth=2, label='95% energy')
ax.axhline(TARGET_ENERGY*100, color='darkgreen', linestyle=':', alpha=0.7, linewidth=2, label=f'{TARGET_ENERGY*100:.1f}% energy (target)')
ax.set_xlabel('Number of POD Modes', fontsize=12)
ax.set_ylabel('Cumulative Energy Captured (%)', fontsize=12)
ax.set_title(f'POD Energy Spectrum ({N_TRAIN} training runs)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
# Scale x-axis to show at least up to R_POD + 20% margin
x_max = max(R_POD * 1.2, min(500, len(S)))
ax.set_xlim(0, x_max)
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "pod_energy.png", dpi=200)
plt.close()

# Project training data to POD basis
print("\nProjecting training data to latent space...")
Y_train = X_train @ U  # (T_total, r)
print(f"✓ Latent training data shape: {Y_train.shape}")

# Save latent trajectories grouped by run
latent_runs = {}
for meta in train_metadata:
    run_id = meta["run_id"]
    run_name = meta["run_name"]
    T = meta["T"]
    
    # Extract this run's latent trajectory
    mask = [idx["run_id"] == run_id for idx in index_map]
    y_run = Y_train[mask, :]  # (T, r)
    
    latent_runs[run_name] = {
        "y": y_run,
        "ic_type": meta["ic_type"],
        "seed": meta["seed"]
    }

np.savez(MVAR_DIR / "latent_trajectories.npz", **{k: v["y"] for k, v in latent_runs.items()})

# Train global MVAR on concatenated latent time series
print(f"\nTraining global MVAR (p={P_LAG}, ridge_alpha={RIDGE_ALPHA})...")

# Build MVAR training data (remove first p_lag steps from each run to avoid boundary issues)
Y_mvar_list = []
for meta in train_metadata:
    run_name = meta["run_name"]
    y_run = latent_runs[run_name]["y"]
    if len(y_run) > P_LAG:
        Y_mvar_list.append(y_run[P_LAG:])  # Skip initial p_lag steps

Y_mvar = np.vstack(Y_mvar_list)  # Concatenated latent data for MVAR
print(f"✓ MVAR training data shape: {Y_mvar.shape}")

# Fit MVAR: Y[t] = A1*Y[t-1] + A2*Y[t-2] + ... + Ap*Y[t-p] + noise
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

timing_info["step2_pod_mvar_training"] = time.time() - step2_start

# =============================================================================
# STEP 3: TEST ENSEMBLE (Stratified by IC type)
# =============================================================================

print("\n" + "="*80)
print("STEP 3: Generating Test Ensemble (Stratified by IC)")
print("="*80)

step3_start = time.time()

# Distribute test runs evenly across IC types
runs_per_ic = M_TEST // len(IC_TYPES)
extra_runs = M_TEST % len(IC_TYPES)

test_metadata = []
test_ic_distribution = {ic: runs_per_ic for ic in IC_TYPES}
# Distribute extra runs
for i in range(extra_runs):
    test_ic_distribution[IC_TYPES[i]] += 1

print(f"\nSimulating {M_TEST} test runs (stratified by IC type):")
for ic, count in test_ic_distribution.items():
    print(f"   {ic}: {count} runs")

test_run_idx = 0
for ic_type in IC_TYPES:
    n_runs = test_ic_distribution[ic_type]
    
    for i in range(n_runs):
        run_name = f"test_{test_run_idx:03d}"
        run_dir = TEST_DIR / run_name
        run_dir.mkdir(exist_ok=True)
        
        seed = 2000 + test_run_idx
        
        # Create config
        config = BASE_CONFIG.copy()
        config["initial_distribution"] = ic_type
        config["seed"] = seed
        
        # Run simulation
        rng = np.random.default_rng(seed)
        result = simulate_backend(config, rng)
        
        # Save trajectories
        np.savez(
            run_dir / "trajectory.npz",
            traj=result["traj"],
            vel=result["vel"],
            times=result["times"]
        )
        
        # Compute KDE densities (same grid as training)
        rho_true, meta = kde_density_movie(
            result["traj"],
            Lx=config["sim"]["Lx"],
            Ly=config["sim"]["Ly"],
            nx=DENSITY_NX,
            ny=DENSITY_NY,
            bandwidth=DENSITY_BANDWIDTH,
            bc=config["sim"]["bc"]
        )
        
        # Compute grid centers
        xgrid = np.linspace(0, config["sim"]["Lx"], DENSITY_NX, endpoint=False) + config["sim"]["Lx"]/(2*DENSITY_NX)
        ygrid = np.linspace(0, config["sim"]["Ly"], DENSITY_NY, endpoint=False) + config["sim"]["Ly"]/(2*DENSITY_NY)
        
        # Save true density
        np.savez(
            run_dir / "density_true.npz",
            rho=rho_true,
            xgrid=xgrid,
            ygrid=ygrid,
            times=result["times"]
        )
        
        # Compute order parameters from velocities
        import pandas as pd
        order_params_list = []
        for t in range(len(result["times"])):
            vel = result["vel"][t]  # (N, 2) velocities at time t
            params = compute_order_params(vel, include_nematic=True)
            params['t'] = result["times"][t]
            order_params_list.append(params)
        
        # Save order parameters
        df_order = pd.DataFrame(order_params_list)
        df_order = df_order[['t', 'phi', 'mean_speed', 'speed_std', 'nematic']]
        df_order.to_csv(run_dir / "order_params.csv", index=False)
        
        # Save run metadata
        with open(run_dir / "metadata.json", "w") as f:
            json.dump({
                "run_id": test_run_idx,
                "ic_type": ic_type,
                "seed": seed
            }, f, indent=2)
        
        # Track metadata
        test_metadata.append({
            "run_id": test_run_idx,
            "run_name": run_name,
            "ic_type": ic_type,
            "seed": seed,
            "T": len(result["times"])
        })
        
        test_run_idx += 1

# Save test metadata
with open(TEST_DIR / "metadata.json", "w") as f:
    json.dump(test_metadata, f, indent=2)

print(f"✓ Generated {M_TEST} test runs")

# =============================================================================
# STEP 4: ROM PREDICTION ON TEST DATA
# =============================================================================

print("\n" + "="*80)
print("STEP 4: MVAR-ROM Prediction on Test Data")
print("="*80)

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

print(f"\nMaking predictions on {M_TEST} test runs...")

test_predictions = {}

for meta in tqdm(test_metadata, desc="Predictions"):
    run_name = meta["run_name"]
    run_dir = TEST_DIR / run_name
    
    # Load true density
    data = np.load(run_dir / "density_true.npz")
    rho_true = data["rho"]
    times = data["times"]
    T, ny, nx = rho_true.shape
    n_space = nx * ny
    
    # Load trajectory for visualization
    traj_data = np.load(run_dir / "trajectory.npz")
    
    # Project true density to latent space
    rho_flat = rho_true.reshape(T, n_space)
    y_true = rho_flat @ U  # (T, r)
    
    # Initialize MVAR with first p_lag steps
    y_init = y_true[:P_LAG]
    
    # Forecast remaining steps
    T_forecast = T - P_LAG
    y_pred = mvar_forecast(y_init, A_matrices, T_forecast)
    
    # Combine init + forecast
    y_full = np.vstack([y_init, y_pred])  # (T, r)
    
    # Reconstruct density from latent
    rho_pred_flat = y_full @ U.T  # (T, n_space)
    rho_pred = rho_pred_flat.reshape(T, ny, nx)
    
    # Save predicted density
    np.savez(
        run_dir / "density_pred.npz",
        rho=rho_pred,
        xgrid=data["xgrid"],
        ygrid=data["ygrid"],
        times=times
    )
    
    # Store for evaluation
    test_predictions[run_name] = {
        "rho_true": rho_true,
        "rho_pred": rho_pred,
        "y_true": y_true,
        "y_pred": y_full,
        "times": times,
        "traj": traj_data["traj"],  # Add trajectory for visualizations
        "ic_type": meta["ic_type"]
    }

print(f"✓ Completed predictions for {M_TEST} test runs")

# =============================================================================
# STEP 5: EVALUATION & METRICS BY IC TYPE
# =============================================================================

print("\n" + "="*80)
print("STEP 5: Evaluation and Metrics by IC Type")
print("="*80)

# Compute training mean for R² calculation
X_train_mean = X_train.mean(axis=0)

# Compute metrics for all test runs
all_metrics = []

for meta in test_metadata:
    run_name = meta["run_name"]
    pred = test_predictions[run_name]
    
    # Flatten densities for metrics
    T, ny, nx = pred["rho_true"].shape
    rho_true_flat = pred["rho_true"].reshape(T, ny * nx)
    rho_pred_flat = pred["rho_pred"].reshape(T, ny * nx)
    
    # Compute frame-wise metrics
    frame_metrics = compute_frame_metrics(rho_true_flat, rho_pred_flat)
    
    # Compute summary metrics
    summary = compute_summary_metrics(
        rho_true_flat,
        rho_pred_flat,
        X_train_mean,
        frame_metrics
    )
    summary["run_name"] = run_name
    summary["ic_type"] = meta["ic_type"]
    
    # Store frame metrics for later use
    pred["frame_metrics"] = frame_metrics
    
    all_metrics.append(summary)

# Convert to DataFrame for easy analysis
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(TEST_DIR / "metrics_all_runs.csv", index=False)

# Overall metrics
print("\n📊 Overall Metrics (all test runs):")
print(f"   R²:              {metrics_df['r2'].mean():.4f} ± {metrics_df['r2'].std():.4f}")
print(f"   Median L² error: {metrics_df['median_e2'].mean():.4f} ± {metrics_df['median_e2'].std():.4f}")
print(f"   Best run:  {metrics_df.loc[metrics_df['r2'].idxmax(), 'run_name']} (R² = {metrics_df['r2'].max():.4f})")
print(f"   Worst run: {metrics_df.loc[metrics_df['r2'].idxmin(), 'run_name']} (R² = {metrics_df['r2'].min():.4f})")

# Metrics by IC type
print("\n📊 Metrics by IC Type:")
ic_metrics = {}

for ic_type in IC_TYPES:
    ic_mask = metrics_df["ic_type"] == ic_type
    ic_data = metrics_df[ic_mask]
    
    if len(ic_data) == 0:
        continue
    
    ic_stats = {
        "ic_type": ic_type,
        "n_runs": len(ic_data),
        "r2_mean": ic_data["r2"].mean(),
        "r2_std": ic_data["r2"].std(),
        "r2_median": ic_data["r2"].median(),
        "r2_p10": ic_data["r2"].quantile(0.1),
        "r2_p90": ic_data["r2"].quantile(0.9),
        "median_e2_mean": ic_data["median_e2"].mean(),
        "median_e2_std": ic_data["median_e2"].std(),
        "best_run": ic_data.loc[ic_data["r2"].idxmax(), "run_name"],
        "best_r2": ic_data["r2"].max(),
    }
    
    ic_metrics[ic_type] = ic_stats
    
    print(f"\n   {ic_type}:")
    print(f"      Runs: {ic_stats['n_runs']}")
    print(f"      R²: {ic_stats['r2_mean']:.4f} ± {ic_stats['r2_std']:.4f} (median={ic_stats['r2_median']:.4f})")
    print(f"      R² range: [{ic_stats['r2_p10']:.4f}, {ic_stats['r2_p90']:.4f}] (P10-P90)")
    print(f"      Best run: {ic_stats['best_run']} (R² = {ic_stats['best_r2']:.4f})")

# Save IC-stratified metrics
pd.DataFrame(ic_metrics.values()).to_csv(TEST_DIR / "metrics_by_ic_type.csv", index=False)

# =============================================================================
# STEP 6: VISUALIZATIONS FOR BEST RUN PER IC TYPE
# =============================================================================

print("\n" + "="*80)
print("STEP 6: Generating Visualizations for Best Runs")
print("="*80)

print("\nGenerating videos and plots for best run per IC type...")

for ic_type in tqdm(IC_TYPES, desc="IC types"):
    ic_stats = ic_metrics.get(ic_type)
    if ic_stats is None:
        continue
    
    best_run = ic_stats["best_run"]
    pred = test_predictions[best_run]
    run_dir = TEST_DIR / best_run
    
    # Create IC-specific output directory
    ic_output_dir = BEST_RUNS_DIR / ic_type
    ic_output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get trajectory from predictions
    traj = pred["traj"]
    times = pred["times"]
    rho_true = pred["rho_true"]
    rho_pred = pred["rho_pred"]
    
    # 1. Trajectory video (ground truth) - renamed to traj_truth.mp4
    trajectory_video(
        path=ic_output_dir,
        traj=traj,
        times=times,
        Lx=BASE_CONFIG["sim"]["Lx"],
        Ly=BASE_CONFIG["sim"]["Ly"],
        name="traj_truth",
        fps=10,
        marker_size=50,
        title=f'Ground Truth Trajectory - {ic_type.replace("_", " ").title()}'
    )
    
    # 2. Density comparison video (side-by-side) - renamed to density_truth_vs_pred.mp4
    frame_metrics = pred["frame_metrics"]
    
    side_by_side_video(
        path=ic_output_dir,
        left_frames=pred["rho_true"],
        right_frames=pred["rho_pred"],
        lower_strip_timeseries=frame_metrics["e2"],
        name="density_truth_vs_pred",
        fps=10,
        cmap='hot',
        titles=(f'Ground Truth', f'MVAR-ROM Prediction')
    )
    
    # 3. Error timeseries plot - renamed to error_time.png
    summary = [m for m in all_metrics if m["run_name"] == best_run][0]
    
    plot_errors_timeseries(
        frame_metrics=frame_metrics,
        summary=summary,
        T0=P_LAG,
        save_path=ic_output_dir / "error_time.png",
        title=f'Error Metrics - {ic_type.replace("_", " ").title()} (R²={ic_stats["best_r2"]:.3f})'
    )
    plt.close('all')
    
    # 4. Error distribution histogram - renamed to error_hist.png
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # L1 error
    e1_final = np.abs(pred["rho_true"][-1] - pred["rho_pred"][-1])
    axes[0].hist(e1_final.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_xlabel('L1 Error', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title(f'L1 Error Distribution (t={pred["times"][-1]:.1f}s)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # L2 error
    e2_final = (pred["rho_true"][-1] - pred["rho_pred"][-1])**2
    axes[1].hist(e2_final.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_xlabel('L2 Error', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title(f'L2 Error Distribution (t={pred["times"][-1]:.1f}s)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Relative error
    rel_error_final = e1_final / (pred["rho_true"][-1] + 1e-10)
    axes[2].hist(rel_error_final.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[2].set_xlabel('Relative Error', fontsize=11)
    axes[2].set_ylabel('Count', fontsize=11)
    axes[2].set_title(f'Relative Error Distribution (t={pred["times"][-1]:.1f}s)', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    fig.suptitle(f'Error Distributions - {ic_type.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(ic_output_dir / "error_hist.png", dpi=200)
    plt.close()
    
    # 5. Order parameters plot - renamed to order_parameters.png
    order_params_path = run_dir / "order_params.csv"
    if order_params_path.exists():
        import pandas as pd
        df_order = pd.read_csv(order_params_path)
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Panel 1: Polarization
        axes[0].plot(df_order['t'], df_order['phi'], 'b-', linewidth=2)
        axes[0].set_ylabel('Polarization Φ', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1.05])
        median_phi = df_order['phi'].iloc[-len(df_order)//4:].median()
        axes[0].axhline(median_phi, color='r', linestyle='--', alpha=0.5,
                       label=f'Final median: {median_phi:.3f}')
        axes[0].legend(loc='best', fontsize=10)
        axes[0].set_title(f'Order Parameters - {ic_type.replace("_", " ").title()} (R²={ic_stats["best_r2"]:.3f})', 
                         fontsize=14, fontweight='bold')
        
        # Panel 2: Mean Speed
        axes[1].plot(df_order['t'], df_order['mean_speed'], 'g-', linewidth=2)
        axes[1].set_ylabel('Mean Speed', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Panel 3: Nematic Order
        axes[2].plot(df_order['t'], df_order['nematic'], 'm-', linewidth=2)
        axes[2].set_ylabel('Nematic Order Q', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Time (s)', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([-0.05, 1.05])
        
        plt.tight_layout()
        order_plot_path = ic_output_dir / "order_parameters.png"
        plt.savefig(order_plot_path, dpi=150, bbox_inches='tight')
        plt.close()

print(f"\n✓ Generated visualizations for {len(IC_TYPES)} IC types")
print(f"\nBest runs organized in: {BEST_RUNS_DIR}/")
for ic in IC_TYPES:
    print(f"   • {ic}/: traj_truth.mp4, density_truth_vs_pred.mp4, error_time.png, error_hist.png, order_parameters.png")

# =============================================================================
# STEP 7: SUMMARY PLOTS
# =============================================================================

print("\n" + "="*80)
print("STEP 7: Generating Summary Plots")
print("="*80)

# 1. R² by IC type (box plot)
fig, ax = plt.subplots(figsize=(10, 6))
ic_r2_data = [metrics_df[metrics_df["ic_type"] == ic]["r2"].values for ic in IC_TYPES]
bp = ax.boxplot(ic_r2_data, labels=IC_TYPES, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax.set_ylabel('R² Score', fontsize=12)
ax.set_xlabel('Initial Condition Type', fontsize=12)
ax.set_title('MVAR-ROM Performance by IC Type', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "r2_by_ic_type.png", dpi=200)
plt.close()

# 2. Median L² error by IC type
fig, ax = plt.subplots(figsize=(10, 6))
ic_e2_data = [metrics_df[metrics_df["ic_type"] == ic]["median_e2"].values for ic in IC_TYPES]
bp = ax.boxplot(ic_e2_data, labels=IC_TYPES, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightcoral')
ax.set_ylabel('Median L² Error', fontsize=12)
ax.set_xlabel('Initial Condition Type', fontsize=12)
ax.set_title('MVAR-ROM Error by IC Type', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "error_by_ic_type.png", dpi=200)
plt.close()

print("  ✓ r2_by_ic_type.png")
print("  ✓ error_by_ic_type.png")

# =============================================================================
# GENERATE SUMMARY JSON
# =============================================================================

# Calculate total pipeline time
total_pipeline_time = time.time() - pipeline_start_time

# Calculate POD compression ratio
n_space = DENSITY_NX * DENSITY_NY
compression_ratio = (1 - R_POD / n_space) * 100  # Percentage compression

# Rank IC types by performance (best to worst R²)
ic_ranking = []
for ic_type in IC_TYPES:
    if ic_type in ic_metrics:
        ic_data = metrics_df[metrics_df["ic_type"] == ic_type]
        ic_ranking.append({
            "ic_type": ic_type,
            "mean_r2": float(ic_data["r2"].mean()),
            "std_r2": float(ic_data["r2"].std()),
            "mean_l2_error": float(ic_data["median_e2"].mean()),
            "best_r2": float(ic_metrics[ic_type]["best_r2"]),
            "best_run": ic_metrics[ic_type]["best_run"]
        })

# Sort by mean R² (descending)
ic_ranking.sort(key=lambda x: x["mean_r2"], reverse=True)

# Create comprehensive summary
summary = {
    "model_parameters": {
        "simulation": {
            "N_particles": int(BASE_CONFIG["sim"]["N"]),
            "domain_size": {
                "Lx": float(BASE_CONFIG["sim"]["Lx"]),
                "Ly": float(BASE_CONFIG["sim"]["Ly"])
            },
            "boundary_conditions": BASE_CONFIG["sim"]["bc"],
            "time": {
                "T_total": float(BASE_CONFIG["sim"]["T"]),
                "dt": float(BASE_CONFIG["sim"]["dt"]),
                "save_every": int(BASE_CONFIG["sim"]["save_every"])
            },
            "model": "vicsek_morse_discrete",
            "speed": float(BASE_CONFIG["model"]["speed"]),
            "interaction_radius": float(BASE_CONFIG["params"]["R"]),
            "noise": {
                "kind": BASE_CONFIG["noise"]["kind"],
                "eta": float(BASE_CONFIG["noise"]["eta"])
            },
            "forces_enabled": BASE_CONFIG["forces"]["enabled"]
        },
        "density_estimation": {
            "method": "KDE",
            "resolution": {"nx": int(DENSITY_NX), "ny": int(DENSITY_NY)},
            "bandwidth": float(DENSITY_BANDWIDTH)
        },
        "initial_conditions": {
            "types": IC_TYPES,
            "description": "Stratified sampling across IC types"
        }
    },
    "training_metrics": {
        "ensemble": {
            "n_simulations": int(N_TRAIN),
            "n_snapshots_total": int(len(index_map)),
            "ic_distribution": {
                ic: sum(1 for m in train_metadata if m["ic_type"] == ic)
                for ic in IC_TYPES
            }
        },
        "pod": {
            "target_energy": float(TARGET_ENERGY),
            "n_modes_selected": int(R_POD),
            "actual_energy_captured": float(actual_energy),
            "spatial_dof": int(n_space),
            "compression_ratio_percent": float(compression_ratio),
            "compression_description": f"{R_POD}/{n_space} modes ({compression_ratio:.2f}% compression)"
        },
        "mvar": {
            "latent_dimension": int(R_POD),
            "lag_order": int(P_LAG),
            "ridge_alpha": float(RIDGE_ALPHA),
            "training_r2": float(train_r2),
            "training_rmse": float(train_rmse)
        },
        "timing": {
            "step1_training_ensemble_sec": float(timing_info.get("step1_training_ensemble", 0)),
            "step2_pod_mvar_training_sec": float(timing_info.get("step2_pod_mvar_training", 0)),
            "total_training_time_sec": float(timing_info.get("step1_training_ensemble", 0) + timing_info.get("step2_pod_mvar_training", 0))
        }
    },
    "test_results": {
        "ensemble": {
            "n_test_runs": int(M_TEST),
            "stratification": "Equal distribution across IC types",
            "runs_per_ic": int(M_TEST // len(IC_TYPES))
        },
        "overall_performance": {
            "mean_r2": float(metrics_df["r2"].mean()),
            "std_r2": float(metrics_df["r2"].std()),
            "median_r2": float(metrics_df["r2"].median()),
            "min_r2": float(metrics_df["r2"].min()),
            "max_r2": float(metrics_df["r2"].max())
        },
        "error_metrics": {
            "mean_l2_error": float(metrics_df["median_e2"].mean()),
            "std_l2_error": float(metrics_df["median_e2"].std()),
            "p10_l2_error": float(metrics_df["p10_e2"].mean()),
            "p90_l2_error": float(metrics_df["p90_e2"].mean()),
            "mean_mass_error": float(metrics_df["mean_mass_error"].mean()),
            "max_mass_error": float(metrics_df["max_mass_error"].max()),
            "mean_tau_tolerance": float(metrics_df["tau_tol"].mean())
        },
        "ic_performance_ranking": ic_ranking,
        "best_overall_run": {
            "run_name": metrics_df.loc[metrics_df["r2"].idxmax(), "run_name"],
            "ic_type": metrics_df.loc[metrics_df["r2"].idxmax(), "ic_type"],
            "r2": float(metrics_df["r2"].max())
        },
        "worst_overall_run": {
            "run_name": metrics_df.loc[metrics_df["r2"].idxmin(), "run_name"],
            "ic_type": metrics_df.loc[metrics_df["r2"].idxmin(), "ic_type"],
            "r2": float(metrics_df["r2"].min())
        }
    },
    "pipeline_metadata": {
        "total_execution_time_sec": float(total_pipeline_time),
        "total_execution_time_formatted": f"{total_pipeline_time//60:.0f}m {total_pipeline_time%60:.1f}s",
        "output_directory": str(OUTPUT_DIR),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "1.0"
    }
}

# Save summary JSON
summary_path = OUTPUT_DIR / "pipeline_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n💾 Saved comprehensive summary: {summary_path.name}")

# =============================================================================
# PIPELINE COMPLETE
# =============================================================================

print("\n" + "="*80)
print("PIPELINE COMPLETE! 🎉")
print("="*80)

print(f"\n📂 All outputs saved to: {OUTPUT_DIR}")
print(f"\n📊 Training:")
print(f"   • {N_TRAIN} simulations")
print(f"   • {len(index_map)} total snapshots")
print(f"   • {R_POD} POD modes ({100*actual_energy:.2f}% energy, target: {TARGET_ENERGY*100:.1f}%)")
print(f"   • MVAR order p={P_LAG}")
print(f"   • Training R²: {train_r2:.4f}")

print(f"\n📈 Test Results ({M_TEST} runs):")
print(f"   • Mean R²:           {metrics_df['r2'].mean():.4f} ± {metrics_df['r2'].std():.4f}")
print(f"   • Median L² error:   {metrics_df['median_e2'].mean():.4f} ± {metrics_df['median_e2'].std():.4f}")

print(f"\n🎬 Best Runs by IC Type ({BEST_RUNS_DIR.name}/):")
for ic_type in IC_TYPES:
    if ic_type in ic_metrics:
        best_r2 = ic_metrics[ic_type]["best_r2"]
        print(f"   • {ic_type}/ (R²={best_r2:.3f})")
        print(f"       - traj_truth.mp4")
        print(f"       - density_truth_vs_pred.mp4")
        print(f"       - error_time.png")
        print(f"       - error_hist.png")
        print(f"       - order_parameters.png")

print(f"\n📊 Summary Plots ({PLOTS_DIR.name}/):")
print(f"   • POD singular values (log scale, showing all {len(S)} modes)")
print(f"   • POD energy spectrum (cumulative %, target line at {TARGET_ENERGY*100:.1f}%)")
print(f"   • R² by IC type (box plot)")
print(f"   • Error by IC type (box plot)")
print(f"   • R² by IC type (box plot)")
print(f"   • Error by IC type (box plot)")
print(f"   • Per-IC best run: error timeseries + error distributions + order parameters")

print(f"\n📄 Data:")
print(f"   • metrics_all_runs.csv - All test run metrics")
print(f"   • metrics_by_ic_type.csv - Aggregated IC-wise statistics")
print(f"   • index_mapping.csv - Global snapshot index mapping")

print("\n✅ Full pipeline executed successfully!")
print("="*80)
