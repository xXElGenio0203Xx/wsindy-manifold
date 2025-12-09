#!/usr/bin/env python3
"""
Complete end-to-end pipeline test:
1. Run light simulations (training data)
2. Train MVAR model on density fields
3. Run test simulations (unseen data)
4. Evaluate MVAR predictions
5. Generate plots and visualizations
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import shutil

# Import rectsim modules
from rectsim.vicsek_discrete import simulate_backend
from rectsim.density import compute_density_grid
from rectsim.mvar import (
    load_density_movies,
    build_global_snapshot_matrix,
    compute_pod,
    project_to_pod,
    fit_mvar_from_runs,
    mvar_forecast,
    reconstruct_from_pod
)
from rectsim.rom_eval_metrics import (
    compute_forecast_metrics,
    compute_relative_errors_timeseries
)

print("="*70)
print("COMPLETE MVAR PIPELINE TEST")
print("="*70)

# Setup directories
output_dir = Path("outputs/pipeline_test")
train_dir = output_dir / "train"
test_dir = output_dir / "test"
mvar_dir = output_dir / "mvar"

# Clean and create directories
if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True)
train_dir.mkdir()
test_dir.mkdir()
mvar_dir.mkdir()

print(f"\n📁 Output directory: {output_dir}")

# =============================================================================
# STEP 1: Generate Training Simulations (3 light runs)
# =============================================================================
print("\n" + "="*70)
print("STEP 1: Generating Training Data (3 simulations)")
print("="*70)

train_config = {
    "sim": {
        "N": 50,          # Small number of particles
        "Lx": 20.0,
        "Ly": 20.0,
        "bc": "periodic",
        "T": 5.0,         # Short duration
        "dt": 0.05,
        "save_every": 2,  # Save every 2 steps
        "neighbor_rebuild": 5,
    },
    "model": {"speed": 1.0},
    "params": {"R": 2.0},
    "noise": {"kind": "gaussian", "eta": 0.3},
}

n_train = 3
train_seeds = [100, 101, 102]
density_resolution = 20  # Low resolution for speed

print(f"Config: N={train_config['sim']['N']}, T={train_config['sim']['T']}, "
      f"dt={train_config['sim']['dt']}, resolution={density_resolution}")

train_densities = {}
for i, seed in enumerate(train_seeds):
    run_name = f"run_{i:03d}"
    print(f"\n  [{i+1}/{n_train}] Simulating {run_name} (seed={seed})...")
    
    rng = np.random.default_rng(seed)
    result = simulate_backend(train_config, rng)
    
    # Compute density fields
    T, N, _ = result["traj"].shape
    densities = []
    
    print(f"    Computing density fields for {T} frames...")
    for t in range(T):
        rho, _, _ = compute_density_grid(
            result["traj"][t],
            nx=density_resolution,
            ny=density_resolution,
            Lx=train_config["sim"]["Lx"],
            Ly=train_config["sim"]["Ly"],
            bandwidth=0.5,
            bc=train_config["sim"]["bc"]
        )
        densities.append(rho)
    
    densities = np.array(densities)
    
    # Save density data in correct format
    run_output_dir = train_dir / run_name
    run_output_dir.mkdir(exist_ok=True)
    np.savez(
        run_output_dir / "density.npz",
        rho=densities,
        times=result["times"],
        metadata={
            "N": N,
            "T": train_config["sim"]["T"],
            "dt": train_config["sim"]["dt"],
            "seed": seed,
        }
    )
    
    train_densities[run_name] = {
        "rho": densities,
        "times": result["times"]
    }
    
    print(f"    ✓ Saved {densities.shape[0]} density fields, shape: {densities.shape[1:]}")

# =============================================================================
# STEP 2: Build POD Basis and Train MVAR
# =============================================================================
print("\n" + "="*70)
print("STEP 2: Training MVAR Model")
print("="*70)

# Load density movies
print("\n  Loading density movies...")
density_dict = load_density_movies([train_dir / f"run_{i:03d}" for i in range(n_train)])
print(f"  ✓ Loaded {len(density_dict)} density movies")

# Build snapshot matrix
print("\n  Building global snapshot matrix...")
X, run_info, mean = build_global_snapshot_matrix(density_dict, subtract_mean=True)
print(f"  ✓ Snapshot matrix shape: {X.shape}")
print(f"    Mean field shape: {mean.shape}")

# Compute POD
print("\n  Computing POD basis...")
r_pod = 10  # Number of POD modes
pod_basis = compute_pod(X, r=r_pod)
print(f"  ✓ Using r={r_pod} POD modes")
print(f"    Cumulative energy: {pod_basis['energy'][r_pod-1]:.2%}")

# Project to POD
print("\n  Projecting density fields to latent space...")
latent_dict = project_to_pod(density_dict, pod_basis["Phi"], mean)

# Train MVAR
print("\n  Training MVAR model...")
p_lag = 3  # Lag order
mvar, mvar_info = fit_mvar_from_runs(latent_dict, order=p_lag, ridge=1e-6, train_frac=1.0)
print(f"  ✓ MVAR model trained with lag p={p_lag}")
print(f"    Latent dimension: {mvar.latent_dim}")
print(f"    Coefficient matrices: {mvar.A.shape}")

# Save MVAR model
mvar_path = mvar_dir / "mvar_model.npz"
mvar.save(mvar_path)
print(f"  ✓ Saved MVAR model to {mvar_path}")

# =============================================================================
# STEP 3: Generate Test Simulations (2 unseen runs)
# =============================================================================
print("\n" + "="*70)
print("STEP 3: Generating Test Data (2 unseen simulations)")
print("="*70)

n_test = 2
test_seeds = [200, 201]

test_densities = {}
for i, seed in enumerate(test_seeds):
    run_name = f"test_{i:03d}"
    print(f"\n  [{i+1}/{n_test}] Simulating {run_name} (seed={seed})...")
    
    rng = np.random.default_rng(seed)
    result = simulate_backend(train_config, rng)
    
    # Compute density fields
    T, N, _ = result["traj"].shape
    densities = []
    
    print(f"    Computing density fields for {T} frames...")
    for t in range(T):
        rho, _, _ = compute_density_grid(
            result["traj"][t],
            nx=density_resolution,
            ny=density_resolution,
            Lx=train_config["sim"]["Lx"],
            Ly=train_config["sim"]["Ly"],
            bandwidth=0.5,
            bc=train_config["sim"]["bc"]
        )
        densities.append(rho)
    
    densities = np.array(densities)
    
    # Save density data in correct format
    run_output_dir = test_dir / run_name
    run_output_dir.mkdir(exist_ok=True)
    np.savez(
        run_output_dir / "density.npz",
        rho=densities,
        times=result["times"],
        metadata={
            "N": N,
            "T": train_config["sim"]["T"],
            "dt": train_config["sim"]["dt"],
            "seed": seed,
        }
    )
    
    test_densities[run_name] = {
        "rho": densities,
        "times": result["times"]
    }
    
    print(f"    ✓ Saved {densities.shape[0]} density fields, shape: {densities.shape[1:]}")

# =============================================================================
# STEP 4: MVAR Prediction on Test Data
# =============================================================================
print("\n" + "="*70)
print("STEP 4: MVAR Prediction on Test Data")
print("="*70)

# Load test density movies
test_density_dict = load_density_movies([test_dir / f"test_{i:03d}" for i in range(n_test)])

# Project test data to latent space
test_latent_dict = project_to_pod(test_density_dict, pod_basis["Phi"], mean)

# Make predictions
print("\n  Making MVAR predictions...")
test_run = "test_000"
Y_test_true = test_latent_dict[test_run]["Y"]
times_test = test_latent_dict[test_run]["times"]

# Use first p_lag timesteps as initial condition, predict the rest
n_ic = p_lag
n_pred = len(Y_test_true) - n_ic

print(f"    Initial condition: {n_ic} timesteps")
print(f"    Prediction horizon: {n_pred} timesteps")

Y_init = Y_test_true[:n_ic]
Y_pred = mvar_forecast(mvar, Y_init, n_pred)

print(f"  ✓ Prediction shape: {Y_pred.shape}")

# Reconstruct density fields
print("\n  Reconstructing density fields...")
ny, nx = density_resolution, density_resolution
rho_true = test_density_dict[test_run]["rho"][n_ic:]
rho_pred = reconstruct_from_pod(Y_pred, pod_basis["Phi"], mean, ny, nx)

print(f"  ✓ True density shape: {rho_true.shape}")
print(f"  ✓ Predicted density shape: {rho_pred.shape}")

# =============================================================================
# STEP 5: Compute Evaluation Metrics
# =============================================================================
print("\n" + "="*70)
print("STEP 5: Evaluation Metrics")
print("="*70)

metrics = compute_forecast_metrics(rho_true, rho_pred, times=times_test[n_ic:], tol=0.05)

print(f"\n  📊 Overall Metrics:")
print(f"    RMSE (mean):  {metrics['rmse_mean']:.4f}")
print(f"    R²:           {metrics['r2']:.4f}")
print(f"    L¹ (median):  {metrics['e1_median']:.4f}")
print(f"    L² (median):  {metrics['e2_median']:.4f}")
print(f"    L∞ (median):  {metrics['einf_median']:.4f}")
print(f"    Mass error:   {metrics['mass_error_mean']:.4f}")
print(f"    τ (5% tol):   {metrics['tau'] if metrics['tau'] else 'N/A'}")

# Time series metrics
error_series = compute_relative_errors_timeseries(rho_true, rho_pred)
rel_errors = error_series["rel_e2"]

print(f"\n  📈 Time Series Statistics:")
print(f"    Mean relative error: {np.mean(rel_errors):.4f}")
print(f"    Max relative error:  {np.max(rel_errors):.4f}")
print(f"    Error std dev:       {np.std(rel_errors):.4f}")

# =============================================================================
# STEP 6: Generate Visualizations
# =============================================================================
print("\n" + "="*70)
print("STEP 6: Generating Visualizations")
print("="*70)

# Plot 1: POD Energy
print("\n  Plotting POD energy spectrum...")
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, len(pod_basis["S"]) + 1), 
        pod_basis["energy"] * 100, 'o-', linewidth=2)
ax.axvline(r_pod, color='r', linestyle='--', label=f'Selected r={r_pod}')
ax.axhline(95, color='g', linestyle=':', alpha=0.5, label='95% energy')
ax.set_xlabel('Number of POD Modes')
ax.set_ylabel('Cumulative Energy (%)')
ax.set_title('POD Energy Spectrum')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_xlim(0, min(30, len(pod_basis["S"])))
plt.tight_layout()
plt.savefig(output_dir / "pod_energy.png", dpi=150)
print(f"    ✓ Saved {output_dir / 'pod_energy.png'}")
plt.close()

# Plot 2: Latent trajectories (first 3 modes)
print("\n  Plotting latent trajectories...")
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
for i in range(3):
    ax = axes[i]
    for j, run_name in enumerate(latent_dict):
        Y = latent_dict[run_name]["Y"]
        times = latent_dict[run_name]["times"]
        ax.plot(times, Y[:, i], label=run_name, alpha=0.7)
    ax.set_ylabel(f'Mode {i+1}')
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(loc='upper right', fontsize=8)
    if i == 2:
        ax.set_xlabel('Time')
axes[0].set_title('Latent Trajectories (Training Data)')
plt.tight_layout()
plt.savefig(output_dir / "latent_trajectories.png", dpi=150)
print(f"    ✓ Saved {output_dir / 'latent_trajectories.png'}")
plt.close()

# Plot 3: Prediction vs Ground Truth (latent space)
print("\n  Plotting latent space predictions...")
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
times_pred = times_test[n_ic:]
for i in range(3):
    ax = axes[i]
    ax.plot(times_pred, Y_test_true[n_ic:, i], 'b-', label='True', linewidth=2)
    ax.plot(times_pred, Y_pred[:, i], 'r--', label='MVAR Pred', linewidth=2)
    ax.set_ylabel(f'Mode {i+1}')
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend()
    if i == 2:
        ax.set_xlabel('Time')
axes[0].set_title(f'MVAR Forecast (Test Run, p={p_lag})')
plt.tight_layout()
plt.savefig(output_dir / "mvar_prediction_latent.png", dpi=150)
print(f"    ✓ Saved {output_dir / 'mvar_prediction_latent.png'}")
plt.close()

# Plot 4: Relative Error Time Series
print("\n  Plotting error time series...")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(times_pred, rel_errors * 100, 'o-', linewidth=2, markersize=4)
ax.axhline(5, color='orange', linestyle='--', alpha=0.5, label='5% threshold')
ax.axhline(10, color='red', linestyle='--', alpha=0.5, label='10% threshold')
ax.set_xlabel('Time')
ax.set_ylabel('Relative Error (%)')
ax.set_title('MVAR Prediction Error Over Time')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(output_dir / "error_timeseries.png", dpi=150)
print(f"    ✓ Saved {output_dir / 'error_timeseries.png'}")
plt.close()

# Plot 5: Density Field Snapshots (comparison)
print("\n  Plotting density field snapshots...")
n_snapshots = min(4, len(rho_true))
snapshot_indices = np.linspace(0, len(rho_true)-1, n_snapshots, dtype=int)

fig, axes = plt.subplots(3, n_snapshots, figsize=(4*n_snapshots, 10))

for col, idx in enumerate(snapshot_indices):
    t = times_pred[idx]
    
    # True density
    im1 = axes[0, col].imshow(rho_true[idx], origin='lower', cmap='hot', 
                              extent=[0, train_config["sim"]["Lx"], 0, train_config["sim"]["Ly"]])
    axes[0, col].set_title(f't={t:.2f}')
    if col == 0:
        axes[0, col].set_ylabel('True Density', fontsize=12)
    
    # Predicted density
    im2 = axes[1, col].imshow(rho_pred[idx], origin='lower', cmap='hot',
                              extent=[0, train_config["sim"]["Lx"], 0, train_config["sim"]["Ly"]])
    if col == 0:
        axes[1, col].set_ylabel('MVAR Prediction', fontsize=12)
    
    # Error
    error = np.abs(rho_true[idx] - rho_pred[idx])
    im3 = axes[2, col].imshow(error, origin='lower', cmap='Reds',
                              extent=[0, train_config["sim"]["Lx"], 0, train_config["sim"]["Ly"]])
    if col == 0:
        axes[2, col].set_ylabel('Absolute Error', fontsize=12)
    axes[2, col].set_xlabel('x')

# Add colorbars
fig.colorbar(im1, ax=axes[0, :], location='right', shrink=0.8, label='Density')
fig.colorbar(im2, ax=axes[1, :], location='right', shrink=0.8, label='Density')
fig.colorbar(im3, ax=axes[2, :], location='right', shrink=0.8, label='Error')

plt.suptitle('Density Field Comparison: True vs MVAR Prediction', fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig(output_dir / "density_snapshots.png", dpi=150)
print(f"    ✓ Saved {output_dir / 'density_snapshots.png'}")
plt.close()

# Plot 6: Summary metrics visualization
print("\n  Plotting summary metrics...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Metrics bar chart
ax = axes[0]
metric_names = ['RMSE', 'L¹', 'L²', 'L∞']
metric_values = [metrics['rmse_mean'], metrics['e1_median'], 
                 metrics['e2_median'], metrics['einf_median']]
colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6']
bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Error Magnitude')
ax.set_title('Forecast Error Metrics')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, metric_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10)

# R² and mass conservation
ax = axes[1]
metric_names = ['R²', 'Mass Error', '1-R²']
metric_values = [metrics['r2'], metrics['mass_error_mean'], 1 - metrics['r2']]
colors = ['#2ecc71', '#e74c3c', '#95a5a6']
bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Value')
ax.set_title('Goodness of Fit')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.0])
for bar, val in zip(bars, metric_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / "summary_metrics.png", dpi=150)
print(f"    ✓ Saved {output_dir / 'summary_metrics.png'}")
plt.close()

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*70)
print("PIPELINE COMPLETE! 🎉")
print("="*70)

print(f"\n📂 All outputs saved to: {output_dir.absolute()}")
print(f"\n📊 Generated plots:")
print(f"   1. pod_energy.png - POD mode energy spectrum")
print(f"   2. latent_trajectories.png - Training data in latent space")
print(f"   3. mvar_prediction_latent.png - Predictions vs truth in latent space")
print(f"   4. error_timeseries.png - Prediction error over time")
print(f"   5. density_snapshots.png - Side-by-side density field comparison")
print(f"   6. summary_metrics.png - Aggregated performance metrics")

print(f"\n📈 Key Results:")
print(f"   • Trained on {n_train} simulations with {r_pod} POD modes")
print(f"   • MVAR lag order: p={p_lag}")
print(f"   • Test RMSE: {metrics['rmse_mean']:.4f}")
print(f"   • Test R²: {metrics['r2']:.4f}")
print(f"   • Mass conservation error: {metrics['mass_error_mean']:.4f}")
if metrics['tau']:
    print(f"   • Accurate prediction up to τ (5% tol) = {metrics['tau']:.2f}")
else:
    print(f"   • Error remained below 5% for entire prediction horizon!")

print(f"\n✅ Full pipeline validated successfully!")
print("="*70)
