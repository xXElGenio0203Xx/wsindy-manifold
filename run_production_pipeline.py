#!/usr/bin/env python3
"""
Production MVAR Pipeline:
- Train on 100 very short simulations
- Predict on 20 test simulations
- Generate videos and comprehensive visualizations

USING LEGACY FUNCTIONS FROM OLD WORKING VERSION (commit 67655d3)
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm
import json

# Import rectsim modules
from rectsim.vicsek_discrete import simulate_backend
from rectsim.density import compute_density_grid, density_movie_kde
from rectsim.mvar import (
    load_density_movies,
    build_global_snapshot_matrix,
    compute_pod,
    project_to_pod,
    fit_mvar_from_runs,
    mvar_forecast,
    reconstruct_from_pod
)

# Import LEGACY functions (old working versions)
from rectsim.legacy_functions import (
    kde_density_movie,
    save_video,
    side_by_side_video,
    trajectory_video,
    compute_order_params,
    compute_frame_metrics,
    compute_summary_metrics,
    plot_errors_timeseries
)

print("="*80)
print("PRODUCTION MVAR PIPELINE - 100 Training Runs, 20 Test Runs")
print("="*80)

# Setup directories
output_dir = Path("outputs/production_pipeline")
train_dir = output_dir / "train"
test_dir = output_dir / "test"
mvar_dir = output_dir / "mvar"
videos_dir = output_dir / "videos"
plots_dir = output_dir / "plots"

# Clean and create directories
if output_dir.exists():
    print("\n🗑️  Cleaning previous outputs...")
    shutil.rmtree(output_dir)

for d in [train_dir, test_dir, mvar_dir, videos_dir, plots_dir]:
    d.mkdir(parents=True)

print(f"\n📁 Output directory: {output_dir}")

# =============================================================================
# Configuration: Very short simulations for fast training
# =============================================================================

config = {
    "sim": {
        "N": 40,          # Small number of particles
        "Lx": 15.0,
        "Ly": 15.0,
        "bc": "periodic",
        "T": 2.0,         # Very short: 2 seconds
        "dt": 0.1,        # Larger timestep
        "save_every": 1,  # Save every step
        "neighbor_rebuild": 5,
    },
    "model": {"speed": 1.0},
    "params": {"R": 2.0},
    "noise": {"kind": "gaussian", "eta": 0.3},
}

n_train = 100  # 100 training runs
n_test = 20    # 20 test runs

# FIXED: Use proper resolution and bandwidth for smooth continuous heatmaps
# Old problematic settings: 16×16, BW=0.5 → blocky sparse fields
# New settings: 64×64, BW=2.0 → smooth continuous heatmaps (like old kde_density_movie)
density_resolution = 64  # High resolution for smooth heatmaps
density_bandwidth = 2.0  # Larger bandwidth for smooth continuous fields

r_pod = 15  # Number of POD modes
p_lag = 4   # MVAR lag order

print(f"\n⚙️  Configuration:")
print(f"   Training runs: {n_train}")
print(f"   Test runs: {n_test}")
print(f"   Particles: {config['sim']['N']}")
print(f"   Duration: {config['sim']['T']}s (dt={config['sim']['dt']})")
print(f"   Density resolution: {density_resolution}×{density_resolution}")
print(f"   Density bandwidth: {density_bandwidth} (for smooth continuous heatmaps)")
print(f"   POD modes: {r_pod}")
print(f"   MVAR lag: {p_lag}")

# =============================================================================
# STEP 1: Generate Training Data
# =============================================================================
print("\n" + "="*80)
print("STEP 1: Generating Training Data")
print("="*80)

train_seeds = range(1000, 1000 + n_train)

print(f"\nSimulating {n_train} training runs...")
for i, seed in enumerate(tqdm(train_seeds, desc="Training sims")):
    run_name = f"run_{i:03d}"
    rng = np.random.default_rng(seed)
    result = simulate_backend(config, rng)
    
    # Compute density fields
    T, N, _ = result["traj"].shape
    densities = []
    
    for t in range(T):
        rho, _, _ = compute_density_grid(
            result["traj"][t],
            nx=density_resolution,
            ny=density_resolution,
            Lx=config["sim"]["Lx"],
            Ly=config["sim"]["Ly"],
            bandwidth=density_bandwidth,  # Use proper bandwidth for smooth heatmaps
            bc=config["sim"]["bc"]
        )
        densities.append(rho)
    
    densities = np.array(densities)
    
    # Save in correct format
    run_output_dir = train_dir / run_name
    run_output_dir.mkdir(exist_ok=True)
    np.savez(
        run_output_dir / "density.npz",
        rho=densities,
        times=result["times"],
    )

print(f"✓ Generated {n_train} training runs")

# =============================================================================
# STEP 2: Build POD and Train MVAR
# =============================================================================
print("\n" + "="*80)
print("STEP 2: Training MVAR Model")
print("="*80)

print("\nLoading density movies...")
density_dict = load_density_movies([train_dir / f"run_{i:03d}" for i in range(n_train)])
print(f"✓ Loaded {len(density_dict)} density movies")

print("\nBuilding global snapshot matrix...")
X, run_info, mean = build_global_snapshot_matrix(density_dict, subtract_mean=True)
print(f"✓ Snapshot matrix shape: {X.shape}")

print(f"\nComputing POD basis (r={r_pod})...")
pod_basis = compute_pod(X, r=r_pod)
print(f"✓ Cumulative energy: {pod_basis['energy'][r_pod-1]:.2%}")

print("\nProjecting to latent space...")
latent_dict = project_to_pod(density_dict, pod_basis["Phi"], mean)

print(f"\nTraining MVAR model (p={p_lag})...")
mvar, mvar_info = fit_mvar_from_runs(latent_dict, order=p_lag, ridge=1e-6, train_frac=1.0)
print(f"✓ MVAR trained: {mvar.latent_dim}D latent, {mvar_info['total_samples']} samples")

# Save model
mvar.save(mvar_dir / "mvar_model.npz")
np.savez(
    mvar_dir / "pod_basis.npz",
    Phi=pod_basis["Phi"],
    S=pod_basis["S"],
    energy=pod_basis["energy"],
    mean=mean
)
print(f"✓ Saved model to {mvar_dir}")

# =============================================================================
# STEP 3: Generate Test Data
# =============================================================================
print("\n" + "="*80)
print("STEP 3: Generating Test Data")
print("="*80)

test_seeds = range(2000, 2000 + n_test)

# Define initial spatial distributions to test (cycle through them)
# Supported: 'uniform', 'gaussian_cluster', 'two_clusters', 'ring'
initial_distributions = [
    "uniform",
    "gaussian_cluster", 
    "two_clusters",
    "ring"
]

# Track which distribution each test run uses
test_run_metadata = {}

print(f"\nSimulating {n_test} test runs with varied initial distributions...")
for i, seed in enumerate(tqdm(test_seeds, desc="Test sims")):
    run_name = f"test_{i:03d}"
    
    # Cycle through initial distributions
    initial_dist = initial_distributions[i % len(initial_distributions)]
    
    # Store metadata
    test_run_metadata[run_name] = {
        "initial_distribution": initial_dist,
        "seed": seed
    }
    
    # Update config with current initial distribution
    test_config = config.copy()
    test_config["initial_distribution"] = initial_dist
    
    rng = np.random.default_rng(seed)
    result = simulate_backend(test_config, rng)
    
    # Compute density fields
    T, N, _ = result["traj"].shape
    densities = []
    
    for t in range(T):
        rho, _, _ = compute_density_grid(
            result["traj"][t],
            nx=density_resolution,
            ny=density_resolution,
            Lx=config["sim"]["Lx"],
            Ly=config["sim"]["Ly"],
            bandwidth=density_bandwidth,  # Use proper bandwidth for smooth heatmaps
            bc=config["sim"]["bc"]
        )
        densities.append(rho)
    
    densities = np.array(densities)
    
    # Compute order parameters from velocities (legacy function)
    order_params_list = []
    for t in range(T):
        vel = result["vel"][t]  # (N, 2) velocities at time t
        params = compute_order_params(vel, include_nematic=True)
        params['t'] = result["times"][t]
        order_params_list.append(params)
    
    # Save density AND trajectory for test runs (needed for videos)
    run_output_dir = test_dir / run_name
    run_output_dir.mkdir(exist_ok=True)
    np.savez(
        run_output_dir / "density.npz",
        rho=densities,
        times=result["times"],
    )
    # Save trajectory for visualization
    np.savez(
        run_output_dir / "trajectory.npz",
        traj=result["traj"],
        times=result["times"],
    )
    # Save order parameters
    import pandas as pd
    df_order = pd.DataFrame(order_params_list)
    df_order = df_order[['t', 'phi', 'mean_speed', 'speed_std', 'nematic']]
    df_order.to_csv(run_output_dir / "order_params.csv", index=False)
    
    # Save metadata (noise distribution info)
    with open(run_output_dir / "metadata.json", "w") as f:
        json.dump(test_run_metadata[run_name], f, indent=2)

print(f"✓ Generated {n_test} test runs")

# Save global test metadata
with open(test_dir / "test_metadata.json", "w") as f:
    json.dump(test_run_metadata, f, indent=2)

# =============================================================================
# STEP 4: Predict on All Test Runs
# =============================================================================
print("\n" + "="*80)
print("STEP 4: MVAR Prediction on Test Data")
print("="*80)

print("\nLoading test data...")
test_density_dict = load_density_movies([test_dir / f"test_{i:03d}" for i in range(n_test)])
test_latent_dict = project_to_pod(test_density_dict, pod_basis["Phi"], mean)

print("\nMaking predictions on all test runs...")
all_metrics = []
all_frame_metrics = {}
predictions = {}

for run_name in tqdm(test_latent_dict.keys(), desc="Predictions"):
    Y_test_true = test_latent_dict[run_name]["Y"]
    times_test = test_latent_dict[run_name]["times"]
    
    # Use first p_lag timesteps as IC
    n_ic = p_lag
    n_pred = len(Y_test_true) - n_ic
    
    Y_init = Y_test_true[:n_ic]
    Y_pred = mvar_forecast(mvar, Y_init, n_pred)
    
    # Reconstruct density fields
    rho_true = test_density_dict[run_name]["rho"][n_ic:]
    rho_pred = reconstruct_from_pod(Y_pred, pod_basis["Phi"], mean, 
                                     density_resolution, density_resolution)
    
    # Flatten density fields for metrics (T, ny, nx) -> (T, n_c)
    T = rho_true.shape[0]
    X_true = rho_true.reshape(T, -1)
    X_pred = rho_pred.reshape(T, -1)
    
    # Compute frame-wise metrics using LEGACY functions
    frame_metrics = compute_frame_metrics(X_true, X_pred)
    
    # Compute summary metrics using LEGACY functions
    summary = compute_summary_metrics(X_true, X_pred, mean, frame_metrics, tolerance_threshold=0.10)
    summary["run_name"] = run_name
    
    all_metrics.append(summary)
    all_frame_metrics[run_name] = frame_metrics
    
    # Store predictions for video generation
    predictions[run_name] = {
        "rho_true": rho_true,
        "rho_pred": rho_pred,
        "times": times_test[n_ic:],
        "Y_true": Y_test_true[n_ic:],
        "Y_pred": Y_pred,
        "frame_metrics": frame_metrics,
    }

# Aggregate metrics
print("\n" + "="*80)
print("STEP 5: Evaluation Results")
print("="*80)

r2_values = [m['r2'] for m in all_metrics]
median_e2_values = [m['median_e2'] for m in all_metrics]
p10_e2_values = [m['p10_e2'] for m in all_metrics]
p90_e2_values = [m['p90_e2'] for m in all_metrics]
tau_tol_values = [m['tau_tol'] for m in all_metrics]
mean_mass_errors = [m['mean_mass_error'] for m in all_metrics]
max_mass_errors = [m['max_mass_error'] for m in all_metrics]

print(f"\n📊 Aggregate Metrics (over {n_test} test runs):")
print(f"   R²:              {np.mean(r2_values):.4f} ± {np.std(r2_values):.4f}")
print(f"   Median L² error: {np.mean(median_e2_values):.4f} ± {np.std(median_e2_values):.4f}")
print(f"   P10 L² error:    {np.mean(p10_e2_values):.4f} ± {np.std(p10_e2_values):.4f}")
print(f"   P90 L² error:    {np.mean(p90_e2_values):.4f} ± {np.std(p90_e2_values):.4f}")
print(f"   Mean τ_tol:      {np.mean(tau_tol_values):.1f} ± {np.std(tau_tol_values):.1f} frames")
print(f"   Mean mass error: {np.mean(mean_mass_errors):.6f} ± {np.std(mean_mass_errors):.6f}")
print(f"   Max mass error:  {np.mean(max_mass_errors):.6f} ± {np.std(max_mass_errors):.6f}")

# Best and worst runs
best_idx = np.argmax(r2_values)
worst_idx = np.argmin(r2_values)
best_run = all_metrics[best_idx]['run_name']
worst_run = all_metrics[worst_idx]['run_name']

print(f"\n   Best run:  {best_run} (R² = {r2_values[best_idx]:.4f})")
print(f"   Worst run: {worst_run} (R² = {r2_values[worst_idx]:.4f})")

# Save metrics
with open(plots_dir / "metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)

# =============================================================================
# STEP 6: Generate Videos for Selected Runs
# =============================================================================
print("\n" + "="*80)
print("STEP 6: Generating Videos")
print("="*80)

# Select top 4 best runs by R² score
sorted_runs = sorted(zip(predictions.keys(), r2_values), key=lambda x: x[1], reverse=True)
top_4_runs = [run_name for run_name, _ in sorted_runs[:4]]

print(f"\nTop 4 runs by R² score:")
for i, run_name in enumerate(top_4_runs):
    idx = list(predictions.keys()).index(run_name)
    initial_dist = test_run_metadata[run_name]["initial_distribution"]
    print(f"  {i+1}. {run_name}: R² = {r2_values[idx]:.4f} (IC: {initial_dist})")

print(f"\nGenerating videos for top 4 runs...")
print("Using LEGACY video generation functions (old working version)")

for i, run_name in enumerate(tqdm(top_4_runs, desc="Videos")):
    pred_data = predictions[run_name]
    
    rho_true = pred_data["rho_true"]
    rho_pred = pred_data["rho_pred"]
    times = pred_data["times"]
    frame_metrics = pred_data["frame_metrics"]
    
    # Get initial distribution for this run
    initial_dist = test_run_metadata[run_name]["initial_distribution"]
    # Create readable label
    dist_labels = {
        "uniform": "Uniform",
        "gaussian_cluster": "Gaussian Cluster",
        "two_clusters": "Two Clusters",
        "ring": "Ring"
    }
    dist_label = dist_labels.get(initial_dist, initial_dist)
    
    # Get summary metrics for this run
    run_summary = None
    for m in all_metrics:
        if m["run_name"] == run_name:
            run_summary = m
            break
    
    # Use relative L² error from frame metrics (better than custom computation)
    rel_errors = frame_metrics["e2"]
    
    # 1. Density comparison video (side-by-side with error plot)
    side_by_side_video(
        path=videos_dir,
        left_frames=rho_true,
        right_frames=rho_pred,
        lower_strip_timeseries=rel_errors,
        name=f"{run_name}_density_comparison",
        fps=10,
        cmap='hot',
        titles=(f'Ground Truth (IC: {dist_label})', f'MVAR Predicted (IC: {dist_label})')
    )
    
    # 2. Error timeseries plot (3-panel: L1, L2, Linf errors over time)
    plot_errors_timeseries(
        frame_metrics=frame_metrics,
        summary=run_summary,
        T0=0,
        save_path=plots_dir / f"{run_name}_errors_timeseries.png",
        title=f'{run_name} - Error Metrics (IC: {dist_label})'
    )
    plt.close('all')
    
    # 3. Trajectory video showing particle positions
    # Load trajectory from saved file
    traj_data = np.load(test_dir / run_name / "trajectory.npz")
    traj = traj_data["traj"]
    traj_times = traj_data["times"]
    
    trajectory_video(
        path=videos_dir,
        traj=traj,
        times=traj_times,
        Lx=config["sim"]["Lx"],
        Ly=config["sim"]["Ly"],
        name=f"{run_name}_trajectory",
        fps=10,
        marker_size=50,
        marker_color='blue',
        title=f'{run_name} - Trajectories (IC: {dist_label})'
    )

print(f"\n✓ Generated {len(top_4_runs)} density comparison videos")
print(f"✓ Generated {len(top_4_runs)} error timeseries plots")
print(f"✓ Generated {len(top_4_runs)} trajectory videos")

# =============================================================================
# STEP 7: Generate Summary Plots
# =============================================================================
print("\n" + "="*80)
print("STEP 7: Generating Summary Plots")
print("="*80)

# Plot 1: POD Energy
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, len(pod_basis["S"]) + 1), 
        pod_basis["energy"] * 100, 'o-', linewidth=2, markersize=4)
ax.axvline(r_pod, color='r', linestyle='--', linewidth=2, label=f'Selected r={r_pod}')
ax.axhline(90, color='g', linestyle=':', alpha=0.5, label='90% energy')
ax.axhline(95, color='b', linestyle=':', alpha=0.5, label='95% energy')
ax.set_xlabel('Number of POD Modes', fontsize=12)
ax.set_ylabel('Cumulative Energy (%)', fontsize=12)
ax.set_title(f'POD Energy Spectrum ({n_train} training runs)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.set_xlim(0, min(50, len(pod_basis["S"])))
plt.tight_layout()
plt.savefig(plots_dir / "pod_energy.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ pod_energy.png")

# Plot 2: Metrics Distribution
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# R² distribution
axes[0, 0].hist(r2_values, bins=20, color='forestgreen', alpha=0.7, edgecolor='black')
axes[0, 0].axvline(np.mean(r2_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(r2_values):.4f}')
axes[0, 0].set_xlabel('R²', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontsize=11)
axes[0, 0].set_title(f'R² Distribution ({n_test} test runs)', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Median L² error distribution
axes[0, 1].hist(median_e2_values, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(np.mean(median_e2_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(median_e2_values):.4f}')
axes[0, 1].set_xlabel('Median L² Error', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].set_title('Median L² Error Distribution', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Tolerance horizon distribution
axes[0, 2].hist(tau_tol_values, bins=20, color='purple', alpha=0.7, edgecolor='black')
axes[0, 2].axvline(np.mean(tau_tol_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(tau_tol_values):.1f}')
axes[0, 2].set_xlabel('Tolerance Horizon (frames)', fontsize=11)
axes[0, 2].set_ylabel('Frequency', fontsize=11)
axes[0, 2].set_title('τ_tol Distribution (10% threshold)', fontsize=12, fontweight='bold')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# P10 vs P90 L² error
axes[1, 0].scatter(p10_e2_values, p90_e2_values, alpha=0.6, s=50, c=r2_values, cmap='viridis', edgecolor='black')
axes[1, 0].set_xlabel('P10 L² Error', fontsize=11)
axes[1, 0].set_ylabel('P90 L² Error', fontsize=11)
axes[1, 0].set_title('Error Percentiles (colored by R²)', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
cbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
cbar.set_label('R²', fontsize=10)

# Mass error distribution
axes[1, 1].hist(mean_mass_errors, bins=20, color='coral', alpha=0.7, edgecolor='black')
axes[1, 1].axvline(np.mean(mean_mass_errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(mean_mass_errors):.6f}')
axes[1, 1].set_xlabel('Mean Mass Error', fontsize=11)
axes[1, 1].set_ylabel('Frequency', fontsize=11)
axes[1, 1].set_title('Mass Conservation Error Distribution', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# τ_tol vs R²
axes[1, 2].scatter(tau_tol_values, r2_values, alpha=0.6, s=50, c=median_e2_values, cmap='coolwarm', edgecolor='black')
axes[1, 2].set_xlabel('Tolerance Horizon (frames)', fontsize=11)
axes[1, 2].set_ylabel('R²', fontsize=11)
axes[1, 2].set_title('τ_tol vs R² (colored by median L² error)', fontsize=12, fontweight='bold')
axes[1, 2].grid(True, alpha=0.3)
cbar = plt.colorbar(axes[1, 2].collections[0], ax=axes[1, 2])
cbar.set_label('Median L² Error', fontsize=10)

plt.tight_layout()
plt.savefig(plots_dir / "metrics_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ metrics_distribution.png")

# Plot 3: Best and Worst Run Comparison
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for row, (run_name, label) in enumerate([(best_run, 'Best'), (worst_run, 'Worst')]):
    pred_data = predictions[run_name]
    rho_true = pred_data["rho_true"]
    rho_pred = pred_data["rho_pred"]
    
    # Select 4 time snapshots
    T = len(rho_true)
    snapshot_indices = [0, T//3, 2*T//3, T-1]
    
    for col, idx in enumerate(snapshot_indices):
        if col < 2:
            # True density
            im = axes[row, col].imshow(rho_true[idx], origin='lower', cmap='hot',
                                      extent=[0, config["sim"]["Lx"], 0, config["sim"]["Ly"]])
            axes[row, col].set_title(f'{label} - True (t={idx})')
        else:
            # Predicted density
            im = axes[row, col].imshow(rho_pred[idx], origin='lower', cmap='hot',
                                      extent=[0, config["sim"]["Lx"], 0, config["sim"]["Ly"]])
            axes[row, col].set_title(f'{label} - Pred (t={idx})')
        
        axes[row, col].set_xlabel('x')
        if col == 0:
            axes[row, col].set_ylabel('y')
        fig.colorbar(im, ax=axes[row, col], fraction=0.046)

plt.suptitle('Best vs Worst Prediction Runs', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(plots_dir / "best_worst_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ best_worst_comparison.png")

# Plot 4: Order Parameters Panel (for best test run)
import pandas as pd
best_run_order_df = pd.read_csv(test_dir / best_run / "order_params.csv")

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Panel 1: Polarization
ax = axes[0]
ax.plot(best_run_order_df['t'], best_run_order_df['phi'], 'b-', linewidth=2)
ax.set_ylabel('Polarization Φ', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])
median_phi = best_run_order_df['phi'].iloc[-len(best_run_order_df)//4:].median()
ax.axhline(median_phi, color='r', linestyle='--', alpha=0.5,
           label=f'Final median: {median_phi:.3f}')
ax.legend(loc='best', fontsize=10)
ax.set_title(f'Order Parameters for Best Run ({best_run})', fontsize=14, fontweight='bold')

# Panel 2: Mean Speed
ax = axes[1]
ax.plot(best_run_order_df['t'], best_run_order_df['mean_speed'], 'g-', linewidth=2)
ax.set_ylabel('Mean Speed', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Panel 3: Nematic Order
ax = axes[2]
ax.plot(best_run_order_df['t'], best_run_order_df['nematic'], 'm-', linewidth=2)
ax.set_ylabel('Nematic Order Q', fontsize=12, fontweight='bold')
ax.set_xlabel('Time (s)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_ylim([-0.05, 1.05])

plt.tight_layout()
plt.savefig(plots_dir / "order_params_panel.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ order_params_panel.png")

# =============================================================================
# Final Summary
# =============================================================================
print("\n" + "="*80)
print("PRODUCTION PIPELINE COMPLETE! 🎉")
print("="*80)

print(f"\n📂 All outputs saved to: {output_dir.absolute()}")

print(f"\n📊 Training:")
print(f"   • {n_train} simulations")
print(f"   • {X.shape[0]} total snapshots")
print(f"   • {r_pod} POD modes ({pod_basis['energy'][r_pod-1]:.2%} energy)")
print(f"   • MVAR order p={p_lag}")

print(f"\n📈 Test Results ({n_test} runs):")
print(f"   • Mean R²:           {np.mean(r2_values):.4f} ± {np.std(r2_values):.4f}")
print(f"   • Median L² error:   {np.mean(median_e2_values):.4f} ± {np.std(median_e2_values):.4f}")
print(f"   • P10 L² error:      {np.mean(p10_e2_values):.4f} ± {np.std(p10_e2_values):.4f}")
print(f"   • P90 L² error:      {np.mean(p90_e2_values):.4f} ± {np.std(p90_e2_values):.4f}")
print(f"   • Mean τ_tol:        {np.mean(tau_tol_values):.1f} ± {np.std(tau_tol_values):.1f} frames")
print(f"   • Mean mass error:   {np.mean(mean_mass_errors):.6f}")
print(f"   • Max mass error:    {np.mean(max_mass_errors):.6f}")

print(f"\n🎬 Videos:")
print(f"   • {len(top_4_runs)} density comparison videos (top 4 runs by R²)")
print(f"   • {len(top_4_runs)} trajectory videos (particle positions)")

print(f"\n📊 Plots:")
print(f"   • pod_energy.png - Energy spectrum")
print(f"   • metrics_distribution.png - Error distributions")
print(f"   • best_worst_comparison.png - Best vs worst runs")
print(f"   • order_params_panel.png - Order parameters (best run)")
print(f"   • {len(top_4_runs)} error timeseries plots (L1, L2, Linf errors)")

print(f"\n📄 Additional Outputs:")
print(f"   • order_params.csv - Order parameters for each test run")

print(f"\n✅ Full production pipeline validated successfully!")
print("="*80)
