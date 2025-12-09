"""
Test Evaluator Module
=====================

Evaluates ROM-MVAR model on test data and saves predictions.
Supports:
- Time-resolved R² analysis
- Multiple R² metrics (reconstructed, latent, POD)
- Mass conservation tracking
"""

import numpy as np
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm


def evaluate_test_runs(
    test_dir, n_test, base_config_test, pod_data, mvar_model, 
    density_nx, density_ny, rom_subsample, eval_config
):
    """
    Evaluate ROM-MVAR model on all test runs.
    
    Parameters
    ----------
    test_dir : Path
        Directory containing test runs
    n_test : int
        Number of test runs
    base_config_test : dict
        Test simulation configuration
    pod_data : dict
        POD data from build_pod_basis
    mvar_model : sklearn model
        Trained MVAR model
    density_nx, density_ny : int
        Density grid resolution
    rom_subsample : int
        Temporal subsampling factor
    eval_config : dict
        Evaluation configuration with keys:
        - save_time_resolved: whether to save R² vs time
        - forecast_start: forecast period start time
        - forecast_end: forecast period end time
    
    Returns
    -------
    pd.DataFrame
        Test results with R² metrics for each run
    """
    
    test_dir = Path(test_dir)
    
    # Extract POD data
    U_r = pod_data['U_r']
    X_mean = pod_data['X_mean']
    R_POD = pod_data['R_POD']
    P_LAG = mvar_model.n_features_in_ // R_POD  # Infer lag from model
    
    # Determine evaluation mode
    save_time_resolved = eval_config.get('save_time_resolved', False)
    forecast_start = eval_config.get('forecast_start', base_config_test['sim']['T'])
    forecast_end = eval_config.get('forecast_end', base_config_test['sim']['T'])
    
    if save_time_resolved:
        print(f"\nTime-resolved evaluation enabled:")
        print(f"   Forecast period: t={forecast_start}s to t={forecast_end}s")
    
    print(f"\nEvaluating {n_test} test runs...")
    
    # Create spatial grids for density output (needed for visualization)
    xgrid = np.linspace(0, base_config_test['sim']['Lx'], density_nx, endpoint=False) + base_config_test['sim']['Lx']/(2*density_nx)
    ygrid = np.linspace(0, base_config_test['sim']['Ly'], density_ny, endpoint=False) + base_config_test['sim']['Ly']/(2*density_ny)
    
    # Evaluation loop
    test_results = []
    
    for test_idx in tqdm(range(n_test), desc="Evaluating"):
        test_run_dir = test_dir / f"test_{test_idx:03d}"
        
        # Load test density (now from density_true.npz)
        test_data = np.load(test_run_dir / "density_true.npz")
        test_density = test_data['rho']  # Changed from 'density' to 'rho'
        test_times = test_data['times']
        
        # Subsample if needed
        if rom_subsample > 1:
            test_density = test_density[::rom_subsample]
            test_times = test_times[::rom_subsample]
        
        T_test = test_density.shape[0]
        test_density_flat = test_density.reshape(T_test, -1)
        
        # Project to latent space
        test_centered = test_density_flat - X_mean
        test_latent = test_centered @ U_r
        
        # Determine initial condition window
        T_train = int(forecast_start / base_config_test['sim']['dt'] / rom_subsample)
        
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
        pred_physical = pred_physical.reshape(-1, density_nx, density_ny)
        
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
        true_reconstructed = true_reconstructed.reshape(-1, density_nx, density_ny)
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
            r2_vs_time = _compute_time_resolved_r2(
                true_physical, pred_physical,
                true_latent, pred_latent,
                true_reconstructed, forecast_times
            )
            
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
    test_results_df.to_csv(test_dir / "test_results.csv", index=False)
    
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
    print(f"\nDetailed results: {test_dir}/test_results.csv")
    
    if save_time_resolved:
        print(f"Time-resolved R²: {test_dir}/test_*/r2_vs_time.csv")
    
    return test_results_df


def _compute_time_resolved_r2(true_physical, pred_physical, true_latent, pred_latent, 
                              true_reconstructed, forecast_times):
    """Compute time-resolved R² metrics."""
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
    
    return r2_vs_time
