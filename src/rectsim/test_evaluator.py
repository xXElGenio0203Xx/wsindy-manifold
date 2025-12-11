"""
Test Evaluator Module
=====================

Evaluates ROM-MVAR model on test data and saves predictions.
Supports:
- Time-resolved R² analysis
- Multiple R² metrics (reconstructed, latent, POD)
- Mass conservation tracking
- Order parameter computation
"""

import numpy as np
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
from rectsim.standard_metrics import compute_metrics_series


def evaluate_test_runs(
    test_dir, n_test, base_config_test, pod_data, forecast_fn, lag,
    density_nx, density_ny, rom_subsample, eval_config, train_T=None,
    model_name="ROM"
):
    """
    Evaluate ROM model on all test runs using generic forecast function.
    
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
    forecast_fn : callable
        Forecast function with signature:
        forecast_fn(y_init_window, n_steps) -> ys_pred
        where y_init_window is [lag, d] and ys_pred is [n_steps, d]
    lag : int
        Lookback window size (number of timesteps)
    density_nx, density_ny : int
        Density grid resolution
    rom_subsample : int
        Temporal subsampling factor
    eval_config : dict
        Evaluation configuration with keys:
        - save_time_resolved: whether to save R² vs time
        - forecast_start: forecast period start time (defaults to train_T)
        - forecast_end: forecast period end time (defaults to test_T)
    train_T : float, optional
        Training trajectory duration (used as default forecast_start)
    model_name : str, optional
        Name of model for logging purposes (default: "ROM")
    
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
    P_LAG = lag  # Use provided lag parameter
    
    # Determine evaluation mode
    save_time_resolved = eval_config.get('save_time_resolved', False)
    test_T = base_config_test['sim']['T']
    
    # Default forecast_start to training time (where forecast begins)
    # Default forecast_end to test time (full test trajectory)
    if train_T is None:
        train_T = test_T  # Fallback if not provided
    
    forecast_start = eval_config.get('forecast_start', train_T)
    forecast_end = eval_config.get('forecast_end', test_T)
    
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
            # Pad with zeros if needed
            if len(ic_window) < P_LAG:
                padding = np.zeros((P_LAG - len(ic_window), R_POD))
                ic_window = np.vstack([padding, ic_window])
        else:
            ic_window = test_latent[T_train-P_LAG:T_train]
        
        # Forecast using generic forecast function
        n_forecast_steps = T_test - T_train
        pred_latent = forecast_fn(ic_window, n_forecast_steps)
        
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
        
        # Compute order parameters from trajectory for visualization
        try:
            # Load trajectory (positions + velocities)
            traj_file = test_run_dir / "trajectory.npz"
            if traj_file.exists():
                traj_data = np.load(traj_file)
                x = traj_data['traj']  # Shape: (T, N, 2) - positions
                
                # Extract velocities from trajectory
                if 'vel' in traj_data:
                    v = traj_data['vel']  # Shape: (T, N, 2)
                elif 'velocities' in traj_data:
                    v = traj_data['velocities']
                else:
                    # Compute velocities from positions if not available
                    dt = base_config_test['sim']['dt']
                    v = np.zeros_like(x)
                    v[:-1] = (x[1:] - x[:-1]) / dt
                    v[-1] = v[-2]  # Repeat last velocity
                
                # Domain bounds
                Lx = base_config_test['sim']['Lx']
                Ly = base_config_test['sim']['Ly']
                domain_bounds = (0, Lx, 0, Ly)
                
                # Compute order parameters for all timesteps
                order_params = compute_metrics_series(
                    x, v, domain_bounds, 
                    resolution=density_nx,
                    verbose=False
                )
                
                # Save to CSV with time column
                op_df = pd.DataFrame({
                    't': test_times,
                    'phi': order_params['polarization'],
                    'mean_speed': order_params['mean_speed'],
                    'angular_momentum': order_params['angular_momentum'],
                    'density_variance': order_params['density_variance'],
                    'total_mass': order_params['total_mass']
                })
                op_df.to_csv(test_run_dir / "order_params.csv", index=False)
                
            # ALSO compute density-based metrics for TRUE vs PREDICTED comparison
            # This allows us to compare ground truth vs prediction
            density_variance_true = np.std(test_density, axis=(1, 2))  # Spatial std per timestep
            density_variance_pred_full = np.zeros(T_test)
            density_variance_pred_full[:T_train] = density_variance_true[:T_train]  # Use true during conditioning
            density_variance_pred_full[T_train:] = np.std(pred_physical, axis=(1, 2))  # Use pred during forecast
            
            mass_true = np.sum(test_density, axis=(1, 2))
            mass_pred_full = np.zeros(T_test)
            mass_pred_full[:T_train] = mass_true[:T_train]
            mass_pred_full[T_train:] = np.sum(pred_physical, axis=(1, 2))
            
            # Save density-based comparison metrics
            density_metrics_df = pd.DataFrame({
                't': test_times,
                'density_variance_true': density_variance_true,
                'density_variance_pred': density_variance_pred_full,
                'mass_true': mass_true,
                'mass_pred': mass_pred_full,
            })
            density_metrics_df.to_csv(test_run_dir / "density_metrics.csv", index=False)
            
        except Exception as e:
            # Don't fail evaluation if order params fail
            print(f"   Warning: Could not compute order parameters for test_{test_idx:03d}: {e}")
        
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
