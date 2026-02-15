"""
Metrics Computation Module
===========================

Computes evaluation metrics for all test runs.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from rectsim.legacy_functions import compute_frame_metrics, compute_summary_metrics


def compute_test_metrics(test_metadata, test_dir, x_train_mean, ic_types, output_dir, model_name='mvar'):
    """
    Compute evaluation metrics for all test runs.
    
    Parameters
    ----------
    test_metadata : list
        List of test run metadata dictionaries
    test_dir : Path
        Directory containing test data
    x_train_mean : ndarray
        Training data mean
    ic_types : list
        List of IC type names
    output_dir : Path
        Directory to save metrics
    model_name : str, optional
        Name of the model ('mvar' or 'lstm'), used for file naming
    
    Returns
    -------
    metrics_df : DataFrame
        Metrics for all runs
    test_predictions : dict
        Dictionary storing predictions for each run
    ic_metrics : dict
        Aggregated metrics by IC type
    """
    
    test_dir = Path(test_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    all_metrics = []
    test_predictions = {}
    
    # Determine IC key name from metadata
    ic_key = 'ic_type' if 'ic_type' in test_metadata[0] else 'distribution'
    
    print(f"\nComputing metrics for {len(test_metadata)} test runs...")
    
    for meta in tqdm(test_metadata, desc="Metrics"):
        run_name = meta["run_name"]
        run_dir = test_dir / run_name
        
        # Load densities
        true_data = np.load(run_dir / "density_true.npz")
        
        # Load model-specific predictions
        pred_file = run_dir / f"density_pred_{model_name}.npz"
        if not pred_file.exists():
            # Fallback to generic density_pred.npz for backward compatibility
            pred_file = run_dir / "density_pred.npz"
        
        pred_data = np.load(pred_file)
        
        rho_true = true_data["rho"]
        rho_pred = pred_data["rho"]
        times_true = true_data["times"]
        times_pred = pred_data["times"]
        
        # Handle case where predictions only cover forecast period (not full time range)
        # Align by using only the overlapping time range
        T_true_full = rho_true.shape[0]
        T_pred = rho_pred.shape[0]
        
        # Find the time index in true data where predictions start
        pred_start_time = times_pred[0]
        true_start_idx = np.argmin(np.abs(times_true - pred_start_time))
        
        # Slice true data to match prediction period
        rho_true = rho_true[true_start_idx:true_start_idx+T_pred]
        times = times_true[true_start_idx:true_start_idx+T_pred]
        
        # Now they should be aligned
        assert rho_true.shape[0] == T_pred, f"Shape mismatch after alignment: {rho_true.shape[0]} vs {T_pred}"
        
        # Flatten for metrics
        T, ny, nx = rho_true.shape
        rho_true_flat = rho_true.reshape(T, ny * nx)
        rho_pred_flat = rho_pred.reshape(T, ny * nx)
        
        # Compute metrics
        frame_metrics = compute_frame_metrics(rho_true_flat, rho_pred_flat)
        summary = compute_summary_metrics(
            rho_true_flat,
            rho_pred_flat,
            x_train_mean,
            frame_metrics
        )
        
        # Add RMSE (root mean squared error)
        mse = np.mean((rho_true_flat - rho_pred_flat) ** 2)
        summary["rmse"] = np.sqrt(mse)
        
        summary["run_name"] = run_name
        summary["ic_type"] = meta[ic_key]
        
        # Load trajectory for later use
        traj_data = np.load(run_dir / "trajectory.npz")
        traj = traj_data["traj"]
        vel = traj_data["vel"] if "vel" in traj_data else None
        traj_times = traj_data["times"]
        
        # Align trajectory to the prediction time window
        # The trajectory covers the full simulation, but predictions only cover the forecast period
        # Find the trajectory indices that match the prediction times
        if traj.shape[0] != T_pred:
            # First try: align by matching time ranges (pred starts at forecast_start)
            pred_start_time = times[0]
            traj_start_idx = np.argmin(np.abs(traj_times - pred_start_time))
            traj = traj[traj_start_idx:traj_start_idx + T_pred]
            if vel is not None:
                vel = vel[traj_start_idx:traj_start_idx + T_pred]
            traj_times = traj_times[traj_start_idx:traj_start_idx + T_pred]
            
            # If still mismatched (e.g., due to ROM subsampling), subsample to match
            if traj.shape[0] != T_pred:
                subsample_factor = max(1, traj.shape[0] // T_pred)
                traj = traj[::subsample_factor][:T_pred]
                if vel is not None:
                    vel = vel[::subsample_factor][:T_pred]
                traj_times = traj_times[::subsample_factor][:T_pred]
        
        # Store for visualization
        test_predictions[run_name] = {
            "rho_true": rho_true,
            "rho_pred": rho_pred,
            "times": times,
            "traj": traj,
            "vel": vel,
            "ic_type": meta[ic_key],
            "frame_metrics": frame_metrics
        }
        
        all_metrics.append(summary)
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    # Save with model-specific name
    metrics_csv = output_dir / f"metrics_all_runs_{model_name}.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    
    # Overall metrics
    print(f"\n📊 Overall Metrics ({model_name.upper()}):")
    print(f"   R²:              {metrics_df['r2'].mean():.4f} ± {metrics_df['r2'].std():.4f}")
    print(f"   Median L² error: {metrics_df['median_e2'].mean():.4f} ± {metrics_df['median_e2'].std():.4f}")
    
    # Metrics by IC type
    print(f"\n📊 Metrics by IC Type ({model_name.upper()}):")
    ic_metrics = {}
    
    for ic_type in ic_types:
        ic_mask = metrics_df["ic_type"] == ic_type
        ic_data = metrics_df[ic_mask]
        
        if len(ic_data) == 0:
            continue
        
        ic_stats = {
            "ic_type": ic_type,
            "n_runs": len(ic_data),
            "mean_r2": ic_data["r2"].mean(),
            "std_r2": ic_data["r2"].std(),
            "median_r2": ic_data["r2"].median(),
            "mean_rmse": ic_data["rmse"].mean(),
            "best_run": ic_data.loc[ic_data["r2"].idxmax(), "run_name"],
            "best_r2": ic_data["r2"].max(),
        }
        
        ic_metrics[ic_type] = ic_stats
        
        print(f"\n   {ic_type}:")
        print(f"      Runs: {ic_stats['n_runs']}")
        print(f"      R²: {ic_stats['mean_r2']:.4f} ± {ic_stats['std_r2']:.4f}")
        print(f"      Best run: {ic_stats['best_run']} (R² = {ic_stats['best_r2']:.4f})")
    
    # Save IC metrics with model-specific name
    ic_csv = output_dir / f"metrics_by_ic_type_{model_name}.csv"
    pd.DataFrame(ic_metrics.values()).to_csv(ic_csv, index=False)
    
    return metrics_df, test_predictions, ic_metrics
