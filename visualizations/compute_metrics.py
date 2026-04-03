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

    status_lookup = {}
    if model_name == 'wsindy':
        status_csv = test_dir.parent / "WSINDy" / "test_results.csv"
        if status_csv.exists():
            try:
                status_df = pd.read_csv(status_csv)
                if "run_name" in status_df.columns:
                    status_lookup = {
                        str(row["run_name"]): row.to_dict()
                        for _, row in status_df.iterrows()
                    }
            except Exception:
                status_lookup = {}
    
    # Determine IC key name from metadata
    ic_key = 'ic_type' if 'ic_type' in test_metadata[0] else 'distribution'
    
    print(f"\nComputing metrics for {len(test_metadata)} test runs...")
    
    skipped_runs = []
    for meta in tqdm(test_metadata, desc="Metrics"):
        run_name = meta["run_name"]
        run_dir = test_dir / run_name
        
        # Load densities — skip incomplete test runs gracefully
        true_file = run_dir / "density_true.npz"
        pred_file = run_dir / f"density_pred_{model_name}.npz"
        if not pred_file.exists():
            # Fallback to generic density_pred.npz for backward compatibility
            pred_file = run_dir / "density_pred.npz"
        
        if not true_file.exists() or not pred_file.exists():
            status_row = status_lookup.get(run_name)
            if status_row is not None and str(status_row.get("forecast_status", "")) == "failed":
                all_metrics.append({
                    "run_name": run_name,
                    "ic_type": meta[ic_key],
                    "r2": np.nan,
                    "rmse": np.nan,
                    "forecast_status": "failed",
                    "failure_reason": status_row.get("failure_reason"),
                    "failure_step": status_row.get("failure_step"),
                    "forecast_method_attempted": status_row.get("forecast_method_attempted"),
                    "forecast_method_used": status_row.get("forecast_method_used"),
                })
                continue
            skipped_runs.append(run_name)
            continue
        
        try:
            true_data = np.load(true_file)
            pred_data = np.load(pred_file)
        except Exception as e:
            print(f"\n  ⚠ Skipping {run_name}: {e}")
            skipped_runs.append(run_name)
            continue
        
        rho_true = true_data["rho"]
        rho_pred = pred_data["rho"]
        times_true = true_data["times"]
        times_pred = pred_data["times"]
        
        # Extract forecast_start_idx if available (stored by test_evaluator)
        forecast_start_idx = int(pred_data["forecast_start_idx"]) if "forecast_start_idx" in pred_data else None
        
        # ---- Fix temporal alignment ----
        # density_true.npz is at raw sim dt (e.g. 0.04s)
        # density_pred.npz is at ROM dt (e.g. 0.12s = dt * rom_subsample)
        # We must subsample truth to ROM resolution before frame-by-frame alignment.
        dt_true = float(times_true[1] - times_true[0]) if len(times_true) > 1 else 1.0
        dt_pred = float(times_pred[1] - times_pred[0]) if len(times_pred) > 1 else dt_true
        rom_subsample = max(1, round(dt_pred / dt_true))
        
        if rom_subsample > 1:
            rho_true = rho_true[::rom_subsample]
            times_true = times_true[::rom_subsample]
        
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
        assert rho_true.shape[0] == T_pred, (
            f"Shape mismatch after alignment: truth {rho_true.shape[0]} vs pred {T_pred}. "
            f"rom_subsample={rom_subsample}, true_start_idx={true_start_idx}"
        )
        
        # Flatten for metrics
        T, ny, nx = rho_true.shape
        rho_true_flat = rho_true.reshape(T, ny * nx)
        rho_pred_flat = rho_pred.reshape(T, ny * nx)
        
        # ---- Compute metrics on FORECAST region only ----
        # Most ROM outputs save conditioning + forecast and store the forecast
        # boundary relative to the saved array. WSINDy exports are forecast-only
        # in practice, but some files still carry an absolute forecast_start_idx
        # from the original simulation. Convert to a relative index after
        # alignment so videos and forecast-only metrics slice the correct region.
        if forecast_start_idx is not None:
            fsi = max(0, int(forecast_start_idx) - int(true_start_idx))
            fsi = min(fsi, T_pred)
        else:
            fsi = 0
        rho_true_fc = rho_true_flat[fsi:]
        rho_pred_fc = rho_pred_flat[fsi:]
        
        # Frame metrics on forecast region only
        frame_metrics_fc = compute_frame_metrics(rho_true_fc, rho_pred_fc)
        summary = compute_summary_metrics(
            rho_true_fc,
            rho_pred_fc,
            x_train_mean,
            frame_metrics_fc
        )
        
        # Also compute full-trajectory frame metrics for visualization
        # (the video shows conditioning + forecast, so we need full metrics)
        frame_metrics = compute_frame_metrics(rho_true_flat, rho_pred_flat)
        
        # Add RMSE (root mean squared error) — forecast only
        mse = np.mean((rho_true_fc - rho_pred_fc) ** 2)
        summary["rmse"] = np.sqrt(mse)
        
        summary["run_name"] = run_name
        summary["ic_type"] = meta[ic_key]
        summary["forecast_start_idx"] = fsi
        summary["T_conditioning"] = fsi
        summary["T_forecast"] = T - fsi
        if run_name in status_lookup:
            status_row = status_lookup[run_name]
            summary["forecast_status"] = status_row.get("forecast_status", "ok")
            summary["failure_reason"] = status_row.get("failure_reason")
            summary["failure_step"] = status_row.get("failure_step")
            summary["forecast_method_attempted"] = status_row.get("forecast_method_attempted")
            summary["forecast_method_used"] = status_row.get("forecast_method_used")
        else:
            summary["forecast_status"] = "ok"

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
            "frame_metrics": frame_metrics,          # Full trajectory (for video)
            "frame_metrics_fc": frame_metrics_fc,     # Forecast-only (for error plots)
            "forecast_start_idx": fsi,
        }
        
        all_metrics.append(summary)
    
    if skipped_runs:
        print(f"\n  ⚠ Skipped {len(skipped_runs)}/{len(test_metadata)} incomplete test runs: {', '.join(skipped_runs[:5])}{'...' if len(skipped_runs) > 5 else ''}")
    
    if len(all_metrics) == 0:
        print("  ✗ No complete test runs found — cannot compute metrics")
        return pd.DataFrame(), {}, {}
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    # Save with model-specific name
    metrics_csv = output_dir / f"metrics_all_runs_{model_name}.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    
    # Overall metrics
    successful_df = metrics_df
    if "forecast_status" in metrics_df.columns:
        successful_df = metrics_df[metrics_df["forecast_status"] != "failed"]

    print(f"\n📊 Overall Metrics ({model_name.upper()}):")
    if len(successful_df) == 0:
        print("   No successful forecasts available")
    else:
        print(f"   R²:              {successful_df['r2'].mean():.4f} ± {successful_df['r2'].std():.4f}")
        print(f"   Median L² error: {successful_df['median_e2'].mean():.4f} ± {successful_df['median_e2'].std():.4f}")
    
    # Metrics by IC type
    print(f"\n📊 Metrics by IC Type ({model_name.upper()}):")
    ic_metrics = {}
    
    for ic_type in ic_types:
        ic_mask = metrics_df["ic_type"] == ic_type
        ic_data = metrics_df[ic_mask]
        ic_success = ic_data
        if "forecast_status" in ic_data.columns:
            ic_success = ic_data[ic_data["forecast_status"] != "failed"]

        if len(ic_data) == 0:
            continue

        ic_stats = {
            "ic_type": ic_type,
            "n_runs": len(ic_data),
            "n_success": len(ic_success),
            "n_failed": len(ic_data) - len(ic_success),
            "mean_r2": ic_success["r2"].mean() if len(ic_success) else np.nan,
            "std_r2": ic_success["r2"].std() if len(ic_success) else np.nan,
            "median_r2": ic_success["r2"].median() if len(ic_success) else np.nan,
            "mean_rmse": ic_success["rmse"].mean() if len(ic_success) else np.nan,
            "best_run": ic_success.loc[ic_success["r2"].idxmax(), "run_name"] if len(ic_success) else None,
            "best_r2": ic_success["r2"].max() if len(ic_success) else np.nan,
        }
        
        ic_metrics[ic_type] = ic_stats
        
        print(f"\n   {ic_type}:")
        print(f"      Runs: {ic_stats['n_runs']}")
        print(f"      Success: {ic_stats['n_success']}  Failed: {ic_stats['n_failed']}")
        if ic_stats["n_success"] > 0:
            print(f"      R²: {ic_stats['mean_r2']:.4f} ± {ic_stats['std_r2']:.4f}")
            print(f"      Best run: {ic_stats['best_run']} (R² = {ic_stats['best_r2']:.4f})")
        else:
            print("      No successful forecasts")
    
    # Save IC metrics with model-specific name
    ic_csv = output_dir / f"metrics_by_ic_type_{model_name}.csv"
    pd.DataFrame(ic_metrics.values()).to_csv(ic_csv, index=False)
    
    return metrics_df, test_predictions, ic_metrics
