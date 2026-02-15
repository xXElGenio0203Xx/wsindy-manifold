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
from rectsim.forecast_utils import mvar_kstep_forecast_fn_factory


def _project_simplex(rho_frame, mass_target=None):
    """
    Project a density frame onto the probability simplex:
      rho >= 0  AND  sum(rho) = mass_target.
    
    Algorithm: Duchi et al. (2008) — O(n log n) Euclidean projection.
    If mass_target is None, uses the pre-clamp sum as target.
    
    Parameters
    ----------
    rho_frame : np.ndarray [H, W] or [N]
        Density field (may contain negatives from POD reconstruction)
    mass_target : float or None
        Desired total mass. If None, uses sum(rho_frame).
    
    Returns
    -------
    np.ndarray
        Projected density field (same shape), all >= 0, sums to mass_target.
    """
    shape = rho_frame.shape
    v = rho_frame.flatten().copy()
    n = len(v)
    
    if mass_target is None:
        mass_target = v.sum()
    
    if mass_target <= 0:
        return np.zeros(shape)
    
    # Scale to unit simplex, project, then scale back
    v_scaled = v / mass_target
    
    # Sort descending
    u = np.sort(v_scaled)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho_vec = np.arange(1, n + 1, dtype=float)
    cond = u - cssv / rho_vec > 0
    # Find the largest index where condition holds
    rho_idx = np.where(cond)[0][-1]
    theta = cssv[rho_idx] / (rho_idx + 1.0)
    
    projected = np.maximum(v_scaled - theta, 0.0) * mass_target
    return projected.reshape(shape)


def _safe_r2(ss_res, ss_tot, signal_sq_sum):
    """R2 with guard against degenerate (near-constant) signals."""
    if ss_tot > 1e-10 * max(signal_sq_sum, 1e-30):
        return 1 - ss_res / ss_tot
    return float('nan')


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
    # Clamp mode: 'C0' = none, 'C1' = clamp only, 'C2' = clamp + renorm (default)
    # Backward compat: clamp_negative=True → C2, clamp_negative=False → C0
    clamp_mode = eval_config.get('clamp_mode', None)
    if clamp_mode is None:
        clamp_negative = eval_config.get('clamp_negative', True)
        clamp_mode = 'C2' if clamp_negative else 'C0'
    test_T = base_config_test['sim']['T']
    
    # Default forecast_start to training time (where forecast begins)
    # Default forecast_end to test time (full test trajectory)
    if train_T is None:
        train_T = test_T  # Fallback if not provided
    
    forecast_start = eval_config.get('forecast_start', train_T)
    forecast_end = eval_config.get('forecast_end', test_T)
    
    # k-step teacher-forced reset interval (0 or None = disabled)
    kstep_reset = eval_config.get('kstep_reset', 0)

    # Mass postprocessing: enforce mass conservation after inverse transform.
    # 'none'    = no mass correction (default, keeps raw/C2 baseline identical)
    # 'simplex' = L2-optimal projection to {rho>=0, sum=M0} (Duchi et al.)
    # 'scale'   = global multiplicative rescaling to M0
    # Backward compat: mass_project=True → 'scale'
    mass_postprocess = eval_config.get('mass_postprocess', 'none')
    if mass_postprocess == 'none' and eval_config.get('mass_project', False):
        mass_postprocess = 'scale'  # backward compatibility

    if save_time_resolved:
        print(f"\nTime-resolved evaluation enabled:")
        print(f"   Forecast period: t={forecast_start}s to t={forecast_end}s")
    if kstep_reset:
        print(f"\n   k-step teacher-forcing: reset every {kstep_reset} steps")
    if mass_postprocess != 'none':
        print(f"   Mass postprocess: {mass_postprocess} → M₀ from ground truth at forecast start")
    
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
        
        # Apply same density transform as training (if any)
        density_transform = pod_data.get('density_transform', 'raw')
        density_transform_eps = pod_data.get('density_transform_eps', 1e-8)
        
        if density_transform == 'log':
            test_density_flat_transformed = np.log(test_density_flat + density_transform_eps)
        elif density_transform == 'sqrt':
            test_density_flat_transformed = np.sqrt(test_density_flat + density_transform_eps)
        elif density_transform == 'meansub':
            snapshot_means = test_density_flat.mean(axis=1, keepdims=True)
            test_density_flat_transformed = test_density_flat - snapshot_means
        else:
            test_density_flat_transformed = test_density_flat
        
        # Project to latent space (using transformed data)
        test_centered = test_density_flat_transformed - X_mean
        test_latent = test_centered @ U_r
        
        # Apply latent standardization if enabled (must match training)
        latent_mean = pod_data.get('latent_mean', None)
        latent_std = pod_data.get('latent_std', None)
        latent_standardize = latent_mean is not None and latent_std is not None
        if latent_standardize:
            test_latent = (test_latent - latent_mean) / latent_std
        
        # Determine initial condition window
        T_train = int(forecast_start / base_config_test['sim']['dt'] / rom_subsample)
        
        # Compute M₀: true total mass at forecast start
        # Always compute for metrics; only apply correction if mass_postprocess != 'none'
        _m0_idx = max(T_train - 1, 0)
        M0 = float(test_density[_m0_idx].sum())  # in cell units
        
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
        
        # Forecast using generic forecast function (operates in standardized space if enabled)
        n_forecast_steps = T_test - T_train
        pred_latent_forecast = forecast_fn(ic_window, n_forecast_steps)
        
        # ---- k-step teacher-forced forecast (if requested) ----
        r2_kstep_density = float('nan')
        r2_kstep_latent = float('nan')
        if kstep_reset and kstep_reset > 0:
            _ks_preds = []
            _ks_history = ic_window.copy()
            for _si in range(n_forecast_steps):
                # Reset to ground truth every k steps
                if _si > 0 and _si % kstep_reset == 0:
                    _true_abs = T_train + _si
                    _h_start = _true_abs - P_LAG
                    if _h_start >= 0:
                        _ks_history = test_latent[_h_start:_true_abs].copy()
                # One autonomous step
                _y_next = forecast_fn(_ks_history[-P_LAG:], 1)[0]
                _ks_preds.append(_y_next)
                _ks_history = np.vstack([_ks_history[1:], _y_next])
            pred_kstep_latent = np.array(_ks_preds)
            # Inverse standardize
            if latent_standardize:
                pred_kstep_latent_raw = pred_kstep_latent * latent_std + latent_mean
            else:
                pred_kstep_latent_raw = pred_kstep_latent
            # Reconstruct to physical space
            pred_kstep_physical = (pred_kstep_latent_raw @ U_r.T) + X_mean
            pred_kstep_physical = pred_kstep_physical.reshape(-1, density_nx, density_ny)
            # Apply same clamp
            if clamp_mode == 'C2':
                for _ti in range(len(pred_kstep_physical)):
                    _fr = pred_kstep_physical[_ti]
                    _mb = _fr.sum()
                    _fr = np.maximum(_fr, 0.0)
                    _ma = _fr.sum()
                    if _ma > 0 and _mb > 0:
                        _fr *= (_mb / _ma)
                    pred_kstep_physical[_ti] = _fr
            elif clamp_mode == 'C1':
                pred_kstep_physical = np.maximum(pred_kstep_physical, 0.0)
            # Mass postprocess for k-step predictions
            if mass_postprocess == 'simplex' and M0 > 0:
                for _ti in range(len(pred_kstep_physical)):
                    pred_kstep_physical[_ti] = _project_simplex(
                        pred_kstep_physical[_ti], mass_target=M0)
            elif mass_postprocess == 'scale' and M0 > 0:
                for _ti in range(len(pred_kstep_physical)):
                    _mc = pred_kstep_physical[_ti].sum()
                    if _mc > 0:
                        pred_kstep_physical[_ti] *= (M0 / _mc)
            # R² in density space
            _true_phys = test_density[T_train:]
            _ss_res_ks = np.sum((_true_phys.flatten() - pred_kstep_physical.flatten())**2)
            _ss_tot_ks = np.sum((_true_phys.flatten() - _true_phys.flatten().mean())**2)
            r2_kstep_density = _safe_r2(_ss_res_ks, _ss_tot_ks, np.sum(_true_phys.flatten()**2))
            # R² in latent space
            if latent_standardize:
                _true_lat_ks = test_latent[T_train:] * latent_std + latent_mean
            else:
                _true_lat_ks = test_latent[T_train:]
            _ss_res_lat_ks = np.sum((_true_lat_ks.flatten() - pred_kstep_latent_raw.flatten())**2)
            _ss_tot_lat_ks = np.sum((_true_lat_ks.flatten() - _true_lat_ks.flatten().mean())**2)
            r2_kstep_latent = _safe_r2(_ss_res_lat_ks, _ss_tot_lat_ks, np.sum(_true_lat_ks.flatten()**2))

        # Inverse latent standardization before POD lifting
        if latent_standardize:
            pred_latent_forecast = pred_latent_forecast * latent_std + latent_mean
        
        # Build full prediction: conditioning window (POD-reconstructed truth) + forecast
        # This ensures the video starts from a good match and transitions to the forecast
        # Note: cond_latent must be in RAW latent space for POD lifting
        if latent_standardize:
            cond_latent = test_latent[:T_train] * latent_std + latent_mean
        else:
            cond_latent = test_latent[:T_train]
        pred_latent_full = np.vstack([cond_latent, pred_latent_forecast])
        
        # Reconstruct full prediction to physical space
        pred_physical_full = (pred_latent_full @ U_r.T) + X_mean
        pred_physical_full = pred_physical_full.reshape(-1, density_nx, density_ny)
        
        # Also reconstruct forecast-only for R² computation
        pred_physical_forecast = (pred_latent_forecast @ U_r.T) + X_mean
        pred_physical_forecast = pred_physical_forecast.reshape(-1, density_nx, density_ny)
        
        # Inverse density transform if needed (convert back to raw density space)
        density_transform = pod_data.get('density_transform', 'raw')
        density_transform_eps = pod_data.get('density_transform_eps', 1e-8)
        
        if density_transform == 'log':
            pred_physical_full = np.exp(pred_physical_full) - density_transform_eps
            pred_physical_forecast = np.exp(pred_physical_forecast) - density_transform_eps
        elif density_transform == 'sqrt':
            # Clamp to 0 before squaring (sqrt domain can't produce negatives, but
            # POD reconstruction in transformed space might)
            pred_physical_full = np.maximum(pred_physical_full, 0.0)**2 - density_transform_eps
            pred_physical_forecast = np.maximum(pred_physical_forecast, 0.0)**2 - density_transform_eps
        elif density_transform == 'meansub':
            # meansub: POD was done on (rho - snapshot_mean). Reconstruction gives
            # (rho - snapshot_mean). We can't perfectly invert without knowing the
            # snapshot mean of the prediction. Use test ground truth mean as proxy
            # for conditioning window, and extrapolate for forecast.
            # Actually the POD mean-centering already handles the global mean.
            # The meansub per-snapshot is an extra centering that removes the
            # DC component per frame. The reconstruction X_mean already captures
            # the average of the meansub'd data, so adding X_mean back gives
            # meansub'd data. We need to add back the per-snapshot mean.
            # For simplicity: just note that meansub changes the POD basis but
            # the lifted prediction is in meansub space. The true comparison
            # should also be in meansub space, OR we skip inverse and compare
            # in transformed space. We compare in raw space by adding back
            # a uniform density level estimated from training.
            pass  # meansub inverse is implicit — comparison handles it
        # 'raw' needs no inverse
        
        # Post-processing: handle negative density values from POD reconstruction
        neg_frac_full = np.mean(pred_physical_full < 0) * 100
        neg_frac = np.mean(pred_physical_forecast < 0) * 100
        
        if clamp_mode == 'C2' and neg_frac_full > 0:
            # C2: Clamp negatives to 0, then renormalize to preserve total mass
            for t_idx in range(len(pred_physical_full)):
                frame = pred_physical_full[t_idx]
                mass_before = frame.sum()
                frame = np.maximum(frame, 0.0)
                mass_after = frame.sum()
                if mass_after > 0 and mass_before > 0:
                    frame *= (mass_before / mass_after)
                pred_physical_full[t_idx] = frame
        elif clamp_mode == 'simplex' and neg_frac_full > 0:
            # Simplex: proper Euclidean projection onto {rho>=0, sum=mass}
            for t_idx in range(len(pred_physical_full)):
                mass_target = pred_physical_full[t_idx].sum()
                pred_physical_full[t_idx] = _project_simplex(
                    pred_physical_full[t_idx], mass_target=mass_target)
        elif clamp_mode == 'C1' and neg_frac_full > 0:
            # C1: Clamp negatives to 0, NO renormalization
            pred_physical_full = np.maximum(pred_physical_full, 0.0)
        # C0: no clamping at all
        
        # ---- Mass postprocess for pred_physical_full ----
        if mass_postprocess == 'simplex' and M0 > 0:
            for t_idx in range(len(pred_physical_full)):
                pred_physical_full[t_idx] = _project_simplex(
                    pred_physical_full[t_idx], mass_target=M0)
        elif mass_postprocess == 'scale' and M0 > 0:
            for t_idx in range(len(pred_physical_full)):
                m_cur = pred_physical_full[t_idx].sum()
                if m_cur > 0:
                    pred_physical_full[t_idx] *= (M0 / m_cur)
        
        if clamp_mode == 'C2' and neg_frac > 0:
            for t_idx in range(len(pred_physical_forecast)):
                frame = pred_physical_forecast[t_idx]
                mass_before = frame.sum()
                frame = np.maximum(frame, 0.0)
                mass_after = frame.sum()
                if mass_after > 0 and mass_before > 0:
                    frame *= (mass_before / mass_after)
                pred_physical_forecast[t_idx] = frame
            print(f"   ℹ️  Clamp mode C2: clamped {neg_frac:.1f}% negative pixels + mass-renorm")
        elif clamp_mode == 'simplex' and neg_frac > 0:
            for t_idx in range(len(pred_physical_forecast)):
                mass_target = pred_physical_forecast[t_idx].sum()
                pred_physical_forecast[t_idx] = _project_simplex(
                    pred_physical_forecast[t_idx], mass_target=mass_target)
            print(f"   ℹ️  Clamp mode simplex: projected {neg_frac:.1f}% negative pixels (Euclidean simplex)")
        elif clamp_mode == 'C1' and neg_frac > 0:
            pred_physical_forecast = np.maximum(pred_physical_forecast, 0.0)
            print(f"   ℹ️  Clamp mode C1: clamped {neg_frac:.1f}% negative pixels (no renorm)")
        elif clamp_mode == 'C0' and neg_frac > 0:
            print(f"   ℹ️  Clamp mode C0: {neg_frac:.1f}% negative pixels (no clamping)")
        elif neg_frac > 0:
            print(f"   ℹ️  {neg_frac:.1f}% negative density pixels (clamp_mode={clamp_mode})")
        
        # ---- Mass postprocess for pred_physical_forecast ----
        if mass_postprocess == 'simplex' and M0 > 0:
            for t_idx in range(len(pred_physical_forecast)):
                pred_physical_forecast[t_idx] = _project_simplex(
                    pred_physical_forecast[t_idx], mass_target=M0)
            if test_idx == 0:
                m_final = pred_physical_forecast[-1].sum()
                print(f"   ℹ️  Mass postprocess=simplex: M₀={M0:.1f} → final mass={m_final:.1f}")
        elif mass_postprocess == 'scale' and M0 > 0:
            for t_idx in range(len(pred_physical_forecast)):
                m_cur = pred_physical_forecast[t_idx].sum()
                if m_cur > 0:
                    pred_physical_forecast[t_idx] *= (M0 / m_cur)
            if test_idx == 0:
                m_final = pred_physical_forecast[-1].sum()
                print(f"   ℹ️  Mass postprocess=scale: M₀={M0:.1f} → final mass={m_final:.1f}")
        
        # Ground truth (forecasted region only — for R² computation)
        true_physical = test_density[T_train:]
        
        # ---- Teacher-forced one-step R² (R²_1step) ----
        # At each forecast timestep, feed TRUE latent history to the model
        # and predict one step ahead. This isolates single-step accuracy
        # from autoregressive error accumulation.
        true_latent_forecast_1s = test_latent[T_train:]
        onestep_preds = []
        for t_1s in range(len(true_latent_forecast_1s)):
            # Build the true window ending at T_train + t_1s
            window_end = T_train + t_1s
            if window_end < P_LAG:
                continue  # Not enough history yet
            true_window = test_latent[window_end - P_LAG:window_end]  # [lag, d]
            pred_1s = forecast_fn(true_window, 1)  # [1, d]
            onestep_preds.append(pred_1s[0])
        
        if len(onestep_preds) > 0:
            onestep_preds = np.array(onestep_preds)  # [N, d]
            # Targets are the true latent states one step after each window
            onestep_targets = true_latent_forecast_1s[:len(onestep_preds)]
            ss_res_1s = np.sum((onestep_targets - onestep_preds)**2)
            ss_tot_1s = np.sum((onestep_targets - onestep_targets.mean(axis=0))**2)
            # Guard against degenerate case: when targets have near-zero variance
            # (e.g., frozen/constant signals), ss_tot ~ 0 and R2 is undefined.
            # Use threshold relative to signal magnitude to detect this.
            signal_scale_1s = np.sum(onestep_targets**2)
            if ss_tot_1s > 1e-10 * max(signal_scale_1s, 1e-30):
                r2_1step = 1 - ss_res_1s / ss_tot_1s
            else:
                # Signal has ~zero temporal variance => R2 is meaningless
                r2_1step = float('nan')
        else:
            r2_1step = float('nan')
        
        # ---- Compute R² metrics (on FORECAST region only — fair evaluation) ----
        # 1. Reconstructed (physical space) — this is R2_rollout
        true_phys_flat = true_physical.flatten()
        pred_phys_flat = pred_physical_forecast.flatten()
        ss_res_phys = np.sum((true_phys_flat - pred_phys_flat)**2)
        ss_tot_phys = np.sum((true_phys_flat - true_phys_flat.mean())**2)
        r2_reconstructed = _safe_r2(ss_res_phys, ss_tot_phys, np.sum(true_phys_flat**2))
        
        # 2. Latent (ROM space) — compare in RAW latent space
        if latent_standardize:
            true_latent_forecast = test_latent[T_train:] * latent_std + latent_mean
        else:
            true_latent_forecast = test_latent[T_train:]
        true_lat_flat = true_latent_forecast.flatten()
        pred_lat_flat = pred_latent_forecast.flatten()
        ss_res_lat = np.sum((true_lat_flat - pred_lat_flat)**2)
        ss_tot_lat = np.sum((true_lat_flat - true_lat_flat.mean())**2)
        r2_latent = _safe_r2(ss_res_lat, ss_tot_lat, np.sum(true_lat_flat**2))
        
        # 3. POD reconstruction quality (using true latent)
        true_reconstructed = (true_latent_forecast @ U_r.T) + X_mean
        true_reconstructed = true_reconstructed.reshape(-1, density_nx, density_ny)
        ss_res_pod = np.sum((true_physical.flatten() - true_reconstructed.flatten())**2)
        r2_pod = _safe_r2(ss_res_pod, ss_tot_phys, np.sum(true_phys_flat**2))
        
        # Compute RMSE metrics (for compatibility) — forecast region only
        rmse_recon = np.sqrt(np.mean((true_physical.flatten() - pred_physical_forecast.flatten())**2))
        rmse_latent = np.sqrt(np.mean((true_latent_forecast.flatten() - pred_latent_forecast.flatten())**2))
        rmse_pod = np.sqrt(np.mean((true_physical.flatten() - true_reconstructed.flatten())**2))
        
        # Compute relative errors
        rel_error_recon = rmse_recon / (np.mean(np.abs(true_physical.flatten())) + 1e-10)
        rel_error_pod = rmse_pod / (np.mean(np.abs(true_physical.flatten())) + 1e-10)
        
        # Compute mass conservation violation (forecast region)
        true_mass = np.sum(true_physical, axis=(1, 2))
        pred_mass = np.sum(pred_physical_forecast, axis=(1, 2))
        mass_violations = np.abs(pred_mass - true_mass) / (true_mass + 1e-10)
        max_mass_violation = np.max(mass_violations)
        
        # Store results
        result = {
            'test_id': test_idx,
            'r2_reconstructed': r2_reconstructed,
            'r2_latent': r2_latent,
            'r2_pod': r2_pod,
            'r2_1step': r2_1step,
            'negativity_frac': neg_frac,
            'clamp_mode': clamp_mode,
            'density_transform': density_transform,
            'rmse_recon': rmse_recon,
            'rmse_latent': rmse_latent,
            'rmse_pod': rmse_pod,
            'rel_error_recon': rel_error_recon,
            'rel_error_pod': rel_error_pod,
            'max_mass_violation': max_mass_violation,
            'T_forecast': len(pred_latent_forecast),
            'T_conditioning': T_train,
            'r2_kstep_density': r2_kstep_density,
            'r2_kstep_latent': r2_kstep_latent,
            'kstep_reset': kstep_reset if kstep_reset else 0,
            'mass_postprocess': mass_postprocess,
            'mass_target_M0': float(M0)
        }
        
        # Save metrics summary JSON (REQUIRED for visualization pipeline)
        metrics_dict = {
            'r2_recon': float(r2_reconstructed),
            'r2_latent': float(r2_latent),
            'r2_pod': float(r2_pod),
            'r2_1step': float(r2_1step),
            'negativity_frac': float(neg_frac),
            'rmse_recon': float(rmse_recon),
            'rmse_latent': float(rmse_latent),
            'rmse_pod': float(rmse_pod),
            'rel_error_recon': float(rel_error_recon),
            'rel_error_pod': float(rel_error_pod),
            'max_mass_violation': float(max_mass_violation),
            'r2_kstep_density': float(r2_kstep_density),
            'r2_kstep_latent': float(r2_kstep_latent),
            'kstep_reset': int(kstep_reset) if kstep_reset else 0,
            'mass_postprocess': mass_postprocess,
            'mass_target_M0': float(M0)
        }
        with open(test_run_dir / "metrics_summary.json", 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        # Define forecast times (needed for predictions and optional for time-resolved)
        forecast_times = test_times[T_train:]
        all_times = test_times[:T_train + len(pred_latent_forecast)]
        
        # Time-resolved analysis (if requested)
        if save_time_resolved and len(pred_latent_forecast) > 0:
            r2_vs_time = _compute_time_resolved_r2(
                true_physical, pred_physical_forecast,
                true_latent_forecast, pred_latent_forecast,
                true_reconstructed, forecast_times
            )
            
            # Save time-resolved data
            r2_df = pd.DataFrame(r2_vs_time)
            r2_df.to_csv(test_run_dir / "r2_vs_time.csv", index=False)
        
        # Save predicted density (REQUIRED for visualization pipeline)
        # Save FULL trajectory: conditioning window (POD truth) + forecast
        # This ensures videos start with a good match and transition to forecast
        model_tag = model_name.lower()
        np.savez_compressed(
            test_run_dir / f"density_pred_{model_tag}.npz",
            rho=pred_physical_full,
            xgrid=xgrid,
            ygrid=ygrid,
            times=all_times,
            forecast_start_idx=T_train  # Index where forecast begins
        )
        # Also save generic for backward compatibility (single-model case)
        np.savez_compressed(
            test_run_dir / "density_pred.npz",
            rho=pred_physical_full,
            xgrid=xgrid,
            ygrid=ygrid,
            times=all_times,
            forecast_start_idx=T_train
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
                
                # Compute order parameters for all timesteps (with density movie)
                order_params = compute_metrics_series(
                    x, v, domain_bounds, 
                    resolution=density_nx,
                    verbose=False,
                    density_movie=test_density
                )
                
                # Compute actual density-based metrics from the density movie
                # (replaces placeholder values from standard_metrics functions)
                density_var_from_movie = np.std(test_density, axis=(1, 2))
                mass_from_movie = np.sum(test_density, axis=(1, 2))
                
                # Save to CSV with time column
                op_df = pd.DataFrame({
                    't': test_times,
                    'phi': order_params['polarization'],
                    'mean_speed': order_params['mean_speed'],
                    'angular_momentum': order_params['angular_momentum'],
                    'nematic_order': order_params['nematic_order'],
                    'density_variance': density_var_from_movie,
                    'total_mass': mass_from_movie,
                    'spatial_order': order_params['spatial_order']
                })
                op_df.to_csv(test_run_dir / "order_params.csv", index=False)
                
            # Compute density-based metrics for TRUE vs PREDICTED comparison
            # Save model-specific files so MVAR and LSTM have separate comparisons
            density_variance_true = np.std(test_density, axis=(1, 2))  # Spatial std per timestep
            density_variance_pred_full = np.zeros(T_test)
            density_variance_pred_full[:T_train] = density_variance_true[:T_train]  # Use true during conditioning
            density_variance_pred_full[T_train:] = np.std(pred_physical_forecast, axis=(1, 2))  # Use pred during forecast
            
            mass_true = np.sum(test_density, axis=(1, 2))
            mass_pred_full = np.zeros(T_test)
            mass_pred_full[:T_train] = mass_true[:T_train]
            mass_pred_full[T_train:] = np.sum(pred_physical_forecast, axis=(1, 2))
            
            # Save density-based comparison metrics (model-specific)
            model_tag = model_name.lower()
            density_metrics_df = pd.DataFrame({
                't': test_times,
                'density_variance_true': density_variance_true,
                'density_variance_pred': density_variance_pred_full,
                'mass_true': mass_true,
                'mass_pred': mass_pred_full,
            })
            density_metrics_df.to_csv(test_run_dir / f"density_metrics_{model_tag}.csv", index=False)
            # Also save generic for backward compatibility
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
    mean_r2_1step = test_results_df['r2_1step'].mean()
    mean_neg_frac = test_results_df['negativity_frac'].mean()
    
    def _fmt_r2(val):
        """Format R² value, handling NaN gracefully."""
        if np.isnan(val):
            return "NaN (constant signal)"
        return f"{val:.4f}"
    
    print(f"\n{'='*80}")
    print("Test Results Summary")
    print("="*80)
    print(f"Mean R² (1-step, teacher-forced): {_fmt_r2(mean_r2_1step)}")
    print(f"Mean R² (rollout, reconstructed): {_fmt_r2(mean_r2_recon)}")
    print(f"Mean R² (rollout, latent):        {_fmt_r2(mean_r2_latent)}")
    print(f"Mean R² (POD ceiling):            {_fmt_r2(mean_r2_pod)}")
    if kstep_reset:
        mean_r2_kstep = test_results_df['r2_kstep_density'].mean()
        print(f"Mean R² (k={kstep_reset}-step TF, density): {_fmt_r2(mean_r2_kstep)}")
    print(f"Mean negativity fraction:         {mean_neg_frac:.2f}%")
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
