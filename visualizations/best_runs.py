"""
Best Runs Visualization Module
===============================

Generates detailed visualizations for top-performing runs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from rectsim.legacy_functions import (
    trajectory_video,
    side_by_side_video,
    plot_errors_timeseries
)


def generate_best_run_visualizations(
    metrics_df,
    test_predictions,
    ic_metrics,
    test_dir,
    best_runs_dir,
    base_config_sim,
    p_lag,
    n_top=4,
    model_name='mvar',
    config_info=None
):
    """
    Generate detailed visualizations for the best test runs.
    
    Parameters
    ----------
    metrics_df : DataFrame
        Metrics for all runs
    test_predictions : dict
        Dictionary of predictions for each run
    ic_metrics : dict
        Metrics aggregated by IC type
    test_dir : Path
        Directory containing test data
    best_runs_dir : Path
        Directory to save best run visualizations
    base_config_sim : dict
        Base simulation configuration
    p_lag : int
        MVAR lag order
    n_top : int, optional
        Number of top runs to visualize (default: 4)
    
    Returns
    -------
    top_runs : DataFrame
        Top N runs by R²
    """
    
    test_dir = Path(test_dir)
    best_runs_dir = Path(best_runs_dir)
    best_runs_dir.mkdir(exist_ok=True, parents=True)
    
    # Select top runs by R² (for return value / summary)
    top_runs = metrics_df.nlargest(n_top, 'r2')
    
    # Generate visualizations for BEST RUN OF EACH IC TYPE (not just top-N overall)
    # This ensures every IC type (uniform, gaussian_cluster, two_clusters, etc.)
    # gets its own subfolder with the best run for that type.
    all_ic_types = sorted(ic_metrics.keys())
    
    print(f"\nGenerating best-run visualizations for each IC type ({model_name.upper()}):")
    for ic_type in all_ic_types:
        ic_stats = ic_metrics[ic_type]
        print(f"   {ic_type}: {ic_stats['best_run']} (R² = {ic_stats['best_r2']:.4f})")
    
    all_metrics_list = metrics_df.to_dict('records')
    
    for ic_type in tqdm(all_ic_types, desc="IC types"):
        ic_stats = ic_metrics[ic_type]
        best_run = ic_stats["best_run"]
        pred = test_predictions[best_run]
        run_dir = test_dir / best_run
        
        ic_output_dir = best_runs_dir / ic_type
        ic_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Find the full metrics entry for this run
        summary = [m for m in all_metrics_list if m["run_name"] == best_run][0]
        
        # Generate visualizations
        _generate_trajectory_video(pred, ic_output_dir, ic_type, base_config_sim)
        _generate_density_comparison(pred, ic_output_dir, model_name=model_name, config_info=config_info)
        _generate_error_timeseries(pred, summary, ic_output_dir, ic_type, ic_stats, p_lag)
        _generate_error_distributions(pred, ic_output_dir, ic_type)
        _generate_order_parameters(pred, ic_output_dir, ic_type, ic_stats, model_name=model_name)
    
    print(f"\n✓ Generated visualizations for {len(all_ic_types)} IC types: {', '.join(all_ic_types)}")
    
    return top_runs


def _generate_trajectory_video(pred, output_dir, ic_type, base_config_sim):
    """Generate trajectory video using saved velocities when available."""
    trajectory_video(
        path=output_dir,
        traj=pred["traj"],
        times=pred["times"],
        Lx=base_config_sim["Lx"],
        Ly=base_config_sim["Ly"],
        name="traj_truth",
        fps=10,
        marker_size=50,
        title=f'Ground Truth Trajectory - {ic_type.replace("_", " ").title()}',
        vel=pred.get("vel"),
    )


def _generate_density_comparison(pred, output_dir, model_name='mvar', config_info=None):
    """Generate side-by-side density comparison video.
    
    Generates two videos:
      1. Forecast-only (from forecast_start_idx onward) — the primary comparison.
      2. Full trajectory (conditioning + forecast) — for debugging only.
    """
    model_label = model_name.upper()
    fsi = pred.get("forecast_start_idx", None)
    
    # --- Forecast-only video (what actually matters — generated FIRST) ---
    if fsi is not None and fsi > 0 and fsi < len(pred["rho_true"]):
        rho_true_fc = pred["rho_true"][fsi:]
        rho_pred_fc = pred["rho_pred"][fsi:]
        # Slice error timeseries to forecast region too
        e2_fc = pred.get("frame_metrics_fc", {}).get("e2", None)
        if e2_fc is None:
            e2_fc = pred["frame_metrics"]["e2"]
            if len(e2_fc) > len(rho_true_fc):
                e2_fc = e2_fc[fsi:]
        
        side_by_side_video(
            path=output_dir,
            left_frames=rho_true_fc,
            right_frames=rho_pred_fc,
            lower_strip_timeseries=e2_fc,
            name="density_forecast_only",
            fps=10,
            cmap='hot',
            titles=('Ground Truth (forecast)', f'{model_label}-ROM Forecast'),
            config_info=config_info
        )
    else:
        # No conditioning window info — treat entire trajectory as forecast
        side_by_side_video(
            path=output_dir,
            left_frames=pred["rho_true"],
            right_frames=pred["rho_pred"],
            lower_strip_timeseries=pred["frame_metrics"]["e2"],
            name="density_forecast_only",
            fps=10,
            cmap='hot',
            titles=('Ground Truth', f'{model_label}-ROM Forecast'),
            config_info=config_info
        )
    
    # --- Full video (conditioning + forecast) — for debugging ---
    side_by_side_video(
        path=output_dir,
        left_frames=pred["rho_true"],
        right_frames=pred["rho_pred"],
        lower_strip_timeseries=pred["frame_metrics"]["e2"],
        name="density_truth_vs_pred_full",
        fps=10,
        cmap='hot',
        titles=('Ground Truth', f'{model_label}-ROM Prediction (incl. conditioning)'),
        config_info=config_info
    )


def _generate_error_timeseries(pred, summary, output_dir, ic_type, ic_stats, p_lag):
    """Generate error timeseries plot (forecast region only)."""
    # Prefer forecast-only frame metrics if available
    fm = pred.get("frame_metrics_fc", pred["frame_metrics"])
    plot_errors_timeseries(
        frame_metrics=fm,
        summary=summary,
        T0=p_lag,
        save_path=output_dir / "error_time.png",
        title=f'Error Metrics (Forecast) - {ic_type.replace("_", " ").title()} (R²={ic_stats["best_r2"]:.3f})'
    )
    plt.close('all')


def _generate_error_distributions(pred, output_dir, ic_type):
    """Generate error distribution histograms."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    e1_final = np.abs(pred["rho_true"][-1] - pred["rho_pred"][-1])
    axes[0].hist(e1_final.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_xlabel('L1 Error', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title(f'L1 Error Distribution (t={pred["times"][-1]:.1f}s)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    e2_final = (pred["rho_true"][-1] - pred["rho_pred"][-1])**2
    axes[1].hist(e2_final.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_xlabel('L2 Error', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title(f'L2 Error Distribution (t={pred["times"][-1]:.1f}s)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    rel_error_final = e1_final / (pred["rho_true"][-1] + 1e-10)
    axes[2].hist(rel_error_final.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[2].set_xlabel('Relative Error', fontsize=11)
    axes[2].set_ylabel('Count', fontsize=11)
    axes[2].set_title(f'Relative Error Distribution (t={pred["times"][-1]:.1f}s)', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    fig.suptitle(f'Error Distributions - {ic_type.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "error_hist.png", dpi=200)
    plt.close()


def _compute_order_params_from_data(vel, traj, rho_true, rho_pred, times):
    """
    Compute order parameters directly from raw trajectory/velocity/density data.
    
    Parameters
    ----------
    vel : ndarray (T, N, 2) or None
        Particle velocities
    traj : ndarray (T, N, 2)
        Particle positions
    rho_true : ndarray (T, ny, nx)
        True density fields
    rho_pred : ndarray (T, ny, nx)
        Predicted density fields
    times : ndarray (T,)
        Time array
    
    Returns
    -------
    order_params : dict of arrays, each length T
    """
    T = len(times)
    params = {'t': times}
    
    # --- Particle-based order parameters (require velocities) ---
    if vel is not None:
        polarization = np.zeros(T)
        nematic = np.zeros(T)
        mean_speed = np.zeros(T)
        angular_momentum = np.zeros(T)
        
        for t in range(T):
            v = vel[t]  # (N, 2)
            r = traj[t]  # (N, 2)
            speeds = np.linalg.norm(v, axis=1)  # (N,)
            mean_speed[t] = np.mean(speeds)
            
            # Polarization: |mean(v_hat)| — velocity alignment
            nonzero = speeds > 1e-10
            if np.any(nonzero):
                v_hat = v[nonzero] / speeds[nonzero, None]
                polarization[t] = np.linalg.norm(np.mean(v_hat, axis=0))
                
                # Nematic: 2*|mean(cos2θ, sin2θ)|  — bidirectional alignment
                angles = np.arctan2(v_hat[:, 1], v_hat[:, 0])
                nematic[t] = np.sqrt(np.mean(np.cos(2 * angles))**2 + 
                                     np.mean(np.sin(2 * angles))**2)
            
            # Angular momentum: normalized |Σ r×v| / (N * <|r|> * <|v|>)
            com = np.mean(r, axis=0)
            r_rel = r - com
            cross = r_rel[:, 0] * v[:, 1] - r_rel[:, 1] * v[:, 0]  # z-component
            r_norms = np.linalg.norm(r_rel, axis=1)
            denom = len(r) * (np.mean(r_norms) * mean_speed[t] + 1e-10)
            angular_momentum[t] = np.abs(np.sum(cross)) / denom
        
        params['polarization'] = polarization
        params['nematic'] = nematic
        params['mean_speed'] = mean_speed
        params['angular_momentum'] = np.clip(angular_momentum, 0, 1)
    
    # --- Density-based order parameters (always available) ---
    spatial_order_true = np.std(rho_true.reshape(T, -1), axis=1)
    spatial_order_pred = np.std(rho_pred.reshape(T, -1), axis=1)
    mass_true = np.sum(rho_true.reshape(T, -1), axis=1)
    mass_pred = np.sum(rho_pred.reshape(T, -1), axis=1)
    mass_error_rel = np.abs(mass_true - mass_pred) / (mass_true + 1e-10)
    
    params['spatial_order_true'] = spatial_order_true
    params['spatial_order_pred'] = spatial_order_pred
    params['mass_true'] = mass_true
    params['mass_pred'] = mass_pred
    params['mass_error_rel'] = mass_error_rel
    
    return params


def _generate_order_parameters(pred, output_dir, ic_type, ic_stats, model_name='mvar'):
    """
    Generate order parameter plots computed directly from prediction data.
    
    Computes particle-based (polarization, nematic, speed, angular momentum)
    and density-based (spatial order, mass conservation) order parameters.
    """
    model_label = model_name.upper()
    
    # Compute order parameters from raw data
    op = _compute_order_params_from_data(
        vel=pred.get('vel'),
        traj=pred['traj'],
        rho_true=pred['rho_true'],
        rho_pred=pred['rho_pred'],
        times=pred['times']
    )
    
    has_particles = 'polarization' in op
    n_panels = 6 if has_particles else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3.0 * n_panels), sharex=True)
    
    panel = 0
    
    # --- Particle-based panels ---
    if has_particles:
        # 1. Polarization
        axes[panel].plot(op['t'], op['polarization'], 'b-', linewidth=2.5, alpha=0.85)
        axes[panel].set_ylabel('Polarization Φ\n(Velocity Alignment)', fontsize=12, fontweight='bold')
        axes[panel].grid(True, alpha=0.3)
        axes[panel].set_ylim([0, 1.05])
        median_phi = np.median(op['polarization'][-len(op['polarization'])//4:])
        axes[panel].axhline(median_phi, color='r', linestyle='--', alpha=0.5,
                           label=f'Final median: {median_phi:.3f}')
        axes[panel].legend(loc='best', fontsize=10)
        axes[panel].set_title(
            f'Order Parameters — {ic_type.replace("_", " ").title()} | '
            f'{model_label} (R²={ic_stats["best_r2"]:.3f})',
            fontsize=14, fontweight='bold')
        panel += 1
        
        # 2. Nematic order
        axes[panel].plot(op['t'], op['nematic'], 'm-', linewidth=2.5, alpha=0.85)
        axes[panel].set_ylabel('Nematic Order Q\n(Bidirectional)', fontsize=12, fontweight='bold')
        axes[panel].grid(True, alpha=0.3)
        axes[panel].set_ylim([-0.05, 1.05])
        panel += 1
        
        # 3. Mean speed
        axes[panel].plot(op['t'], op['mean_speed'], 'g-', linewidth=2.5, alpha=0.85)
        axes[panel].set_ylabel('Mean Speed\n(Kinetic Energy)', fontsize=12, fontweight='bold')
        axes[panel].grid(True, alpha=0.3)
        axes[panel].axhline(np.mean(op['mean_speed']), color='k', linestyle='--', alpha=0.4,
                           label=f'Mean: {np.mean(op["mean_speed"]):.3f}')
        axes[panel].legend(loc='best', fontsize=10)
        panel += 1
        
        # 4. Angular momentum
        axes[panel].plot(op['t'], op['angular_momentum'], 'c-', linewidth=2.5, alpha=0.85)
        axes[panel].set_ylabel('Angular Momentum\n(Milling/Rotation)', fontsize=12, fontweight='bold')
        axes[panel].grid(True, alpha=0.3)
        axes[panel].set_ylim([-0.05, 1.05])
        panel += 1
    else:
        axes[panel].set_title(
            f'Order Parameters (Density-Based) — {ic_type.replace("_", " ").title()} | '
            f'{model_label} (R²={ic_stats["best_r2"]:.3f})',
            fontsize=14, fontweight='bold')
    
    # --- Density-based panels (always shown) ---
    # Spatial order: true vs predicted
    axes[panel].plot(op['t'], op['spatial_order_true'], 'b-', linewidth=2.5, alpha=0.85, label='Ground Truth')
    axes[panel].plot(op['t'], op['spatial_order_pred'], 'r--', linewidth=2.5, alpha=0.85, label=f'Predicted ({model_label})')
    axes[panel].set_ylabel('Spatial Order\nσ(ρ) per frame', fontsize=12, fontweight='bold')
    axes[panel].grid(True, alpha=0.3)
    axes[panel].legend(loc='best', fontsize=11)
    panel += 1
    
    # Mass conservation: true vs predicted
    axes[panel].plot(op['t'], op['mass_true'], 'b-', linewidth=2.5, alpha=0.85, label='Ground Truth')
    axes[panel].plot(op['t'], op['mass_pred'], 'r--', linewidth=2.5, alpha=0.85, label=f'Predicted ({model_label})')
    max_err = np.max(op['mass_error_rel']) * 100
    axes[panel].set_ylabel('Total Mass\nΣ ρ(x,y)·dx·dy', fontsize=12, fontweight='bold')
    axes[panel].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    axes[panel].grid(True, alpha=0.3)
    axes[panel].legend(loc='upper left', fontsize=11)
    # Add mass error annotation
    ax_err = axes[panel].twinx()
    ax_err.plot(op['t'], op['mass_error_rel'] * 100, 'orange', linewidth=1.5, alpha=0.6, label=f'Mass error (max {max_err:.2f}%)')
    ax_err.set_ylabel('Mass Error (%)', fontsize=10, color='orange')
    ax_err.tick_params(axis='y', labelcolor='orange')
    ax_err.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "order_parameters.png", dpi=150, bbox_inches='tight')
    plt.close()

