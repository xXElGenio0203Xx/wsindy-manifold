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
    n_top=4
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
    
    # Select top runs by R²
    top_runs = metrics_df.nlargest(n_top, 'r2')
    top_ic_types = top_runs['ic_type'].tolist()
    
    print(f"\nGenerating videos and plots for top {n_top} runs by R²:")
    for idx, row in top_runs.iterrows():
        print(f"   {row['run_name']}: {row['ic_type']} (R² = {row['r2']:.4f})")
    
    for ic_type in tqdm(top_ic_types, desc="IC types"):
        ic_stats = ic_metrics.get(ic_type)
        if ic_stats is None:
            continue
        
        best_run = ic_stats["best_run"]
        pred = test_predictions[best_run]
        run_dir = test_dir / best_run
        
        ic_output_dir = best_runs_dir / ic_type
        ic_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Find the full metrics entry for this run
        all_metrics = metrics_df.to_dict('records')
        summary = [m for m in all_metrics if m["run_name"] == best_run][0]
        
        # Generate visualizations
        _generate_trajectory_video(pred, ic_output_dir, ic_type, base_config_sim)
        _generate_density_comparison(pred, ic_output_dir)
        _generate_error_timeseries(pred, summary, ic_output_dir, ic_type, ic_stats, p_lag)
        _generate_error_distributions(pred, ic_output_dir, ic_type)
        _generate_order_parameters(run_dir, ic_output_dir, ic_type, ic_stats)
    
    print(f"\n✓ Generated visualizations for top {n_top} runs")
    
    return top_runs


def _generate_trajectory_video(pred, output_dir, ic_type, base_config_sim):
    """Generate trajectory video."""
    trajectory_video(
        path=output_dir,
        traj=pred["traj"],
        times=pred["times"],
        Lx=base_config_sim["Lx"],
        Ly=base_config_sim["Ly"],
        name="traj_truth",
        fps=10,
        marker_size=50,
        title=f'Ground Truth Trajectory - {ic_type.replace("_", " ").title()}'
    )


def _generate_density_comparison(pred, output_dir):
    """Generate side-by-side density comparison video."""
    side_by_side_video(
        path=output_dir,
        left_frames=pred["rho_true"],
        right_frames=pred["rho_pred"],
        lower_strip_timeseries=pred["frame_metrics"]["e2"],
        name="density_truth_vs_pred",
        fps=10,
        cmap='hot',
        titles=('Ground Truth', 'MVAR-ROM Prediction')
    )


def _generate_error_timeseries(pred, summary, output_dir, ic_type, ic_stats, p_lag):
    """Generate error timeseries plot."""
    plot_errors_timeseries(
        frame_metrics=pred["frame_metrics"],
        summary=summary,
        T0=p_lag,
        save_path=output_dir / "error_time.png",
        title=f'Error Metrics - {ic_type.replace("_", " ").title()} (R²={ic_stats["best_r2"]:.3f})'
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


def _generate_order_parameters(run_dir, output_dir, ic_type, ic_stats):
    """Generate order parameter plots if available."""
    # Check for density-based order parameters
    order_params_path = run_dir / "order_params_density.csv"
    if order_params_path.exists():
        df_order = pd.read_csv(order_params_path)
        has_particles = 'polarization' in df_order.columns
        
        if has_particles:
            _plot_full_order_parameters(df_order, output_dir, ic_type, ic_stats)
            _plot_mass_conservation(df_order, output_dir, ic_type)
        else:
            _plot_density_order_parameters(df_order, output_dir, ic_type, ic_stats)
    
    # Check for particle-based order parameters
    order_params_traj_path = run_dir / "order_params.csv"
    if order_params_traj_path.exists():
        _plot_particle_order_parameters(
            order_params_traj_path, output_dir, ic_type, ic_stats
        )


def _plot_full_order_parameters(df_order, output_dir, ic_type, ic_stats):
    """Plot full order parameters (particle + density)."""
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    
    # Polarization
    axes[0].plot(df_order['t'], df_order['polarization'], 'b-', linewidth=2.5, alpha=0.85)
    axes[0].set_ylabel('Polarization Φ\n(Velocity Alignment)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])
    median_phi = df_order['polarization'].iloc[-len(df_order)//4:].median()
    axes[0].axhline(median_phi, color='r', linestyle='--', alpha=0.5,
                   label=f'Final median: {median_phi:.3f}')
    axes[0].legend(loc='best', fontsize=10)
    axes[0].set_title(f'Order Parameters - {ic_type.replace("_", " ").title()} (R²={ic_stats["best_r2"]:.3f})', 
                     fontsize=14, fontweight='bold')
    
    # Nematic order
    axes[1].plot(df_order['t'], df_order['nematic'], 'm-', linewidth=2.5, alpha=0.85)
    axes[1].set_ylabel('Nematic Order Q\n(Bidirectional)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([-0.05, 1.05])
    
    # Mean speed
    axes[2].plot(df_order['t'], df_order['mean_speed'], 'g-', linewidth=2.5, alpha=0.85)
    axes[2].set_ylabel('Mean Speed\n(Kinetic Energy)', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(df_order['mean_speed'].mean(), color='k', linestyle='--', alpha=0.4)
    
    # Angular momentum
    axes[3].plot(df_order['t'], df_order['angular_momentum'], 'c-', linewidth=2.5, alpha=0.85)
    axes[3].set_ylabel('Angular Momentum\n(Milling/Rotation)', fontsize=12, fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_ylim([-0.05, 1.05])
    
    # Spatial order
    axes[4].plot(df_order['t'], df_order['spatial_order_true'], 'b-', linewidth=2.5, label='True', alpha=0.8)
    axes[4].plot(df_order['t'], df_order['spatial_order_pred'], 'r--', linewidth=2.5, label='Predicted', alpha=0.8)
    axes[4].set_ylabel('Spatial Order\n(Density Std)', fontsize=12, fontweight='bold')
    axes[4].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    axes[4].grid(True, alpha=0.3)
    axes[4].legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / "order_parameters.png", dpi=150, bbox_inches='tight')
    plt.close()


def _plot_mass_conservation(df_order, output_dir, ic_type):
    """Plot mass conservation metrics."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Mass conservation
    axes[0].plot(df_order['t'], df_order['mass_true'], 'b-', linewidth=2, label='True', alpha=0.8)
    axes[0].plot(df_order['t'], df_order['mass_pred'], 'g--', linewidth=2, label='Predicted (Corrected)', alpha=0.8)
    axes[0].set_ylabel('Total Mass', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=10)
    axes[0].set_title(f'Mass Conservation - {ic_type.replace("_", " ").title()}', 
                     fontsize=14, fontweight='bold')
    
    # Mass error
    axes[1].semilogy(df_order['t'], df_order['mass_error_rel'] * 100, 'r-', linewidth=2)
    axes[1].set_ylabel('Mass Error (%)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    max_err = df_order['mass_error_rel'].max() * 100
    axes[1].axhline(max_err, color='k', linestyle='--', alpha=0.5,
                   label=f'Max: {max_err:.2e}%')
    axes[1].legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / "mass_conservation.png", dpi=150, bbox_inches='tight')
    plt.close()


def _plot_density_order_parameters(df_order, output_dir, ic_type, ic_stats):
    """Plot density-based order parameters only."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Spatial order
    axes[0].plot(df_order['t'], df_order['spatial_order_true'], 'b-', linewidth=2, label='True', alpha=0.8)
    axes[0].plot(df_order['t'], df_order['spatial_order_pred'], 'r--', linewidth=2, label='Predicted', alpha=0.8)
    axes[0].set_ylabel('Spatial Order\n(Density Std)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=10)
    axes[0].set_title(f'Order Parameters (Density-Based) - {ic_type.replace("_", " ").title()} (R²={ic_stats["best_r2"]:.3f})', 
                     fontsize=14, fontweight='bold')
    
    # Mass conservation
    axes[1].plot(df_order['t'], df_order['mass_true'], 'b-', linewidth=2, label='True', alpha=0.8)
    axes[1].plot(df_order['t'], df_order['mass_pred'], 'g--', linewidth=2, label='Predicted (Corrected)', alpha=0.8)
    axes[1].set_ylabel('Total Mass', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', fontsize=10)
    
    # Mass error
    axes[2].semilogy(df_order['t'], df_order['mass_error_rel'] * 100, 'r-', linewidth=2)
    axes[2].set_ylabel('Mass Error (%)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    max_err = df_order['mass_error_rel'].max() * 100
    axes[2].axhline(max_err, color='k', linestyle='--', alpha=0.5,
                   label=f'Max: {max_err:.2e}%')
    axes[2].legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / "order_parameters.png", dpi=150, bbox_inches='tight')
    plt.close()


def _plot_particle_order_parameters(order_params_path, output_dir, ic_type, ic_stats):
    """Plot particle-based order parameters."""
    df_order_traj = pd.read_csv(order_params_path)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    axes[0].plot(df_order_traj['t'], df_order_traj['phi'], 'b-', linewidth=2)
    axes[0].set_ylabel('Polarization Φ', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])
    median_phi = df_order_traj['phi'].iloc[-len(df_order_traj)//4:].median()
    axes[0].axhline(median_phi, color='r', linestyle='--', alpha=0.5,
                   label=f'Final median: {median_phi:.3f}')
    axes[0].legend(loc='best', fontsize=10)
    axes[0].set_title(f'Order Parameters (Particles) - {ic_type.replace("_", " ").title()} (R²={ic_stats["best_r2"]:.3f})', 
                     fontsize=14, fontweight='bold')
    
    axes[1].plot(df_order_traj['t'], df_order_traj['mean_speed'], 'g-', linewidth=2)
    axes[1].set_ylabel('Mean Speed', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(df_order_traj['t'], df_order_traj['nematic'], 'm-', linewidth=2)
    axes[2].set_ylabel('Nematic Order Q', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([-0.05, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_dir / "order_parameters_particles.png", dpi=150, bbox_inches='tight')
    plt.close()
