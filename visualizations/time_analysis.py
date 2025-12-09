"""
Time-Resolved Analysis Module
==============================

Generates time-resolved R² degradation and window contamination analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import yaml
from pathlib import Path


def generate_time_resolved_analysis(
    test_metadata,
    test_dir,
    mvar_dir,
    data_dir,
    time_analysis_dir
):
    """
    Generate comprehensive time-resolved R² degradation analysis.
    
    Parameters
    ----------
    test_metadata : list
        List of test run metadata dictionaries
    test_dir : Path
        Directory containing test data
    mvar_dir : Path
        Directory containing MVAR model
    data_dir : Path
        Root data directory (contains config_used.yaml)
    time_analysis_dir : Path
        Directory to save time analysis outputs
    
    Returns
    -------
    degradation_info : dict
        Degradation statistics and temporal analysis
    """
    
    test_dir = Path(test_dir)
    mvar_dir = Path(mvar_dir)
    data_dir = Path(data_dir)
    time_analysis_dir = Path(time_analysis_dir)
    time_analysis_dir.mkdir(exist_ok=True, parents=True)
    
    # Determine IC key name from metadata
    ic_key = 'ic_type' if 'ic_type' in test_metadata[0] else 'distribution'
    
    # Load all R² time series
    print(f"\nAnalyzing R² evolution over time for {len(test_metadata)} test runs...")
    all_r2_timeseries = []
    
    for meta in test_metadata:
        run_name = meta["run_name"]
        r2_file = test_dir / run_name / "r2_vs_time.csv"
        if r2_file.exists():
            df = pd.read_csv(r2_file)
            df['test_id'] = run_name
            df['ic_type'] = meta[ic_key]
            all_r2_timeseries.append(df)
    
    if not all_r2_timeseries:
        print("⚠ No R² time series found - skipping time-resolved analysis")
        return {}
    
    r2_combined = pd.concat(all_r2_timeseries, ignore_index=True)
    times = sorted(r2_combined['time'].unique())
    
    # Compute time statistics
    time_stats_df = _compute_time_statistics(r2_combined, times, time_analysis_dir)
    
    # Generate plots
    _plot_mean_r2_over_time(time_stats_df, times, time_analysis_dir)
    _plot_r2_reconstructed_detailed(time_stats_df, times, time_analysis_dir)
    _plot_survival_curves(r2_combined, times, time_analysis_dir)
    
    # Compute degradation info
    recon_stats = time_stats_df[time_stats_df['metric'] == 'r2_reconstructed']
    degradation_info = _compute_degradation_info(recon_stats)
    
    print(f"✓ Time-resolved analysis complete")
    print(f"   Best: t={degradation_info['best_time']:.1f}s (R²={degradation_info['best_r2']:.3f})")
    print(f"   Worst: t={degradation_info['worst_time']:.1f}s (R²={degradation_info['worst_r2']:.3f})")
    
    # Additional analysis: Window count and contamination
    print("\nGenerating R² degradation analysis with window count...")
    temporal_analysis = _generate_window_analysis(
        r2_combined, time_stats_df, times, mvar_dir, data_dir, time_analysis_dir
    )
    
    degradation_info['temporal_analysis'] = temporal_analysis
    
    return degradation_info


def _compute_time_statistics(r2_combined, times, output_dir):
    """Compute statistics at each time point."""
    time_stats = []
    for t in times:
        subset = r2_combined[r2_combined['time'] == t]
        for metric in ['r2_reconstructed', 'r2_latent', 'r2_pod']:
            values = subset[metric].values
            time_stats.append({
                'time': t,
                'metric': metric,
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'p25': np.percentile(values, 25),
                'p75': np.percentile(values, 75),
                'min': np.min(values),
                'max': np.max(values)
            })
    
    time_stats_df = pd.DataFrame(time_stats)
    time_stats_df.to_csv(output_dir / "r2_statistics_over_time.csv", index=False)
    return time_stats_df


def _plot_mean_r2_over_time(time_stats_df, times, output_dir):
    """Plot mean R² over time for all metrics."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    metrics_plot = ['r2_reconstructed', 'r2_latent', 'r2_pod']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    labels = ['R² Reconstructed (Physical Space)', 'R² Latent (ROM Space)', 'R² POD (Basis Quality)']
    
    for metric, color, label in zip(metrics_plot, colors, labels):
        subset = time_stats_df[time_stats_df['metric'] == metric]
        ax.plot(subset['time'], subset['mean'], color=color, linewidth=3, 
               label=label, alpha=0.9, zorder=3)
        ax.fill_between(subset['time'], subset['p25'], subset['p75'], 
                        color=color, alpha=0.2, zorder=1)
    
    # Threshold lines
    ax.axhline(0.8, color='green', linestyle='--', linewidth=2, alpha=0.5, 
              label='Good (R²=0.8)', zorder=2)
    ax.axhline(0.5, color='orange', linestyle='--', linewidth=2, alpha=0.5, 
              label='Acceptable (R²=0.5)', zorder=2)
    ax.axhline(0.0, color='red', linestyle='--', linewidth=2, alpha=0.5, 
              label='Baseline (R²=0)', zorder=2)
    
    ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=14, fontweight='bold')
    ax.set_title('Time-Resolved R² - Mean ± IQR Across All Test Runs', 
                fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_xlim([min(times), max(times)])
    plt.tight_layout()
    plt.savefig(output_dir / "r2_mean_over_time.png", dpi=250, bbox_inches='tight')
    plt.close()


def _plot_r2_reconstructed_detailed(time_stats_df, times, output_dir):
    """Plot detailed R² reconstructed with all statistics."""
    fig, ax = plt.subplots(figsize=(14, 8))
    recon_stats = time_stats_df[time_stats_df['metric'] == 'r2_reconstructed']
    
    ax.plot(recon_stats['time'], recon_stats['mean'], color='#2E86AB', 
           linewidth=3.5, label='Mean', alpha=1.0, zorder=5)
    ax.plot(recon_stats['time'], recon_stats['median'], color='darkblue', 
           linewidth=2.5, linestyle='--', label='Median', alpha=0.8, zorder=4)
    ax.fill_between(recon_stats['time'], recon_stats['p25'], recon_stats['p75'], 
                   color='#2E86AB', alpha=0.25, label='IQR (25th-75th)', zorder=2)
    ax.fill_between(recon_stats['time'], recon_stats['min'], recon_stats['max'], 
                   color='#2E86AB', alpha=0.1, label='Min-Max Range', zorder=1)
    
    ax.axhline(0.8, color='green', linestyle=':', linewidth=2.5, alpha=0.6, label='Good (R²=0.8)', zorder=3)
    ax.axhline(0.5, color='orange', linestyle=':', linewidth=2.5, alpha=0.6, label='Acceptable (R²=0.5)', zorder=3)
    ax.axhline(0.0, color='red', linestyle=':', linewidth=2.5, alpha=0.6, label='Baseline (R²=0)', zorder=3)
    
    ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('R² Reconstructed', fontsize=14, fontweight='bold')
    ax.set_title('R² Reconstructed (Physical Space) - Detailed Time Evolution', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_xlim([min(times), max(times)])
    plt.tight_layout()
    plt.savefig(output_dir / "r2_reconstructed_detailed.png", dpi=250, bbox_inches='tight')
    plt.close()


def _plot_survival_curves(r2_combined, times, output_dir):
    """Plot performance survival curves."""
    fig, ax = plt.subplots(figsize=(14, 8))
    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    colors_survival = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(thresholds)))
    
    for threshold, color in zip(thresholds, colors_survival):
        survival = []
        for t in times:
            subset = r2_combined[r2_combined['time'] == t]['r2_reconstructed']
            fraction_above = np.mean(subset >= threshold)
            survival.append(fraction_above * 100)
        ax.plot(times, survival, linewidth=2.5, label=f'R² ≥ {threshold:.1f}', 
               color=color, alpha=0.9)
    
    ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage of Runs Above Threshold (%)', fontsize=14, fontweight='bold')
    ax.set_title('Performance Survival Curves - When Do Runs Fail?', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=10, ncol=2, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([min(times), max(times)])
    ax.set_ylim([0, 105])
    plt.tight_layout()
    plt.savefig(output_dir / "survival_curves.png", dpi=250, bbox_inches='tight')
    plt.close()


def _compute_degradation_info(recon_stats):
    """Compute degradation threshold information."""
    degradation_info = {}
    
    for threshold in [0.8, 0.5, 0.3, 0.0]:
        below_threshold = recon_stats[recon_stats['mean'] < threshold]
        if len(below_threshold) > 0:
            first_time = float(below_threshold.iloc[0]['time'])
            degradation_info[f'mean_r2_below_{threshold}'] = first_time
    
    # Best and worst time points
    best_idx = recon_stats['mean'].idxmax()
    worst_idx = recon_stats['mean'].idxmin()
    degradation_info['best_time'] = float(recon_stats.loc[best_idx, 'time'])
    degradation_info['best_r2'] = float(recon_stats.loc[best_idx, 'mean'])
    degradation_info['worst_time'] = float(recon_stats.loc[worst_idx, 'time'])
    degradation_info['worst_r2'] = float(recon_stats.loc[worst_idx, 'mean'])
    
    return degradation_info


def _generate_window_analysis(r2_combined, time_stats_df, times, mvar_dir, data_dir, output_dir):
    """Generate window count and contamination analysis."""
    # Load MVAR config
    mvar_data = np.load(mvar_dir / "mvar_model.npz")
    mvar_lag = int(mvar_data['p'])
    
    # Load simulation config
    try:
        with open(data_dir / "config_used.yaml", "r") as f:
            config = yaml.safe_load(f)
            dt = config['sim']['dt']
            T_train = config['sim']['T']
    except:
        dt = 0.1
        T_train = times[0] if len(times) > 0 else 2.0
    
    window_duration = mvar_lag * dt
    
    # Plot R² with window count
    _plot_r2_with_window_count(
        r2_combined, time_stats_df, times, mvar_lag, dt, T_train, window_duration, output_dir
    )
    
    # Generate temporal analysis
    temporal_analysis = _create_temporal_analysis(
        time_stats_df, times, mvar_lag, dt, T_train, window_duration, output_dir
    )
    
    return temporal_analysis


def _plot_r2_with_window_count(r2_combined, time_stats_df, times, mvar_lag, dt, T_train, window_duration, output_dir):
    """Plot R² degradation with window count on secondary axis."""
    fig, ax1 = plt.subplots(figsize=(16, 10))
    
    recon_stats = time_stats_df[time_stats_df['metric'] == 'r2_reconstructed']
    color = '#2E86AB'
    
    ax1.plot(recon_stats['time'], recon_stats['mean'], color=color, linewidth=3.5, 
            label='Mean R² Reconstructed', alpha=0.9, zorder=3, marker='o', markersize=6)
    ax1.fill_between(recon_stats['time'], recon_stats['p25'], recon_stats['p75'], 
                    color=color, alpha=0.2, zorder=1, label='25th-75th percentile')
    
    # Add individual runs (sample)
    for test_id in r2_combined['test_id'].unique()[:20]:
        subset = r2_combined[r2_combined['test_id'] == test_id]
        ax1.plot(subset['time'], subset['r2_reconstructed'], color=color, 
                linewidth=0.5, alpha=0.15, zorder=0)
    
    # Threshold lines
    ax1.axhline(0.8, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Good (R²=0.8)', zorder=2)
    ax1.axhline(0.5, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Acceptable (R²=0.5)', zorder=2)
    ax1.axhline(0.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Baseline (R²=0.0)', zorder=2)
    
    ax1.set_xlabel('Time (s)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('R² Reconstructed (Physical Space)', fontsize=16, fontweight='bold', color=color)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.set_xlim([min(times), max(times)])
    ax1.set_ylim([min(-0.5, recon_stats['mean'].min() - 0.1), 1.05])
    ax1.grid(True, alpha=0.3, zorder=0)
    ax1.legend(loc='upper left', fontsize=12, framealpha=0.95)
    
    # Secondary x-axis for window count
    ax2 = ax1.twiny()
    window_counts = [(t - T_train) / window_duration if t > T_train else 0 for t in times]
    ax2.set_xlim([min(window_counts), max(window_counts)])
    ax2.set_xlabel('Number of Training Windows (Forecast Duration / Window Size)', 
                  fontsize=14, fontweight='bold', color='#A23B72')
    ax2.tick_params(axis='x', labelcolor='#A23B72', labelsize=11)
    
    # Annotations for key degradation points
    for threshold in [0.8, 0.5, 0.0]:
        below = recon_stats[recon_stats['mean'] < threshold]
        if len(below) > 0:
            t_thresh = below.iloc[0]['time']
            windows_thresh = (t_thresh - T_train) / window_duration if t_thresh > T_train else 0
            ax1.axvline(t_thresh, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, zorder=1)
            ax1.annotate(f'R²<{threshold:.1f}\nt={t_thresh:.1f}s\n{windows_thresh:.1f} windows',
                       xy=(t_thresh, threshold), xytext=(10, 15),
                       textcoords='offset points', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1.5))
    
    plt.title(f'R² Degradation Over Time and Training Windows\n(Window Size = {mvar_lag} lags × {dt}s = {window_duration:.1f}s, Training Duration = {T_train:.1f}s)',
             fontsize=17, fontweight='bold', pad=40)
    plt.tight_layout()
    plt.savefig(output_dir / "r2_degradation_with_windows.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ R² degradation analysis with window count saved")


def _create_temporal_analysis(time_stats_df, times, mvar_lag, dt, T_train, window_duration, output_dir):
    """Create and save temporal analysis with contamination metrics."""
    recon_stats = time_stats_df[time_stats_df['metric'] == 'r2_reconstructed']
    
    steps_per_train_run = int(T_train / dt)
    samples_per_train_run = steps_per_train_run - mvar_lag
    
    temporal_analysis = {
        'config': {
            'dt': float(dt),
            'mvar_lag': int(mvar_lag),
            'T_train': float(T_train),
            'steps_per_train_run': int(steps_per_train_run),
            'window_duration': float(window_duration),
            'samples_per_train_run': int(samples_per_train_run)
        },
        'degradation_timeline': {},
        'autoregressive_contamination': []
    }
    
    # Degradation timeline
    for threshold in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
        below = recon_stats[recon_stats['mean'] < threshold]
        if len(below) > 0:
            t_thresh = float(below.iloc[0]['time'])
            time_beyond = t_thresh - T_train
            ratio_to_window = time_beyond / window_duration if window_duration > 0 else 0
            temporal_analysis['degradation_timeline'][f'r2_below_{threshold:.1f}'] = {
                'time': t_thresh,
                'time_beyond_training': time_beyond,
                'ratio_to_window': ratio_to_window
            }
    
    # Autoregressive contamination
    for t_pred in times[::max(1, len(times)//20)]:
        if t_pred <= T_train:
            continue
        steps_ahead = int((t_pred - T_train) / dt)
        predicted_in_window = min(steps_ahead, mvar_lag)
        true_in_window = max(0, mvar_lag - steps_ahead)
        
        temporal_analysis['autoregressive_contamination'].append({
            'time': float(t_pred),
            'steps_ahead': int(steps_ahead),
            'true_in_window': int(true_in_window),
            'predicted_in_window': int(predicted_in_window),
            'window_purity': float(true_in_window / mvar_lag) if mvar_lag > 0 else 0.0
        })
    
    # Save JSON
    with open(output_dir / "temporal_windows_analysis.json", 'w') as f:
        json.dump(temporal_analysis, f, indent=2)
    
    # Plot window contamination
    if len(temporal_analysis['autoregressive_contamination']) > 0:
        contam_df = pd.DataFrame(temporal_analysis['autoregressive_contamination'])
        _plot_window_contamination(contam_df, times, mvar_lag, window_duration, output_dir)
    
    print("✓ Temporal windows analysis complete")
    
    return temporal_analysis


def _plot_window_contamination(contam_df, times, mvar_lag, window_duration, output_dir):
    """Plot window contamination analysis."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Window purity
    ax1.plot(contam_df['time'], contam_df['window_purity'] * 100, 
            linewidth=3, color='#2E86AB', marker='o', markersize=5)
    ax1.fill_between(contam_df['time'], 0, contam_df['window_purity'] * 100,
                    color='#2E86AB', alpha=0.3)
    ax1.axhline(100, color='green', linestyle='--', linewidth=2, alpha=0.5, label='100% Ground Truth')
    ax1.axhline(50, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='50% Contaminated')
    ax1.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='100% Predicted')
    ax1.set_ylabel('Window Purity (%)\n(Ground Truth in Lag Window)', fontsize=13, fontweight='bold')
    ax1.set_title(f'Autoregressive Error Contamination Analysis\n(MVAR Lag = {mvar_lag}, Window = {window_duration:.1f}s)', 
                 fontsize=15, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])
    
    # Composition of lag window
    ax2.bar(contam_df['time'], contam_df['true_in_window'], 
           label='True Values', color='green', alpha=0.7, width=(times[1]-times[0])*0.8)
    ax2.bar(contam_df['time'], contam_df['predicted_in_window'], 
           bottom=contam_df['true_in_window'],
           label='Predicted Values', color='red', alpha=0.7, width=(times[1]-times[0])*0.8)
    ax2.axhline(mvar_lag, color='black', linestyle='--', linewidth=2, alpha=0.5, 
               label=f'Total Window Size ({mvar_lag})')
    ax2.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Values in Lag Window', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, mvar_lag * 1.1])
    
    plt.tight_layout()
    plt.savefig(output_dir / "window_contamination_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Window contamination analysis saved")
