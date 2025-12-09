"""
Analyze R² degradation over time across all test runs.
Shows when the model maintains acceptable performance and when it fails.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Paths
TEST_DIR = Path("oscar_output/best_run_extended_test/test")
OUTPUT_DIR = Path("predictions/best_run_extended_test/time_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("R² DEGRADATION ANALYSIS - Time-Resolved Performance")
print("="*80)

# Load all R² time series
all_r2_data = []
test_dirs = sorted([d for d in TEST_DIR.iterdir() if d.is_dir() and d.name.startswith("test_")])

print(f"\nLoading R² time series from {len(test_dirs)} test runs...")

for test_dir in test_dirs:
    r2_file = test_dir / "r2_vs_time.csv"
    if r2_file.exists():
        df = pd.read_csv(r2_file)
        df['test_id'] = test_dir.name
        all_r2_data.append(df)

r2_combined = pd.concat(all_r2_data, ignore_index=True)

# Get unique times
times = sorted(r2_combined['time'].unique())
n_runs = len(test_dirs)

print(f"✓ Loaded {n_runs} test runs")
print(f"✓ Time range: {min(times):.2f}s to {max(times):.2f}s ({len(times)} time points)")

# ============================================================================
# Analysis 1: Mean R² evolution over time
# ============================================================================

print("\n" + "="*80)
print("Time-Resolved Statistics")
print("="*80)

# Compute statistics at each time point
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

# Find when R² drops below thresholds
thresholds = [0.8, 0.5, 0.3, 0.0]

print("\nR² Reconstructed (Physical Space) Degradation:")
print("-" * 60)

recon_stats = time_stats_df[time_stats_df['metric'] == 'r2_reconstructed']

for threshold in thresholds:
    below_threshold = recon_stats[recon_stats['mean'] < threshold]
    if len(below_threshold) > 0:
        first_time = below_threshold.iloc[0]['time']
        print(f"  Mean R² < {threshold:0.1f}: t = {first_time:.1f}s")
    else:
        print(f"  Mean R² < {threshold:0.1f}: Never (always above)")

# Check best time point
best_idx = recon_stats['mean'].idxmax()
best_time = recon_stats.loc[best_idx, 'time']
best_r2 = recon_stats.loc[best_idx, 'mean']
print(f"\n  Best performance: t = {best_time:.1f}s (R² = {best_r2:.4f})")

# Check worst time point
worst_idx = recon_stats['mean'].idxmin()
worst_time = recon_stats.loc[worst_idx, 'time']
worst_r2 = recon_stats.loc[worst_idx, 'mean']
print(f"  Worst performance: t = {worst_time:.1f}s (R² = {worst_r2:.4f})")

# ============================================================================
# Plot 1: Mean R² over time (all metrics)
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 8))

metrics = ['r2_reconstructed', 'r2_latent', 'r2_pod']
colors = ['#2E86AB', '#A23B72', '#F18F01']
labels = ['R² Reconstructed (Physical Space)', 'R² Latent (ROM Space)', 'R² POD (Basis Quality)']

for metric, color, label in zip(metrics, colors, labels):
    subset = time_stats_df[time_stats_df['metric'] == metric]
    
    # Plot mean
    ax.plot(subset['time'], subset['mean'], color=color, linewidth=3, 
           label=label, alpha=0.9, zorder=3)
    
    # Shade IQR (25th-75th percentile)
    ax.fill_between(subset['time'], subset['p25'], subset['p75'], 
                    color=color, alpha=0.2, zorder=1)

# Add threshold lines
ax.axhline(0.8, color='green', linestyle='--', linewidth=2, alpha=0.5, 
          label='Good (R²=0.8)', zorder=2)
ax.axhline(0.5, color='orange', linestyle='--', linewidth=2, alpha=0.5, 
          label='Acceptable (R²=0.5)', zorder=2)
ax.axhline(0.0, color='red', linestyle='--', linewidth=2, alpha=0.5, 
          label='Baseline (R²=0)', zorder=2)

ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
ax.set_ylabel('R² Score', fontsize=14, fontweight='bold')
ax.set_title('Time-Resolved R² - Mean ± IQR Across 40 Test Runs', 
            fontsize=16, fontweight='bold')
ax.legend(loc='best', fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3, zorder=0)
ax.set_xlim([min(times), max(times)])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "r2_mean_over_time.png", dpi=250, bbox_inches='tight')
print(f"\n✓ Saved: {OUTPUT_DIR / 'r2_mean_over_time.png'}")
plt.close()

# ============================================================================
# Plot 2: Focused on R² Reconstructed with percentiles
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 8))

recon_stats = time_stats_df[time_stats_df['metric'] == 'r2_reconstructed']

# Plot mean
ax.plot(recon_stats['time'], recon_stats['mean'], color='#2E86AB', 
       linewidth=3.5, label='Mean', alpha=1.0, zorder=5)

# Plot median
ax.plot(recon_stats['time'], recon_stats['median'], color='darkblue', 
       linewidth=2.5, linestyle='--', label='Median', alpha=0.8, zorder=4)

# Shade quartiles
ax.fill_between(recon_stats['time'], recon_stats['p25'], recon_stats['p75'], 
               color='#2E86AB', alpha=0.25, label='IQR (25th-75th)', zorder=2)

# Shade min-max range
ax.fill_between(recon_stats['time'], recon_stats['min'], recon_stats['max'], 
               color='#2E86AB', alpha=0.1, label='Min-Max Range', zorder=1)

# Threshold lines
ax.axhline(0.8, color='green', linestyle=':', linewidth=2.5, alpha=0.6, 
          label='Good (R²=0.8)', zorder=3)
ax.axhline(0.5, color='orange', linestyle=':', linewidth=2.5, alpha=0.6, 
          label='Acceptable (R²=0.5)', zorder=3)
ax.axhline(0.0, color='red', linestyle=':', linewidth=2.5, alpha=0.6, 
          label='Baseline (R²=0)', zorder=3)

ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
ax.set_ylabel('R² Reconstructed', fontsize=14, fontweight='bold')
ax.set_title('R² Reconstructed (Physical Space) - Detailed Time Evolution', 
            fontsize=16, fontweight='bold')
ax.legend(loc='best', fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3, zorder=0)
ax.set_xlim([min(times), max(times)])

# Add text annotation for key statistics
text_str = f"Best: t={best_time:.1f}s (R²={best_r2:.3f})\n"
text_str += f"Worst: t={worst_time:.1f}s (R²={worst_r2:.3f})\n"
text_str += f"Runs: {n_runs}"
ax.text(0.98, 0.02, text_str, transform=ax.transAxes,
       verticalalignment='bottom', horizontalalignment='right',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85),
       fontsize=11)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "r2_reconstructed_detailed.png", dpi=250, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR / 'r2_reconstructed_detailed.png'}")
plt.close()

# ============================================================================
# Plot 3: Heatmap of R² over time per run
# ============================================================================

fig, ax = plt.subplots(figsize=(16, 10))

# Create matrix: rows=runs, cols=time points
pivot = r2_combined.pivot(index='test_id', columns='time', values='r2_reconstructed')
run_names = pivot.index.tolist()

# Sort runs by final R²
final_r2 = pivot.iloc[:, -1].values
sort_idx = np.argsort(final_r2)[::-1]  # Descending order
pivot_sorted = pivot.iloc[sort_idx]

# Plot heatmap
im = ax.imshow(pivot_sorted.values, aspect='auto', cmap='RdYlGn', 
              vmin=-2, vmax=1.0, interpolation='nearest')

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cbar.set_label('R² Score', fontsize=13, fontweight='bold')
cbar.ax.axhline(0.8, color='white', linestyle='--', linewidth=2, alpha=0.7)
cbar.ax.axhline(0.5, color='white', linestyle='--', linewidth=2, alpha=0.7)
cbar.ax.axhline(0.0, color='white', linestyle='--', linewidth=2, alpha=0.7)

# Labels
ax.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
ax.set_ylabel('Test Run (sorted by final R²)', fontsize=13, fontweight='bold')
ax.set_title('R² Reconstructed Heatmap - All Test Runs Over Time', 
            fontsize=15, fontweight='bold')

# Set x-axis ticks to show time
n_ticks = min(10, len(times))
tick_indices = np.linspace(0, len(times)-1, n_ticks, dtype=int)
ax.set_xticks(tick_indices)
ax.set_xticklabels([f"{times[i]:.1f}" for i in tick_indices], fontsize=10)

# Set y-axis ticks (show every 5th run)
y_tick_indices = np.arange(0, len(run_names), 5)
ax.set_yticks(y_tick_indices)
ax.set_yticklabels([pivot_sorted.index[i] for i in y_tick_indices], fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "r2_heatmap_all_runs.png", dpi=250, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR / 'r2_heatmap_all_runs.png'}")
plt.close()

# ============================================================================
# Plot 4: When does each run fail? (Survival curve)
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 8))

# For each threshold, compute survival: fraction of runs still above threshold
thresholds_detailed = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
colors_survival = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(thresholds_detailed)))

for threshold, color in zip(thresholds_detailed, colors_survival):
    survival = []
    for t in times:
        subset = r2_combined[r2_combined['time'] == t]['r2_reconstructed']
        fraction_above = np.mean(subset >= threshold)
        survival.append(fraction_above * 100)
    
    ax.plot(times, survival, linewidth=2.5, label=f'R² ≥ {threshold:.1f}', 
           color=color, alpha=0.9)

ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
ax.set_ylabel('Percentage of Runs Above Threshold (%)', fontsize=14, fontweight='bold')
ax.set_title('Performance Survival Curves - When Do Runs Fail?', 
            fontsize=16, fontweight='bold')
ax.legend(loc='best', fontsize=10, ncol=2, framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.set_xlim([min(times), max(times)])
ax.set_ylim([0, 105])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "survival_curves.png", dpi=250, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR / 'survival_curves.png'}")
plt.close()

# ============================================================================
# Save detailed statistics CSV
# ============================================================================

time_stats_df.to_csv(OUTPUT_DIR / "r2_statistics_over_time.csv", index=False)
print(f"✓ Saved: {OUTPUT_DIR / 'r2_statistics_over_time.csv'}")

# ============================================================================
# Print detailed report
# ============================================================================

print("\n" + "="*80)
print("DETAILED TIME-RESOLVED ANALYSIS")
print("="*80)

print("\nR² Reconstructed at Key Time Points:")
print("-" * 60)

key_times = [min(times), 2.0, 4.0, 6.0, 8.0, max(times)]
for t in key_times:
    if t in recon_stats['time'].values:
        row = recon_stats[recon_stats['time'] == t].iloc[0]
        print(f"  t = {t:5.1f}s: Mean = {row['mean']:7.4f} ± {row['std']:6.4f}  "
              f"[{row['min']:7.4f}, {row['max']:7.4f}]  Median = {row['median']:7.4f}")

print("\n" + "="*80)
print("SURVIVAL ANALYSIS - Percentage of Runs Above Threshold")
print("="*80)

survival_thresholds = [0.8, 0.5, 0.0]
survival_data = []

for threshold in survival_thresholds:
    print(f"\nR² ≥ {threshold:.1f}:")
    print("-" * 60)
    
    for t in [min(times), 2.0, 4.0, 6.0, 8.0, max(times)]:
        if t in times:
            subset = r2_combined[r2_combined['time'] == t]['r2_reconstructed']
            fraction_above = np.mean(subset >= threshold) * 100
            n_above = np.sum(subset >= threshold)
            print(f"  t = {t:5.1f}s: {fraction_above:5.1f}% ({n_above}/{n_runs} runs)")
            
            survival_data.append({
                'time': t,
                'threshold': threshold,
                'percentage': fraction_above,
                'n_runs_above': n_above,
                'n_runs_total': n_runs
            })

survival_df = pd.DataFrame(survival_data)
survival_df.to_csv(OUTPUT_DIR / "survival_analysis.csv", index=False)
print(f"\n✓ Saved: {OUTPUT_DIR / 'survival_analysis.csv'}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\n📂 All plots and data saved to: {OUTPUT_DIR}")
print("\nKey Files:")
print(f"  • r2_mean_over_time.png - Overview of all metrics")
print(f"  • r2_reconstructed_detailed.png - Detailed R² reconstructed")
print(f"  • r2_heatmap_all_runs.png - Individual run performance")
print(f"  • survival_curves.png - Failure analysis")
print(f"  • r2_statistics_over_time.csv - Detailed stats")
print(f"  • survival_analysis.csv - Survival data")

# Open output directory
import subprocess
subprocess.run(["open", str(OUTPUT_DIR)])
