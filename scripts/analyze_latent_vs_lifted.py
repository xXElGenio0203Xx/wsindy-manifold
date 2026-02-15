#!/usr/bin/env python3
"""
Latent vs Lifted R² Analysis
=============================

Cross-experiment comparison of R² in latent (ROM) space vs lifted (density) space.
Shows the "lifting gap" — how much accuracy is lost when projecting back to physical space.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import json

# Experiments to analyze
EXPERIMENTS = {
    'V2.2\n(d=10,scl=0.97)': 'synthesis_v2_2',
    'V2.3\n(d=10,no scl,no clp)': 'synthesis_v2_3',
    'V2.4 🏆\n(d=10,no scl,clp)': 'synthesis_v2_4',
    'V2.4.2\n(α=0.01)': 'synthesis_v2_4_2',
    'V2.5\n(d=25,scl=0.97)': 'synthesis_v2_5',
    'V3.1\n(d=43,p=5)': 'synthesis_v3_1',
    'V3.2\n(mod,d=10)': 'synthesis_v3_2',
    'V3.3\n(mod,d=20)': 'synthesis_v3_3',
    'V3.4\n(ext,d=10)': 'synthesis_v3_4',
}


def load_experiment_data(base_dir):
    """Load MVAR test_results.csv and r2_vs_time.csv for all experiments."""
    results = {}
    
    for label, exp in EXPERIMENTS.items():
        exp_dir = base_dir / exp
        mvar_csv = exp_dir / "MVAR" / "test_results.csv"
        
        if not mvar_csv.exists():
            print(f"  ⚠ Skipping {exp}: no MVAR/test_results.csv")
            continue
        
        df = pd.read_csv(mvar_csv)
        
        # Load time-resolved data
        test_dir = exp_dir / "test"
        time_dfs = []
        for _, row in df.iterrows():
            test_id = row['test_id']
            # test_id can be int (0,1,2) or string (test_000) — normalize
            test_id_str = f"test_{int(test_id):03d}" if isinstance(test_id, (int, float, np.integer, np.floating)) else str(test_id)
            r2_file = test_dir / test_id_str / "r2_vs_time.csv"
            if r2_file.exists():
                tdf = pd.read_csv(r2_file)
                tdf['test_id'] = test_id_str
                time_dfs.append(tdf)
        
        # Compute spectral radius
        mvar_npz = exp_dir / "MVAR" / "mvar_model.npz"
        rho = None
        if mvar_npz.exists():
            data = np.load(mvar_npz)
            A_coef = data['A_companion']
            R_POD = int(data['r'])
            P_LAG = int(data['p'])
            companion_dim = P_LAG * R_POD
            C = np.zeros((companion_dim, companion_dim))
            C[:R_POD, :] = A_coef
            for k in range(P_LAG - 1):
                C[(k+1)*R_POD:(k+2)*R_POD, k*R_POD:(k+1)*R_POD] = np.eye(R_POD)
            rho = np.max(np.abs(np.linalg.eigvals(C)))
        
        results[label] = {
            'exp_name': exp,
            'summary': df,
            'time_resolved': pd.concat(time_dfs, ignore_index=True) if time_dfs else None,
            'spectral_radius': rho,
            'r2_lifted': df['r2_reconstructed'].mean(),
            'r2_latent': df['r2_latent'].mean(),
            'r2_pod': df['r2_pod'].mean(),
        }
        
        rho_str = f'ρ={rho:.4f}' if rho is not None else 'ρ=N/A'
        print(f"  ✓ {exp}: R²_lift={results[label]['r2_lifted']:.4f}, "
              f"R²_lat={results[label]['r2_latent']:.4f}, "
              f"R²_pod={results[label]['r2_pod']:.4f}, "
              f"{rho_str}")
    
    return results


def plot_latent_vs_lifted_bars(results, output_dir):
    """Bar chart comparing R²_latent, R²_lifted, and R²_pod across experiments."""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    labels = list(results.keys())
    x = np.arange(len(labels))
    width = 0.25
    
    r2_lifted = [results[l]['r2_lifted'] for l in labels]
    r2_latent = [results[l]['r2_latent'] for l in labels]
    r2_pod = [results[l]['r2_pod'] for l in labels]
    
    bars1 = ax.bar(x - width, r2_lifted, width, label='R² Lifted (Density Space)', 
                   color='#2E86AB', alpha=0.85, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, r2_latent, width, label='R² Latent (ROM Space)',
                   color='#A23B72', alpha=0.85, edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, r2_pod, width, label='R² POD (Basis Quality)',
                   color='#F18F01', alpha=0.85, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height - 0.03,
                       f'{height:.3f}', ha='center', va='top', fontsize=7.5, fontweight='bold')
    
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_xlabel('Experiment', fontsize=13, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=13, fontweight='bold')
    ax.set_title('Latent vs Lifted R² Across All Experiments (MVAR)', 
                fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, ha='center')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / "latent_vs_lifted_r2_bars.png", dpi=250, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: latent_vs_lifted_r2_bars.png")


def plot_lifting_gap_analysis(results, output_dir):
    """Scatter plot showing gap = R²_latent - R²_lifted vs spectral radius."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    labels = list(results.keys())
    gaps = [results[l]['r2_latent'] - results[l]['r2_lifted'] for l in labels]
    rhos = [results[l]['spectral_radius'] for l in labels]
    r2_lifted = [results[l]['r2_lifted'] for l in labels]
    r2_pod = [results[l]['r2_pod'] for l in labels]
    
    # Panel 1: Gap vs spectral radius
    ax = axes[0]
    colors = ['red' if r > 1.0 else 'green' for r in rhos]
    for i, (rho, gap, label) in enumerate(zip(rhos, gaps, labels)):
        ax.scatter(rho, gap, s=150, c=colors[i], edgecolor='black', 
                  linewidth=1.5, zorder=5)
        short_label = label.split('\n')[0]
        ax.annotate(short_label, (rho, gap), textcoords="offset points",
                   xytext=(8, 8), fontsize=8.5, fontweight='bold')
    
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Stability boundary (ρ=1)')
    ax.set_xlabel('Spectral Radius ρ', fontsize=13, fontweight='bold')
    ax.set_ylabel('Lifting Gap (R²_latent − R²_lifted)', fontsize=13, fontweight='bold')
    ax.set_title('Does Lifting Hurt or Help?', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: R²_lifted vs R²_pod (how much of the POD ceiling is achieved)
    ax = axes[1]
    for i, (pod, lift, label) in enumerate(zip(r2_pod, r2_lifted, labels)):
        efficiency = lift / pod * 100 if pod > 0 else 0
        ax.scatter(pod, lift, s=150, c='#2E86AB', edgecolor='black', 
                  linewidth=1.5, zorder=5)
        short_label = label.split('\n')[0]
        ax.annotate(f'{short_label}\n({efficiency:.0f}%)', (pod, lift), 
                   textcoords="offset points", xytext=(8, 8), fontsize=8, fontweight='bold')
    
    # Diagonal (perfect = lifted matches POD ceiling)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect lifting (R²_lift = R²_pod)')
    ax.set_xlabel('R² POD (Basis Quality Ceiling)', fontsize=13, fontweight='bold')
    ax.set_ylabel('R² Lifted (Achieved)', fontsize=13, fontweight='bold')
    ax.set_title('Lifting Efficiency: Achieved vs Ceiling', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "lifting_gap_analysis.png", dpi=250, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: lifting_gap_analysis.png")


def plot_time_resolved_latent_vs_lifted(results, output_dir):
    """Plot R² over time in both latent and lifted space for each experiment."""
    # Select experiments with time-resolved data
    exps_with_time = {k: v for k, v in results.items() if v['time_resolved'] is not None}
    
    if not exps_with_time:
        print("  ⚠ No time-resolved data available")
        return
    
    n_exps = len(exps_with_time)
    n_cols = min(3, n_exps)
    n_rows = (n_exps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5.5*n_rows), squeeze=False)
    
    for idx, (label, data) in enumerate(exps_with_time.items()):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        
        tdf = data['time_resolved']
        times = sorted(tdf['time'].unique())
        
        # Compute mean across test runs at each time
        means_lifted = []
        means_latent = []
        means_pod = []
        iqr_lifted_lo, iqr_lifted_hi = [], []
        iqr_latent_lo, iqr_latent_hi = [], []
        
        for t in times:
            subset = tdf[tdf['time'] == t]
            means_lifted.append(subset['r2_reconstructed'].mean())
            means_latent.append(subset['r2_latent'].mean())
            means_pod.append(subset['r2_pod'].mean())
            iqr_lifted_lo.append(subset['r2_reconstructed'].quantile(0.25))
            iqr_lifted_hi.append(subset['r2_reconstructed'].quantile(0.75))
            iqr_latent_lo.append(subset['r2_latent'].quantile(0.25))
            iqr_latent_hi.append(subset['r2_latent'].quantile(0.75))
        
        ax.plot(times, means_lifted, color='#2E86AB', linewidth=2.5, 
               label='R² Lifted', alpha=0.9, zorder=3)
        ax.fill_between(times, iqr_lifted_lo, iqr_lifted_hi,
                        color='#2E86AB', alpha=0.15, zorder=1)
        
        ax.plot(times, means_latent, color='#A23B72', linewidth=2.5, 
               label='R² Latent', alpha=0.9, zorder=3)
        ax.fill_between(times, iqr_latent_lo, iqr_latent_hi,
                        color='#A23B72', alpha=0.15, zorder=1)
        
        ax.plot(times, means_pod, color='#F18F01', linewidth=2, 
               label='R² POD', linestyle='--', alpha=0.7, zorder=2)
        
        ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.4)
        ax.axhline(0.5, color='orange', linestyle=':', linewidth=1, alpha=0.4)
        
        short_label = label.replace('\n', ' ')
        rho = data['spectral_radius']
        rho_str = f', ρ={rho:.3f}' if rho else ''
        ax.set_title(f'{short_label}{rho_str}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('R²', fontsize=10)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([min(times), max(times)])
    
    # Hide unused axes
    for idx in range(n_exps, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)
    
    plt.suptitle('Time-Resolved R²: Latent vs Lifted vs POD Ceiling (MVAR)',
                fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "time_resolved_latent_vs_lifted.png", dpi=250, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: time_resolved_latent_vs_lifted.png")


def plot_spectral_radius_vs_performance(results, output_dir):
    """Scatter plot: spectral radius vs all R² metrics."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    labels = list(results.keys())
    rhos = [results[l]['spectral_radius'] for l in labels]
    r2_lifted = [results[l]['r2_lifted'] for l in labels]
    r2_latent = [results[l]['r2_latent'] for l in labels]
    
    ax.scatter(rhos, r2_lifted, s=180, c='#2E86AB', edgecolor='black',
              linewidth=1.5, zorder=5, label='R² Lifted', marker='o')
    ax.scatter(rhos, r2_latent, s=180, c='#A23B72', edgecolor='black',
              linewidth=1.5, zorder=5, label='R² Latent', marker='s')
    
    for i, label in enumerate(labels):
        short = label.split('\n')[0]
        ax.annotate(short, (rhos[i], r2_lifted[i]), textcoords="offset points",
                   xytext=(-10, 12), fontsize=8, color='#2E86AB', fontweight='bold')
        ax.annotate(short, (rhos[i], r2_latent[i]), textcoords="offset points",
                   xytext=(-10, -15), fontsize=8, color='#A23B72', fontweight='bold')
    
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, 
              label='Stability boundary (ρ=1)')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    ax.set_xlabel('Spectral Radius ρ', fontsize=14, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=14, fontweight='bold')
    ax.set_title('Spectral Radius vs Performance: Why Scaling to 0.97 Hurts', 
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "spectral_radius_vs_performance.png", dpi=250, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: spectral_radius_vs_performance.png")


def main():
    base_dir = Path("oscar_output")
    output_dir = Path("predictions") / "cross_experiment_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("LATENT vs LIFTED R² — Cross-Experiment Analysis")
    print("=" * 80)
    
    print("\nLoading experiment data...")
    results = load_experiment_data(base_dir)
    
    if not results:
        print("❌ No experiment data found!")
        return
    
    print(f"\n✓ Loaded {len(results)} experiments")
    
    # Summary table
    print("\n" + "=" * 100)
    print(f"{'Experiment':<28} {'R²_lifted':>10} {'R²_latent':>10} {'R²_pod':>10} "
          f"{'Gap':>10} {'ρ':>8} {'Scaling?':>10}")
    print("-" * 100)
    for label, data in results.items():
        short = label.replace('\n', ' ')
        gap = data['r2_latent'] - data['r2_lifted']
        rho = data['spectral_radius']
        rho_str = f'{rho:8.4f}' if rho is not None else '     N/A'
        scaled = 'YES' if rho is not None and abs(rho - 0.97) < 0.001 else 'NO'
        print(f"{short:<28} {data['r2_lifted']:>+10.4f} {data['r2_latent']:>+10.4f} "
              f"{data['r2_pod']:>10.4f} {gap:>+10.4f} {rho_str} {scaled:>10}")
    print("=" * 100)
    
    print("\nGenerating plots...")
    plot_latent_vs_lifted_bars(results, output_dir)
    plot_lifting_gap_analysis(results, output_dir)
    plot_time_resolved_latent_vs_lifted(results, output_dir)
    plot_spectral_radius_vs_performance(results, output_dir)
    
    print(f"\n✅ All plots saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
