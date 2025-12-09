"""
POD Visualization Module
========================

Generates POD singular value and energy spectrum plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def generate_pod_plots(pod_data, plots_dir, n_train):
    """
    Generate POD singular value and energy spectrum plots.
    
    Parameters
    ----------
    pod_data : dict
        Dictionary containing POD data with keys:
        - 'U': POD basis vectors
        - 'singular_values': Selected singular values
        - 'all_singular_values': All singular values
        - 'energy_ratio': Energy captured by selected modes
        - 'cumulative_ratio': Cumulative energy ratio array
    plots_dir : Path
        Directory to save plots
    n_train : int
        Number of training runs
    
    Returns
    -------
    None
        Saves plots to disk
    """
    
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract data
    singular_values = pod_data['singular_values']
    all_singular_values = pod_data['all_singular_values']
    cumulative_energy = pod_data['cumulative_energy']
    
    R_POD = len(singular_values)
    actual_energy = cumulative_energy[R_POD - 1]
    
    # Plot 1: POD singular values (log scale)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(range(1, len(all_singular_values)+1), all_singular_values, 'o-', 
                linewidth=2, markersize=2, color='steelblue', alpha=0.6)
    ax.axvline(R_POD, color='r', linestyle='--', linewidth=2, 
               label=f'Selected r={R_POD} ({actual_energy*100:.2f}% energy)')
    ax.set_xlabel('POD Mode Index', fontsize=12)
    ax.set_ylabel('Singular Value (log scale)', fontsize=12)
    ax.set_title('POD Singular Values', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10)
    x_max = max(R_POD * 1.2, min(500, len(all_singular_values)))
    ax.set_xlim(0, x_max)
    plt.tight_layout()
    plt.savefig(plots_dir / "pod_singular_values.png", dpi=200)
    plt.close()
    
    # Plot 2: Cumulative energy
    fig, ax = plt.subplots(figsize=(10, 6))
    cumulative_energy_pct = 100 * cumulative_energy
    ax.plot(range(1, len(all_singular_values)+1), cumulative_energy_pct, 'o-', 
            linewidth=2, markersize=2, color='forestgreen', alpha=0.7)
    ax.axvline(R_POD, color='r', linestyle='--', linewidth=2, label=f'Selected r={R_POD}')
    ax.axhline(90, color='orange', linestyle=':', alpha=0.7, linewidth=2, label='90% energy')
    ax.axhline(95, color='purple', linestyle=':', alpha=0.7, linewidth=2, label='95% energy')
    ax.axhline(actual_energy*100, color='darkgreen', linestyle=':', alpha=0.7, linewidth=2, 
               label=f'{actual_energy*100:.1f}% energy (target)')
    ax.set_xlabel('Number of POD Modes', fontsize=12)
    ax.set_ylabel('Cumulative Energy Captured (%)', fontsize=12)
    ax.set_title(f'POD Energy Spectrum ({n_train} training runs)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    x_max = max(R_POD * 1.2, min(500, len(all_singular_values)))
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(plots_dir / "pod_energy.png", dpi=200)
    plt.close()
    
    print("✓ POD plots saved")
