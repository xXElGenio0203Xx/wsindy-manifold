"""
Summary Plots Module
====================

Generates aggregated summary plots across all runs.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def generate_summary_plots(metrics_df, ic_types, plots_dir):
    """
    Generate summary plots aggregated across all runs.
    
    Parameters
    ----------
    metrics_df : DataFrame
        Metrics for all runs
    ic_types : list
        List of IC type names
    plots_dir : Path
        Directory to save plots
    
    Returns
    -------
    None
        Saves plots to disk
    """
    
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # R² by IC type
    fig, ax = plt.subplots(figsize=(10, 6))
    ic_r2_data = [metrics_df[metrics_df["ic_type"] == ic]["r2"].values for ic in ic_types]
    bp = ax.boxplot(ic_r2_data, tick_labels=ic_types, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_xlabel('Initial Condition Type', fontsize=12)
    ax.set_title('MVAR-ROM Performance by IC Type', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(plots_dir / "r2_by_ic_type.png", dpi=200)
    plt.close()
    
    # Median L² error by IC type
    fig, ax = plt.subplots(figsize=(10, 6))
    ic_e2_data = [metrics_df[metrics_df["ic_type"] == ic]["median_e2"].values for ic in ic_types]
    bp = ax.boxplot(ic_e2_data, tick_labels=ic_types, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightcoral')
    ax.set_ylabel('Median L² Error', fontsize=12)
    ax.set_xlabel('Initial Condition Type', fontsize=12)
    ax.set_title('MVAR-ROM Error by IC Type', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(plots_dir / "error_by_ic_type.png", dpi=200)
    plt.close()
    
    print("✓ Summary plots saved")
