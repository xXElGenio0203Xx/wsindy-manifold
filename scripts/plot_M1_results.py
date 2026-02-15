#!/usr/bin/env python3
"""
SUITE M1 — Visualization of mass constraint results.
Produces:
  1. R² vs time curves (raw vs each fix) — one panel per horizon
  2. Mass error % vs time curves
  3. Bar chart: ΔR² by method and horizon
"""

import numpy as np
import csv
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'oscar_output'

# ROM_dt = 0.12s (dt=0.04 × subsample=3)
ROM_DT = 0.12
FORECAST_START_TIME = 0.60  # 5 frames × 0.12

EXPERIMENTS = {
    'H162': {
        'base': 'K4_v1_sqrtD19_p5_H162',
        'methods': {
            'scale':        'M1-1_eval_massFix_scale',
            'offset+clamp': 'M1-2_eval_massFix_offsetClamp',
            'simplex L2':   'M1-3_eval_massFix_simplex',
        },
    },
    'H312': {
        'base': 'K5_v1_sqrtD19_p5_H312',
        'methods': {
            'scale':        'M1-4_eval_massFix_scale_H312',
            'offset+clamp': 'M1-5_eval_massFix_offsetClamp_H312',
            'simplex L2':   'M1-6_eval_massFix_simplex_H312',
        },
    },
}

COLORS = {
    'raw':          '#888888',
    'scale':        '#e74c3c',
    'offset+clamp': '#2ecc71',
    'simplex L2':   '#3498db',
}


def load_timeseries(exp_name):
    """Load metrics_vs_time.csv → dict of arrays."""
    path = OUT / exp_name / 'metrics_vs_time.csv'
    data = {'frame': [], 'r2_raw': [], 'r2_fix': [],
            'mass_err_raw': [], 'mass_err_fix': [],
            'neg_pct_raw': [], 'neg_pct_fix': []}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in data:
                data[k].append(float(row[k]))
    return {k: np.array(v) for k, v in data.items()}


def plot_r2_vs_time():
    """One panel per horizon, all methods overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, (horizon, cfg) in zip(axes, EXPERIMENTS.items()):
        # Load each method
        first_ts = None
        for method, exp_name in cfg['methods'].items():
            ts = load_timeseries(exp_name)
            frames = ts['frame']
            times = frames * ROM_DT + FORECAST_START_TIME

            if first_ts is None:
                first_ts = ts
                # Plot raw (same for all methods)
                ax.plot(times, ts['r2_raw'], color=COLORS['raw'],
                        linewidth=2, label='uncorrected', zorder=1)

            ax.plot(times, ts['r2_fix'], color=COLORS[method],
                    linewidth=2, label=method, zorder=2)

        ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_title(f'{horizon}  (base: {cfg["base"]})', fontsize=13)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('R² (per-frame, avg over 26 tests)', fontsize=12)
    fig.suptitle('Suite M1 — R² vs Time: Mass Constraint Strategies', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / 'M1_r2_vs_time.png', dpi=150, bbox_inches='tight')
    print(f"  → Saved {OUT / 'M1_r2_vs_time.png'}")
    plt.close(fig)


def plot_mass_vs_time():
    """Mass error % vs time."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, (horizon, cfg) in zip(axes, EXPERIMENTS.items()):
        first_ts = None
        for method, exp_name in cfg['methods'].items():
            ts = load_timeseries(exp_name)
            frames = ts['frame']
            times = frames * ROM_DT + FORECAST_START_TIME

            if first_ts is None:
                first_ts = ts
                ax.plot(times, ts['mass_err_raw'], color=COLORS['raw'],
                        linewidth=2, label='uncorrected', zorder=1)

            ax.plot(times, ts['mass_err_fix'], color=COLORS[method],
                    linewidth=1.5, label=method, zorder=2, linestyle='--')

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_title(f'{horizon}', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Mass error %', fontsize=12)
    fig.suptitle('Suite M1 — Mass Error vs Time', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / 'M1_mass_error_vs_time.png', dpi=150, bbox_inches='tight')
    print(f"  → Saved {OUT / 'M1_mass_error_vs_time.png'}")
    plt.close(fig)


def plot_delta_r2_bar():
    """Bar chart of ΔR² by method and horizon."""
    combined_path = OUT / 'M1_combined_results.json'
    with open(combined_path) as f:
        summaries = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 5))

    methods = ['scale', 'offset+clamp', 'simplex L2']
    method_labels = {
        'scale_to_M0': 'scale',
        'offset_then_clamp_to_M0': 'offset+clamp',
        'simplex_L2_projection': 'simplex L2',
    }

    x = np.arange(2)  # H162, H312
    width = 0.25

    for i, method in enumerate(methods):
        deltas = []
        for horizon in ['H162', 'H312']:
            # Find matching summary
            for s in summaries:
                if method_labels.get(s['constraint']) == method and horizon.lower() in s['experiment_name'].lower():
                    deltas.append(s['r2_delta'])
                    break
                # Also check by explicit horizon for H162 (no "h162" in name for first 3)
                if method_labels.get(s['constraint']) == method and 'H312' not in s['experiment_name'] and horizon == 'H162':
                    deltas.append(s['r2_delta'])
                    break

        ax.bar(x + i * width, deltas, width, label=method, color=COLORS[method])

    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel('ΔR² (fix − raw)', fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels(['H162\n(K4, √ρ d=19)', 'H312\n(K5, √ρ d=19)'], fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Suite M1 — ΔR² by Mass Constraint Method', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / 'M1_delta_r2_bar.png', dpi=150, bbox_inches='tight')
    print(f"  → Saved {OUT / 'M1_delta_r2_bar.png'}")
    plt.close(fig)


if __name__ == '__main__':
    plot_r2_vs_time()
    plot_mass_vs_time()
    plot_delta_r2_bar()
    print("\nAll M1 visualizations complete.")
