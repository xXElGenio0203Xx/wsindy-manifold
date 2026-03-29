#!/usr/bin/env python3
"""
sPOD Diagnostics — Shift-alignment motivation plots
====================================================

Produces three key figures that make the alignment argument airtight
(cf. Reiss et al. 2018, arXiv:1512.01985v3):

1. **Singular value decay** — Raw vs Aligned
   Shows that alignment concentrates energy in fewer modes (fast decay).

2. **Cumulative energy** — How many modes are needed for X% energy
   Directly quantifies the rank reduction.

3. **Phase dynamics Δ(t)** — 1D shift time-series analysis
   Treats the translational shift as a separate dynamical variable.
   Demonstrates that shifts have simple (smooth, low-frequency) dynamics
   that can be modeled with a low-order AR model.

Usage:
    python scripts/spod_diagnostics.py [--output_dir artifacts/thesis_figures]

Requires local data:
    CUR1 (aligned) + CUR2 (no-align)  — fast curvature dynamics
    DYN experiments (all use alignment) — for DYN-suite SVD comparison
    ABL3/4 and ABL7/8 (if available)   — standard Vicsek align vs no-align
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import json
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

# ─── Color palette ──────────────────────────────────────────────────────────
COLORS = {
    'aligned': '#2196F3',     # Blue
    'noalign': '#F44336',     # Red
    'aligned2': '#1565C0',    # Darker blue
    'noalign2': '#C62828',    # Darker red
    'dyn': '#4CAF50',         # Green for DYN suite
    'phase': '#FF9800',       # Orange for phase dynamics
    'shape': '#9C27B0',       # Purple for shape dynamics
}

FIGSIZE_WIDE = (14, 5)
FIGSIZE_TALL = (10, 12)
FIGSIZE_SINGLE = (7, 5)


def load_singular_values(exp_dir):
    """Load singular values from pod_basis.npz."""
    pod_path = Path(exp_dir) / 'rom_common' / 'pod_basis.npz'
    if not pod_path.exists():
        pod_path = Path(exp_dir) / 'mvar' / 'pod_basis.npz'
    if not pod_path.exists():
        return None
    d = np.load(pod_path)
    if 'all_singular_values' in d:
        return d['all_singular_values']
    return d['singular_values']


def load_shift_data(exp_dir):
    """Load shift alignment data from shift_align.npz."""
    sa_path = Path(exp_dir) / 'rom_common' / 'shift_align.npz'
    if not sa_path.exists():
        return None
    return np.load(sa_path, allow_pickle=True)


def load_r2_vs_time(exp_dir, model='mvar'):
    """Load time-resolved R² — aggregate across test runs if per-run files exist."""
    import pandas as pd
    
    # Try model-specific aggregate first
    for pattern in [f'r2_vs_time_{model}.csv', 'r2_vs_time.csv']:
        path = Path(exp_dir) / 'MVAR' / pattern
        if not path.exists():
            path = Path(exp_dir) / 'mvar' / pattern
        if path.exists():
            return pd.read_csv(path)
    
    # Fall back: aggregate per-test-run files
    test_dir = Path(exp_dir) / 'test'
    if not test_dir.exists():
        return None
    
    dfs = []
    for run_dir in sorted(test_dir.iterdir()):
        r2_file = run_dir / 'r2_vs_time.csv'
        if not r2_file.exists():
            r2_file = run_dir / f'r2_vs_time_{model}.csv'
        if r2_file.exists():
            df = pd.read_csv(r2_file)
            dfs.append(df)
    
    if len(dfs) == 0:
        return None
    
    # Aggregate: mean and std across runs at each timestep
    # All runs should have the same 'time' column
    ref_times = dfs[0]['time'].values
    r2_col = 'r2_reconstructed' if 'r2_reconstructed' in dfs[0].columns else dfs[0].columns[1]
    
    r2_matrix = np.array([df[r2_col].values[:len(ref_times)] for df in dfs])
    
    result = pd.DataFrame({
        'time': ref_times,
        'mean_r2': r2_matrix.mean(axis=0),
        'std_r2': r2_matrix.std(axis=0),
        'min_r2': r2_matrix.min(axis=0),
        'max_r2': r2_matrix.max(axis=0),
        'n_runs': len(dfs),
    })
    return result


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Singular Value Decay — Raw vs Aligned
# ═════════════════════════════════════════════════════════════════════════════

def plot_svd_decay(pairs, output_dir, max_modes=60):
    """
    Plot singular value decay for each aligned/unaligned pair.
    
    pairs: list of dicts with keys:
        'label', 'aligned_dir', 'noalign_dir', 'transform'
    """
    n_pairs = len(pairs)
    fig, axes = plt.subplots(1, n_pairs + 1, figsize=(5 * (n_pairs + 1), 5))
    if n_pairs == 0:
        print("  No pairs to plot.")
        return
    if n_pairs + 1 == 1:
        axes = [axes]
    
    all_aligned = []
    all_noalign = []
    
    for i, pair in enumerate(pairs):
        ax = axes[i]
        sv_align = load_singular_values(pair['aligned_dir'])
        sv_noalign = load_singular_values(pair['noalign_dir'])
        
        if sv_align is None or sv_noalign is None:
            ax.text(0.5, 0.5, f"Missing data\n{pair['label']}", 
                   transform=ax.transAxes, ha='center', va='center')
            continue
        
        n = min(max_modes, len(sv_align), len(sv_noalign))
        modes = np.arange(1, n + 1)
        
        # Normalize by first SV for comparison
        sv_align_norm = (sv_align[:n] / sv_align[0]) ** 2
        sv_noalign_norm = (sv_noalign[:n] / sv_noalign[0]) ** 2
        
        all_aligned.append(sv_align_norm)
        all_noalign.append(sv_noalign_norm)
        
        ax.semilogy(modes, sv_noalign_norm, 'o-', color=COLORS['noalign'],
                    markersize=3, linewidth=1.5, label='No alignment', alpha=0.8)
        ax.semilogy(modes, sv_align_norm, 's-', color=COLORS['aligned'],
                    markersize=3, linewidth=1.5, label='Aligned', alpha=0.8)
        
        ax.set_xlabel('Mode index $i$')
        ax.set_ylabel(r'$(\sigma_i / \sigma_1)^2$')
        ax.set_title(pair['label'], fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, n)
        ax.set_ylim(bottom=1e-6)
    
    # Summary panel: cumulative energy comparison
    ax = axes[-1]
    for i, pair in enumerate(pairs):
        sv_align = load_singular_values(pair['aligned_dir'])
        sv_noalign = load_singular_values(pair['noalign_dir'])
        if sv_align is None or sv_noalign is None:
            continue
        
        n = min(max_modes, len(sv_align), len(sv_noalign))
        cum_align = np.cumsum(sv_align[:n]**2) / np.sum(sv_align**2)
        cum_noalign = np.cumsum(sv_noalign[:n]**2) / np.sum(sv_noalign**2)
        
        alpha = 0.6 + 0.4 * i / max(1, n_pairs - 1)
        ax.plot(np.arange(1, n+1), cum_noalign, '--', color=COLORS['noalign'],
               alpha=alpha, linewidth=1.5, label=f'No align ({pair["label"]})')
        ax.plot(np.arange(1, n+1), cum_align, '-', color=COLORS['aligned'],
               alpha=alpha, linewidth=1.5, label=f'Aligned ({pair["label"]})')
    
    ax.axhline(0.99, color='gray', linestyle=':', alpha=0.5, label='99% energy')
    ax.axhline(0.995, color='gray', linestyle='--', alpha=0.5, label='99.5% energy')
    ax.set_xlabel('Number of modes $d$')
    ax.set_ylabel('Cumulative energy ratio')
    ax.set_title('Energy Concentration', fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, min(40, n))
    ax.set_ylim(0.8, 1.005)
    
    fig.suptitle('Singular Value Decay: Raw Frame vs Shift-Aligned Frame\n'
                 r'(sPOD effect: alignment $\Rightarrow$ faster energy concentration)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    
    path = output_dir / 'svd_decay_raw_vs_aligned.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    path_pdf = output_dir / 'svd_decay_raw_vs_aligned.pdf'
    fig.savefig(path_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {path}")
    return path


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2: DYN Suite — SVD comparison across dynamics regimes
# ═════════════════════════════════════════════════════════════════════════════

def plot_dyn_suite_svd(dyn_experiments, output_dir, max_modes=30):
    """
    Plot SVD decay for all DYN experiments together.
    Shows how different dynamics create different spectral structures.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    
    cmap = plt.cm.tab10
    exp_data = []
    
    for i, (exp_name, exp_dir) in enumerate(dyn_experiments):
        sv = load_singular_values(exp_dir)
        if sv is None:
            continue
        n = min(max_modes, len(sv))
        sv_norm = (sv[:n] / sv[0]) ** 2
        cum = np.cumsum(sv[:n]**2) / np.sum(sv**2)
        
        color = cmap(i / max(1, len(dyn_experiments) - 1))
        short_name = exp_name.replace('_v2', '').replace('DYN', 'D')
        
        ax1.semilogy(np.arange(1, n+1), sv_norm, 'o-', color=color,
                    markersize=3, linewidth=1.5, label=short_name, alpha=0.8)
        ax2.plot(np.arange(1, n+1), cum, '-', color=color,
                linewidth=1.5, label=short_name, alpha=0.8)
        
        # Find mode count for 99% energy
        idx_99 = np.searchsorted(cum, 0.99)
        exp_data.append((short_name, idx_99 + 1, cum[min(18, n-1)]))
    
    ax1.set_xlabel('Mode index $i$')
    ax1.set_ylabel(r'$(\sigma_i / \sigma_1)^2$')
    ax1.set_title('Singular Value Decay by Dynamics Type')
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, max_modes)
    
    ax2.axhline(0.99, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(0.995, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Number of modes $d$')
    ax2.set_ylabel('Cumulative energy ratio')
    ax2.set_title('Cumulative Energy by Dynamics Type')
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, max_modes)
    ax2.set_ylim(0.85, 1.005)
    
    fig.suptitle('DYN Suite — Spectral Structure Across Dynamics Regimes',
                fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    
    path = output_dir / 'dyn_suite_svd_comparison.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    fig.savefig(output_dir / 'dyn_suite_svd_comparison.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {path}")
    
    # Print summary table
    print("\n  DYN Suite — Modes for 99% energy (d=19 used):")
    for name, n99, e19 in exp_data:
        print(f"    {name:20s}  d_99%={n99:3d}  energy@19={e19:.4f}")
    
    return path


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Phase Dynamics Δ(t) — Shift time-series analysis
# ═════════════════════════════════════════════════════════════════════════════

def plot_phase_dynamics(exp_dirs_with_shifts, output_dir):
    """
    Analyze the shift Δ(t) time-series from aligned experiments.
    
    Shows:
    (a) Raw Δ_x(t), Δ_y(t) trajectories
    (b) Power spectral density of shifts → shows low-frequency content
    (c) AR(p) predictability of shifts → shifts are simple dynamics
    (d) Phase vs shape coordinate variance decomposition
    
    Parameters
    ----------
    exp_dirs_with_shifts : list of (name, exp_dir) tuples
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    ax_traj = fig.add_subplot(gs[0, 0])
    ax_psd = fig.add_subplot(gs[0, 1])
    ax_ar = fig.add_subplot(gs[0, 2])
    ax_var = fig.add_subplot(gs[1, 0])
    ax_autocorr = fig.add_subplot(gs[1, 1])
    ax_summary = fig.add_subplot(gs[1, 2])
    
    cmap = plt.cm.Set2
    all_ar_r2 = []
    
    for idx, (name, exp_dir) in enumerate(exp_dirs_with_shifts):
        sa = load_shift_data(exp_dir)
        if sa is None:
            continue
        
        shifts = sa['shifts']  # (M*T, 2) or (M, T, 2)
        if shifts.ndim == 2:
            # Need to figure out M and T
            # Try to load train metadata for M
            meta_path = Path(exp_dir) / 'train' / 'metadata.json'
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                M = len(meta) if isinstance(meta, list) else meta.get('n_train', 0)
            else:
                M = 112  # default
            T = shifts.shape[0] // M if M > 0 else shifts.shape[0]
            shifts_3d = shifts.reshape(M, T, 2)
        else:
            shifts_3d = shifts
            M, T = shifts_3d.shape[:2]
        
        color = cmap(idx / max(1, len(exp_dirs_with_shifts) - 1))
        short_name = name.replace('_v2', '').split('_')[0]
        
        # (a) Sample shift trajectories (first 5 runs)
        for run_i in range(min(5, M)):
            ax_traj.plot(shifts_3d[run_i, :, 1], shifts_3d[run_i, :, 0],
                        alpha=0.3, linewidth=0.5, color=color)
        # Mean shift trajectory
        mean_shift = shifts_3d.mean(axis=0)
        ax_traj.plot(mean_shift[:, 1], mean_shift[:, 0], '-', color=color,
                    linewidth=2, label=short_name, alpha=0.9)
        
        # (b) Power spectral density of shifts (all runs averaged)
        from scipy import signal
        for dim, dim_label in [(0, 'y'), (1, 'x')]:
            psds = []
            for run_i in range(M):
                f, psd = signal.welch(shifts_3d[run_i, :, dim], fs=1.0, 
                                     nperseg=min(128, T//2))
                psds.append(psd)
            mean_psd = np.mean(psds, axis=0)
            ls = '-' if dim == 1 else '--'
            ax_psd.semilogy(f, mean_psd, ls, color=color, linewidth=1.5,
                           label=f'{short_name} Δ{dim_label}', alpha=0.8)
        
        # (c) AR predictability — fit AR(p) to shifts and compute R²
        from numpy.linalg import lstsq
        ar_orders = [1, 2, 3, 5, 10]
        ar_r2_by_p = []
        for p in ar_orders:
            r2s = []
            for dim in [0, 1]:
                # Build AR design matrix from all runs
                X_list, y_list = [], []
                for run_i in range(M):
                    s = shifts_3d[run_i, :, dim]
                    for t in range(p, T):
                        X_list.append(s[t-p:t][::-1])
                        y_list.append(s[t])
                X_ar = np.array(X_list)
                y_ar = np.array(y_list)
                # Fit
                coef, _, _, _ = lstsq(X_ar, y_ar, rcond=None)
                y_pred = X_ar @ coef
                ss_res = np.sum((y_ar - y_pred)**2)
                ss_tot = np.sum((y_ar - y_ar.mean())**2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                r2s.append(r2)
            ar_r2_by_p.append(np.mean(r2s))
        
        ax_ar.plot(ar_orders, ar_r2_by_p, 'o-', color=color, linewidth=1.5,
                  markersize=6, label=short_name)
        all_ar_r2.append((short_name, dict(zip(ar_orders, ar_r2_by_p))))
        
        # (d) Variance decomposition: phase vs shape
        # Phase variance = var of shifts across runs at each time
        phase_var = np.var(shifts_3d, axis=0).mean()  # mean across time & dims
        # Total variance = variance of all shifts
        total_var = np.var(shifts_3d)
        ax_var.bar(idx, phase_var, color=color, alpha=0.7, label=short_name)
        
        # (e) Autocorrelation of shifts
        for dim in [0, 1]:
            mean_s = shifts_3d[:, :, dim].mean(axis=0)
            autocorr = np.correlate(mean_s - mean_s.mean(), 
                                   mean_s - mean_s.mean(), mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr /= autocorr[0] if autocorr[0] > 0 else 1
            max_lag = min(50, len(autocorr))
            ls = '-' if dim == 1 else '--'
            ax_autocorr.plot(np.arange(max_lag), autocorr[:max_lag], ls,
                           color=color, linewidth=1.5, alpha=0.8)
    
    # Formatting
    ax_traj.set_xlabel(r'$\Delta_x$ (pixels)')
    ax_traj.set_ylabel(r'$\Delta_y$ (pixels)')
    ax_traj.set_title('(a) Shift Trajectories $\\Delta(t)$')
    ax_traj.legend(fontsize=8)
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_aspect('equal', adjustable='datalim')
    
    ax_psd.set_xlabel('Frequency')
    ax_psd.set_ylabel('PSD')
    ax_psd.set_title(r'(b) Power Spectrum of $\Delta(t)$')
    ax_psd.legend(fontsize=7, ncol=2)
    ax_psd.grid(True, alpha=0.3)
    
    ax_ar.set_xlabel('AR order $p$')
    ax_ar.set_ylabel(r'$R^2$')
    ax_ar.set_title(r'(c) AR($p$) Predictability of $\Delta$')
    ax_ar.legend(fontsize=8)
    ax_ar.grid(True, alpha=0.3)
    ax_ar.set_ylim(0.5, 1.01)
    ax_ar.set_xticks(ar_orders)
    
    ax_var.set_xlabel('Experiment')
    ax_var.set_ylabel('Mean shift variance (px²)')
    ax_var.set_title('(d) Phase Dynamics Variance')
    ax_var.grid(True, alpha=0.3, axis='y')
    
    ax_autocorr.set_xlabel('Lag (timesteps)')
    ax_autocorr.set_ylabel('Autocorrelation')
    ax_autocorr.set_title(r'(e) Autocorrelation of mean $\Delta(t)$')
    ax_autocorr.grid(True, alpha=0.3)
    ax_autocorr.axhline(0, color='gray', linewidth=0.5)
    
    # Summary text panel
    ax_summary.axis('off')
    txt = "Phase Dynamics Summary\n" + "="*30 + "\n\n"
    for name, r2_dict in all_ar_r2:
        txt += f"{name}:\n"
        for p, r2 in r2_dict.items():
            txt += f"  AR({p}): R²={r2:.4f}\n"
        txt += "\n"
    txt += "Key insight: shifts Δ(t) are\nhighly predictable with AR(≤5),\n"
    txt += "confirming they represent\nsimple phase dynamics that can\n"
    txt += "be factored from shape dynamics."
    ax_summary.text(0.05, 0.95, txt, transform=ax_summary.transAxes,
                   fontsize=9, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    fig.suptitle('Phase Dynamics Analysis: Shift $\\Delta(t)$ as Separate Dynamical Variable\n'
                 '(sPOD / Moving-Frame Decomposition)',
                 fontsize=13, fontweight='bold')
    
    path = output_dir / 'phase_dynamics_analysis.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    fig.savefig(output_dir / 'phase_dynamics_analysis.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {path}")
    return path


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Multi-step R²(h) — Forecast quality vs horizon
# ═════════════════════════════════════════════════════════════════════════════

def plot_r2_vs_horizon_comparison(experiments, output_dir):
    """
    Compare R²(h) degradation curves between experiments.
    
    experiments: list of (name, exp_dir, style_dict) tuples.
    style_dict: {'color': ..., 'linestyle': ...}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    
    for name, exp_dir, style in experiments:
        r2_data = load_r2_vs_time(exp_dir, model='mvar')
        if r2_data is None:
            print(f"  ⚠ No r2_vs_time data for {name}")
            continue
        
        # Columns: time, mean_r2, std_r2, etc.
        if 'time' in r2_data.columns and 'mean_r2' in r2_data.columns:
            t = r2_data['time'].values
            r2 = r2_data['mean_r2'].values
            r2_std = r2_data.get('std_r2', r2_data.get('r2_std', None))
        elif 'timestep' in r2_data.columns:
            t = r2_data['timestep'].values
            r2 = r2_data.iloc[:, 1].values
            r2_std = None
        else:
            print(f"  ⚠ Unknown columns in r2_vs_time for {name}: {list(r2_data.columns)}")
            continue
        
        ax1.plot(t, r2, style.get('linestyle', '-'), color=style['color'],
                linewidth=1.5, label=name, alpha=0.8)
        if r2_std is not None:
            r2_std_vals = r2_std.values if hasattr(r2_std, 'values') else r2_std
            ax1.fill_between(t, r2 - r2_std_vals, r2 + r2_std_vals,
                           color=style['color'], alpha=0.15)
        
        # Normalized plot: R²(h) / R²(0)
        r2_0 = r2[0] if r2[0] > 0 else 1
        ax2.plot(t, r2 / r2_0, style.get('linestyle', '-'), color=style['color'],
                linewidth=1.5, label=name, alpha=0.8)
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(r'$R^2(t)$')
    ax1.set_title('Forecast R² Over Time')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.05)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel(r'$R^2(t) / R^2(0)$')
    ax2.set_title('Normalized R² Degradation')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)
    
    fig.suptitle('Multi-Step R² Degradation: Comparison Across Experiments',
                fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    
    path = output_dir / 'r2_vs_horizon_comparison.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    fig.savefig(output_dir / 'r2_vs_horizon_comparison.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {path}")
    return path


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='sPOD Diagnostics')
    parser.add_argument('--output_dir', type=str, default='artifacts/thesis_figures',
                       help='Output directory for figures')
    parser.add_argument('--data_dir', type=str, default='oscar_output',
                       help='Base data directory')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    
    print("="*80)
    print("sPOD DIAGNOSTICS — Shift-Alignment Motivation Plots")
    print("="*80)
    
    # ─── Discover available experiments ─────────────────────────────────────
    
    # Alignment pairs: (label, aligned_dir, noalign_dir, transform)
    alignment_pairs = []
    
    # CUR pair: fast curvature dynamics
    cur1 = data_dir / 'CUR1_N500_fast_aligned_sqrtSimplex_H300'
    cur2 = data_dir / 'CUR2_N500_fast_NOalign_sqrtSimplex_H300'
    if cur1.exists() and cur2.exists():
        alignment_pairs.append({
            'label': 'Curvature (sqrt+simplex)',
            'aligned_dir': cur1,
            'noalign_dir': cur2,
            'transform': 'sqrt+simplex',
        })
    
    # ABL pairs: standard Vicsek
    abl3 = data_dir / 'ABL3_N200_raw_simplex_noAlign_H300_v2'
    abl4 = data_dir / 'ABL4_N200_raw_simplex_align_H300_v2'
    if abl3.exists() and abl4.exists():
        if load_singular_values(abl3) is not None:
            alignment_pairs.append({
                'label': 'Vicsek (raw+simplex)',
                'aligned_dir': abl4,
                'noalign_dir': abl3,
                'transform': 'raw+simplex',
            })
    
    abl7 = data_dir / 'ABL7_N200_sqrt_simplex_noAlign_H300_v2'
    abl8 = data_dir / 'ABL8_N200_sqrt_simplex_align_H300_v2'
    if abl7.exists() and abl8.exists():
        if load_singular_values(abl7) is not None:
            alignment_pairs.append({
                'label': 'Vicsek (sqrt+simplex)',
                'aligned_dir': abl8,
                'noalign_dir': abl7,
                'transform': 'sqrt+simplex',
            })
    
    # LST pair: LSTM experiments
    lst5 = data_dir / 'LST5_raw_none_noAlign_h64_L2'
    lst7 = data_dir / 'LST7_raw_none_align_h128_L2'
    if lst5.exists() and lst7.exists():
        if load_singular_values(lst5) is not None and load_singular_values(lst7) is not None:
            alignment_pairs.append({
                'label': 'LSTM (raw+none)',
                'aligned_dir': lst7,
                'noalign_dir': lst5,
                'transform': 'raw+none',
            })
    
    print(f"\n  Found {len(alignment_pairs)} alignment pairs:")
    for p in alignment_pairs:
        print(f"    • {p['label']}: {p['aligned_dir'].name} vs {p['noalign_dir'].name}")
    
    # DYN experiments
    dyn_experiments = []
    for i in range(1, 8):
        for suffix in ['_v2', '']:
            d = data_dir / f'DYN{i}_{["", "gentle", "hypervelocity", "hypernoisy", "blackhole", "supernova", "varspeed", "pure_vicsek"][i]}{suffix}'
            if d.exists() and load_singular_values(d) is not None:
                dyn_experiments.append((d.name, d))
                break
    
    print(f"\n  Found {len(dyn_experiments)} DYN experiments:")
    for name, _ in dyn_experiments:
        print(f"    • {name}")
    
    # Experiments with shift data
    shift_experiments = []
    for d in sorted(data_dir.iterdir()):
        if d.is_dir() and load_shift_data(d) is not None:
            shift_experiments.append((d.name, d))
    
    print(f"\n  Found {len(shift_experiments)} experiments with shift data:")
    for name, _ in shift_experiments:
        print(f"    • {name}")
    
    # ─── FIGURE 1: SVD Decay ───────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("FIGURE 1: Singular Value Decay — Raw vs Aligned")
    print("="*80)
    
    if len(alignment_pairs) > 0:
        plot_svd_decay(alignment_pairs, output_dir)
    else:
        print("  ⚠ No alignment pairs found. Need CUR1+CUR2 or ABL3+ABL4 etc.")
    
    # ─── FIGURE 2: DYN Suite SVD ──────────────────────────────────────────
    print(f"\n{'='*80}")
    print("FIGURE 2: DYN Suite — Spectral Structure Comparison")
    print("="*80)
    
    if len(dyn_experiments) > 0:
        plot_dyn_suite_svd(dyn_experiments, output_dir)
    else:
        print("  ⚠ No DYN experiments found.")
    
    # ─── FIGURE 3: Phase Dynamics ─────────────────────────────────────────
    print(f"\n{'='*80}")
    print("FIGURE 3: Phase Dynamics Δ(t) Analysis")
    print("="*80)
    
    if len(shift_experiments) > 0:
        plot_phase_dynamics(shift_experiments[:6], output_dir)  # limit to 6
    else:
        print("  ⚠ No experiments with shift data found.")
    
    # ─── FIGURE 4: R²(h) Degradation ─────────────────────────────────────
    print(f"\n{'='*80}")
    print("FIGURE 4: Multi-Step R²(h) Comparison")
    print("="*80)
    
    # Collect experiments with R² vs time data
    r2h_experiments = []
    for name, exp_dir in dyn_experiments[:7]:
        r2_data = load_r2_vs_time(exp_dir)
        if r2_data is not None:
            short = name.replace('_v2', '').split('_')[0]
            color_idx = int(short.replace('DYN', '')) - 1
            r2h_experiments.append((name.replace('_v2',''), exp_dir, 
                                   {'color': plt.cm.tab10(color_idx/7), 'linestyle': '-'}))
    
    if len(r2h_experiments) > 0:
        plot_r2_vs_horizon_comparison(r2h_experiments, output_dir)
    else:
        print("  ⚠ No R²-vs-time data found. Run visualizations with time analysis first.")
    
    print(f"\n{'='*80}")
    print(f"All figures saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
