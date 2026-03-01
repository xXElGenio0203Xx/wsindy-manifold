#!/usr/bin/env python3
"""
Cross-Experiment Analysis Pipeline
===================================

Scans oscar_output/ for completed experiment folders and produces
a comprehensive set of comparison and per-experiment analysis plots
in the Analyses/ directory.

Designed for systematic MVAR vs LSTM comparison across dynamical regimes.

Outputs (all saved to Analyses/):
    Cross-experiment (one plot, all experiments on same axes):
        - POD vs sPOD singular value spectra + cumulative energy
        - Multi-step R^2 degradation (all regimes, one plot)
        - Relative L2 error vs time
        - Relative L1 error vs time
        - Relative Linf error vs time
        - Predicted total mass vs time
    Per-experiment:
        - KDE alignment check
        - Spatial order parameter prediction (true vs pred)
    Phase dynamics:
        - Phase dynamics analysis (shift trajectories, PSD, AR, variance, autocorr)

Usage:
    python plot_pipeline.py [--data_dir oscar_output] [--output_dir Analyses]
                            [--experiments EXP1 EXP2 ...]
                            [--models mvar lstm]
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
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'src'))

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
CMAP = plt.cm.tab10
CMAP20 = plt.cm.tab20


def _color(idx, n):
    return CMAP(idx / max(1, n - 1)) if n <= 10 else CMAP20(idx / max(1, n - 1))


def _short(name):
    """Shorten experiment name for legend labels."""
    return name.replace('_v2', '').replace('_v3', '')


# ============================================================================
# Data loading helpers
# ============================================================================

def load_singular_values(exp_dir):
    """Load all singular values from pod_basis.npz (aligned / sPOD)."""
    for sub in ['rom_common', 'mvar']:
        pod_path = Path(exp_dir) / sub / 'pod_basis.npz'
        if pod_path.exists():
            d = np.load(pod_path)
            return d.get('all_singular_values', d.get('singular_values'))
    return None


def load_unaligned_singular_values(exp_dir):
    """Load unaligned (raw POD) singular values if saved."""
    for sub in ['rom_common', 'mvar']:
        path = Path(exp_dir) / sub / 'pod_basis_unaligned.npz'
        if path.exists():
            d = np.load(path)
            return d.get('all_singular_values', d.get('singular_values'))
    return None


def load_shift_data(exp_dir):
    """Load shift alignment data."""
    sa_path = Path(exp_dir) / 'rom_common' / 'shift_align.npz'
    if sa_path.exists():
        return np.load(sa_path, allow_pickle=True)
    return None


def load_r2_vs_time(exp_dir, model='mvar'):
    """Load time-resolved R^2, aggregated across test runs."""
    model_upper = model.upper()
    test_dir = Path(exp_dir) / 'test'
    if not test_dir.exists():
        return None

    dfs = []
    for run_dir in sorted(test_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        for pattern in [f'r2_vs_time_{model}.csv', 'r2_vs_time.csv']:
            r2_file = run_dir / pattern
            if r2_file.exists():
                dfs.append(pd.read_csv(r2_file))
                break

    if not dfs:
        return None

    ref_times = dfs[0]['time'].values
    r2_col = 'r2_reconstructed' if 'r2_reconstructed' in dfs[0].columns else dfs[0].columns[1]
    r2_matrix = np.array([df[r2_col].values[:len(ref_times)] for df in dfs])
    return pd.DataFrame({
        'time': ref_times,
        'mean_r2': r2_matrix.mean(axis=0),
        'std_r2': r2_matrix.std(axis=0),
        'min_r2': r2_matrix.min(axis=0),
        'max_r2': r2_matrix.max(axis=0),
    })


def load_density_pair(exp_dir, test_idx=0, model='mvar'):
    """Load (rho_true, rho_pred, times) for a single test run."""
    test_run = Path(exp_dir) / 'test' / f'test_{test_idx:03d}'
    true_path = test_run / 'density_true.npz'
    pred_path = test_run / f'density_pred_{model}.npz'
    if not pred_path.exists():
        pred_path = test_run / 'density_pred.npz'
    if not true_path.exists() or not pred_path.exists():
        return None, None, None
    dt = np.load(true_path)
    dp = np.load(pred_path)
    # Align to prediction time range
    rho_true = dt['rho']
    times_true = dt['times']
    rho_pred = dp['rho']
    times_pred = dp['times']
    # Find start index in true data corresponding to pred start
    start_idx = int(dp.get('forecast_start_idx', 0))
    T_pred = rho_pred.shape[0]
    rho_true_aligned = rho_true[start_idx:start_idx + T_pred]
    times_aligned = times_true[start_idx:start_idx + T_pred]
    if rho_true_aligned.shape[0] != T_pred:
        minT = min(rho_true_aligned.shape[0], T_pred)
        rho_true_aligned = rho_true_aligned[:minT]
        rho_pred = rho_pred[:minT]
        times_aligned = times_aligned[:minT]
    return rho_true_aligned, rho_pred, times_aligned


def load_test_trajectory(exp_dir, test_idx=0):
    """Load particle trajectory for a test run (if available)."""
    traj_path = Path(exp_dir) / 'test' / f'test_{test_idx:03d}' / 'trajectory.npz'
    if not traj_path.exists():
        return None
    return np.load(traj_path)


# ============================================================================
# Error computation helpers
# ============================================================================

def compute_error_timeseries(rho_true, rho_pred):
    """Compute per-timestep relative L1, L2, Linf errors and mass."""
    T = rho_true.shape[0]
    e1, e2, einf, mass_true, mass_pred = [], [], [], [], []
    for t in range(T):
        diff = rho_true[t] - rho_pred[t]
        norm_true = np.sqrt(np.sum(rho_true[t] ** 2))
        e2_t = np.sqrt(np.sum(diff ** 2)) / (norm_true + 1e-12)
        e1_t = np.sum(np.abs(diff)) / (np.sum(np.abs(rho_true[t])) + 1e-12)
        einf_t = np.max(np.abs(diff)) / (np.max(np.abs(rho_true[t])) + 1e-12)
        e2.append(e2_t)
        e1.append(e1_t)
        einf.append(einf_t)
        mass_true.append(np.sum(rho_true[t]))
        mass_pred.append(np.sum(rho_pred[t]))
    return {
        'rel_e1': np.array(e1),
        'rel_e2': np.array(e2),
        'rel_einf': np.array(einf),
        'mass_true': np.array(mass_true),
        'mass_pred': np.array(mass_pred),
    }


def compute_spatial_order_timeseries(rho):
    """std(rho) per timestep as a spatial order parameter."""
    return np.array([np.std(rho[t]) for t in range(rho.shape[0])])


# ============================================================================
# Discovery
# ============================================================================

def discover_experiments(data_dir, experiment_filter=None):
    """Find all valid experiment folders in data_dir.

    A valid experiment has at least rom_common/pod_basis.npz or mvar/pod_basis.npz
    and a test/ directory.
    """
    data_dir = Path(data_dir)
    experiments = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        if experiment_filter and d.name not in experiment_filter:
            continue
        has_pod = (d / 'rom_common' / 'pod_basis.npz').exists() or (d / 'mvar' / 'pod_basis.npz').exists()
        has_test = (d / 'test').exists()
        if has_pod and has_test:
            experiments.append(d)
    return experiments


def detect_models(exp_dir):
    """Return list of available models for this experiment."""
    models = []
    if (Path(exp_dir) / 'MVAR' / 'test_results.csv').exists() or (Path(exp_dir) / 'mvar' / 'mvar_model.npz').exists():
        models.append('mvar')
    if (Path(exp_dir) / 'LSTM').exists() and any((Path(exp_dir) / 'LSTM').iterdir()):
        models.append('lstm')
    return models


# ============================================================================
# Plot 1: POD vs sPOD spectra and cumulative energy
# ============================================================================

def plot_svd_spectra(experiments, output_dir, max_modes=40):
    """Singular value decay and cumulative energy for all experiments.

    Produces TWO figures:
      1) Per-experiment sPOD spectra (colored by experiment)
      2) POD vs sPOD contrast: all sPOD lines in blue, all POD lines in red
    """
    n = len(experiments)

    # ---- Figure 1: Per-experiment sPOD spectra ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for i, exp_dir in enumerate(experiments):
        sv = load_singular_values(exp_dir)
        if sv is None:
            continue
        nm = min(max_modes, len(sv))
        modes = np.arange(1, nm + 1)
        sv_norm = (sv[:nm] / sv[0]) ** 2
        cum = np.cumsum(sv[:nm] ** 2) / np.sum(sv ** 2)
        color = _color(i, n)
        label = _short(exp_dir.name)
        ax1.semilogy(modes, sv_norm, 'o-', color=color, markersize=2,
                      linewidth=1.2, label=label, alpha=0.8)
        ax2.plot(modes, cum, '-', color=color, linewidth=1.2, label=label, alpha=0.8)

    ax1.set_xlabel('Mode index')
    ax1.set_ylabel(r'$(\sigma_i/\sigma_1)^2$')
    ax1.set_title('Singular Value Decay (sPOD — aligned)')
    ax1.legend(fontsize=6, ncol=3)
    ax1.grid(True, alpha=0.3)
    ax2.axhline(0.99, color='gray', ls=':', alpha=0.5, label='99%')
    ax2.axhline(0.999, color='gray', ls='--', alpha=0.5, label='99.9%')
    ax2.set_xlabel('Number of modes $d$')
    ax2.set_ylabel('Cumulative energy')
    ax2.set_title('Cumulative Energy (sPOD — aligned)')
    ax2.legend(fontsize=6, ncol=3)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.85, 1.005)
    fig.suptitle('sPOD Spectra Across Experiments', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = output_dir / 'svd_spectra_comparison.pdf'
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(output_dir / 'svd_spectra_comparison.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")

    # ---- Figure 2: POD vs sPOD contrast ----
    # All sPOD lines in one color family, all raw POD in another,
    # so the faster eigenvalue decay of sPOD is visually obvious.
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))
    spod_color = '#1f77b4'  # blue family
    pod_color = '#d62728'   # red family
    has_any_unaligned = False

    for i, exp_dir in enumerate(experiments):
        sv_aligned = load_singular_values(exp_dir)
        sv_raw = load_unaligned_singular_values(exp_dir)
        if sv_aligned is None:
            continue
        nm = min(max_modes, len(sv_aligned))
        modes = np.arange(1, nm + 1)

        # sPOD (aligned)
        sv_norm_a = (sv_aligned[:nm] / sv_aligned[0]) ** 2
        cum_a = np.cumsum(sv_aligned[:nm] ** 2) / np.sum(sv_aligned ** 2)
        lab_a = 'sPOD (aligned)' if i == 0 else None
        ax3.semilogy(modes, sv_norm_a, '-', color=spod_color,
                      linewidth=0.8, alpha=0.35, label=lab_a)
        ax4.plot(modes, cum_a, '-', color=spod_color, linewidth=0.8, alpha=0.35,
                 label=lab_a)

        # Raw POD (unaligned)
        if sv_raw is not None:
            has_any_unaligned = True
            nm_r = min(max_modes, len(sv_raw))
            sv_norm_r = (sv_raw[:nm_r] / sv_raw[0]) ** 2
            cum_r = np.cumsum(sv_raw[:nm_r] ** 2) / np.sum(sv_raw ** 2)
            lab_r = 'POD (unaligned)' if i == 0 else None
            ax3.semilogy(modes[:nm_r], sv_norm_r, '-', color=pod_color,
                          linewidth=0.8, alpha=0.35, label=lab_r)
            ax4.plot(modes[:nm_r], cum_r, '-', color=pod_color,
                     linewidth=0.8, alpha=0.35, label=lab_r)

    ax3.set_xlabel('Mode index')
    ax3.set_ylabel(r'$(\sigma_i/\sigma_1)^2$')
    ax3.set_title('Eigenvalue Decay: sPOD (blue) vs POD (red)')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax4.axhline(0.99, color='gray', ls=':', alpha=0.5)
    ax4.axhline(0.999, color='gray', ls='--', alpha=0.5)
    ax4.set_xlabel('Number of modes $d$')
    ax4.set_ylabel('Cumulative energy')
    ax4.set_title('Cumulative Energy: sPOD vs POD')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0.70, 1.005)
    subtitle = 'Alignment Accelerates Eigenvalue Decay'
    if not has_any_unaligned:
        subtitle += ' (no unaligned data yet — run pipeline first)'
    fig2.suptitle(subtitle, fontsize=13, fontweight='bold')
    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    path2 = output_dir / 'pod_vs_spod_contrast.pdf'
    fig2.savefig(path2, bbox_inches='tight')
    fig2.savefig(output_dir / 'pod_vs_spod_contrast.png', dpi=200, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved: {path2.name}")


# ============================================================================
# Plot 2: R^2 degradation over time (all regimes, per model)
# ============================================================================

def _split_into_groups(experiments, n_groups=4):
    """Split experiments into n roughly-equal groups."""
    k = len(experiments)
    size = max(1, (k + n_groups - 1) // n_groups)
    return [experiments[i:i + size] for i in range(0, k, size)]


def plot_r2_degradation(experiments, output_dir, models):
    """R^2 vs prediction horizon, split into 4 subplots (~14 experiments each).

    One figure per model, each with a 2x2 grid of subplots.
    """
    groups = _split_into_groups(experiments, 4)

    for model in models:
        n_groups = len(groups)
        nrows = 2 if n_groups > 2 else 1
        ncols = 2 if n_groups > 1 else 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5 * nrows), squeeze=False)

        for gi, grp in enumerate(groups):
            ax = axes[gi // ncols, gi % ncols]
            n_grp = len(grp)
            for i, exp_dir in enumerate(grp):
                r2 = load_r2_vs_time(exp_dir, model=model)
                if r2 is None:
                    continue
                t = r2['time'].values
                color = _color(i, n_grp)
                ax.plot(t, r2['mean_r2'], '-', color=color, linewidth=1.2,
                        label=_short(exp_dir.name), alpha=0.8)
                ax.fill_between(t, r2['mean_r2'] - r2['std_r2'],
                                r2['mean_r2'] + r2['std_r2'],
                                color=color, alpha=0.1)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(r'$R^2$')
            ax.set_title(f'Group {gi + 1}')
            ax.legend(fontsize=6, ncol=2)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.1, 1.05)

        # Hide unused subplots
        for gi in range(len(groups), nrows * ncols):
            axes[gi // ncols, gi % ncols].set_visible(False)

        fig.suptitle(f'{model.upper()} — R$^2$ Degradation Across Regimes',
                     fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        path = output_dir / f'r2_degradation_{model}.pdf'
        fig.savefig(path, bbox_inches='tight')
        fig.savefig(output_dir / f'r2_degradation_{model}.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path.name}")


# ============================================================================
# Plot 3: Relative error norms (L1, L2, Linf) over time
# ============================================================================

def plot_error_norms(experiments, output_dir, models, test_idx=0):
    """Relative L1, L2, Linf error vs time.

    One figure per (model × error norm), each split into 4 subplots.
    Produces: error_L2_mvar.pdf, error_L1_mvar.pdf, error_Linf_mvar.pdf, etc.
    """
    norm_info = [
        ('L2', 'rel_e2', r'Relative $L^2$ Error'),
        ('L1', 'rel_e1', r'Relative $L^1$ Error'),
        ('Linf', 'rel_einf', r'Relative $L^\infty$ Error'),
    ]
    groups = _split_into_groups(experiments, 4)

    # Pre-compute error data to avoid reloading per norm
    error_cache = {}  # exp_name -> {model -> errs_dict, times}
    for exp_dir in experiments:
        error_cache[exp_dir.name] = {}
        for model in models:
            rho_true, rho_pred, times = load_density_pair(exp_dir, test_idx, model)
            if rho_true is not None:
                errs = compute_error_timeseries(rho_true, rho_pred)
                error_cache[exp_dir.name][model] = (errs, times)

    for model in models:
        for norm_tag, key, title_str in norm_info:
            n_groups = len(groups)
            nrows = 2 if n_groups > 2 else 1
            ncols = 2 if n_groups > 1 else 1
            fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5 * nrows),
                                     squeeze=False)

            for gi, grp in enumerate(groups):
                ax = axes[gi // ncols, gi % ncols]
                n_grp = len(grp)
                for i, exp_dir in enumerate(grp):
                    cached = error_cache.get(exp_dir.name, {}).get(model)
                    if cached is None:
                        continue
                    errs, times = cached
                    color = _color(i, n_grp)
                    ax.plot(times, errs[key], '-', color=color, linewidth=1,
                            label=_short(exp_dir.name), alpha=0.8)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Relative Error')
                ax.set_title(f'Group {gi + 1}')
                ax.legend(fontsize=6, ncol=2)
                ax.grid(True, alpha=0.3)

            for gi in range(len(groups), nrows * ncols):
                axes[gi // ncols, gi % ncols].set_visible(False)

            fig.suptitle(f'{model.upper()} — {title_str} (test {test_idx})',
                         fontsize=14, fontweight='bold')
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            path = output_dir / f'error_{norm_tag}_{model}.pdf'
            fig.savefig(path, bbox_inches='tight')
            fig.savefig(output_dir / f'error_{norm_tag}_{model}.png', dpi=200,
                        bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {path.name}")


# ============================================================================
# Plot 4: Predicted total mass over time (all experiments, same plot)
# ============================================================================

def plot_mass_conservation(experiments, output_dir, models, test_idx=0):
    """Predicted vs true total mass over time, all experiments on same axes."""
    n = len(experiments)
    for model in models:
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, exp_dir in enumerate(experiments):
            rho_true, rho_pred, times = load_density_pair(exp_dir, test_idx, model)
            if rho_true is None:
                continue
            errs = compute_error_timeseries(rho_true, rho_pred)
            color = _color(i, n)
            label = _short(exp_dir.name)
            ax.plot(times, errs['mass_pred'], '-', color=color, linewidth=1,
                    label=label, alpha=0.8)
            # Plot true mass as dashed (should be nearly identical across experiments
            # if same N is used)
            if i == 0:
                ax.plot(times, errs['mass_true'], '--', color='black', linewidth=1,
                        label='True mass', alpha=0.6)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Total Mass (sum of density)')
        ax.set_title(f'{model.upper()} Predicted Mass Over Time (test {test_idx})')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        path = output_dir / f'mass_conservation_{model}.pdf'
        fig.savefig(path, bbox_inches='tight')
        fig.savefig(output_dir / f'mass_conservation_{model}.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path.name}")


# ============================================================================
# Plot 5: Spatial order parameter (per experiment)
# ============================================================================

# ============================================================================
# Plot 5+6: KDE alignment + spatial order (merged, per experiment)
# ============================================================================

def plot_kde_alignment(experiments, output_dir, models, test_idx=0, n_frames=4):
    """Per-experiment diagnostic: density frames + spatial order parameter.

    Layout (3 rows):
      Row 0: Raw density snapshots at sampled times
      Row 1: Reference field (if shift-aligned) or duplicate
      Row 2: Spatial order (std of density) — true vs predicted per model
    """
    per_exp_dir = output_dir / 'per_experiment'
    per_exp_dir.mkdir(exist_ok=True)

    for exp_dir in experiments:
        sa = load_shift_data(exp_dir)
        test_run = Path(exp_dir) / 'test' / f'test_{test_idx:03d}'
        true_path = test_run / 'density_true.npz'
        if not true_path.exists():
            continue
        dt = np.load(true_path)
        rho = dt['rho']
        T_total = rho.shape[0]
        frame_idxs = np.linspace(0, T_total - 1, n_frames, dtype=int)

        fig = plt.figure(figsize=(4 * n_frames, 10))
        gs = GridSpec(3, n_frames, figure=fig, height_ratios=[1, 1, 0.8],
                      hspace=0.30, wspace=0.15)

        # Row 0: Raw density frames
        for j, fidx in enumerate(frame_idxs):
            ax = fig.add_subplot(gs[0, j])
            frame = rho[fidx]
            ax.imshow(frame, origin='lower', cmap='hot', aspect='auto')
            ax.set_title(f't={dt["times"][fidx]:.1f}s', fontsize=9)
            ax.axis('off')
            if j == 0:
                ax.set_ylabel('Raw frames', fontsize=10)

        # Row 1: Reference / aligned
        for j, fidx in enumerate(frame_idxs):
            ax = fig.add_subplot(gs[1, j])
            if sa is not None:
                ref = sa['ref']
                ax.imshow(ref, origin='lower', cmap='hot', aspect='auto')
                ax.set_title('Reference (mean)', fontsize=8)
            else:
                ax.imshow(rho[fidx], origin='lower', cmap='hot', aspect='auto')
                ax.set_title('No alignment', fontsize=8)
            ax.axis('off')
            if j == 0:
                ax.set_ylabel('Ref / aligned', fontsize=10)

        # Row 2 (span all columns): Spatial order parameter
        ax_so = fig.add_subplot(gs[2, :])
        so_true = compute_spatial_order_timeseries(rho)
        times_true = dt['times']
        ax_so.plot(times_true, so_true, 'k-', linewidth=1.5, label='True', alpha=0.8)

        model_colors = {'mvar': 'C0', 'lstm': 'C1'}
        for model in models:
            rho_true_a, rho_pred, times_pred = load_density_pair(exp_dir, test_idx, model)
            if rho_pred is not None:
                so_pred = compute_spatial_order_timeseries(rho_pred)
                ax_so.plot(times_pred, so_pred, '-', color=model_colors.get(model, 'C2'),
                           linewidth=1.2, label=f'{model.upper()} pred', alpha=0.8)

        ax_so.set_xlabel('Time (s)')
        ax_so.set_ylabel('Spatial Order (std ρ)')
        ax_so.legend(fontsize=8, ncol=3)
        ax_so.grid(True, alpha=0.3)

        fig.suptitle(f'{_short(exp_dir.name)} — Alignment & Spatial Order',
                     fontsize=12, fontweight='bold')
        path = per_exp_dir / f'alignment_spatial_{exp_dir.name}.pdf'
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
    print(f"  Saved alignment + spatial order diagnostics")


# ============================================================================
# Plot 7: Phase dynamics analysis (shift trajectories, PSD, AR, variance, autocorr)
# ============================================================================

def plot_phase_dynamics(experiments, output_dir):
    """Phase dynamics analysis of shift sequences. Replicates spod_diagnostics."""
    from scipy import signal
    from numpy.linalg import lstsq

    # Only include experiments with shift data
    shift_exps = [(exp, load_shift_data(exp)) for exp in experiments]
    shift_exps = [(exp, sa) for exp, sa in shift_exps if sa is not None]
    if not shift_exps:
        print("  No experiments with shift data found, skipping phase dynamics.")
        return

    n = len(shift_exps)
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
    ax_traj = fig.add_subplot(gs[0, 0])
    ax_psd = fig.add_subplot(gs[0, 1])
    ax_ar = fig.add_subplot(gs[0, 2])
    ax_var = fig.add_subplot(gs[1, 0])
    ax_autocorr = fig.add_subplot(gs[1, 1])
    ax_summary = fig.add_subplot(gs[1, 2])

    all_ar_r2 = []
    for idx, (exp_dir, sa) in enumerate(shift_exps):
        shifts = sa['shifts']
        # Reshape to (M, T, 2)
        meta_path = Path(exp_dir) / 'train' / 'metadata.json'
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            M = len(meta) if isinstance(meta, list) else 112
        else:
            M = 112
        T_shift = shifts.shape[0] // M if M > 0 else shifts.shape[0]
        if shifts.ndim == 2:
            shifts_3d = shifts[:M * T_shift].reshape(M, T_shift, 2)
        else:
            shifts_3d = shifts
            M, T_shift = shifts_3d.shape[:2]

        color = _color(idx, n)
        label = _short(exp_dir.name)

        # (a) Shift trajectories (mean)
        mean_shift = shifts_3d.mean(axis=0)
        ax_traj.plot(mean_shift[:, 1], mean_shift[:, 0], '-', color=color,
                     linewidth=1.5, label=label, alpha=0.8)

        # (b) PSD
        for dim, dim_label in [(0, 'y'), (1, 'x')]:
            psds = []
            for run_i in range(min(M, 50)):
                f_psd, psd = signal.welch(shifts_3d[run_i, :, dim], fs=1.0,
                                          nperseg=min(128, T_shift // 2))
                psds.append(psd)
            mean_psd = np.mean(psds, axis=0)
            ls = '-' if dim == 1 else '--'
            ax_psd.semilogy(f_psd, mean_psd, ls, color=color, linewidth=1, alpha=0.7)

        # (c) AR predictability
        ar_orders = [1, 2, 3, 5, 10]
        ar_r2_by_p = []
        for p in ar_orders:
            r2s = []
            for dim in [0, 1]:
                X_list, y_list = [], []
                for run_i in range(M):
                    s = shifts_3d[run_i, :, dim]
                    for t in range(p, T_shift):
                        X_list.append(s[t - p:t][::-1])
                        y_list.append(s[t])
                X_ar = np.array(X_list)
                y_ar = np.array(y_list)
                coef, _, _, _ = lstsq(X_ar, y_ar, rcond=None)
                y_pred = X_ar @ coef
                ss_res = np.sum((y_ar - y_pred) ** 2)
                ss_tot = np.sum((y_ar - y_ar.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                r2s.append(r2)
            ar_r2_by_p.append(np.mean(r2s))
        ax_ar.plot(ar_orders, ar_r2_by_p, 'o-', color=color, linewidth=1.2,
                   markersize=5, label=label)
        all_ar_r2.append((label, dict(zip(ar_orders, ar_r2_by_p))))

        # (d) Phase variance
        phase_var = np.var(shifts_3d, axis=0).mean()
        ax_var.bar(idx, phase_var, color=color, alpha=0.7, label=label)

        # (e) Autocorrelation of mean shift
        for dim in [0, 1]:
            mean_s = shifts_3d[:, :, dim].mean(axis=0)
            ac = np.correlate(mean_s - mean_s.mean(), mean_s - mean_s.mean(), mode='full')
            ac = ac[len(ac) // 2:]
            ac /= ac[0] if ac[0] > 0 else 1
            max_lag = min(50, len(ac))
            ls = '-' if dim == 1 else '--'
            ax_autocorr.plot(np.arange(max_lag), ac[:max_lag], ls, color=color,
                             linewidth=1, alpha=0.7)

    ax_traj.set_xlabel(r'$\Delta_x$ (px)')
    ax_traj.set_ylabel(r'$\Delta_y$ (px)')
    ax_traj.set_title('(a) Mean Shift Trajectories')
    ax_traj.legend(fontsize=6, ncol=2)
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_aspect('equal', adjustable='datalim')

    ax_psd.set_xlabel('Frequency')
    ax_psd.set_ylabel('PSD')
    ax_psd.set_title(r'(b) Power Spectrum of $\Delta(t)$')
    ax_psd.grid(True, alpha=0.3)

    ax_ar.set_xlabel('AR order $p$')
    ax_ar.set_ylabel(r'$R^2$')
    ax_ar.set_title(r'(c) AR($p$) Predictability')
    ax_ar.legend(fontsize=6, ncol=2)
    ax_ar.grid(True, alpha=0.3)
    ax_ar.set_ylim(0.5, 1.01)
    ax_ar.set_xticks(ar_orders)

    ax_var.set_xlabel('Experiment')
    ax_var.set_ylabel('Mean shift variance (px$^2$)')
    ax_var.set_title('(d) Phase Variance')
    ax_var.grid(True, alpha=0.3, axis='y')
    ax_var.set_xticks(range(n))
    ax_var.set_xticklabels([_short(e.name) for e, _ in shift_exps], rotation=45, ha='right', fontsize=7)

    ax_autocorr.set_xlabel('Lag (timesteps)')
    ax_autocorr.set_ylabel('Autocorrelation')
    ax_autocorr.set_title(r'(e) Autocorrelation of mean $\Delta(t)$')
    ax_autocorr.grid(True, alpha=0.3)
    ax_autocorr.axhline(0, color='gray', linewidth=0.5)

    # Summary text
    ax_summary.axis('off')
    txt = "AR Predictability Summary\n" + "=" * 30 + "\n\n"
    for name, r2_dict in all_ar_r2:
        txt += f"{name}: AR(1)={r2_dict[1]:.3f}  AR(5)={r2_dict[5]:.3f}\n"
    ax_summary.text(0.05, 0.95, txt, transform=ax_summary.transAxes,
                    fontsize=8, va='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('Phase Dynamics Analysis', fontsize=14, fontweight='bold')
    path = output_dir / 'phase_dynamics_analysis.pdf'
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(output_dir / 'phase_dynamics_analysis.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Cross-Experiment Analysis Pipeline')
    parser.add_argument('--data_dir', type=str, default='oscar_output',
                        help='Base directory with experiment outputs')
    parser.add_argument('--output_dir', type=str, default='Analyses',
                        help='Output directory for all analysis plots')
    parser.add_argument('--experiments', nargs='*', default=None,
                        help='Specific experiment names to include (default: all)')
    parser.add_argument('--models', nargs='*', default=None,
                        help='Models to analyze (default: auto-detect)')
    parser.add_argument('--test_idx', type=int, default=0,
                        help='Test run index for per-test plots (default: 0)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CROSS-EXPERIMENT ANALYSIS PIPELINE")
    print("=" * 80)

    # Discover experiments
    experiments = discover_experiments(data_dir, args.experiments)
    if not experiments:
        print(f"\nNo valid experiments found in {data_dir}")
        return

    print(f"\nFound {len(experiments)} experiments:")
    for exp in experiments:
        print(f"  {exp.name}")

    # Detect available models
    if args.models:
        models = args.models
    else:
        all_models = set()
        for exp in experiments:
            all_models.update(detect_models(exp))
        models = sorted(all_models)
    print(f"\nModels: {', '.join(m.upper() for m in models)}")
    print(f"Output: {output_dir}/")

    # ---- Cross-experiment plots ----
    print(f"\n{'=' * 60}")
    print("1. SVD Spectra & Cumulative Energy")
    print("=" * 60)
    plot_svd_spectra(experiments, output_dir)

    print(f"\n{'=' * 60}")
    print("2. R^2 Degradation Over Time")
    print("=" * 60)
    plot_r2_degradation(experiments, output_dir, models)

    print(f"\n{'=' * 60}")
    print("3. Relative Error Norms (L1, L2, Linf)")
    print("=" * 60)
    plot_error_norms(experiments, output_dir, models, test_idx=args.test_idx)

    print(f"\n{'=' * 60}")
    print("4. Mass Conservation")
    print("=" * 60)
    plot_mass_conservation(experiments, output_dir, models, test_idx=args.test_idx)

    print(f"\n{'=' * 60}")
    print("5. Alignment + Spatial Order (per experiment)")
    print("=" * 60)
    plot_kde_alignment(experiments, output_dir, models, test_idx=args.test_idx)

    print(f"\n{'=' * 60}")
    print("6. Phase Dynamics Analysis")
    print("=" * 60)
    plot_phase_dynamics(experiments, output_dir)

    # ---- Summary ----
    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll plots saved to: {output_dir}/")
    print(f"  Cross-experiment plots: {output_dir}/*.pdf")
    print(f"  Per-experiment plots:   {output_dir}/per_experiment/*.pdf")


if __name__ == '__main__':
    main()
