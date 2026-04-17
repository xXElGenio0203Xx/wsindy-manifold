#!/usr/bin/env python3
"""
3-Model IC-Stratified Cross-Experiment Analysis Pipeline
=========================================================

Extension of plot_pipeline.py to support all three ROM models:
    MVAR  — Multivariate Autoregressive (ROM-based)
    LSTM  — Long Short-Term Memory (ROM-based)
    WSINDy — Weak SINDy PDE discovery (equation-based)

Produces all the same plots as plot_pipeline.py but with 3-model panels
where applicable, plus WSINDy-specific analysis plots.

Model availability:
  - MVAR/LSTM: require rom_common/pod_basis.npz (sPOD, POD decay, phase)
  - WSINDy:    requires WSINDy/ directory with model artifacts
  - All plots gracefully degrade if a model is missing

Per (IC x group) plots [3-panel: MVAR | LSTM | WSINDy]:
    - R² degradation over time
    - Normalized R²(t)/R²(t₀)
    - Relative L¹, L², L∞ error vs time
    - Mass conservation
    - KDE density snapshots (True | MVAR | LSTM | WSINDy)
    - Spatial order parameter

Cross-IC (regime-independent) plots:
    - sPOD singular value spectra + cumulative energy  [ROM only]
    - POD vs sPOD eigenvalue decay contrast            [ROM only]
    - Phase dynamics (shift trajectories, PSD, AR)     [ROM only]
    - Runtime comparison (all 3 models)

WSINDy-specific plots (per experiment, cross-IC):
    - Discovered PDE coefficients
    - Bootstrap confidence intervals
    - Inclusion probability

Usage:
    python 3_models_plots.py [--data_dir oscar_output] [--output_dir Analyses_3models]
                             [--experiments EXP1 EXP2 ...]
                             [--ics gaussian uniform ring two_clusters]
                             [--groups CS_alignment VS_repulsive ...]
                             [--skip_kde] [--skip_phase] [--skip_wsindy_detail]
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import time as _time

# ============================================================================
# Regime grouping (identical to plot_pipeline.py)
# ============================================================================

REGIME_GROUPS = [
    {
        'key': 'CS_alignment',
        'label': 'Constant Speed \u2014 Alignment-Dominated',
        'members': [
            'NDYN01_crawl', 'NDYN02_flock', 'NDYN04_gas',
            'NDYN08_pure_vicsek', 'NDYN09_longrange', 'NDYN13_chaos',
        ],
    },
    {
        'key': 'CS_attractive',
        'label': 'Constant Speed \u2014 Attractive',
        'members': [
            'NDYN03_sprint', 'NDYN05_blackhole', 'NDYN07_crystal',
            'NDYN10_shortrange', 'NDYN11_noisy_collapse',
        ],
    },
    {
        'key': 'CS_repulsive',
        'label': 'Constant Speed \u2014 Repulsive',
        'members': [
            'NDYN06_supernova', 'NDYN12_fast_explosion',
            'DO_EC01_esccol_C2_l3', 'DO_EC02_esccol_C3_l05',
            'DO_ES01_escsym_C3_l09', 'DO_EU01_escuns_C2_l2',
            'DO_EU02_escuns_C3_l3',
        ],
    },
    {
        'key': 'CS_collective',
        'label': 'Constant Speed \u2014 Collective Patterns',
        'members': [
            'DO_CS01_swarm_C01_l05', 'DO_CS02_swarm_C05_l3',
            'DO_CS03_swarm_C09_l3', 'DO_DM01_dmill_C09_l05',
            'DO_DR01_dring_C01_l01', 'DO_DR02_dring_C09_l09',
            'DO_SM01_mill_C05_l01', 'DO_SM02_mill_C3_l01',
            'DO_SM03_mill_C2_l05',
        ],
    },
    {
        'key': 'VS_alignment',
        'label': 'Variable Speed \u2014 Alignment-Dominated',
        'members': [
            'NDYN01_crawl_VS', 'NDYN02_flock_VS', 'NDYN04_gas_VS',
            'NDYN09_longrange_VS', 'NDYN13_chaos_VS', 'NDYN14_varspeed',
        ],
    },
    {
        'key': 'VS_attractive',
        'label': 'Variable Speed \u2014 Attractive',
        'members': [
            'NDYN03_sprint_VS', 'NDYN05_blackhole_VS', 'NDYN07_crystal_VS',
            'NDYN10_shortrange_VS', 'NDYN11_noisy_collapse_VS',
        ],
    },
    {
        'key': 'VS_repulsive',
        'label': 'Variable Speed \u2014 Repulsive',
        'members': [
            'NDYN06_supernova_VS', 'NDYN12_fast_explosion_VS',
            'DO_EC01_esccol_C2_l3_VS', 'DO_EC02_esccol_C3_l05_VS',
            'DO_ES01_escsym_C3_l09_VS', 'DO_EU01_escuns_C2_l2_VS',
            'DO_EU02_escuns_C3_l3_VS',
        ],
    },
    {
        'key': 'VS_collective',
        'label': 'Variable Speed \u2014 Collective Patterns',
        'members': [
            'DO_CS01_swarm_C01_l05_VS', 'DO_CS02_swarm_C05_l3_VS',
            'DO_CS03_swarm_C09_l3_VS', 'DO_DM01_dmill_C09_l05_VS',
            'DO_DR01_dring_C01_l01_VS', 'DO_DR02_dring_C09_l09_VS',
            'DO_SM01_mill_C05_l01_VS', 'DO_SM02_mill_C3_l01_VS',
            'DO_SM03_mill_C2_l05_VS',
        ],
    },
]

IC_NAMES = ['gaussian', 'uniform', 'ring', 'two_clusters']

IC_DISPLAY = {
    'gaussian': 'Gaussian',
    'uniform': 'Uniform',
    'ring': 'Ring',
    'two_clusters': 'Double Gaussian',
}

# Forecasting models (WSINDy is PDE-discovery only, not a forecaster)
ALL_MODELS = ['mvar', 'lstm']
MODEL_DISPLAY = {'mvar': 'MVAR', 'lstm': 'LSTM'}
MODEL_COLORS = {'mvar': '#1f77b4', 'lstm': '#d62728'}

# ── Thesis figure styling ────────────────────────────────────────────────────
THESIS_MVAR_COLOR = '#1f77b4'   # navy
THESIS_LSTM_COLOR = '#d62728'   # red
THESIS_MODELS = ['mvar', 'lstm']
THESIS_COLORS = {
    'mvar': THESIS_MVAR_COLOR,
    'lstm': THESIS_LSTM_COLOR,
}
THESIS_LABELS = {'mvar': 'MVAR', 'lstm': 'LSTM'}
THESIS_MARKERS = {'mvar': 'o', 'lstm': 's'}
MODEL_LINESTYLES = {'mvar': '-', 'lstm': '--'}

# Y-axis display floor for R²: values below this are clamped for visualization
R2_DISPLAY_FLOOR = -1.0
R2_DISPLAY_CEIL = 1.05

# Per-experiment marker pool (34+ distinct markers)
EXPERIMENT_MARKERS = [
    'o', 's', '^', 'v', '<', '>', 'D', 'd', 'p', 'P',
    '*', 'h', 'H', '8', 'X', '+', 'x', '|', '_',
    '1', '2', '3', '4',
    (3, 0, 0), (4, 0, 45), (5, 0, 0), (6, 0, 0),
    (3, 1, 0), (4, 1, 0), (5, 1, 0), (6, 1, 0),
    (4, 2, 0), (5, 2, 0), (6, 2, 0),
]

# ============================================================================
# Color and label helpers
# ============================================================================


def _color(i, n):
    cmap = plt.cm.get_cmap('tab10' if n <= 10 else 'tab20')
    return cmap(i / max(n - 1, 1))


def _short(name):
    for prefix in ['NDYN', 'DO_']:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


# ============================================================================
# Data loaders
# ============================================================================

def load_singular_values(exp_dir):
    for subdir in ['rom_common', 'mvar']:
        path = Path(exp_dir) / subdir / 'pod_basis.npz'
        if path.exists():
            data = np.load(path)
            return data['all_singular_values']
    return None


def load_unaligned_singular_values(exp_dir):
    for subdir in ['rom_common', 'mvar']:
        path = Path(exp_dir) / subdir / 'pod_basis_unaligned.npz'
        if path.exists():
            data = np.load(path)
            return data['all_singular_values']
    return None


def load_shift_data(exp_dir):
    for subdir in ['rom_common', 'mvar']:
        # Try both naming conventions (pipeline saves shift_align.npz)
        for fname in ['shift_align.npz', 'shift_align_data.npz']:
            path = Path(exp_dir) / subdir / fname
            if path.exists():
                data = np.load(path)
                return {'ref': data['ref'], 'shifts': data['shifts']}
    return None


def load_r2_single_test(exp_dir, test_idx, model='mvar'):
    """Load R² vs time for a single test run and model."""
    test_run = Path(exp_dir) / 'test' / f'test_{test_idx:03d}'
    if not test_run.exists():
        return None
    model_tag = model.lower()
    for pattern in [f'r2_vs_time_{model_tag}.csv', 'r2_vs_time.csv']:
        f = test_run / pattern
        if f.exists():
            return pd.read_csv(f)
    return None


def load_density_pair(exp_dir, test_idx=0, model='mvar'):
    """Load (rho_true, rho_pred, times) for a single test run."""
    test_run = Path(exp_dir) / 'test' / f'test_{test_idx:03d}'
    true_path = test_run / 'density_true.npz'
    model_tag = model.lower()
    pred_path = test_run / f'density_pred_{model_tag}.npz'
    if not pred_path.exists():
        pred_path = test_run / 'density_pred.npz'
    if not true_path.exists() or not pred_path.exists():
        return None, None, None
    dt = np.load(true_path)
    dp = np.load(pred_path)
    rho_true = dt['rho']
    times_true = dt['times']
    rho_pred = dp['rho']
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


def load_wsindy_model_info(exp_dir):
    """Load WSINDy discovered PDE info from summary.json or WSINDy/ dir."""
    ws_dir = Path(exp_dir) / 'WSINDy'
    if not ws_dir.exists():
        return None

    info = {}

    # Try scalar model
    scalar_path = ws_dir / 'wsindy_model.npz'
    if scalar_path.exists():
        d = np.load(scalar_path, allow_pickle=True)
        info['mode'] = 'scalar'
        info['col_names'] = list(d['col_names'])
        info['w'] = d['w']
        info['active'] = d['active']
        return info

    # Try multifield model
    mf_path = ws_dir / 'multifield_model.json'
    if mf_path.exists():
        with open(mf_path) as f:
            info = json.load(f)
        info['mode'] = 'multifield'
        return info

    # Try per-equation npz
    for eq in ['rho', 'px', 'py']:
        eq_path = ws_dir / f'wsindy_model_{eq}.npz'
        if eq_path.exists():
            d = np.load(eq_path, allow_pickle=True)
            info.setdefault('equations', {})[eq] = {
                'col_names': list(d['col_names']),
                'w': d['w'],
                'active': d['active'],
            }
    if info:
        info['mode'] = 'multifield'
    return info if info else None


def load_wsindy_bootstrap(exp_dir, equation=None):
    """Load WSINDy bootstrap results."""
    ws_dir = Path(exp_dir) / 'WSINDy'
    if equation:
        path = ws_dir / f'bootstrap_{equation}.npz'
    else:
        path = ws_dir / 'bootstrap.npz'
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    return {
        'col_names': list(d['col_names']),
        'coeff_mean': d['coeff_mean'],
        'coeff_std': d['coeff_std'],
        'inclusion_probability': d['inclusion_probability'],
        'ci_lo': d.get('ci_lo'),
        'ci_hi': d.get('ci_hi'),
    }


# ============================================================================
# Error computation helpers
# ============================================================================

def compute_error_timeseries(rho_true, rho_pred):
    T = rho_true.shape[0]
    e1, e2, einf, mass_true, mass_pred = [], [], [], [], []
    for t in range(T):
        diff = rho_true[t] - rho_pred[t]
        norm_true = np.sqrt(np.sum(rho_true[t] ** 2))
        e2.append(np.sqrt(np.sum(diff ** 2)) / (norm_true + 1e-12))
        e1.append(np.sum(np.abs(diff)) / (np.sum(np.abs(rho_true[t])) + 1e-12))
        einf.append(np.max(np.abs(diff)) / (np.max(np.abs(rho_true[t])) + 1e-12))
        mass_true.append(np.sum(rho_true[t]))
        mass_pred.append(np.sum(rho_pred[t]))
    return {
        'rel_e1': np.array(e1), 'rel_e2': np.array(e2),
        'rel_einf': np.array(einf),
        'mass_true': np.array(mass_true), 'mass_pred': np.array(mass_pred),
    }


def compute_spatial_order_timeseries(rho):
    return np.array([np.std(rho[t]) for t in range(rho.shape[0])])


# ============================================================================
# Discovery
# ============================================================================

def discover_experiments(data_dir, experiment_filter=None):
    data_dir = Path(data_dir)
    experiments = {}
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        if experiment_filter and d.name not in experiment_filter:
            continue
        # Accept experiments with ROM basis
        has_pod = (d / 'rom_common' / 'pod_basis.npz').exists() or \
                  (d / 'mvar' / 'pod_basis.npz').exists()
        has_test = (d / 'test').exists()
        if has_pod and has_test:
            experiments[d.name] = d
    return experiments


def detect_available_models(exp_dir):
    """Return list of model tags with available test predictions."""
    models = []
    p = Path(exp_dir)
    if (p / 'MVAR' / 'test_results.csv').exists() or \
       (p / 'mvar' / 'mvar_model.npz').exists():
        models.append('mvar')
    if (p / 'LSTM').exists():
        try:
            if any((p / 'LSTM').iterdir()):
                models.append('lstm')
        except (StopIteration, PermissionError):
            pass
    return models


def get_ic_test_idx_map(exp_dir):
    meta_path = Path(exp_dir) / 'test' / 'metadata.json'
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, IOError):
            return _default_ic_map()
        ic_map = {}
        for entry in meta:
            dist = entry.get('distribution', entry.get('ic_type', 'unknown')).lower()
            idx = entry.get('run_id', entry.get('test_idx', len(ic_map)))
            if 'gaussian' in dist or 'gauss' in dist:
                ic_map.setdefault('gaussian', idx)
            elif 'uniform' in dist:
                ic_map.setdefault('uniform', idx)
            elif 'ring' in dist:
                ic_map.setdefault('ring', idx)
            elif 'two_cluster' in dist or 'cluster' in dist:
                ic_map.setdefault('two_clusters', idx)
        if ic_map:
            return ic_map
    return _default_ic_map()


def _default_ic_map():
    return {'gaussian': 0, 'uniform': 1, 'ring': 2, 'two_clusters': 3}


def _save_fig(fig, output_dir, stem):
    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'{stem}.{ext}',
                    bbox_inches='tight', dpi=200 if ext == 'png' else None)
    plt.close(fig)


def _get_active_models(group_exps, ic_name, ic_maps):
    """Determine which models actually have data for this (group, IC) combo."""
    available = set()
    for exp_name, exp_dir in group_exps:
        tidx = ic_maps.get(exp_name, {}).get(ic_name)
        if tidx is None:
            continue
        for model in ALL_MODELS:
            r2_df = load_r2_single_test(exp_dir, tidx, model)
            if r2_df is not None:
                available.add(model)
    return [m for m in ALL_MODELS if m in available]


# ============================================================================
# PLOT 1: SVD spectra (ROM only — cross-IC)
# ============================================================================

def plot_svd_spectra(experiments, output_dir, max_modes=40):
    """Singular value decay + cumulative energy (sPOD). ROM-only."""
    exp_list = list(experiments.values())
    n = len(exp_list)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for i, exp_dir in enumerate(exp_list):
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
        ax2.plot(modes, cum, '-', color=color, linewidth=1.2,
                 label=label, alpha=0.8)

    ax1.set_xlabel('Mode index')
    ax1.set_ylabel(r'$(\sigma_i/\sigma_1)^2$')
    ax1.set_title('Singular Value Decay (sPOD)')
    ax1.legend(fontsize=5, ncol=4)
    ax1.grid(True, alpha=0.3)
    ax2.axhline(0.99, color='gray', ls=':', alpha=0.5, label='99 %')
    ax2.axhline(0.999, color='gray', ls='--', alpha=0.5, label='99.9 %')
    ax2.set_xlabel('Number of modes $d$')
    ax2.set_ylabel('Cumulative energy')
    ax2.set_title('Cumulative Energy (sPOD)')
    ax2.legend(fontsize=5, ncol=4)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.85, 1.005)
    fig.suptitle('sPOD Spectra Across Experiments', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig, output_dir, 'svd_spectra_comparison')
    print(f"  Saved: svd_spectra_comparison.pdf")

    # POD vs sPOD contrast
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))
    spod_color, pod_color = '#1f77b4', '#d62728'
    has_unaligned = False
    for i, exp_dir in enumerate(exp_list):
        sv_a = load_singular_values(exp_dir)
        sv_r = load_unaligned_singular_values(exp_dir)
        if sv_a is None:
            continue
        nm = min(max_modes, len(sv_a))
        modes = np.arange(1, nm + 1)
        sv_norm_a = (sv_a[:nm] / sv_a[0]) ** 2
        cum_a = np.cumsum(sv_a[:nm] ** 2) / np.sum(sv_a ** 2)
        lab_a = 'sPOD (aligned)' if i == 0 else None
        ax3.semilogy(modes, sv_norm_a, '-', color=spod_color,
                     linewidth=0.8, alpha=0.35, label=lab_a)
        ax4.plot(modes, cum_a, '-', color=spod_color,
                 linewidth=0.8, alpha=0.35, label=lab_a)
        if sv_r is not None:
            has_unaligned = True
            nm_r = min(max_modes, len(sv_r))
            sv_norm_r = (sv_r[:nm_r] / sv_r[0]) ** 2
            cum_r = np.cumsum(sv_r[:nm_r] ** 2) / np.sum(sv_r ** 2)
            lab_r = 'POD (unaligned)' if i == 0 else None
            ax3.semilogy(modes[:nm_r], sv_norm_r, '-', color=pod_color,
                         linewidth=0.8, alpha=0.35, label=lab_r)
            ax4.plot(modes[:nm_r], cum_r, '-', color=pod_color,
                     linewidth=0.8, alpha=0.35, label=lab_r)
    ax3.set_xlabel('Mode index')
    ax3.set_ylabel(r'$(\sigma_i/\sigma_1)^2$')
    ax3.set_title('Eigenvalue Decay: sPOD vs POD')
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
    if not has_unaligned:
        subtitle += ' (no unaligned data yet)'
    fig2.suptitle(subtitle, fontsize=13, fontweight='bold')
    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig2, output_dir, 'pod_vs_spod_contrast')
    print(f"  Saved: pod_vs_spod_contrast.pdf")


# ============================================================================
# PLOT 2: R² degradation (3-panel: MVAR | LSTM | WSINDy)
# ============================================================================

def plot_r2_degradation_group(group_exps, group_label, ic_name,
                              ic_maps, output_dir):
    """R² over forecast horizon — one panel per available model."""
    active_models = _get_active_models(group_exps, ic_name, ic_maps)
    n_panels = len(active_models)
    if n_panels == 0:
        return
    n = len(group_exps)
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 5), sharey=True,
                             squeeze=False)
    for mi, model in enumerate(active_models):
        ax = axes[0, mi]
        for i, (exp_name, exp_dir) in enumerate(group_exps):
            tidx = ic_maps.get(exp_name, {}).get(ic_name)
            if tidx is None:
                continue
            r2_df = load_r2_single_test(exp_dir, tidx, model)
            if r2_df is None:
                continue
            r2_col = 'r2_reconstructed' if 'r2_reconstructed' in r2_df.columns \
                else r2_df.columns[1]
            color = _color(i, n)
            ax.plot(r2_df['time'], r2_df[r2_col], '-', color=color,
                    linewidth=1.2, label=_short(exp_name), alpha=0.85)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(r'$R^2$')
        ax.set_title(MODEL_DISPLAY[model])
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.05)
    fig.suptitle(f'{group_label} \u2014 $R^2$ Degradation ({IC_DISPLAY[ic_name]} IC)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig, output_dir, 'r2_degradation')


# ============================================================================
# PLOT 2b: Normalized R²
# ============================================================================

def plot_normalized_r2_group(group_exps, group_label, ic_name,
                              ic_maps, output_dir):
    active_models = _get_active_models(group_exps, ic_name, ic_maps)
    n_panels = len(active_models)
    if n_panels == 0:
        return
    n = len(group_exps)
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 5), sharey=True,
                             squeeze=False)
    for mi, model in enumerate(active_models):
        ax = axes[0, mi]
        for i, (exp_name, exp_dir) in enumerate(group_exps):
            tidx = ic_maps.get(exp_name, {}).get(ic_name)
            if tidx is None:
                continue
            r2_df = load_r2_single_test(exp_dir, tidx, model)
            if r2_df is None:
                continue
            r2_col = 'r2_reconstructed' if 'r2_reconstructed' in r2_df.columns \
                else r2_df.columns[1]
            r2_vals = r2_df[r2_col].values
            r2_init = r2_vals[0] if len(r2_vals) > 0 and abs(r2_vals[0]) > 0.01 else 1.0
            r2_norm = r2_vals / r2_init
            color = _color(i, n)
            ax.plot(r2_df['time'], r2_norm, '-', color=color,
                    linewidth=1.2, label=_short(exp_name), alpha=0.85)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(r'$R^2(t)\;/\;R^2(t_0)$')
        ax.set_title(MODEL_DISPLAY[model])
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.5, 1.15)
        ax.axhline(1.0, color='gray', ls=':', alpha=0.4)
        ax.axhline(0.0, color='gray', ls='--', alpha=0.4)
    fig.suptitle(f'{group_label} \u2014 Normalized $R^2$ ({IC_DISPLAY[ic_name]} IC)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig, output_dir, 'normalized_r2')


# ============================================================================
# PLOT 3: Error norms (L¹, L², L∞) — 3-panel
# ============================================================================

def plot_error_norms_group(group_exps, group_label, ic_name,
                            ic_maps, output_dir):
    NORMS = [
        ('L2', 'rel_e2', r'Relative $L^2$ Error'),
        ('L1', 'rel_e1', r'Relative $L^1$ Error'),
        ('Linf', 'rel_einf', r'Relative $L^\infty$ Error'),
    ]
    active_models = _get_active_models(group_exps, ic_name, ic_maps)
    n_panels = len(active_models)
    if n_panels == 0:
        return
    n = len(group_exps)

    # Pre-load errors
    cache = {}
    for exp_name, exp_dir in group_exps:
        tidx = ic_maps.get(exp_name, {}).get(ic_name)
        if tidx is None:
            continue
        cache[exp_name] = {}
        for model in active_models:
            rho_t, rho_p, times = load_density_pair(exp_dir, tidx, model)
            if rho_t is not None:
                cache[exp_name][model] = (compute_error_timeseries(rho_t, rho_p), times)

    for norm_tag, key, title_str in NORMS:
        fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 5),
                                 sharey=True, squeeze=False)
        for mi, model in enumerate(active_models):
            ax = axes[0, mi]
            for i, (exp_name, _) in enumerate(group_exps):
                cached = cache.get(exp_name, {}).get(model)
                if cached is None:
                    continue
                errs, times = cached
                color = _color(i, n)
                ax.plot(times, errs[key], '-', color=color, linewidth=1,
                        label=_short(exp_name), alpha=0.85)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Relative Error')
            ax.set_title(MODEL_DISPLAY[model])
            ax.legend(fontsize=6, ncol=2)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f'{group_label} \u2014 {title_str} ({IC_DISPLAY[ic_name]} IC)',
                     fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        _save_fig(fig, output_dir, f'error_{norm_tag}')


# ============================================================================
# PLOT 4: Mass conservation — 3-panel
# ============================================================================

def plot_mass_conservation_group(group_exps, group_label, ic_name,
                                  ic_maps, output_dir):
    active_models = _get_active_models(group_exps, ic_name, ic_maps)
    n_panels = len(active_models)
    if n_panels == 0:
        return
    n = len(group_exps)
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 5), squeeze=False)
    for mi, model in enumerate(active_models):
        ax = axes[0, mi]
        true_plotted = False
        for i, (exp_name, exp_dir) in enumerate(group_exps):
            tidx = ic_maps.get(exp_name, {}).get(ic_name)
            if tidx is None:
                continue
            rho_t, rho_p, times = load_density_pair(exp_dir, tidx, model)
            if rho_t is None:
                continue
            errs = compute_error_timeseries(rho_t, rho_p)
            color = _color(i, n)
            ax.plot(times, errs['mass_pred'], '-', color=color, linewidth=1,
                    label=_short(exp_name), alpha=0.8)
            if not true_plotted:
                ax.plot(times, errs['mass_true'], '--', color='black',
                        linewidth=1, label='True', alpha=0.6)
                true_plotted = True
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Total Mass')
        ax.set_title(MODEL_DISPLAY[model])
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f'{group_label} \u2014 Mass Conservation ({IC_DISPLAY[ic_name]} IC)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig, output_dir, 'mass_conservation')


# ============================================================================
# PLOT 5: KDE density snapshots (True | MVAR | LSTM | WSINDy)
# ============================================================================

def plot_kde_snapshots_group(group_exps, group_label, ic_name,
                              ic_maps, output_dir, n_frames=4):
    """Density snapshot grid: rows=experiments, column groups per model."""
    active_models = _get_active_models(group_exps, ic_name, ic_maps)
    n_model = len(active_models)
    if n_model == 0:
        return

    valid = []
    for exp_name, exp_dir in group_exps:
        tidx = ic_maps.get(exp_name, {}).get(ic_name)
        if tidx is None:
            continue
        true_path = Path(exp_dir) / 'test' / f'test_{tidx:03d}' / 'density_true.npz'
        if true_path.exists():
            valid.append((exp_name, exp_dir, tidx))
    if not valid:
        return

    n_exp = len(valid)
    # Columns: True snapshots + each model's snapshots
    n_col = n_frames * (1 + n_model)

    fig, axes = plt.subplots(n_exp, n_col,
                             figsize=(2.5 * n_col, 2.2 * n_exp),
                             squeeze=False)

    for ei, (exp_name, exp_dir, tidx) in enumerate(valid):
        dt = np.load(Path(exp_dir) / 'test' / f'test_{tidx:03d}' / 'density_true.npz')
        rho_true_full = dt['rho']
        times_full = dt['times']
        T_total = rho_true_full.shape[0]
        start_f = T_total // 4
        frame_idxs = np.linspace(start_f, T_total - 1, n_frames, dtype=int)

        preds = {}
        for model in active_models:
            _, rho_p, times_p = load_density_pair(exp_dir, tidx, model)
            if rho_p is not None:
                preds[model] = (rho_p, times_p)

        for fi, fidx in enumerate(frame_idxs):
            t_val = times_full[fidx]

            # True density
            ax = axes[ei, fi]
            ax.imshow(rho_true_full[fidx], origin='lower', cmap='hot', aspect='auto')
            ax.axis('off')
            if ei == 0:
                ax.set_title(f'True\nt={t_val:.1f}s', fontsize=7)
            if fi == 0:
                ax.set_ylabel(_short(exp_name), fontsize=7, rotation=0, labelpad=50)

            # Each model
            for mi, model in enumerate(active_models):
                ax = axes[ei, (1 + mi) * n_frames + fi]
                if model in preds:
                    rho_p, times_p = preds[model]
                    p_idx = np.argmin(np.abs(times_p - t_val))
                    if p_idx < len(rho_p):
                        ax.imshow(rho_p[p_idx], origin='lower', cmap='hot', aspect='auto')
                ax.axis('off')
                if ei == 0:
                    ax.set_title(f'{MODEL_DISPLAY[model]}\nt={t_val:.1f}s', fontsize=7)

    fig.suptitle(f'{group_label} \u2014 Density Snapshots ({IC_DISPLAY[ic_name]} IC)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, output_dir, 'kde_snapshots')


# ============================================================================
# PLOT 6: Spatial order — 3-panel
# ============================================================================

def plot_spatial_order_group(group_exps, group_label, ic_name,
                              ic_maps, output_dir):
    active_models = _get_active_models(group_exps, ic_name, ic_maps)
    n_panels = len(active_models)
    if n_panels == 0:
        return
    n = len(group_exps)
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 5),
                             sharey=True, squeeze=False)
    for mi, model in enumerate(active_models):
        ax = axes[0, mi]
        for i, (exp_name, exp_dir) in enumerate(group_exps):
            tidx = ic_maps.get(exp_name, {}).get(ic_name)
            if tidx is None:
                continue
            rho_t, rho_p, times = load_density_pair(exp_dir, tidx, model)
            if rho_t is None:
                continue
            so_true = compute_spatial_order_timeseries(rho_t)
            so_pred = compute_spatial_order_timeseries(rho_p)
            color = _color(i, n)
            ax.plot(times, so_true, '-', color=color, linewidth=1.2, alpha=0.5)
            ax.plot(times, so_pred, '--', color=color, linewidth=1.2,
                    label=_short(exp_name), alpha=0.85)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(r'Spatial Order (std $\rho$)')
        ax.set_title(f'{MODEL_DISPLAY[model]} (solid=true, dashed=pred)')
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f'{group_label} \u2014 Spatial Order ({IC_DISPLAY[ic_name]} IC)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig, output_dir, 'spatial_order')


# ============================================================================
# PLOT 7: Phase dynamics (ROM-only, cross-IC)
# ============================================================================

def plot_phase_dynamics(experiments, output_dir):
    """Phase dynamics analysis: shift trajectory, PSD, AR, variance, autocorr."""
    from scipy import signal
    from numpy.linalg import lstsq

    exp_list = list(experiments.values())
    shift_exps = [(exp, load_shift_data(exp)) for exp in exp_list]
    shift_exps = [(exp, sa) for exp, sa in shift_exps if sa is not None]
    if not shift_exps:
        print("  No shift data found \u2014 skipping phase dynamics.")
        return

    n = len(shift_exps)
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
    ax_traj = fig.add_subplot(gs[0, 0])
    ax_psd = fig.add_subplot(gs[0, 1])
    ax_ar = fig.add_subplot(gs[0, 2])
    ax_var = fig.add_subplot(gs[1, 0])
    ax_ac = fig.add_subplot(gs[1, 1])
    ax_sum = fig.add_subplot(gs[1, 2])
    ar_orders = [1, 2, 3, 5, 10]
    all_ar = []

    for idx, (exp_dir, sa) in enumerate(shift_exps):
        shifts = sa['shifts']
        meta_path = Path(exp_dir) / 'train' / 'metadata.json'
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                M = len(meta) if isinstance(meta, list) else 112
            except Exception:
                M = 112
        else:
            M = 112
        T_s = shifts.shape[0] // M if M > 0 else shifts.shape[0]
        if T_s < 4:
            continue
        if shifts.ndim == 2:
            shifts_3d = shifts[:M * T_s].reshape(M, T_s, 2)
        else:
            shifts_3d = shifts
            M, T_s = shifts_3d.shape[:2]

        color = _color(idx, n)
        label = _short(exp_dir.name)

        mean_s = shifts_3d.mean(axis=0)
        ax_traj.plot(mean_s[:, 1], mean_s[:, 0], '-', color=color,
                     linewidth=1.5, label=label, alpha=0.8)

        for dim in [0, 1]:
            psds = []
            for ri in range(min(M, 50)):
                f_psd, psd = signal.welch(shifts_3d[ri, :, dim], fs=1.0,
                                          nperseg=min(128, T_s // 2))
                psds.append(psd)
            mean_psd = np.mean(psds, axis=0)
            ls = '-' if dim == 1 else '--'
            ax_psd.semilogy(f_psd, mean_psd, ls, color=color,
                            linewidth=1, alpha=0.7)

        ar_r2_p = []
        for p in ar_orders:
            r2s = []
            for dim in [0, 1]:
                Xl, yl = [], []
                for ri in range(M):
                    s = shifts_3d[ri, :, dim]
                    for t in range(p, T_s):
                        Xl.append(s[t - p:t][::-1])
                        yl.append(s[t])
                Xa = np.array(Xl)
                ya = np.array(yl)
                coef, _, _, _ = lstsq(Xa, ya, rcond=None)
                yp = Xa @ coef
                ss_res = np.sum((ya - yp) ** 2)
                ss_tot = np.sum((ya - ya.mean()) ** 2)
                r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else 0)
            ar_r2_p.append(np.mean(r2s))
        ax_ar.plot(ar_orders, ar_r2_p, 'o-', color=color,
                   linewidth=1.2, markersize=5, label=label)
        all_ar.append((label, dict(zip(ar_orders, ar_r2_p))))

        ax_var.bar(idx, np.var(shifts_3d, axis=0).mean(),
                   color=color, alpha=0.7, label=label)

        for dim in [0, 1]:
            ms = shifts_3d[:, :, dim].mean(axis=0)
            ac = np.correlate(ms - ms.mean(), ms - ms.mean(), mode='full')
            ac = ac[len(ac) // 2:]
            ac /= ac[0] if ac[0] > 0 else 1
            ml = min(50, len(ac))
            ls = '-' if dim == 1 else '--'
            ax_ac.plot(np.arange(ml), ac[:ml], ls, color=color,
                       linewidth=1, alpha=0.7)

    ax_traj.set_xlabel(r'$\Delta_x$ (px)')
    ax_traj.set_ylabel(r'$\Delta_y$ (px)')
    ax_traj.set_title('(a) Mean Shift Trajectories')
    ax_traj.legend(fontsize=5, ncol=3)
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_aspect('equal', adjustable='datalim')
    ax_psd.set_xlabel('Frequency')
    ax_psd.set_ylabel('PSD')
    ax_psd.set_title(r'(b) Power Spectrum of $\Delta(t)$')
    ax_psd.grid(True, alpha=0.3)
    ax_ar.set_xlabel('AR order $p$')
    ax_ar.set_ylabel(r'$R^2$')
    ax_ar.set_title(r'(c) AR($p$) Predictability')
    ax_ar.legend(fontsize=5, ncol=3)
    ax_ar.grid(True, alpha=0.3)
    ax_ar.set_ylim(0.5, 1.01)
    ax_ar.set_xticks(ar_orders)
    ax_var.set_xlabel('Experiment')
    ax_var.set_ylabel(r'Mean shift variance (px$^2$)')
    ax_var.set_title('(d) Phase Variance')
    ax_var.grid(True, alpha=0.3, axis='y')
    ax_var.set_xticks(range(n))
    ax_var.set_xticklabels(
        [_short(e.name) for e, _ in shift_exps],
        rotation=45, ha='right', fontsize=5,
    )
    ax_ac.set_xlabel('Lag (timesteps)')
    ax_ac.set_ylabel('Autocorrelation')
    ax_ac.set_title(r'(e) Autocorrelation of mean $\Delta(t)$')
    ax_ac.grid(True, alpha=0.3)
    ax_ac.axhline(0, color='gray', linewidth=0.5)
    ax_sum.axis('off')
    txt = "AR Predictability Summary\n" + "=" * 30 + "\n\n"
    for name, r2d in all_ar[:20]:
        txt += f"{name}: AR(1)={r2d[1]:.3f}  AR(5)={r2d[5]:.3f}\n"
    ax_sum.text(0.05, 0.95, txt, transform=ax_sum.transAxes,
                fontsize=6, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    fig.suptitle('Phase Dynamics Analysis (ROM only)', fontsize=14, fontweight='bold')
    _save_fig(fig, output_dir, 'phase_dynamics_analysis')
    print(f"  Saved: phase_dynamics_analysis.pdf")


# ============================================================================
# PLOT 8: Runtime comparison — all 3 models
# ============================================================================

def plot_runtime_comparison(experiments, output_dir):
    """Bar chart of training time, inference speed, parameter count (all 3 models)."""
    records = []
    for exp_name, exp_dir in experiments.items():
        for model in ALL_MODELS:
            profile_path = exp_dir / model.upper() / 'runtime_profile.json'
            # WSINDy uses WSINDy/ directory
            if not profile_path.exists() and model == 'wsindy':
                profile_path = exp_dir / 'WSINDy' / 'runtime_profile.json'
            if not profile_path.exists():
                continue
            with open(profile_path) as f:
                prof = json.load(f)
            records.append({
                'experiment': _short(exp_name),
                'model': MODEL_DISPLAY[model],
                'training_s': prof.get('training_time_seconds', 0),
                'inference_us': prof.get('inference', {}).get(
                    'single_step', {}).get('mean_seconds', 0) * 1e6,
                'params': prof.get('model_params', 0),
            })
    if not records:
        print("  No runtime profiles found \u2014 skipping.")
        return
    df = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ('training_s', 'Training / Discovery Time (s)'),
        ('inference_us', 'Inference (\u00b5s / step)'),
        ('params', 'Parameters / Active Terms'),
    ]
    for ax, (col, ylabel) in zip(axes, metrics):
        pivot = df.pivot_table(index='experiment', columns='model',
                               values=col, aggfunc='mean')
        pivot.plot.bar(ax=ax, rot=45, alpha=0.8)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    fig.suptitle('Runtime Comparison: MVAR vs LSTM vs WSINDy',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig, output_dir, 'runtime_comparison_3models')
    print(f"  Saved: runtime_comparison_3models.pdf")


# ============================================================================
# WSINDY-SPECIFIC PLOTS
# ============================================================================

def plot_wsindy_coefficients(experiments, output_dir):
    """Per-experiment bar chart of discovered PDE coefficients."""
    ws_exps = []
    for exp_name, exp_dir in experiments.items():
        info = load_wsindy_model_info(exp_dir)
        if info is not None:
            ws_exps.append((exp_name, exp_dir, info))
    if not ws_exps:
        print("  No WSINDy model info found \u2014 skipping coefficients.")
        return

    for exp_name, exp_dir, info in ws_exps:
        if info.get('mode') == 'scalar':
            w = info['w']
            active = info['active'].astype(bool)
            col_names = info['col_names']
            active_idx = np.where(active)[0]
            if len(active_idx) == 0:
                continue

            fig, ax = plt.subplots(figsize=(max(6, len(active_idx) * 0.8), 4))
            names = [col_names[i] for i in active_idx]
            coeffs = w[active_idx]
            x = np.arange(len(names))
            colors = ['steelblue' if c >= 0 else '#d62728' for c in coeffs]
            ax.bar(x, coeffs, color=colors, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(names, fontsize=9, rotation=45, ha='right')
            ax.set_ylabel('Coefficient')
            ax.set_title(f'WSINDy Discovered PDE \u2014 {_short(exp_name)}',
                         fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(0, color='k', linewidth=0.5)
            fig.tight_layout()
            _save_fig(fig, output_dir, f'wsindy_coefficients_{_short(exp_name)}')

        elif info.get('mode') == 'multifield':
            # Multi-field: one subplot per equation
            eqs = info.get('equations', {})

            # Try to load identification_summary.json for error bars + Pi_m
            id_summary = None
            id_summary_path = Path(exp_dir) / 'WSINDy' / 'identification_summary.json'
            if id_summary_path.exists():
                with open(id_summary_path) as f:
                    id_summary = json.load(f)

            if not eqs:
                # Try discovered_pde from JSON
                dp = info.get('discovered_pde', {})
                if dp:
                    n_eq = len(dp)
                    fig, axes = plt.subplots(1, n_eq, figsize=(6 * n_eq, 4.5),
                                            squeeze=False)
                    for ei, (eq_name, eq_data) in enumerate(dp.items()):
                        ax = axes[0, ei]
                        if isinstance(eq_data, dict):
                            terms = eq_data.get('active_terms', [])
                            coeffs_dict = eq_data.get('coefficients', {})
                            names = list(coeffs_dict.keys())
                            vals = [coeffs_dict[n] for n in names]
                        else:
                            continue
                        if not names:
                            ax.text(0.5, 0.5, 'No active terms',
                                    ha='center', va='center', transform=ax.transAxes)
                            ax.set_title(eq_name)
                            continue
                        x = np.arange(len(names))

                        # Sign-aware coloring: green = physically expected sign,
                        # red = unexpected/constrained, steelblue = neutral
                        _expect_neg = {
                            'px': {'px', 'lap_px', 'dx_rho2'},
                            'py': {'py', 'lap_py', 'dy_rho2'},
                        }
                        neg_set = _expect_neg.get(eq_name, set())
                        colors = []
                        for n_i, v_i in zip(names, vals):
                            if n_i in neg_set:
                                colors.append('#2ca02c' if v_i < 0 else '#d62728')
                            else:
                                colors.append('steelblue' if v_i >= 0 else '#d62728')

                        # Error bars from identification_summary
                        yerr = None
                        if id_summary is not None:
                            eq_sum = id_summary.get('equations', {}).get(
                                eq_name if '_t' in eq_name else f'{eq_name}_t', {})
                            std_c = eq_sum.get('std_coefficients', {})
                            if std_c:
                                yerr = [std_c.get(n_i, 0.0) for n_i in names]

                        ax.bar(x, vals, color=colors, alpha=0.8,
                               yerr=yerr, capsize=3, ecolor='gray')
                        ax.set_xticks(x)
                        ax.set_xticklabels(names, fontsize=7, rotation=45, ha='right')
                        ax.set_ylabel('Coefficient')
                        ax.set_title(f'{eq_name} equation')
                        ax.grid(True, alpha=0.3, axis='y')
                        ax.axhline(0, color='k', linewidth=0.5)

                        # Annotate Pi_tilde_m above each bar
                        if id_summary is not None:
                            eq_sum = id_summary.get('equations', {}).get(
                                eq_name if '_t' in eq_name else f'{eq_name}_t', {})
                            pi_norm = eq_sum.get('mean_dominant_balance', {})
                            if pi_norm:
                                for xi, n_i in enumerate(names):
                                    pi_val = pi_norm.get(n_i)
                                    if pi_val is not None and pi_val > 0.01:
                                        y_pos = vals[xi]
                                        offset = 0.02 * (max(abs(v) for v in vals) or 1)
                                        ax.annotate(
                                            f'\u03a0\u0303={pi_val:.2f}',
                                            (xi, y_pos + (offset if y_pos >= 0 else -offset)),
                                            ha='center', va='bottom' if y_pos >= 0 else 'top',
                                            fontsize=6, color='#555555')

                    fig.suptitle(f'WSINDy Multi-Field PDE \u2014 {_short(exp_name)}',
                                 fontsize=12, fontweight='bold')
                    fig.tight_layout(rect=[0, 0, 1, 0.93])
                    _save_fig(fig, output_dir, f'wsindy_coefficients_{_short(exp_name)}')
                continue

            n_eq = len(eqs)
            fig, axes = plt.subplots(1, n_eq, figsize=(6 * n_eq, 4), squeeze=False)
            for ei, (eq_name, eq_data) in enumerate(eqs.items()):
                ax = axes[0, ei]
                w_eq = eq_data['w']
                active_eq = eq_data['active'].astype(bool)
                names_eq = eq_data['col_names']
                active_idx = np.where(active_eq)[0]
                if len(active_idx) == 0:
                    ax.text(0.5, 0.5, 'No active terms',
                            ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(eq_name)
                    continue
                names = [names_eq[i] for i in active_idx]
                coeffs = w_eq[active_idx]
                x = np.arange(len(names))
                colors = ['steelblue' if c >= 0 else '#d62728' for c in coeffs]
                ax.bar(x, coeffs, color=colors, alpha=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels(names, fontsize=7, rotation=45, ha='right')
                ax.set_ylabel('Coefficient')
                ax.set_title(f'{eq_name} equation')
                ax.grid(True, alpha=0.3, axis='y')
                ax.axhline(0, color='k', linewidth=0.5)
            fig.suptitle(f'WSINDy Multi-Field PDE \u2014 {_short(exp_name)}',
                         fontsize=12, fontweight='bold')
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            _save_fig(fig, output_dir, f'wsindy_coefficients_{_short(exp_name)}')

    print(f"  Saved: wsindy_coefficients_*.pdf ({len(ws_exps)} experiments)")


def plot_wsindy_bootstrap(experiments, output_dir):
    """Bootstrap confidence intervals and inclusion probability."""
    for exp_name, exp_dir in experiments.items():
        ws_dir = Path(exp_dir) / 'WSINDy'
        if not ws_dir.exists():
            continue

        # Determine equations to plot
        equations = [None]  # scalar default
        for eq in ['rho', 'px', 'py']:
            if (ws_dir / f'bootstrap_{eq}.npz').exists():
                if equations == [None]:
                    equations = []
                equations.append(eq)

        for eq in equations:
            boot = load_wsindy_bootstrap(exp_dir, eq)
            if boot is None:
                continue

            col_names = boot['col_names']
            inc_prob = boot['inclusion_probability']
            means = boot['coeff_mean']
            stds = boot['coeff_std']
            ci_lo = boot.get('ci_lo')
            ci_hi = boot.get('ci_hi')

            mask = inc_prob > 0.01
            if not np.any(mask):
                continue
            idx = np.where(mask)[0]
            order = np.argsort(-inc_prob[idx])
            idx = idx[order]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Panel 1: Coefficients + CI
            x = np.arange(len(idx))
            if ci_lo is not None and ci_hi is not None:
                yerr_lo = means[idx] - ci_lo[idx]
                yerr_hi = ci_hi[idx] - means[idx]
                ax1.errorbar(x, means[idx],
                             yerr=[yerr_lo, yerr_hi],
                             fmt='o', capsize=5, color='steelblue',
                             markersize=8, label='Mean \u00b1 95% CI')
            else:
                ax1.errorbar(x, means[idx],
                             yerr=1.96 * stds[idx],
                             fmt='o', capsize=5, color='steelblue',
                             markersize=8, label='Mean \u00b1 1.96\u03c3')
            ax1.set_xticks(x)
            ax1.set_xticklabels([col_names[i] for i in idx], fontsize=8,
                                rotation=45, ha='right')
            ax1.set_ylabel('Coefficient')

            eq_label = f' ({eq})' if eq else ''
            ax1.set_title(f'Bootstrap Coefficients{eq_label}', fontweight='bold')
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.axhline(0, color='k', linewidth=0.5)

            # Panel 2: Inclusion probability
            colors = ['seagreen' if inc_prob[i] >= 0.5 else 'salmon' for i in idx]
            ax2.bar(x, inc_prob[idx], color=colors, alpha=0.85)
            ax2.axhline(0.5, color='k', ls='--', linewidth=1.5,
                        label='50% threshold')
            ax2.set_xticks(x)
            ax2.set_xticklabels([col_names[i] for i in idx], fontsize=8,
                                rotation=45, ha='right')
            ax2.set_ylabel('Inclusion probability')
            ax2.set_title(f'Term Selection Frequency{eq_label}', fontweight='bold')
            ax2.set_ylim(0, 1.05)
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3, axis='y')

            suffix = f'_{eq}' if eq else ''
            fig.suptitle(f'WSINDy Bootstrap UQ \u2014 {_short(exp_name)}{eq_label}',
                         fontsize=13, fontweight='bold')
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            _save_fig(fig, output_dir,
                      f'wsindy_bootstrap_{_short(exp_name)}{suffix}')

    print(f"  Saved: wsindy_bootstrap_*.pdf")


# ============================================================================
# PLOT: Dominant balance (horizontal bar chart per equation)
# ============================================================================

def plot_dominant_balance(experiments, output_dir):
    """Horizontal bar chart of normalised dominant balance Pi_tilde per term."""
    for exp_name, exp_dir in experiments.items():
        id_path = Path(exp_dir) / 'WSINDy' / 'identification_summary.json'
        if not id_path.exists():
            continue
        with open(id_path) as f:
            id_sum = json.load(f)
        eqs = id_sum.get('equations', {})
        if not eqs:
            continue

        n_eq = len(eqs)
        fig, axes = plt.subplots(1, n_eq, figsize=(5 * n_eq, 4), squeeze=False)
        for ei, (eq_name, eq_data) in enumerate(eqs.items()):
            ax = axes[0, ei]
            pi_norm = eq_data.get('mean_dominant_balance', {})
            if not pi_norm:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes)
                ax.set_title(eq_name)
                continue
            # Sort descending
            items = sorted(pi_norm.items(), key=lambda kv: kv[1], reverse=True)
            names = [k for k, v in items]
            vals = [v for k, v in items]
            y = np.arange(len(names))
            ax.barh(y, vals, color='steelblue', alpha=0.8)
            ax.set_yticks(y)
            ax.set_yticklabels(names, fontsize=8)
            ax.set_xlabel(r'$\tilde{\Pi}_m$')
            ax.set_title(f'{eq_name}')
            ax.set_xlim(0, 1.05)
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')
            # Annotate values
            for yi, vi in enumerate(vals):
                if vi > 0.01:
                    ax.text(vi + 0.01, yi, f'{vi:.3f}', va='center', fontsize=7)
        fig.suptitle(f'Dominant Balance — {_short(exp_name)}',
                     fontsize=12, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        _save_fig(fig, output_dir, f'dominant_balance_{_short(exp_name)}')

    print(f"  Saved: dominant_balance_*.pdf")


# ============================================================================
# PLOT: Regime comparison heatmap (terms × regimes, cells = Pi_tilde)
# ============================================================================

def plot_regime_comparison_heatmap(experiments, output_dir):
    """Heatmap: rows=regimes, cols=library terms, cells=Pi_tilde_m."""
    # Collect per-equation data across all experiments
    eq_data = {}  # eq_name -> {exp_name: {term: Pi_tilde}}
    for exp_name, exp_dir in experiments.items():
        id_path = Path(exp_dir) / 'WSINDy' / 'identification_summary.json'
        if not id_path.exists():
            continue
        with open(id_path) as f:
            id_sum = json.load(f)
        for eq_name, eq_info in id_sum.get('equations', {}).items():
            pi_norm = eq_info.get('mean_dominant_balance', {})
            if pi_norm:
                eq_data.setdefault(eq_name, {})[exp_name] = pi_norm

    if not eq_data:
        print("  No identification summaries found — skipping regime heatmap.")
        return

    for eq_name, regime_dict in eq_data.items():
        # Union of all terms
        all_terms = sorted(set().union(*(d.keys() for d in regime_dict.values())))
        regime_names = list(regime_dict.keys())
        if len(regime_names) < 2 or not all_terms:
            continue

        mat = np.zeros((len(regime_names), len(all_terms)))
        for ri, rn in enumerate(regime_names):
            for ti, tn in enumerate(all_terms):
                mat[ri, ti] = regime_dict[rn].get(tn, 0.0)

        fig, ax = plt.subplots(figsize=(max(6, len(all_terms) * 0.9),
                                        max(3, len(regime_names) * 0.7)))
        im = ax.imshow(mat, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(all_terms)))
        ax.set_xticklabels(all_terms, fontsize=7, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(regime_names)))
        ax.set_yticklabels([_short(r) for r in regime_names], fontsize=8)
        # Annotate cells
        for ri in range(len(regime_names)):
            for ti in range(len(all_terms)):
                v = mat[ri, ti]
                if v > 0.01:
                    ax.text(ti, ri, f'{v:.2f}', ha='center', va='center',
                            fontsize=6, color='white' if v > 0.5 else 'black')
        fig.colorbar(im, ax=ax, shrink=0.8, label=r'$\tilde{\Pi}_m$')
        eq_label = eq_name.replace('_t', '')
        ax.set_title(f'Dominant Balance — {eq_label} equation', fontweight='bold')
        fig.tight_layout()
        _save_fig(fig, output_dir, f'regime_comparison_{eq_label}')

    print(f"  Saved: regime_comparison_*.pdf")


# ============================================================================
# PLOT: Condition numbers (grouped bar chart, log scale)
# ============================================================================

def plot_condition_numbers(experiments, output_dir):
    """Grouped bar chart of cond(G) for rho/px/py across regimes."""
    records = []
    for exp_name, exp_dir in experiments.items():
        diag_path = Path(exp_dir) / 'WSINDy' / 'multifield_diagnostics.json'
        if not diag_path.exists():
            continue
        with open(diag_path) as f:
            diag = json.load(f)
        fit_diag = diag.get('fit_diagnostics', {})
        row = {'regime': _short(exp_name)}
        for eq in ['rho', 'px', 'py']:
            row[eq] = fit_diag.get(eq, {}).get('condition_number', float('nan'))
        records.append(row)

    if not records:
        print("  No diagnostics found — skipping condition numbers.")
        return

    regimes = [r['regime'] for r in records]
    eq_names = ['rho', 'px', 'py']
    x = np.arange(len(regimes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(6, len(regimes) * 1.5), 5))
    for i, eq in enumerate(eq_names):
        vals = [r[eq] for r in records]
        ax.bar(x + i * width, vals, width, label=f'{eq}', alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(regimes, fontsize=8, rotation=30, ha='right')
    ax.set_ylabel(r'$\kappa(G)$')
    ax.set_yscale('log')
    ax.set_title('Condition Numbers by Regime and Equation', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    _save_fig(fig, output_dir, 'condition_numbers')
    print(f"  Saved: condition_numbers.pdf")


# ============================================================================
# PLOT: Model comparison summary (R² bar chart across all models)
# ============================================================================

def plot_r2_summary_bars(experiments, ic_maps, ic_types, output_dir):
    """For each IC, create a grouped bar chart: experiments × models → mean R²."""
    active_models = [m for m in ALL_MODELS if any(
        (Path(d) / m.upper() / 'test_results.csv').exists()
        for d in experiments.values())]
    if not active_models:
        print("  No model data found — skipping r2_summary_bars.")
        return
    for ic_name in ic_types:
        records = []
        for exp_name, exp_dir in experiments.items():
            tidx = ic_maps.get(exp_name, {}).get(ic_name)
            if tidx is None:
                continue
            for model in active_models:
                r2_df = load_r2_single_test(exp_dir, tidx, model)
                if r2_df is None:
                    continue
                r2_col = 'r2_reconstructed' if 'r2_reconstructed' in r2_df.columns \
                    else r2_df.columns[1]
                mean_r2 = r2_df[r2_col].mean()
                records.append({
                    'experiment': _short(exp_name),
                    'model': MODEL_DISPLAY[model],
                    'mean_r2': mean_r2,
                })
        if not records:
            continue
        df = pd.DataFrame(records)
        pivot = df.pivot_table(index='experiment', columns='model',
                               values='mean_r2', aggfunc='mean')
        fig, ax = plt.subplots(figsize=(max(10, len(pivot) * 0.6), 5))
        pivot.plot.bar(ax=ax, rot=45, alpha=0.85,
                       color=[MODEL_COLORS.get(m.lower(), 'gray')
                              for m in pivot.columns])
        ax.set_ylabel(r'Mean $R^2$')
        ax.set_title(f'Mean $R^2$ by Model \u2014 {IC_DISPLAY[ic_name]} IC',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(min(0, ax.get_ylim()[0]), 1.05)
        fig.tight_layout()
        _save_fig(fig, output_dir, f'r2_summary_{ic_name}')
    print(f"  Saved: r2_summary_*.pdf")


# ############################################################################
#                     THESIS-MODE AGGREGATE FIGURES
# ############################################################################

def _build_experiment_style_map(experiments):
    """Assign a unique (color, marker) pair to each experiment.

    Returns dict[exp_name -> {'color': str, 'marker': marker, 'index': int}].
    Deterministic: sorted experiment names give stable assignment.
    """
    cmap = plt.cm.get_cmap('tab20')
    names = sorted(experiments.keys())
    style_map = {}
    for i, name in enumerate(names):
        color = cmap(i % 20 / 19) if len(names) > 1 else cmap(0.0)
        marker = EXPERIMENT_MARKERS[i % len(EXPERIMENT_MARKERS)]
        style_map[name] = {'color': color, 'marker': marker, 'index': i}
    return style_map


def _draw_experiment_legend(ax, style_map, ncol=4, fontsize=6, loc='lower center',
                            bbox_to_anchor=None, markersize=5):
    """Add a compact legend mapping each experiment's color+marker to short name."""
    handles = []
    for exp_name in sorted(style_map.keys()):
        s = style_map[exp_name]
        h = plt.Line2D([], [], marker=s['marker'], color=s['color'],
                        linestyle='None', markersize=markersize,
                        label=_short(exp_name))
        handles.append(h)
    if bbox_to_anchor is None:
        bbox_to_anchor = (0.5, -0.15)
    ax.legend(handles=handles, ncol=ncol, fontsize=fontsize, loc=loc,
              bbox_to_anchor=bbox_to_anchor, frameon=True, fancybox=True,
              framealpha=0.8)


def _save_experiment_legend(style_map, output_dir, stem, ncol=5, fontsize=8,
                            markersize=7):
    """Save a standalone legend image mapping each experiment's color+marker."""
    handles = []
    for exp_name in sorted(style_map.keys()):
        s = style_map[exp_name]
        h = plt.Line2D([], [], marker=s['marker'], color=s['color'],
                        linestyle='None', markersize=markersize,
                        label=_short(exp_name))
        handles.append(h)
    if not handles:
        return
    n_rows = (len(handles) + ncol - 1) // ncol
    fig_leg = plt.figure(figsize=(ncol * 1.8, n_rows * 0.45))
    fig_leg.legend(handles=handles, ncol=ncol, fontsize=fontsize, loc='center',
                   frameon=True, fancybox=True, framealpha=0.9,
                   edgecolor='gray')
    fig_leg.tight_layout()
    _save_fig(fig_leg, output_dir, stem)


def _available_thesis_models(exp_dir):
    """Return list of thesis model tags that have prediction data for an experiment."""
    p = Path(exp_dir)
    avail = []
    for model in THESIS_MODELS:
        tag = model.upper()
        if (p / tag / 'test_results.csv').exists():
            avail.append(model)
    return avail


def _apply_thesis_style():
    """Set matplotlib rcParams for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.labelweight': 'bold',
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


# ---------------------------------------------------------------------------
# Helper: collect all r2_vs_time traces for a model across all experiments
# ---------------------------------------------------------------------------

def _collect_all_r2_traces(experiments, model):
    """Return list of (time_array, r2_array, exp_name) for every test run."""
    traces = []
    model_tag = model.lower()
    for exp_name, exp_dir in experiments.items():
        test_dir = Path(exp_dir) / 'test'
        if not test_dir.exists():
            continue
        for run_dir in sorted(test_dir.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith('test_'):
                continue
            # Try model-specific file first, then generic
            for pattern in [f'r2_vs_time_{model_tag}.csv', 'r2_vs_time.csv']:
                fpath = run_dir / pattern
                if fpath.exists():
                    try:
                        df = pd.read_csv(fpath)
                    except Exception:
                        continue
                    r2_col = 'r2_reconstructed' if 'r2_reconstructed' in df.columns \
                        else df.columns[1]
                    traces.append((df['time'].values, df[r2_col].values, exp_name))
                    break  # don't double-count
    return traces


# ---------------------------------------------------------------------------
# THESIS FIG 1: R² degradation — model-level aggregate with IQR
# ---------------------------------------------------------------------------

def plot_thesis_r2_degradation(experiments, output_dir):
    """Multi-panel R²(t) degradation: one per model, all experiments overlaid.

    Individual traces are semi-faint; bold mean + IQR (25–75 %) shaded band.
    Panels are created only for models with data.
    """
    print(f"\n{'=' * 60}")
    print("THESIS FIG 1: R² Degradation (all experiments, model-level)")
    print("=" * 60)

    # Determine which models have data
    model_traces = {}
    for model in THESIS_MODELS:
        traces = _collect_all_r2_traces(experiments, model)
        if traces:
            model_traces[model] = traces
    active_models = list(model_traces.keys())
    if not active_models:
        print("  No R² data found — skipping.")
        return

    n_panels = len(active_models)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5),
                             squeeze=False)

    for mi, model in enumerate(active_models):
        ax = axes[0, mi]
        traces = model_traces[model]

        exp_names = sorted(set(t[2] for t in traces))
        all_times = np.unique(np.concatenate([t[0] for t in traces]))
        r2_matrix = []
        for t_arr, r2_arr, _ in traces:
            r2_interp = np.interp(all_times, t_arr, r2_arr,
                                  left=np.nan, right=np.nan)
            r2_matrix.append(r2_interp)
        r2_matrix = np.array(r2_matrix)

        color = THESIS_COLORS[model]

        # Semi-faint individual lines (more visible than before)
        for row in r2_matrix:
            ax.plot(all_times, row, '-', color=color,
                    linewidth=0.8, alpha=0.3)

        # Statistics
        with np.errstate(all='ignore'):
            mean_r2 = np.nanmean(r2_matrix, axis=0)
            p25 = np.nanpercentile(r2_matrix, 25, axis=0)
            p75 = np.nanpercentile(r2_matrix, 75, axis=0)

        ax.fill_between(all_times, p25, p75, color=color, alpha=0.25,
                        label='IQR (25–75%)')
        ax.plot(all_times, mean_r2, '-', color=color, linewidth=2.5,
                label='Mean')

        ax.axhline(0.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel(r'$R^2$ (density-space)')
        ax.set_title(THESIS_LABELS[model], fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='lower left')
        ax.grid(True, alpha=0.3)

        # Clip to something readable; use display floor for all panels
        if model == 'mvar':
            ax.set_ylim(-0.1, 1.05)
        else:
            ymin = max(np.nanpercentile(r2_matrix, 2), R2_DISPLAY_FLOOR)
            ax.set_ylim(ymin, 1.05)

        n_exp = len(exp_names)
        n_traces = len(traces)
        ax.annotate(f'{n_traces} runs, {n_exp} experiments',
                    xy=(0.98, 0.02), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=8, color='gray')

    fig.suptitle(r'$R^2$ Degradation Over Forecast Horizon',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save_fig(fig, output_dir, 'thesis_r2_degradation')
    print(f"  Saved: thesis_r2_degradation.pdf/png")


# ---------------------------------------------------------------------------
# THESIS FIG 1b: R² degradation — per-experiment detail
# ---------------------------------------------------------------------------

def plot_thesis_r2_degradation_detail(experiments, output_dir, style_map):
    """Per-experiment R²(t) traces: each experiment's test runs averaged ± 1σ.

    One figure per model.  Color + marker from style_map identifies experiment;
    markers placed every ~10 % of the time span for readability.
    """
    print(f"\n{'=' * 60}")
    print("THESIS FIG 1b: R² Degradation Detail (per-experiment)")
    print("=" * 60)

    for model in _get_active_thesis_models(experiments):
        all_traces = _collect_all_r2_traces(experiments, model)
        if not all_traces:
            continue

        # Group traces by experiment
        from collections import defaultdict
        exp_traces = defaultdict(list)
        for t_arr, r2_arr, exp_name in all_traces:
            exp_traces[exp_name].append((t_arr, r2_arr))

        fig, ax = plt.subplots(figsize=(12, 7))

        for exp_name in sorted(exp_traces.keys()):
            traces = exp_traces[exp_name]
            sty = style_map.get(exp_name, {'color': 'gray', 'marker': 'o'})

            # Build common time grid for this experiment's runs
            exp_times = np.unique(np.concatenate([t[0] for t in traces]))
            r2_mat = []
            for t_arr, r2_arr in traces:
                r2_mat.append(np.interp(exp_times, t_arr, r2_arr,
                                        left=np.nan, right=np.nan))
            r2_mat = np.array(r2_mat)

            with np.errstate(all='ignore'):
                mean_r2 = np.nanmean(r2_mat, axis=0)
                std_r2 = np.nanstd(r2_mat, axis=0)

            ax.plot(exp_times, mean_r2, '-', color=sty['color'],
                    linewidth=1.2, alpha=0.85)
            ax.fill_between(exp_times, mean_r2 - std_r2, mean_r2 + std_r2,
                            color=sty['color'], alpha=0.12)

            # Place markers at ~10 evenly-spaced indices
            n_marks = max(1, len(exp_times) // 10)
            mark_idx = np.linspace(0, len(exp_times) - 1, n_marks, dtype=int)
            ax.scatter(exp_times[mark_idx], mean_r2[mark_idx],
                       marker=sty['marker'], color=sty['color'],
                       s=30, zorder=4, edgecolors='white', linewidths=0.4)

        ax.axhline(0.0, color='gray', ls='--', lw=0.8, alpha=0.5)
        ax.axhline(1.0, color='gray', ls=':', lw=0.8, alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(r'$R^2$ (density-space)')
        ax.set_title(f'{THESIS_LABELS[model]} — Per-Experiment $R^2$ Degradation',
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(R2_DISPLAY_FLOOR, R2_DISPLAY_CEIL)

        # Save experiment legend as separate image
        _save_experiment_legend({k: style_map[k] for k in exp_traces
                                 if k in style_map},
                                output_dir, f'thesis_r2_detail_{model}_legend')
        fig.tight_layout()
        _save_fig(fig, output_dir, f'thesis_r2_detail_{model}')
        print(f"  Saved: thesis_r2_detail_{model}.pdf/png")


# ---------------------------------------------------------------------------
# THESIS FIG 2: Per-experiment MVAR vs LSTM R² degradation (side by side)
# ---------------------------------------------------------------------------

def plot_thesis_r2_degradation_per_exp(experiments, output_dir, style_map):
    """Per-experiment R²(t): 2 panels side by side — MVAR (left) | LSTM (right).

    All experiments overlaid with unique color+marker.  One figure saved as
    thesis_r2_per_exp.pdf/png.
    """
    print(f"\n{'=' * 60}")
    print("THESIS FIG 2: Per-Experiment R² Degradation (MVAR | LSTM)")
    print("=" * 60)

    active_models = _get_active_thesis_models(experiments)
    if not active_models:
        print("  No active models found — skipping.")
        return

    fig, axes = plt.subplots(1, len(active_models),
                             figsize=(8 * len(active_models), 6),
                             sharey=True, squeeze=False)

    for mi, model in enumerate(active_models):
        ax = axes[0, mi]
        all_traces = _collect_all_r2_traces(experiments, model)
        if not all_traces:
            ax.set_title(f'{THESIS_LABELS[model]} — no data')
            continue

        from collections import defaultdict
        exp_traces = defaultdict(list)
        for t_arr, r2_arr, exp_name in all_traces:
            exp_traces[exp_name].append((t_arr, r2_arr))

        for exp_name in sorted(exp_traces.keys()):
            traces = exp_traces[exp_name]
            sty = style_map.get(exp_name, {'color': 'gray', 'marker': 'o'})

            exp_times = np.unique(np.concatenate([t[0] for t in traces]))
            r2_mat = []
            for t_arr, r2_arr in traces:
                r2_mat.append(np.interp(exp_times, t_arr, r2_arr,
                                        left=np.nan, right=np.nan))
            r2_mat = np.array(r2_mat)

            with np.errstate(all='ignore'):
                mean_r2 = np.nanmean(r2_mat, axis=0)
                std_r2 = np.nanstd(r2_mat, axis=0)

            ax.plot(exp_times, mean_r2, '-', color=sty['color'],
                    linewidth=1.2, alpha=0.85, label=_short(exp_name))
            ax.fill_between(exp_times, mean_r2 - std_r2, mean_r2 + std_r2,
                            color=sty['color'], alpha=0.12)

            n_marks = max(1, len(exp_times) // 10)
            mark_idx = np.linspace(0, len(exp_times) - 1, n_marks, dtype=int)
            ax.scatter(exp_times[mark_idx], mean_r2[mark_idx],
                       marker=sty['marker'], color=sty['color'],
                       s=30, zorder=4, edgecolors='white', linewidths=0.4)

        ax.axhline(0.0, color='gray', ls='--', lw=0.8, alpha=0.5)
        ax.axhline(1.0, color='gray', ls=':', lw=0.8, alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_title(THESIS_LABELS[model], fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(R2_DISPLAY_FLOOR, R2_DISPLAY_CEIL)
        if mi == 0:
            ax.set_ylabel(r'$R^2$ (density-space)')
        if mi == len(active_models) - 1:
            ax.legend(fontsize=7, ncol=2, loc='lower left')

    fig.suptitle('Per-Experiment $R^2$ Degradation Over Forecast Horizon',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, output_dir, 'thesis_r2_per_exp')
    print(f"  Saved: thesis_r2_per_exp.pdf/png")


# ---------------------------------------------------------------------------
# THESIS FIG 2: Density-space vs Latent-space R² scatter
# ---------------------------------------------------------------------------

def plot_thesis_density_latent_scatter(experiments, output_dir, style_map):
    """Scatter of R²_density vs R²_latent — per-experiment marker + per-model fill.

    Each experiment gets a unique color+marker from style_map.
    Model identity: MVAR = filled markers, LSTM = open/hollow.
    """
    print(f"\n{'=' * 60}")
    print("THESIS FIG 2: Density vs Latent R² Scatter")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Only plot models that actually have data
    active_models = _get_active_thesis_models(experiments)

    # Model fill styles: filled, hollow, cross-hatched
    _model_fill = {
        'mvar':   {'facecolors': None, 'edgecolors': None, 'linewidths': 0.6},
        'lstm':   {'facecolors': 'none', 'edgecolors': None, 'linewidths': 1.2},
    }
    # For the bold mean markers
    _model_fill_bold = {
        'mvar':   {'facecolors': None, 'edgecolors': 'black', 'linewidths': 1.0},
        'lstm':   {'facecolors': 'none', 'edgecolors': None, 'linewidths': 1.5},
    }

    model_counts = {m: 0 for m in active_models}
    exps_plotted = set()

    for model in active_models:
        model_tag = model.upper()
        fill = _model_fill[model]
        fill_bold = _model_fill_bold[model]

        for exp_name, exp_dir in sorted(experiments.items()):
            sty = style_map.get(exp_name, {'color': 'gray', 'marker': 'o'})
            csv_path = Path(exp_dir) / model_tag / 'test_results.csv'
            if not csv_path.exists():
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            if 'r2_reconstructed' not in df.columns or 'r2_latent' not in df.columns:
                continue
            x = df['r2_reconstructed'].values
            y = df['r2_latent'].values
            model_counts[model] += len(x)
            exps_plotted.add(exp_name)

            fc = sty['color'] if fill['facecolors'] is None else fill['facecolors']
            ec = sty['color'] if fill['edgecolors'] is None else fill['edgecolors']

            # Individual test points (faint)
            ax.scatter(x, y, marker=sty['marker'], c=[fc] * len(x),
                       s=25, alpha=0.35, edgecolors=ec,
                       linewidths=fill['linewidths'])

            # Per-experiment mean (bold)
            fc_b = sty['color'] if fill_bold['facecolors'] is None else fill_bold['facecolors']
            ec_b = sty['color'] if fill_bold['edgecolors'] is None else fill_bold['edgecolors']
            ax.scatter(np.mean(x), np.mean(y), marker=sty['marker'],
                       c=[fc_b], s=120, edgecolors=ec_b,
                       linewidths=fill_bold['linewidths'], zorder=5)

    # Model legend entries
    for model in active_models:
        if model_counts[model] == 0:
            continue
        fc = THESIS_COLORS[model] if model != 'lstm' else 'none'
        ec = 'black' if model != 'lstm' else THESIS_COLORS[model]
        ax.scatter([], [], marker=THESIS_MARKERS[model], c=[fc],
                   s=60, edgecolors=ec, linewidths=1.0,
                   label=f'{THESIS_LABELS[model]} (n={model_counts[model]})')

    # Reference diagonal
    ax.plot([-1, 1.05], [-1, 1.05], '--', color='gray', alpha=0.5,
            linewidth=1, label=r'$R^2_\mathrm{density} = R^2_\mathrm{latent}$')

    ax.set_xlabel(r'$R^2$ Density-Space (reconstructed)', fontsize=12)
    ax.set_ylabel(r'$R^2$ Latent-Space', fontsize=12)
    ax.set_title('Density vs Latent R² — Per-Test Run',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, 1.05)
    ax.set_ylim(-0.55, 1.05)

    # Save experiment legend as separate image
    _save_experiment_legend({k: style_map[k] for k in exps_plotted
                             if k in style_map},
                            output_dir, 'thesis_density_vs_latent_scatter_legend')
    fig.tight_layout()
    _save_fig(fig, output_dir, 'thesis_density_vs_latent_scatter')
    print(f"  Saved: thesis_density_vs_latent_scatter.pdf/png")


# ---------------------------------------------------------------------------
# THESIS FIG 3: sPOD vs POD singular value decay (median + IQR)
# ---------------------------------------------------------------------------

def plot_thesis_svd_decay(experiments, output_dir, max_modes=40):
    """Eigenvalue decay + cumulative energy: sPOD vs POD, averaged + IQR."""
    print(f"\n{'=' * 60}")
    print("THESIS FIG 3: sPOD vs POD Eigenvalue Decay (IQR)")
    print("=" * 60)

    # Collect all singular value arrays
    spod_curves, pod_curves = [], []    # normalised (σ/σ₁)²
    spod_cum, pod_cum = [], []          # cumulative energy

    for exp_name, exp_dir in experiments.items():
        sv_a = load_singular_values(exp_dir)
        sv_r = load_unaligned_singular_values(exp_dir)
        if sv_a is not None:
            nm = min(max_modes, len(sv_a))
            norm_a = (sv_a[:nm] / sv_a[0]) ** 2
            cum_a = np.cumsum(sv_a[:nm] ** 2) / np.sum(sv_a ** 2)
            spod_curves.append(norm_a)
            spod_cum.append(cum_a)
        if sv_r is not None:
            nm = min(max_modes, len(sv_r))
            norm_r = (sv_r[:nm] / sv_r[0]) ** 2
            cum_r = np.cumsum(sv_r[:nm] ** 2) / np.sum(sv_r ** 2)
            pod_curves.append(norm_r)
            pod_cum.append(cum_r)

    if not spod_curves and not pod_curves:
        print("  No SVD data found — skipping.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    spod_c, pod_c = THESIS_MVAR_COLOR, THESIS_LSTM_COLOR  # blue vs red

    def _pad_to_matrix(curves, max_modes):
        """Pad curves of different lengths into a (n, max_modes) matrix."""
        matrix = np.full((len(curves), max_modes), np.nan)
        for i, c in enumerate(curves):
            matrix[i, :len(c)] = c
        return matrix

    modes = np.arange(1, max_modes + 1)

    # sPOD
    if spod_curves:
        mat = _pad_to_matrix(spod_curves, max_modes)
        mat_cum = _pad_to_matrix(spod_cum, max_modes)
        # Faint individual
        for row in mat:
            ax1.semilogy(modes, row, '-', color=spod_c, lw=0.8, alpha=0.25)
        for row in mat_cum:
            ax2.plot(modes, row, '-', color=spod_c, lw=0.8, alpha=0.25)
        # Median + IQR
        med = np.nanmedian(mat, axis=0)
        p25, p75 = np.nanpercentile(mat, 25, axis=0), np.nanpercentile(mat, 75, axis=0)
        ax1.semilogy(modes, med, '-', color=spod_c, lw=2.5,
                     label=f'sPOD (n={len(spod_curves)})')
        ax1.fill_between(modes, p25, p75, color=spod_c, alpha=0.2)

        med_c = np.nanmedian(mat_cum, axis=0)
        p25_c, p75_c = np.nanpercentile(mat_cum, 25, axis=0), np.nanpercentile(mat_cum, 75, axis=0)
        ax2.plot(modes, med_c, '-', color=spod_c, lw=2.5,
                 label=f'sPOD (n={len(spod_cum)})')
        ax2.fill_between(modes, p25_c, p75_c, color=spod_c, alpha=0.2)

    # POD (unaligned)
    if pod_curves:
        mat = _pad_to_matrix(pod_curves, max_modes)
        mat_cum = _pad_to_matrix(pod_cum, max_modes)
        for row in mat:
            ax1.semilogy(modes, row, '-', color=pod_c, lw=0.8, alpha=0.25)
        for row in mat_cum:
            ax2.plot(modes, row, '-', color=pod_c, lw=0.8, alpha=0.25)
        med = np.nanmedian(mat, axis=0)
        p25, p75 = np.nanpercentile(mat, 25, axis=0), np.nanpercentile(mat, 75, axis=0)
        ax1.semilogy(modes, med, '-', color=pod_c, lw=2.5,
                     label=f'POD (n={len(pod_curves)})')
        ax1.fill_between(modes, p25, p75, color=pod_c, alpha=0.2)

        med_c = np.nanmedian(mat_cum, axis=0)
        p25_c, p75_c = np.nanpercentile(mat_cum, 25, axis=0), np.nanpercentile(mat_cum, 75, axis=0)
        ax2.plot(modes, med_c, '-', color=pod_c, lw=2.5,
                 label=f'POD (n={len(pod_cum)})')
        ax2.fill_between(modes, p25_c, p75_c, color=pod_c, alpha=0.2)

    ax1.set_xlabel('Mode index $k$')
    ax1.set_ylabel(r'$(\sigma_k / \sigma_1)^2$')
    ax1.set_title('Eigenvalue Decay', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.axhline(0.99, color='gray', ls=':', alpha=0.5, label='99%')
    ax2.axhline(0.999, color='gray', ls='--', alpha=0.5, label='99.9%')
    ax2.set_xlabel('Number of modes $d$')
    ax2.set_ylabel('Cumulative energy ratio')
    ax2.set_title('Cumulative Energy', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.70, 1.005)

    fig.suptitle('sPOD vs POD — Singular Value Spectra',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save_fig(fig, output_dir, 'thesis_svd_decay')
    print(f"  Saved: thesis_svd_decay.pdf/png")


# ---------------------------------------------------------------------------
# THESIS FIG 3b: SVD decay — per-experiment detail with identifiable traces
# ---------------------------------------------------------------------------

def plot_thesis_svd_decay_detail(experiments, output_dir, style_map, max_modes=40):
    """Per-experiment SVD decay: each experiment identifiable via color+marker.

    2-panel figure (eigenvalue decay + cumulative energy).
    sPOD = solid lines, POD = dashed lines, same color per experiment.
    Dots at each mode index for identification.
    """
    print(f"\n{'=' * 60}")
    print("THESIS FIG 3b: SVD Decay Detail (per-experiment)")
    print("=" * 60)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    modes = np.arange(1, max_modes + 1)
    exps_plotted = set()

    for exp_name, exp_dir in sorted(experiments.items()):
        sty = style_map.get(exp_name, {'color': 'gray', 'marker': 'o'})
        sv_a = load_singular_values(exp_dir)
        sv_r = load_unaligned_singular_values(exp_dir)

        if sv_a is not None:
            nm = min(max_modes, len(sv_a))
            norm_a = (sv_a[:nm] / sv_a[0]) ** 2
            cum_a = np.cumsum(sv_a[:nm] ** 2) / np.sum(sv_a ** 2)
            m = modes[:nm]
            ax1.semilogy(m, norm_a, '-', color=sty['color'], lw=1.0, alpha=0.7)
            ax1.scatter(m, norm_a, marker=sty['marker'], color=sty['color'],
                        s=12, alpha=0.7, zorder=3)
            ax2.plot(m, cum_a, '-', color=sty['color'], lw=1.0, alpha=0.7)
            ax2.scatter(m, cum_a, marker=sty['marker'], color=sty['color'],
                        s=12, alpha=0.7, zorder=3)
            exps_plotted.add(exp_name)

        if sv_r is not None:
            nm = min(max_modes, len(sv_r))
            norm_r = (sv_r[:nm] / sv_r[0]) ** 2
            cum_r = np.cumsum(sv_r[:nm] ** 2) / np.sum(sv_r ** 2)
            m = modes[:nm]
            ax1.semilogy(m, norm_r, '--', color=sty['color'], lw=0.8, alpha=0.5)
            ax2.plot(m, cum_r, '--', color=sty['color'], lw=0.8, alpha=0.5)
            exps_plotted.add(exp_name)

    if not exps_plotted:
        plt.close(fig)
        print("  No SVD data — skipping detail.")
        return

    # Style guide entries
    ax1.plot([], [], '-', color='gray', lw=1.5, label='sPOD (solid)')
    ax1.plot([], [], '--', color='gray', lw=1.0, label='POD (dashed)')

    ax1.set_xlabel('Mode index $k$')
    ax1.set_ylabel(r'$(\sigma_k / \sigma_1)^2$')
    ax1.set_title('Eigenvalue Decay', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.axhline(0.99, color='gray', ls=':', alpha=0.5, label='99%')
    ax2.axhline(0.999, color='gray', ls='--', alpha=0.5, label='99.9%')
    ax2.set_xlabel('Number of modes $d$')
    ax2.set_ylabel('Cumulative energy ratio')
    ax2.set_title('Cumulative Energy', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.70, 1.005)

    fig.suptitle('sPOD vs POD — Per-Experiment Singular Value Spectra',
                 fontsize=14, fontweight='bold')

    # Save experiment legend as separate image
    _save_experiment_legend({k: style_map[k] for k in exps_plotted
                             if k in style_map},
                            output_dir, 'thesis_svd_decay_detail_legend')
    fig.tight_layout()
    _save_fig(fig, output_dir, 'thesis_svd_decay_detail')
    print(f"  Saved: thesis_svd_decay_detail.pdf/png")


# ---------------------------------------------------------------------------
# THESIS FIG 4: KDE alignment snapshots (from pre-extracted data)
# ---------------------------------------------------------------------------

def plot_thesis_kde_snapshots(experiments, output_dir, data_dir):
    """KDE density snapshots at t=0.6, 3, 25, 75, 200.

    Layout per experiment:
      Col 0:  True density + red particles
      Col 1..N: One column per available model (MVAR, LSTM, WSINDy) — prediction
      Rightmost: Spatial order parameter time-series (all models overlaid)

    Uses pre-extracted kde_snapshots.npz (from Oscar) if available,
    otherwise falls back to loading full density + trajectory files.
    WSINDy column only shown when data exists for that experiment.
    """
    import yaml

    print(f"\n{'=' * 60}")
    print("THESIS FIG 4: KDE Alignment Snapshots (restructured)")
    print("=" * 60)

    TARGET_TIMES = [0.6, 3.0, 25.0, 75.0, 200.0]
    base_kde_dir = Path('kde_snapshots')
    n_generated = 0

    for exp_name, exp_dir in experiments.items():
        exp_kde_dir = base_kde_dir / exp_name
        exp_kde_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = Path(exp_dir) / 'kde_snapshots.npz'
        test_run = Path(exp_dir) / 'test' / 'test_000'

        # Determine domain bounds
        cfg_path = Path(exp_dir) / 'config_used.yaml'
        if not cfg_path.exists():
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        Lx = cfg['sim']['Lx']
        Ly = cfg['sim']['Ly']

        if snapshot_path.exists():
            snap = np.load(snapshot_path, allow_pickle=True)
            actual_times = snap['times_actual']
            particles = {float(f'{t:.1f}'): snap[f'particles_t{t:.1f}']
                         for t in actual_times if f'particles_t{t:.1f}' in snap}
            rho_true_frames = {float(f'{t:.1f}'): snap[f'rho_true_t{t:.1f}']
                               for t in actual_times if f'rho_true_t{t:.1f}' in snap}
            rho_pred = {}
            for model in THESIS_MODELS:
                rho_pred[model] = {}
                for t in actual_times:
                    key = f'rho_pred_{model}_t{t:.1f}'
                    if key in snap:
                        rho_pred[model][float(f'{t:.1f}')] = snap[key]
        else:
            traj_path = test_run / 'trajectory.npz'
            true_path = test_run / 'density_true.npz'
            if not traj_path.exists() or not true_path.exists():
                continue

            traj_data = np.load(traj_path)
            dens_true = np.load(true_path)
            x_traj = traj_data['traj']
            times_traj = traj_data['times']
            rho_true_all = dens_true['rho']
            times_dens = dens_true['times']

            particles = {}
            rho_true_frames = {}
            actual_times = []
            for target_t in TARGET_TIMES:
                traj_idx = np.argmin(np.abs(times_traj - target_t))
                dens_idx = np.argmin(np.abs(times_dens - target_t))
                actual_t = float(f'{times_dens[dens_idx]:.1f}')
                actual_times.append(actual_t)
                particles[actual_t] = x_traj[traj_idx]
                rho_true_frames[actual_t] = rho_true_all[dens_idx]

            rho_pred = {}
            for model in THESIS_MODELS:
                rho_pred[model] = {}
                pred_path = test_run / f'density_pred_{model}.npz'
                if not pred_path.exists():
                    pred_path = test_run / 'density_pred.npz'
                if not pred_path.exists():
                    continue
                dp = np.load(pred_path)
                rho_p = dp['rho']
                times_p = dp['times']
                for actual_t in actual_times:
                    pidx = np.argmin(np.abs(times_p - actual_t))
                    rho_pred[model][actual_t] = rho_p[pidx]

        # Determine which models have prediction data for this experiment
        active_models = [m for m in THESIS_MODELS
                         if m in rho_pred and rho_pred[m]]
        if not active_models:
            continue

        # Load spatial order data if available
        so_path = Path(exp_dir) / 'spatial_order.npz'
        has_spatial = so_path.exists()
        so_data = np.load(so_path, allow_pickle=True) if has_spatial else None

        # Column layout: True+particles | model_1 | model_2 | ... | SpatialOrder
        n_times = len(actual_times)
        n_model_cols = len(active_models)
        n_cols = 1 + n_model_cols + (1 if has_spatial else 0)
        fig, axes = plt.subplots(n_times, n_cols,
                                 figsize=(5.5 * n_cols, 5 * n_times))
        if n_times == 1:
            axes = axes[np.newaxis, :]
        if n_cols == 1:
            axes = axes[:, np.newaxis]

        for row, t in enumerate(actual_times):
            t_key = float(f'{t:.1f}') if isinstance(t, (float, np.floating)) else t
            pos = particles.get(t_key)
            rho_t = rho_true_frames.get(t_key)

            # Col 0: True density + particles
            ax = axes[row, 0]
            if rho_t is not None:
                vmax = max(rho_t.max(), 1e-6)
                im = ax.imshow(rho_t, origin='lower',
                               extent=[0, Lx, 0, Ly],
                               cmap='viridis', vmin=0, vmax=vmax)
                plt.colorbar(im, ax=ax, fraction=0.046)
            if pos is not None:
                ax.scatter(pos[:, 0], pos[:, 1], s=6, c='red', alpha=0.7,
                           edgecolors='darkred', linewidths=0.3)
            ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
            ax.set_aspect('equal')
            ax.set_xlabel('x'); ax.set_ylabel('y')
            ax.set_title(f'True + particles (t={t_key:.1f}s)')

            # Model prediction columns
            for mi, model in enumerate(active_models):
                ax = axes[row, 1 + mi]
                rho_p = rho_pred[model].get(t_key)
                if rho_p is not None:
                    vmax_p = max(rho_p.max(), 1e-6)
                    im = ax.imshow(rho_p, origin='lower',
                                   extent=[0, Lx, 0, Ly],
                                   cmap='viridis', vmin=0, vmax=vmax_p)
                    plt.colorbar(im, ax=ax, fraction=0.046)
                ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
                ax.set_aspect('equal')
                ax.set_xlabel('x'); ax.set_ylabel('y')
                ax.set_title(f'{THESIS_LABELS[model]} (t={t_key:.1f}s)')

            # Spatial order column (rightmost) — same plot per row but
            # highlight current time with a vertical line
            if has_spatial:
                ax = axes[row, -1]
                so_times = so_data['times'] if 'times' in so_data else None
                if so_times is not None:
                    # True spatial order uses full time axis (from t=0)
                    t_true = so_data['times_true'] if 'times_true' in so_data else so_times
                    ax.plot(t_true, so_data['so_true'], '-', color='black',
                            lw=1.5, alpha=0.8, label='True')
                    for m in active_models:
                        key = f'so_pred_{m}'
                        if key in so_data:
                            ax.plot(so_times, so_data[key],
                                    MODEL_LINESTYLES.get(m, '-'),
                                    color=THESIS_COLORS[m], lw=1.2,
                                    alpha=0.8, label=THESIS_LABELS[m])
                    ax.axvline(t_key, color='red', ls=':', lw=1, alpha=0.6)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel(r'Spatial order (std $\rho$)')
                ax.set_title(f'Spatial order (t={t_key:.1f}s)')
                if row == 0:
                    ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)

        fig.suptitle(f'{exp_name} — KDE Alignment\n'
                     f'Red dots = particle positions',
                     fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        _save_fig(fig, exp_kde_dir, f'kde_{exp_name}')
        n_generated += 1

    print(f"  Generated {n_generated} KDE alignment figures")


# ---------------------------------------------------------------------------
# THESIS FIG 5: Mass conservation — model-level aggregate
# ---------------------------------------------------------------------------

def plot_thesis_mass_conservation(experiments, output_dir, data_dir):
    """Aggregate mass conservation: all experiments overlaid per model.

    Uses pre-extracted mass_timeseries.npz if available, otherwise loads
    full density files (expensive).  Dynamic panels: only models with data.
    """
    print(f"\n{'=' * 60}")
    print("THESIS FIG 5: Mass Conservation (model-level aggregate)")
    print("=" * 60)

    mass_data = {m: [] for m in THESIS_MODELS}

    for exp_name, exp_dir in experiments.items():
        mass_path = Path(exp_dir) / 'mass_timeseries.npz'
        if mass_path.exists():
            mdata = np.load(mass_path)
            times = mdata['times']
            m_true = mdata['mass_true']
            for model in THESIS_MODELS:
                key = f'mass_pred_{model}'
                if key in mdata:
                    m_pred = mdata[key]
                    M0 = m_true[0] if m_true[0] != 0 else 1.0
                    mass_data[model].append((times, m_true / M0, m_pred / M0,
                                             exp_name))
            continue

        for model in THESIS_MODELS:
            rho_t, rho_p, times = load_density_pair(exp_dir, test_idx=0,
                                                     model=model)
            if rho_t is None:
                continue
            m_true = np.array([np.sum(rho_t[t]) for t in range(rho_t.shape[0])])
            m_pred = np.array([np.sum(rho_p[t]) for t in range(rho_p.shape[0])])
            M0 = m_true[0] if m_true[0] != 0 else 1.0
            mass_data[model].append((times, m_true / M0, m_pred / M0, exp_name))

    # Only create panels for models with data
    active_models = [m for m in THESIS_MODELS if mass_data[m]]
    if not active_models:
        print("  No mass data found — skipping.")
        return

    n_panels = len(active_models)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5),
                             squeeze=False)

    for mi, model in enumerate(active_models):
        ax = axes[0, mi]
        traces = mass_data[model]
        color = THESIS_COLORS[model]

        for times, m_true_ratio, m_pred_ratio, _ in traces:
            ax.plot(times, m_pred_ratio, '-', color=color,
                    linewidth=0.5, alpha=0.15)

        all_times = np.unique(np.concatenate([t[0] for t in traces]))
        pred_matrix = []
        for times, _, m_pred_ratio, _ in traces:
            interp = np.interp(all_times, times, m_pred_ratio,
                               left=np.nan, right=np.nan)
            pred_matrix.append(interp)
        pred_matrix = np.array(pred_matrix)

        with np.errstate(all='ignore'):
            med = np.nanmedian(pred_matrix, axis=0)
            p25 = np.nanpercentile(pred_matrix, 25, axis=0)
            p75 = np.nanpercentile(pred_matrix, 75, axis=0)

        ax.fill_between(all_times, p25, p75, color=color, alpha=0.25,
                        label='IQR (25–75%)')
        ax.plot(all_times, med, '-', color=color, linewidth=2.5,
                label='Median')
        # Ground truth: constant at 1.0
        ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5,
                    alpha=0.7, label='True mass')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel(r'Relative Mass $M(t)/M_0$')
        ax.set_title(THESIS_LABELS[model], fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.annotate(f'{len(traces)} experiments',
                    xy=(0.98, 0.02), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=8, color='gray')

    fig.suptitle('Mass Conservation Over Forecast Horizon',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save_fig(fig, output_dir, 'thesis_mass_conservation')
    print(f"  Saved: thesis_mass_conservation.pdf/png")


# ---------------------------------------------------------------------------
# THESIS: Mass conservation table (replaces plot in thesis flow)
# ---------------------------------------------------------------------------

def generate_thesis_mass_table(experiments, output_dir, data_dir):
    """Per-experiment mean mass deviation table: MVAR vs LSTM.

    Reads mass_timeseries.npz when available, otherwise falls back to
    loading full density files.  Outputs mass_table.csv and mass_table.tex.
    """
    print(f"\n{'=' * 60}")
    print("THESIS: Mass Conservation Table")
    print("=" * 60)

    rows = []
    for exp_name, exp_dir in sorted(experiments.items()):
        row = {'Experiment': _short(exp_name)}
        mass_path = Path(exp_dir) / 'mass_timeseries.npz'
        if mass_path.exists():
            mdata = np.load(mass_path)
            m_true = mdata['mass_true']
            M0 = m_true[0] if m_true[0] != 0 else 1.0
            for model in THESIS_MODELS:
                key = f'mass_pred_{model}'
                if key in mdata:
                    rel = mdata[key] / M0
                    row[f'{THESIS_LABELS[model]} Δmass'] = float(
                        np.mean(np.abs(rel - 1.0)))
        else:
            for model in THESIS_MODELS:
                rho_t, rho_p, times = load_density_pair(
                    exp_dir, test_idx=0, model=model)
                if rho_t is None:
                    continue
                m_true = np.array([np.sum(rho_t[t])
                                   for t in range(rho_t.shape[0])])
                m_pred = np.array([np.sum(rho_p[t])
                                   for t in range(rho_p.shape[0])])
                M0 = m_true[0] if m_true[0] != 0 else 1.0
                rel = m_pred / M0
                row[f'{THESIS_LABELS[model]} Δmass'] = float(
                    np.mean(np.abs(rel - 1.0)))
        rows.append(row)

    if not rows:
        print("  No mass data found — skipping.")
        return

    table = pd.DataFrame(rows)
    csv_path = output_dir / 'mass_table.csv'
    table.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"  Saved: mass_table.csv ({len(table)} experiments)")

    tex_path = output_dir / 'mass_table.tex'
    with open(tex_path, 'w') as f:
        f.write(table.to_latex(index=False, escape=False,
                               float_format='%.4f', na_rep='---'))
    print(f"  Saved: mass_table.tex")


# ---------------------------------------------------------------------------
# THESIS FIG 7: Phase dynamics — aggregate with per-experiment styling
# ---------------------------------------------------------------------------

def plot_thesis_phase_dynamics(experiments, output_dir, style_map):
    """Phase dynamics aggregate: 3-panel (trajectory | AR | autocorrelation).

    Adaptation of plot_phase_dynamics() with consistent per-experiment
    color+marker from style_map.
    """
    from scipy import signal
    from numpy.linalg import lstsq

    print(f"\n{'=' * 60}")
    print("THESIS FIG 7: Phase Dynamics (aggregate, per-experiment styled)")
    print("=" * 60)

    exp_list = [(name, exp_dir) for name, exp_dir in sorted(experiments.items())]
    shift_exps = []
    for name, exp_dir in exp_list:
        sa = load_shift_data(exp_dir)
        if sa is not None:
            shift_exps.append((name, exp_dir, sa))
    if not shift_exps:
        print("  No shift data found — skipping phase dynamics.")
        return

    fig, (ax_traj, ax_ar, ax_ac) = plt.subplots(1, 3, figsize=(16, 5))
    ar_orders = [1, 2, 3, 5, 10]

    for idx, (exp_name, exp_dir, sa) in enumerate(shift_exps):
        sty = style_map.get(exp_name, {'color': _color(idx, len(shift_exps)),
                                        'marker': 'o'})
        shifts = sa['shifts']
        meta_path = Path(exp_dir) / 'train' / 'metadata.json'
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                M = len(meta) if isinstance(meta, list) else 112
            except Exception:
                M = 112
        else:
            M = 112
        T_s = shifts.shape[0] // M if M > 0 else shifts.shape[0]
        if T_s < 4:
            continue
        if shifts.ndim == 2:
            shifts_3d = shifts[:M * T_s].reshape(M, T_s, 2)
        else:
            shifts_3d = shifts
            M, T_s = shifts_3d.shape[:2]

        color = sty['color']
        marker = sty['marker']

        # 1. Shift trajectory (mean over realizations)
        mean_s = shifts_3d.mean(axis=0)
        ax_traj.plot(mean_s[:, 1], mean_s[:, 0], '-', color=color,
                     linewidth=1.5, alpha=0.8)
        ax_traj.scatter(mean_s[0, 1], mean_s[0, 0], marker=marker,
                        color=color, s=30, zorder=4)

        # 2. AR fit
        ar_r2_p = []
        for p in ar_orders:
            r2s = []
            for dim in [0, 1]:
                Xl, yl = [], []
                for ri in range(M):
                    s = shifts_3d[ri, :, dim]
                    for t in range(p, T_s):
                        Xl.append(s[t - p:t][::-1])
                        yl.append(s[t])
                Xa = np.array(Xl)
                ya = np.array(yl)
                coef, _, _, _ = lstsq(Xa, ya, rcond=None)
                yp = Xa @ coef
                ss_res = np.sum((ya - yp) ** 2)
                ss_tot = np.sum((ya - ya.mean()) ** 2)
                r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else 0)
            ar_r2_p.append(np.mean(r2s))
        ax_ar.plot(ar_orders, ar_r2_p, '-', color=color, linewidth=1.2,
                   alpha=0.8, label=_short(exp_name))
        ax_ar.scatter(ar_orders, ar_r2_p, marker=marker, color=color, s=25,
                      zorder=3)

        # 3. Autocorrelation
        max_lag = min(T_s - 1, 50)
        lags = np.arange(max_lag)
        acorr = np.zeros(max_lag)
        count = 0
        for dim in [0, 1]:
            for ri in range(min(M, 30)):
                s = shifts_3d[ri, :, dim]
                s = s - s.mean()
                var_s = np.var(s)
                if var_s < 1e-12:
                    continue
                for lag in range(max_lag):
                    if lag < len(s):
                        acorr[lag] += np.mean(s[:len(s) - lag] * s[lag:]) / var_s
                count += 1
        if count > 0:
            acorr /= count
        ax_ac.plot(lags, acorr, '-', color=color, linewidth=1, alpha=0.7,
                   label=_short(exp_name))

    ax_traj.set_xlabel(r'Shift $\Delta y$')
    ax_traj.set_ylabel(r'Shift $\Delta x$')
    ax_traj.set_title('Mean Shift Trajectory')
    ax_traj.grid(True, alpha=0.3)

    ax_ar.set_xlabel('AR order $p$')
    ax_ar.set_ylabel(r'$R^2$')
    ax_ar.set_title('AR Predictability')
    ax_ar.legend(fontsize=6, ncol=2, frameon=True)
    ax_ar.grid(True, alpha=0.3)

    ax_ac.set_xlabel('Lag')
    ax_ac.set_ylabel('Autocorrelation')
    ax_ac.set_title('Shift Autocorrelation')
    ax_ac.axhline(0, color='gray', ls='--', lw=0.8)
    ax_ac.grid(True, alpha=0.3)

    fig.suptitle('Phase Dynamics — Shift Analysis',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, output_dir, 'thesis_phase_dynamics')
    print(f"  Saved: thesis_phase_dynamics.pdf/png")


# ---------------------------------------------------------------------------
# THESIS FIG 7b: Phase dynamics — per-experiment detail
# ---------------------------------------------------------------------------

def plot_thesis_phase_dynamics_detail(experiments, output_dir, style_map):
    """One 6-panel phase dynamics figure per experiment."""
    from scipy import signal
    from numpy.linalg import lstsq

    print(f"\n{'=' * 60}")
    print("THESIS FIG 7b: Phase Dynamics Detail (per-experiment)")
    print("=" * 60)

    n_generated = 0
    phase_dir = output_dir / 'phase_detail'
    phase_dir.mkdir(parents=True, exist_ok=True)

    for exp_name, exp_dir in sorted(experiments.items()):
        sa = load_shift_data(exp_dir)
        if sa is None:
            continue
        sty = style_map.get(exp_name, {'color': 'steelblue', 'marker': 'o'})

        shifts = sa['shifts']
        meta_path = Path(exp_dir) / 'train' / 'metadata.json'
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                M = len(meta) if isinstance(meta, list) else 112
            except Exception:
                M = 112
        else:
            M = 112
        T_s = shifts.shape[0] // M if M > 0 else shifts.shape[0]
        if T_s < 4:
            continue
        if shifts.ndim == 2:
            shifts_3d = shifts[:M * T_s].reshape(M, T_s, 2)
        else:
            shifts_3d = shifts
            M, T_s = shifts_3d.shape[:2]

        color = sty['color']

        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
        ax_traj = fig.add_subplot(gs[0, 0])
        ax_psd = fig.add_subplot(gs[0, 1])
        ax_ar = fig.add_subplot(gs[0, 2])
        ax_var = fig.add_subplot(gs[1, 0])
        ax_ac = fig.add_subplot(gs[1, 1])
        ax_info = fig.add_subplot(gs[1, 2])

        # 1. Individual shift trajectories (faint) + mean (bold)
        for ri in range(min(M, 30)):
            ax_traj.plot(shifts_3d[ri, :, 1], shifts_3d[ri, :, 0],
                         '-', color=color, lw=0.3, alpha=0.15)
        mean_s = shifts_3d.mean(axis=0)
        ax_traj.plot(mean_s[:, 1], mean_s[:, 0], '-', color=color, lw=2)
        ax_traj.scatter(mean_s[0, 1], mean_s[0, 0], marker='o', color='red',
                        s=40, zorder=5, label='Start')
        ax_traj.set_xlabel(r'Shift $\Delta y$')
        ax_traj.set_ylabel(r'Shift $\Delta x$')
        ax_traj.set_title('Shift Trajectories')
        ax_traj.legend(fontsize=7)
        ax_traj.grid(True, alpha=0.3)

        # 2. PSD (x and y dims)
        for dim, ls, dlabel in [(0, '--', 'x'), (1, '-', 'y')]:
            psds = []
            for ri in range(min(M, 50)):
                f_psd, psd = signal.welch(shifts_3d[ri, :, dim], fs=1.0,
                                          nperseg=min(128, T_s // 2))
                psds.append(psd)
            mean_psd = np.mean(psds, axis=0)
            ax_psd.semilogy(f_psd, mean_psd, ls, color=color, lw=1.2,
                            label=f'dim-{dlabel}')
        ax_psd.set_xlabel('Frequency')
        ax_psd.set_ylabel('PSD')
        ax_psd.set_title('Power Spectral Density')
        ax_psd.legend(fontsize=7)
        ax_psd.grid(True, alpha=0.3)

        # 3. AR fit R²
        ar_orders = [1, 2, 3, 5, 10]
        ar_r2_p = []
        for p in ar_orders:
            r2s = []
            for dim in [0, 1]:
                Xl, yl = [], []
                for ri in range(M):
                    s = shifts_3d[ri, :, dim]
                    for t in range(p, T_s):
                        Xl.append(s[t - p:t][::-1])
                        yl.append(s[t])
                Xa = np.array(Xl)
                ya = np.array(yl)
                coef, _, _, _ = lstsq(Xa, ya, rcond=None)
                yp = Xa @ coef
                ss_res = np.sum((ya - yp) ** 2)
                ss_tot = np.sum((ya - ya.mean()) ** 2)
                r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else 0)
            ar_r2_p.append(np.mean(r2s))
        ax_ar.plot(ar_orders, ar_r2_p, 'o-', color=color, lw=1.5, markersize=6)
        ax_ar.set_xlabel('AR order $p$')
        ax_ar.set_ylabel(r'$R^2$')
        ax_ar.set_title('AR Predictability')
        ax_ar.grid(True, alpha=0.3)

        # 4. Per-realization variance histogram
        vars_per_real = [np.var(shifts_3d[ri]) for ri in range(M)]
        ax_var.hist(vars_per_real, bins=20, color=color, alpha=0.7,
                    edgecolor='black', linewidth=0.5)
        ax_var.set_xlabel('Shift Variance')
        ax_var.set_ylabel('Count')
        ax_var.set_title(f'Variance Distribution (M={M})')
        ax_var.grid(True, alpha=0.3, axis='y')

        # 5. Autocorrelation
        max_lag = min(T_s - 1, 50)
        lags = np.arange(max_lag)
        acorr = np.zeros(max_lag)
        count = 0
        for dim in [0, 1]:
            for ri in range(min(M, 30)):
                s = shifts_3d[ri, :, dim]
                s = s - s.mean()
                var_s = np.var(s)
                if var_s < 1e-12:
                    continue
                for lag in range(max_lag):
                    if lag < len(s):
                        acorr[lag] += np.mean(s[:len(s) - lag] * s[lag:]) / var_s
                count += 1
        if count > 0:
            acorr /= count
        ax_ac.plot(lags, acorr, '-', color=color, lw=1.5)
        ax_ac.axhline(0, color='gray', ls='--', lw=0.8)
        ax_ac.set_xlabel('Lag')
        ax_ac.set_ylabel('Autocorrelation')
        ax_ac.set_title('Shift Autocorrelation')
        ax_ac.grid(True, alpha=0.3)

        # 6. Info panel
        ax_info.axis('off')
        info_text = (
            f"Experiment: {exp_name}\n"
            f"Realizations: {M}\n"
            f"Time steps: {T_s}\n"
            f"Mean variance: {np.mean(vars_per_real):.4f}\n"
            f"AR(5) R²: {dict(zip(ar_orders, ar_r2_p)).get(5, 0):.3f}\n"
            f"AR(10) R²: {dict(zip(ar_orders, ar_r2_p)).get(10, 0):.3f}"
        )
        ax_info.text(0.1, 0.5, info_text, transform=ax_info.transAxes,
                     fontsize=10, verticalalignment='center',
                     fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.suptitle(f'{exp_name} — Phase Dynamics',
                     fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        _save_fig(fig, phase_dir, f'phase_{exp_name}')
        n_generated += 1

    print(f"  Generated {n_generated} phase dynamics detail figures")


# ---------------------------------------------------------------------------
# THESIS FIG 8: L^p error time-series
# ---------------------------------------------------------------------------

def plot_thesis_lp_errors(experiments, output_dir, style_map):
    """L^p error time-series: one figure per model.

    Each figure has 3 line types (L1, L2, L∞) on same axes.
    Per-experiment traces (mean of 4 tests) with faint ±1σ shading.
    Global mean + IQR band overlaid bold.
    Loads from lp_errors.npz (Oscar extraction) or falls back to computing
    from full density files.
    """
    print(f"\n{'=' * 60}")
    print("THESIS FIG 8: L^p Error Time-Series")
    print("=" * 60)

    error_norms = {'rel_e1': 'L¹', 'rel_e2': 'L²', 'rel_einf': 'L∞'}
    norm_styles = {'rel_e1': '-', 'rel_e2': '--', 'rel_einf': ':'}

    for model in THESIS_MODELS:
        model_color = THESIS_COLORS[model]

        # Collect per-experiment error traces: {exp_name: {norm: [(times, vals)]}}
        exp_errors = {}

        for exp_name, exp_dir in sorted(experiments.items()):
            lp_path = Path(exp_dir) / 'lp_errors.npz'
            if lp_path.exists():
                lpdata = np.load(lp_path, allow_pickle=True)
                # Try multi-test keys first: rel_e1_mvar_test000, etc.
                test_traces = {}
                for norm_key in error_norms:
                    test_traces[norm_key] = []
                    # Multi-test format
                    for ti in range(20):  # up to 20 test runs
                        tkey = f'{norm_key}_{model}_test{ti:03d}'
                        timeskey = f'times_{model}_test{ti:03d}'
                        if tkey in lpdata:
                            t_arr = lpdata[timeskey] if timeskey in lpdata else lpdata.get('times', None)
                            if t_arr is not None:
                                test_traces[norm_key].append((t_arr, lpdata[tkey]))
                    # Single-test fallback
                    if not test_traces[norm_key]:
                        skey = f'{norm_key}_{model}'
                        if skey in lpdata and 'times' in lpdata:
                            test_traces[norm_key].append(
                                (lpdata['times'], lpdata[skey]))
                if any(test_traces[k] for k in error_norms):
                    exp_errors[exp_name] = test_traces
                continue

            # Fallback: compute from full density files
            test_dir = Path(exp_dir) / 'test'
            if not test_dir.exists():
                continue
            test_traces = {k: [] for k in error_norms}
            for run_dir in sorted(test_dir.iterdir()):
                if not run_dir.is_dir() or not run_dir.name.startswith('test_'):
                    continue
                rho_t, rho_p, times = load_density_pair(exp_dir,
                    test_idx=int(run_dir.name.split('_')[1]), model=model)
                if rho_t is None:
                    continue
                errs = compute_error_timeseries(rho_t, rho_p)
                for norm_key in error_norms:
                    test_traces[norm_key].append((times, errs[norm_key]))
            if any(test_traces[k] for k in error_norms):
                exp_errors[exp_name] = test_traces

        if not exp_errors:
            continue

        fig, ax = plt.subplots(figsize=(12, 7))

        # Global traces per norm for IQR computation
        global_per_norm = {k: [] for k in error_norms}

        for exp_name, test_traces in sorted(exp_errors.items()):
            sty = style_map.get(exp_name, {'color': 'gray', 'marker': 'o'})

            for norm_key, ls in norm_styles.items():
                traces = test_traces[norm_key]
                if not traces:
                    continue
                # Average over test runs
                all_t = np.unique(np.concatenate([t[0] for t in traces]))
                mat = []
                for t_arr, vals in traces:
                    mat.append(np.interp(all_t, t_arr, vals,
                                         left=np.nan, right=np.nan))
                mat = np.array(mat)
                with np.errstate(all='ignore'):
                    mean_err = np.nanmean(mat, axis=0)
                    std_err = np.nanstd(mat, axis=0)

                ax.plot(all_t, mean_err, ls, color=sty['color'],
                        linewidth=0.8, alpha=0.6)
                ax.fill_between(all_t,
                                np.maximum(mean_err - std_err, 1e-8),
                                mean_err + std_err,
                                color=sty['color'], alpha=0.05)

                global_per_norm[norm_key].append((all_t, mean_err))

        # Global mean + IQR per norm
        for norm_key, norm_label in error_norms.items():
            traces = global_per_norm[norm_key]
            if not traces:
                continue
            all_t_global = np.unique(np.concatenate([t[0] for t in traces]))
            mat_g = []
            for t_arr, vals in traces:
                mat_g.append(np.interp(all_t_global, t_arr, vals,
                                       left=np.nan, right=np.nan))
            mat_g = np.array(mat_g)
            with np.errstate(all='ignore'):
                gmed = np.nanmedian(mat_g, axis=0)
                gp25 = np.nanpercentile(mat_g, 25, axis=0)
                gp75 = np.nanpercentile(mat_g, 75, axis=0)
            ls = norm_styles[norm_key]
            ax.plot(all_t_global, gmed, ls, color=model_color, lw=2.5,
                    label=f'{norm_label} median')
            ax.fill_between(all_t_global, gp25, gp75, color=model_color,
                            alpha=0.15)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Relative Error')
        ax.set_title(f'{THESIS_LABELS[model]} — Relative $L^p$ Errors',
                     fontsize=13, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')

        # Save experiment legend as separate image
        plotted = {k: style_map[k] for k in exp_errors if k in style_map}
        _save_experiment_legend(plotted, output_dir, f'thesis_lp_errors_{model}_legend')
        fig.tight_layout()
        _save_fig(fig, output_dir, f'thesis_lp_errors_{model}')
        print(f"  Saved: thesis_lp_errors_{model}.pdf/png")


# ---------------------------------------------------------------------------
# THESIS FIG 9: Runtime comparison — training, inference, complexity, cost
# ---------------------------------------------------------------------------

def plot_thesis_runtime(experiments, output_dir):
    """Comprehensive runtime comparison across all 3 models.

    Single figure with 4 panels:
      (a) Training time (log-scale bar + strip)
      (b) Inference throughput (steps/sec, log-scale)
      (c) Model complexity (parameter count, log-scale)
      (d) Cost-effectiveness: R² vs training time scatter
    """
    print(f"\n{'=' * 60}")
    print("THESIS FIG 9: Runtime Comparison (Training / Inference / Complexity)")
    print("=" * 60)

    # Collect data from summary.json runtime_analysis profiles
    records = []
    for exp_name, exp_dir in experiments.items():
        summary_path = Path(exp_dir) / 'summary.json'
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            summary = json.load(f)

        profiles = summary.get('runtime_analysis', {}).get('profiles', [])
        for prof in profiles:
            model_name = prof.get('model_name', '').lower()
            if model_name not in ('mvar', 'lstm'):
                continue

            train_s = prof.get('training', {}).get('total_seconds', np.nan)
            infer_step = prof.get('inference', {}).get(
                'single_step', {}).get('mean_seconds', np.nan)
            infer_traj = prof.get('inference', {}).get(
                'full_trajectory', {}).get('mean_seconds', np.nan)
            throughput = prof.get('throughput', {}).get(
                'steps_per_second', np.nan)
            params = prof.get('memory', {}).get('model_parameters', np.nan)

            # Get R² from summary top-level model section
            model_summary = summary.get(model_name, {})
            r2 = model_summary.get('mean_r2_test', np.nan)

            records.append({
                'experiment': _short(exp_name),
                'model': model_name,
                'training_s': train_s,
                'inference_step_s': infer_step,
                'inference_traj_s': infer_traj,
                'throughput_sps': throughput,
                'parameters': params,
                'r2': r2,
            })

    # Fallback: if no runtime_analysis, use summary-level training_time_s
    if not records:
        for exp_name, exp_dir in experiments.items():
            summary_path = Path(exp_dir) / 'summary.json'
            if not summary_path.exists():
                continue
            with open(summary_path) as f:
                summary = json.load(f)
            for model_name in THESIS_MODELS:
                model_summary = summary.get(model_name, {})
                train_s = model_summary.get('training_time_s', np.nan)
                r2 = model_summary.get('mean_r2_test', np.nan)
                if np.isnan(train_s) and np.isnan(r2):
                    continue
                records.append({
                    'experiment': _short(exp_name),
                    'model': model_name,
                    'training_s': train_s,
                    'inference_step_s': np.nan,
                    'inference_traj_s': np.nan,
                    'throughput_sps': np.nan,
                    'parameters': np.nan,
                    'r2': r2,
                })

    if not records:
        print("  No runtime data found — skipping.")
        return

    df = pd.DataFrame(records)

    # Determine which panels have data
    has_inference = df['throughput_sps'].notna().any()
    has_params = df['parameters'].notna().any()
    has_r2 = df['r2'].notna().any()

    # Adaptive layout: 2×2 if all data, fewer panels otherwise
    n_panels = 1 + int(has_inference) + int(has_params) + int(has_r2)
    if n_panels >= 3:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
        if n_panels == 1:
            axes = [axes]

    panel_idx = 0
    active_models = [m for m in THESIS_MODELS if m in df['model'].values]

    # ── Panel (a): Training Time ──
    ax = axes[panel_idx]
    panel_idx += 1
    train_data = df.dropna(subset=['training_s'])
    if not train_data.empty:
        model_means = train_data.groupby('model')['training_s'].agg(['median', 'mean', 'std'])
        positions = []
        labels = []
        for i, model in enumerate(active_models):
            if model not in model_means.index:
                continue
            vals = train_data[train_data['model'] == model]['training_s'].values
            color = THESIS_COLORS[model]
            # Box plot
            bp = ax.boxplot([vals], positions=[i], widths=0.5, patch_artist=True,
                           boxprops=dict(facecolor=color, alpha=0.3),
                           medianprops=dict(color=color, linewidth=2),
                           whiskerprops=dict(color=color),
                           capprops=dict(color=color),
                           flierprops=dict(marker='o', markerfacecolor=color,
                                          markersize=3, alpha=0.5))
            # Strip (jittered points)
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                      color=color, s=15, alpha=0.5, zorder=5,
                      edgecolors='white', linewidths=0.3)
            positions.append(i)
            labels.append(THESIS_LABELS[model])
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_yscale('log')
        ax.set_ylabel('Training Time (seconds)')
        ax.set_title('(a) Training Time', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both', axis='y')

    # ── Panel (b): Inference Throughput ──
    if has_inference:
        ax = axes[panel_idx]
        panel_idx += 1
        infer_data = df.dropna(subset=['throughput_sps'])
        if not infer_data.empty:
            positions = []
            labels = []
            for i, model in enumerate(active_models):
                vals = infer_data[infer_data['model'] == model]['throughput_sps'].values
                if len(vals) == 0:
                    continue
                color = THESIS_COLORS[model]
                bp = ax.boxplot([vals], positions=[i], widths=0.5, patch_artist=True,
                               boxprops=dict(facecolor=color, alpha=0.3),
                               medianprops=dict(color=color, linewidth=2),
                               whiskerprops=dict(color=color),
                               capprops=dict(color=color),
                               flierprops=dict(marker='o', markerfacecolor=color,
                                              markersize=3, alpha=0.5))
                jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
                ax.scatter(np.full(len(vals), i) + jitter, vals,
                          color=color, s=15, alpha=0.5, zorder=5,
                          edgecolors='white', linewidths=0.3)
                positions.append(i)
                labels.append(THESIS_LABELS[model])
            ax.set_xticks(positions)
            ax.set_xticklabels(labels)
            ax.set_yscale('log')
            ax.set_ylabel('Steps / second')
            ax.set_title('(b) Inference Throughput', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, which='both', axis='y')

    # ── Panel (c): Model Complexity ──
    if has_params:
        ax = axes[panel_idx]
        panel_idx += 1
        param_data = df.dropna(subset=['parameters'])
        if not param_data.empty:
            positions = []
            labels = []
            for i, model in enumerate(active_models):
                vals = param_data[param_data['model'] == model]['parameters'].values
                if len(vals) == 0:
                    continue
                color = THESIS_COLORS[model]
                bp = ax.boxplot([vals], positions=[i], widths=0.5, patch_artist=True,
                               boxprops=dict(facecolor=color, alpha=0.3),
                               medianprops=dict(color=color, linewidth=2),
                               whiskerprops=dict(color=color),
                               capprops=dict(color=color),
                               flierprops=dict(marker='o', markerfacecolor=color,
                                              markersize=3, alpha=0.5))
                jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
                ax.scatter(np.full(len(vals), i) + jitter, vals,
                          color=color, s=15, alpha=0.5, zorder=5,
                          edgecolors='white', linewidths=0.3)
                positions.append(i)
                labels.append(THESIS_LABELS[model])
            ax.set_xticks(positions)
            ax.set_xticklabels(labels)
            ax.set_yscale('log')
            ax.set_ylabel('Number of Parameters')
            ax.set_title('(c) Model Complexity', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, which='both', axis='y')

    # ── Panel (d): Cost-effectiveness (R² vs Training Time) ──
    if has_r2:
        ax = axes[panel_idx]
        panel_idx += 1
        scatter_data = df.dropna(subset=['training_s', 'r2'])
        # Filter out catastrophic LSTM failures for readability
        scatter_data = scatter_data[scatter_data['r2'] > -5]
        for model in active_models:
            mdata = scatter_data[scatter_data['model'] == model]
            if mdata.empty:
                continue
            ax.scatter(mdata['training_s'], mdata['r2'],
                      color=THESIS_COLORS[model],
                      marker=THESIS_MARKERS[model],
                      s=50, alpha=0.7, edgecolors='white', linewidths=0.5,
                      label=THESIS_LABELS[model], zorder=5)
        ax.set_xscale('log')
        ax.set_xlabel('Training Time (seconds)')
        ax.set_ylabel(r'Test $R^2$')
        ax.set_title('(d) Accuracy vs Training Cost', fontsize=12, fontweight='bold')
        ax.axhline(0.0, color='gray', ls='--', lw=0.8, alpha=0.5)
        ax.axhline(1.0, color='gray', ls=':', lw=0.8, alpha=0.3)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused panels
    for idx in range(panel_idx, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Runtime Comparison: MVAR vs LSTM vs WSINDy',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, output_dir, 'thesis_runtime_comparison')
    print(f"  Saved: thesis_runtime_comparison.pdf/png")


# ---------------------------------------------------------------------------
# Helper: collect mean R² per experiment per model from summary.json
# ---------------------------------------------------------------------------

def _get_active_thesis_models(experiments):
    """Return the subset of THESIS_MODELS that actually have results."""
    active = set()
    for exp_dir in experiments.values():
        p = Path(exp_dir)
        summary_path = p / 'summary.json'
        if summary_path.exists():
            try:
                with open(summary_path) as f:
                    summary = json.load(f)
                for model in THESIS_MODELS:
                    section = summary.get(model, {})
                    r2 = section.get('mean_r2_test')
                    if r2 is not None and not (isinstance(r2, float) and np.isnan(r2)):
                        active.add(model)
            except (json.JSONDecodeError, IOError):
                pass
        # Also check for test_results.csv directories
        for model in THESIS_MODELS:
            dir_tag = model.upper()
            if (p / dir_tag / 'test_results.csv').exists():
                active.add(model)
    return [m for m in THESIS_MODELS if m in active]


def _collect_summary_r2(experiments):
    """Return DataFrame with columns [experiment, group, model, r2].

    Reads mean_r2_test from summary.json for each model.
    Only includes models that actually have data.
    """
    active_models = _get_active_thesis_models(experiments)
    rows = []
    for exp_name, exp_dir in sorted(experiments.items()):
        summary_path = Path(exp_dir) / 'summary.json'
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            summary = json.load(f)

        # Determine regime group
        group_key = 'other'
        for g in REGIME_GROUPS:
            if exp_name in g['members']:
                group_key = g['key']
                break

        for model in active_models:
            model_section = summary.get(model, {})
            r2 = model_section.get('mean_r2_test', np.nan)
            if r2 is None:
                r2 = np.nan
            rows.append({
                'experiment': exp_name,
                'short_name': _short(exp_name),
                'group': group_key,
                'model': model,
                'r2': float(r2),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# THESIS FIG 10: R² Heatmap — all regimes × models in one compact figure
# ---------------------------------------------------------------------------

def plot_thesis_r2_heatmap(experiments, output_dir):
    """Heatmap: rows = experiments (sorted by group), columns = MVAR/LSTM/WSINDy.

    Color = R² value; annotated with numbers.  Best single-figure overview.
    """
    print(f"\n{'=' * 60}")
    print("THESIS FIG 10: R² Heatmap (experiments x models)")
    print("=" * 60)

    df = _collect_summary_r2(experiments)
    if df.empty:
        print("  No data — skipping.")
        return

    # Pivot: rows = experiments, columns = models
    pivot = df.pivot_table(index='experiment', columns='model', values='r2')

    # Order experiments by group, then alphabetically within group
    group_order = [g['key'] for g in REGIME_GROUPS] + ['other']
    exp_group = df.drop_duplicates('experiment').set_index('experiment')['group']

    ordered_exps = []
    group_labels = []
    group_boundaries = []
    for gkey in group_order:
        members = sorted(exp_group[exp_group == gkey].index.tolist())
        if not members:
            continue
        group_boundaries.append(len(ordered_exps))
        for m in members:
            if m in pivot.index:
                ordered_exps.append(m)
                group_labels.append(gkey)

    if not ordered_exps:
        print("  No valid experiments in pivot — skipping.")
        return

    pivot = pivot.reindex(ordered_exps)
    # Reorder columns to active models only
    col_order = [m for m in ['mvar', 'lstm'] if m in pivot.columns]
    pivot = pivot[col_order]

    # Short names for display
    short_names = [_short(e) for e in ordered_exps]

    # Clamp for colormap readability
    vmin = max(pivot.min().min(), R2_DISPLAY_FLOOR)
    vmax = R2_DISPLAY_CEIL

    n_rows = len(pivot)
    fig_height = max(6, n_rows * 0.32 + 2)
    fig, ax = plt.subplots(figsize=(5.5, fig_height))

    import matplotlib.colors as mcolors
    # Diverging: red (bad) -> white (0) -> blue/green (good)
    cmap = plt.cm.RdYlGn
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    data = pivot.values
    im = ax.imshow(data, aspect='auto', cmap=cmap, norm=norm,
                   interpolation='nearest')

    # Annotate cells — show true value, mark clamped with arrow
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                ax.text(j, i, '—', ha='center', va='center',
                        fontsize=7, color='gray')
            elif val < R2_DISPLAY_FLOOR:
                color = 'white'
                ax.text(j, i, f'↓{val:.0f}', ha='center', va='center',
                        fontsize=6, fontweight='bold', color=color)
            else:
                color = 'white' if val < -0.3 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, fontweight='bold', color=color)

    # Draw group separators
    for boundary in group_boundaries[1:]:
        ax.axhline(boundary - 0.5, color='black', linewidth=1.5)

    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels([THESIS_LABELS[m] for m in col_order], fontsize=11,
                       fontweight='bold')
    ax.set_yticks(range(len(short_names)))
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_xlabel('')
    ax.set_title(r'Test $R^2$ by Experiment and Model', fontsize=13,
                 fontweight='bold', pad=12)

    # Group labels on right margin
    prev_b = 0
    for idx, boundary in enumerate(group_boundaries):
        next_b = group_boundaries[idx + 1] if idx + 1 < len(group_boundaries) \
            else len(ordered_exps)
        mid = (boundary + next_b - 1) / 2.0
        gkey = group_order[idx] if idx < len(group_order) else ''
        # Find the matching group label
        for g in REGIME_GROUPS:
            if g['key'] == gkey:
                gkey = g['key'].replace('_', '\n')
                break
        ax.text(len(col_order) + 0.3, mid, gkey, ha='left', va='center',
                fontsize=6, style='italic', color='gray')

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.15)
    cbar.set_label(r'$R^2$', fontsize=11)

    fig.tight_layout()
    _save_fig(fig, output_dir, 'thesis_r2_heatmap')
    print(f"  Saved: thesis_r2_heatmap.pdf/png  ({len(pivot)} experiments)")


# ---------------------------------------------------------------------------
# THESIS FIG 11: R² Distribution — violin + box per model
# ---------------------------------------------------------------------------

def plot_thesis_r2_distribution(experiments, output_dir):
    """Violin + box + strip plot of R² distribution across regimes per model.

    Compact: single figure, one violin per model.
    """
    print(f"\n{'=' * 60}")
    print("THESIS FIG 11: R² Distribution (violin per model)")
    print("=" * 60)

    df = _collect_summary_r2(experiments)
    if df.empty:
        print("  No data — skipping.")
        return

    # Drop NaN R²
    df = df.dropna(subset=['r2'])
    active_models = [m for m in THESIS_MODELS if m in df['model'].values]
    if not active_models:
        print("  No models with R² data — skipping.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    positions = list(range(len(active_models)))
    violin_data = []
    for model in active_models:
        vals = df[df['model'] == model]['r2'].values
        # Clamp extreme negatives for visualization
        vals_clamped = np.clip(vals, R2_DISPLAY_FLOOR, 1.5)
        violin_data.append(vals_clamped)

    # Violin
    parts = ax.violinplot(violin_data, positions=positions,
                          showmeans=False, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(THESIS_COLORS[active_models[i]])
        pc.set_alpha(0.3)
        pc.set_edgecolor(THESIS_COLORS[active_models[i]])

    # Box overlay
    bp = ax.boxplot(violin_data, positions=positions, widths=0.2,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', linewidth=2))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(THESIS_COLORS[active_models[i]])
        patch.set_alpha(0.5)

    # Strip (jittered individual points)
    rng = np.random.default_rng(42)
    for i, model in enumerate(active_models):
        vals = df[df['model'] == model]['r2'].values
        vals_clamped = np.clip(vals, R2_DISPLAY_FLOOR, 1.5)
        n_clipped = np.sum(vals < R2_DISPLAY_FLOOR)
        jitter = rng.uniform(-0.08, 0.08, len(vals_clamped))
        ax.scatter(np.full(len(vals_clamped), i) + jitter, vals_clamped,
                   color=THESIS_COLORS[model], s=18, alpha=0.6,
                   edgecolors='white', linewidths=0.3, zorder=5)

        # Annotate stats
        median_val = np.median(vals)
        n_pos = np.sum(vals > 0)
        clipped_txt = f'\n({n_clipped} clipped)' if n_clipped > 0 else ''
        ax.annotate(f'med={median_val:.2f}\n{n_pos}/{len(vals)} > 0{clipped_txt}',
                    xy=(i, R2_DISPLAY_FLOOR - 0.08), ha='center', va='top',
                    fontsize=7,
                    color=THESIS_COLORS[model], fontweight='bold')

    ax.set_xticks(positions)
    ax.set_xticklabels([THESIS_LABELS[m] for m in active_models], fontsize=12,
                       fontweight='bold')
    ax.axhline(0.0, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.axhline(1.0, color='gray', ls=':', lw=0.8, alpha=0.3)
    ax.set_ylabel(r'Test $R^2$', fontsize=12)
    ax.set_title(r'$R^2$ Distribution Across All Experiments', fontsize=13,
                 fontweight='bold')
    ax.set_ylim(R2_DISPLAY_FLOOR - 0.25, 1.15)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    _save_fig(fig, output_dir, 'thesis_r2_distribution')
    print(f"  Saved: thesis_r2_distribution.pdf/png")


# ---------------------------------------------------------------------------
# THESIS FIG 12: MVAR vs LSTM scatter — head-to-head per regime
# ---------------------------------------------------------------------------

def plot_thesis_mvar_vs_lstm(experiments, output_dir):
    """Scatter: x = MVAR R², y = LSTM R², one point per experiment.

    Color by regime group.  Shows where LSTM matches/beats/fails vs MVAR.
    """
    print(f"\n{'=' * 60}")
    print("THESIS FIG 12: MVAR vs LSTM Head-to-Head Scatter")
    print("=" * 60)

    df = _collect_summary_r2(experiments)
    if df.empty:
        print("  No data — skipping.")
        return

    mvar_df = df[df['model'] == 'mvar'][['experiment', 'group', 'r2']].rename(
        columns={'r2': 'mvar_r2'})
    lstm_df = df[df['model'] == 'lstm'][['experiment', 'r2']].rename(
        columns={'r2': 'lstm_r2'})

    merged = mvar_df.merge(lstm_df, on='experiment', how='inner')
    merged = merged.dropna(subset=['mvar_r2', 'lstm_r2'])

    if merged.empty:
        print("  No experiments with both MVAR and LSTM — skipping.")
        return

    # Clip extreme values for display; annotate clipped points
    clip_lo = R2_DISPLAY_FLOOR
    merged['mvar_disp'] = np.clip(merged['mvar_r2'].values, clip_lo, None)
    merged['lstm_disp'] = np.clip(merged['lstm_r2'].values, clip_lo, None)
    clipped = merged[(merged['mvar_r2'] < clip_lo) | (merged['lstm_r2'] < clip_lo)]

    # Group colors
    group_colors = {}
    cmap_disc = plt.cm.tab10
    unique_groups = [g['key'] for g in REGIME_GROUPS]
    for i, gkey in enumerate(unique_groups):
        group_colors[gkey] = cmap_disc(i / max(len(unique_groups) - 1, 1))
    group_colors['other'] = 'gray'

    group_display = {g['key']: g['label'] for g in REGIME_GROUPS}
    group_display['other'] = 'Other'

    fig, ax = plt.subplots(figsize=(8, 7))

    for gkey in unique_groups + ['other']:
        subset = merged[merged['group'] == gkey]
        if subset.empty:
            continue
        ax.scatter(subset['mvar_disp'], subset['lstm_disp'],
                   color=group_colors[gkey], s=50, alpha=0.8,
                   edgecolors='white', linewidths=0.5,
                   label=group_display.get(gkey, gkey), zorder=5)

    # Annotate clipped points with true R² value
    for _, row in clipped.iterrows():
        true_mvar = row['mvar_r2']
        true_lstm = row['lstm_r2']
        disp_x = row['mvar_disp']
        disp_y = row['lstm_disp']
        lbl_parts = []
        if true_mvar < clip_lo:
            lbl_parts.append(f'M={true_mvar:.0f}')
        if true_lstm < clip_lo:
            lbl_parts.append(f'L={true_lstm:.0f}')
        ax.annotate(', '.join(lbl_parts), (disp_x, disp_y),
                    textcoords='offset points', xytext=(5, -10),
                    fontsize=6, color='red', fontstyle='italic')

    # Identity line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, '--', color='gray', alpha=0.5, lw=1,
            label='MVAR = LSTM')

    # Reference lines
    ax.axhline(0, color='gray', ls=':', lw=0.8, alpha=0.3)
    ax.axvline(0, color='gray', ls=':', lw=0.8, alpha=0.3)

    ax.set_xlabel(r'MVAR $R^2$', fontsize=12)
    ax.set_ylabel(r'LSTM $R^2$', fontsize=12)
    ax.set_title('MVAR vs LSTM — Per-Experiment Comparison', fontsize=13,
                 fontweight='bold')
    ax.legend(fontsize=7, loc='lower right', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Clip for readability — use consistent floor
    ax.set_xlim(R2_DISPLAY_FLOOR - 0.05, R2_DISPLAY_CEIL)
    ax.set_ylim(R2_DISPLAY_FLOOR - 0.05, R2_DISPLAY_CEIL)

    # Note clipped count
    n_clipped = len(clipped)
    if n_clipped > 0:
        ax.annotate(f'{n_clipped} point(s) clipped to floor={R2_DISPLAY_FLOOR}',
                    xy=(0.02, 0.02), xycoords='axes fraction',
                    fontsize=7, color='red', fontstyle='italic')

    fig.tight_layout()
    _save_fig(fig, output_dir, 'thesis_mvar_vs_lstm')
    print(f"  Saved: thesis_mvar_vs_lstm.pdf/png  ({len(merged)} experiments)")


# ---------------------------------------------------------------------------
# THESIS FIG 13: R² by Regime Group — grouped bars
# ---------------------------------------------------------------------------

def plot_thesis_r2_by_group(experiments, output_dir):
    """Grouped bar chart: x = regime groups, bars = MVAR/LSTM(/WSINDy) mean R².

    Compact overview of how models compare across behavioral categories.
    """
    print(f"\n{'=' * 60}")
    print("THESIS FIG 13: R² by Regime Group (grouped bars)")
    print("=" * 60)

    df = _collect_summary_r2(experiments)
    if df.empty:
        print("  No data — skipping.")
        return

    df = df.dropna(subset=['r2'])
    active_models = [m for m in THESIS_MODELS if m in df['model'].values]

    group_order = [g['key'] for g in REGIME_GROUPS]
    group_display = {g['key']: g['label'] for g in REGIME_GROUPS}

    # Compute mean + std per group × model
    stats = df.groupby(['group', 'model'])['r2'].agg(['mean', 'std', 'count'])
    stats = stats.reset_index()

    fig, ax = plt.subplots(figsize=(12, 5))

    present_groups = [g for g in group_order if g in stats['group'].values]
    n_groups = len(present_groups)
    n_models = len(active_models)
    bar_width = 0.8 / n_models
    x = np.arange(n_groups)

    for mi, model in enumerate(active_models):
        means = []
        stds = []
        for gkey in present_groups:
            row = stats[(stats['group'] == gkey) & (stats['model'] == model)]
            if not row.empty:
                means.append(row['mean'].values[0])
                stds.append(row['std'].values[0])
            else:
                means.append(np.nan)
                stds.append(0)

        offset = (mi - (n_models - 1) / 2) * bar_width
        bars = ax.bar(x + offset, means, bar_width, yerr=stds,
                      color=THESIS_COLORS[model], alpha=0.8,
                      edgecolor='white', linewidth=0.5,
                      capsize=3, error_kw={'linewidth': 1},
                      label=THESIS_LABELS[model])

        # Annotate bar values
        for xi, val in zip(x + offset, means):
            if not np.isnan(val):
                ax.text(xi, val + 0.02, f'{val:.2f}', ha='center', va='bottom',
                        fontsize=6, fontweight='bold', color=THESIS_COLORS[model])

    # Wrap long group labels
    short_labels = []
    for gkey in present_groups:
        lbl = group_display.get(gkey, gkey)
        # "Constant Speed — Alignment-Dominated" -> "CS\nAlignment"
        lbl = lbl.replace('Constant Speed', 'CS').replace('Variable Speed', 'VS')
        lbl = lbl.replace(' \u2014 ', '\n').replace('-Dominated', '')
        short_labels.append(lbl)

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=8, ha='center')
    ax.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.set_ylabel(r'Mean Test $R^2$', fontsize=12)
    ax.set_title(r'$R^2$ by Regime Group', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    raw_min = min(0, ax.get_ylim()[0] - 0.1)
    ax.set_ylim(max(R2_DISPLAY_FLOOR - 0.15, raw_min), 1.15)

    fig.tight_layout()
    _save_fig(fig, output_dir, 'thesis_r2_by_group')
    print(f"  Saved: thesis_r2_by_group.pdf/png  ({n_groups} groups)")


# ---------------------------------------------------------------------------
# THESIS FIG 6: Per-experiment summary table (CSV + LaTeX)
# ---------------------------------------------------------------------------

def generate_thesis_table(experiments, output_dir):
    """Produce a per-experiment results table with all model metrics."""
    print(f"\n{'=' * 60}")
    print("THESIS FIG 6: Per-Experiment Summary Table")
    print("=" * 60)

    rows = []
    for exp_name, exp_dir in sorted(experiments.items()):
        row = {'Experiment': exp_name}

        # Determine regime from experiment name
        if exp_name.startswith('DO_'):
            parts = exp_name.split('_')
            row['Regime'] = parts[1] if len(parts) > 1 else 'unknown'
        elif exp_name.startswith('NDYN'):
            row['Regime'] = 'NDYN'
        else:
            row['Regime'] = exp_name.split('_')[0] if '_' in exp_name else 'other'

        for model in THESIS_MODELS:
            tag = model.upper()
            csv_path = Path(exp_dir) / tag / 'test_results.csv'
            if not csv_path.exists():
                # Model not available for this experiment — skip entirely
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                row[f'{tag} R²'] = np.nan
                row[f'{tag} R²_std'] = np.nan
                row[f'{tag} RMSE'] = np.nan
                row[f'{tag} Mass Viol.'] = np.nan
                continue

            if 'r2_reconstructed' in df.columns:
                row[f'{tag} R²'] = df['r2_reconstructed'].mean()
                row[f'{tag} R²_std'] = df['r2_reconstructed'].std()
            else:
                row[f'{tag} R²'] = np.nan
                row[f'{tag} R²_std'] = np.nan

            if 'rmse_recon' in df.columns:
                row[f'{tag} RMSE'] = df['rmse_recon'].mean()
            else:
                row[f'{tag} RMSE'] = np.nan

            if 'max_mass_violation' in df.columns:
                row[f'{tag} Mass Viol.'] = df['max_mass_violation'].max()
            else:
                row[f'{tag} Mass Viol.'] = np.nan

        rows.append(row)

    if not rows:
        print("  No data for table.")
        return

    table = pd.DataFrame(rows)

    # Sort by MVAR R² descending
    table = table.sort_values('MVAR R²', ascending=False, na_position='last')

    # Save CSV
    csv_path = output_dir / 'results_table.csv'
    table.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"  Saved: results_table.csv ({len(table)} experiments)")

    # Save LaTeX
    tex_path = output_dir / 'results_table.tex'
    # Format for LaTeX — keep it readable
    fmt_table = table.copy()
    for tag in ['MVAR', 'LSTM']:
        r2_col = f'{tag} R²'
        std_col = f'{tag} R²_std'
        if r2_col in fmt_table.columns and std_col in fmt_table.columns:
            fmt_table[f'{tag} R²±std'] = fmt_table.apply(
                lambda r: f"${r[r2_col]:.3f} \\pm {r[std_col]:.3f}$"
                if pd.notna(r[r2_col]) else '---', axis=1)
            fmt_table = fmt_table.drop(columns=[r2_col, std_col])

    tex_cols = ['Experiment', 'Regime']
    for tag in ['MVAR', 'LSTM']:
        if f'{tag} R²±std' in fmt_table.columns:
            tex_cols.append(f'{tag} R²±std')
        if f'{tag} RMSE' in fmt_table.columns:
            tex_cols.append(f'{tag} RMSE')
        if f'{tag} Mass Viol.' in fmt_table.columns:
            tex_cols.append(f'{tag} Mass Viol.')
    tex_subset = fmt_table[[c for c in tex_cols if c in fmt_table.columns]]

    with open(tex_path, 'w') as f:
        f.write(tex_subset.to_latex(index=False, escape=False,
                                     float_format='%.4f',
                                     na_rep='---'))
    print(f"  Saved: results_table.tex")


# ---------------------------------------------------------------------------
# THESIS orchestrator
# ---------------------------------------------------------------------------

def thesis_figures(experiments, ic_maps, output_dir, data_dir, skip_kde=False,
                   skip_phase=False, skip_lp=False):
    """Generate all thesis-quality aggregate figures."""
    print("\n" + "=" * 80)
    print("THESIS MODE — Aggregate Model-Level Figures (MVAR vs LSTM)")
    print("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build consistent per-experiment style map (foundation for all figures)
    style_map = _build_experiment_style_map(experiments)
    print(f"  Style map: {len(style_map)} experiments assigned color+marker")

    # Fig 1: R² degradation (aggregate — all experiments, all IC types)
    plot_thesis_r2_degradation(experiments, output_dir)

    # Fig 2: Per-experiment MVAR | LSTM R² degradation (side by side)
    plot_thesis_r2_degradation_per_exp(experiments, output_dir, style_map)

    # Fig 3: sPOD vs POD decay (aggregate)
    plot_thesis_svd_decay(experiments, output_dir)

    # Fig 4: Density vs Latent R² scatter
    plot_thesis_density_latent_scatter(experiments, output_dir, style_map)

    # Fig 5: Mass conservation table (CSV + LaTeX — no plot)
    generate_thesis_mass_table(experiments, output_dir, data_dir)

    # Fig 6: Summary results table
    generate_thesis_table(experiments, output_dir)

    # Fig 7: Phase dynamics (3 panels: trajectory | AR | autocorrelation)
    if not skip_phase:
        plot_thesis_phase_dynamics(experiments, output_dir, style_map)
    else:
        print(f"\n  Skipping phase dynamics (--skip_phase)")

    # Fig 8: sPOD vs POD decay per-experiment detail
    plot_thesis_svd_decay_detail(experiments, output_dir, style_map)

    # Fig 9: KDE snapshots per experiment → kde_snapshots/<exp_name>/
    if not skip_kde:
        plot_thesis_kde_snapshots(experiments, output_dir, data_dir)
    else:
        print(f"\n  Skipping KDE snapshots (--skip_kde)")

    print(f"\n{'=' * 80}")
    print("THESIS FIGURES COMPLETE")
    print("=" * 80)
    print(f"  Output: {output_dir}/")
    print(f"  KDE snapshots: kde_snapshots/<experiment_name>/")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='3-Model IC-Stratified Cross-Experiment Analysis Pipeline')
    parser.add_argument('--data_dir', type=str, default='oscar_output',
                        help='Base directory with experiment outputs')
    parser.add_argument('--output_dir', type=str, default='Analyses_3models',
                        help='Output directory')
    parser.add_argument('--experiments', nargs='*', default=None,
                        help='Only include these experiment names')
    parser.add_argument('--ics', nargs='*', default=None,
                        help='IC types to analyze (default: all 4)')
    parser.add_argument('--groups', nargs='*', default=None,
                        help='Regime group keys (default: all 8)')
    parser.add_argument('--skip_kde', action='store_true',
                        help='Skip KDE snapshot grids (large figures)')
    parser.add_argument('--skip_phase', action='store_true',
                        help='Skip phase dynamics (slow)')
    parser.add_argument('--skip_lp', action='store_true',
                        help='Skip L^p error figures (requires extracted data)')
    parser.add_argument('--systematic', action='store_true',
                        help='Only include systematic experiments (DO_*/NDYN_*)')
    parser.add_argument('--thesis', action='store_true',
                        help='Generate thesis-quality aggregate figures (MVAR vs LSTM)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = _time.time()

    print("=" * 80)
    print("3-MODEL IC-STRATIFIED CROSS-EXPERIMENT ANALYSIS PIPELINE")
    print("  Models: MVAR | LSTM")
    print("=" * 80)

    experiments = discover_experiments(data_dir, args.experiments)
    # Filter to systematic experiments if requested
    if args.systematic:
        experiments = {k: v for k, v in experiments.items()
                       if k.startswith('DO_') or k.startswith('NDYN')}
    if not experiments:
        print(f"\n  No valid experiments found in {data_dir}/")
        return
    print(f"\nFound {len(experiments)} experiments in {data_dir}/")

    # Detect model availability per experiment
    model_avail = {}
    for exp_name, exp_dir in experiments.items():
        model_avail[exp_name] = detect_available_models(exp_dir)
    n_mvar = sum(1 for v in model_avail.values() if 'mvar' in v)
    n_lstm = sum(1 for v in model_avail.values() if 'lstm' in v)
    print(f"Model availability: MVAR={n_mvar}, LSTM={n_lstm}")

    ic_maps = {}
    for exp_name, exp_dir in experiments.items():
        ic_maps[exp_name] = get_ic_test_idx_map(exp_dir)

    ic_types = args.ics if args.ics else IC_NAMES
    active_groups = REGIME_GROUPS
    if args.groups:
        active_groups = [g for g in REGIME_GROUPS if g['key'] in args.groups]
    print(f"IC types:  {', '.join(IC_DISPLAY.get(ic, ic) for ic in ic_types)}")
    print(f"Groups:    {len(active_groups)}")
    print(f"Output:    {output_dir}/")

    # ==================================================================
    # THESIS MODE — aggregate model-level figures
    # ==================================================================
    if args.thesis:
        _apply_thesis_style()
        thesis_figures(experiments, ic_maps, output_dir, data_dir,
                       skip_kde=args.skip_kde,
                       skip_phase=args.skip_phase,
                       skip_lp=args.skip_lp)
        elapsed = _time.time() - t0
        print(f"\n  Wall-clock time:     {elapsed / 60:.1f} min")
        print("=" * 80)
        return

    # ==================================================================
    # LEGACY MODE — per-IC × per-group 3-panel plots
    # ==================================================================

    # Cross-IC plots
    cross_dir = output_dir / 'cross_ic'
    cross_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("1. SVD Spectra & Cumulative Energy (ROM only)")
    print("=" * 60)
    plot_svd_spectra(experiments, cross_dir)

    if not args.skip_phase:
        print(f"\n{'=' * 60}")
        print("2. Phase Dynamics (ROM only)")
        print("=" * 60)
        plot_phase_dynamics(experiments, cross_dir)

    print(f"\n{'=' * 60}")
    print("3. Runtime Comparison (all 3 models)")
    print("=" * 60)
    plot_runtime_comparison(experiments, cross_dir)

    print(f"\n{'=' * 60}")
    print("4. R\u00b2 Summary Bars (all 3 models)")
    print("=" * 60)
    plot_r2_summary_bars(experiments, ic_maps, ic_types, cross_dir)

    # Per-IC x Per-Group plots (3-panel)
    total_generated = 0

    for ic_name in ic_types:
        ic_dir = output_dir / f'IC_{ic_name}'
        ic_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 80}")
        print(f"IC TYPE: {IC_DISPLAY[ic_name]}")
        print("=" * 80)

        for group in active_groups:
            gkey = group['key']
            glabel = group['label']
            gmembers = group['members']

            group_exps = [
                (m, experiments[m]) for m in gmembers if m in experiments
            ]
            if not group_exps:
                continue

            gdir = ic_dir / f'group_{gkey}'
            gdir.mkdir(parents=True, exist_ok=True)

            print(f"\n  {glabel}  ({len(group_exps)} experiments)")

            print(f"    \u2022 R\u00b2 degradation \u2026")
            plot_r2_degradation_group(group_exps, glabel, ic_name,
                                     ic_maps, gdir)
            total_generated += 1

            print(f"    \u2022 Normalized R\u00b2 \u2026")
            plot_normalized_r2_group(group_exps, glabel, ic_name,
                                    ic_maps, gdir)
            total_generated += 1

            print(f"    \u2022 Error norms (L1, L2, Linf) \u2026")
            plot_error_norms_group(group_exps, glabel, ic_name,
                                  ic_maps, gdir)
            total_generated += 3

            print(f"    \u2022 Mass conservation \u2026")
            plot_mass_conservation_group(group_exps, glabel, ic_name,
                                        ic_maps, gdir)
            total_generated += 1

            if not args.skip_kde:
                print(f"    \u2022 KDE density snapshots \u2026")
                plot_kde_snapshots_group(group_exps, glabel, ic_name,
                                        ic_maps, gdir)
                total_generated += 1

            print(f"    \u2022 Spatial order \u2026")
            plot_spatial_order_group(group_exps, glabel, ic_name,
                                    ic_maps, gdir)
            total_generated += 1

    # Summary
    elapsed = _time.time() - t0
    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\n  Output directory:    {output_dir}/")
    print(f"  Cross-IC plots:      {output_dir}/cross_ic/")
    for ic in ic_types:
        n_grp = sum(
            1 for g in active_groups
            if any(m in experiments for m in g['members'])
        )
        print(f"  {IC_DISPLAY[ic]:15s}  -> {output_dir}/IC_{ic}/  "
              f"({n_grp} groups)")
    print(f"\n  Models detected:     MVAR={n_mvar}  LSTM={n_lstm}")
    print(f"  Figures generated:   ~{total_generated}")
    print(f"  Wall-clock time:     {elapsed / 60:.1f} min")
    print("=" * 80)


if __name__ == '__main__':
    main()
