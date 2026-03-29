#!/usr/bin/env python3
"""
IC-Stratified Cross-Experiment Analysis Pipeline
=================================================

Scans oscar_output/ for completed experiments from configs/systematic/
and produces thesis-quality comparison plots, stratified by IC type and
grouped by dynamical regime similarity.

Design
------
Each systematic experiment has 4 test runs (1 per IC type):
    test_000 = gaussian, test_001 = uniform, test_002 = ring, test_003 = two_clusters

Regimes are batched into ~7-member groups by physics:
    Constant-speed alignment, attractive, repulsive, collective patterns
    Variable-speed  alignment, attractive, repulsive, collective patterns

For EVERY (IC type x regime group) pair, the pipeline produces:
    - R^2 degradation over time  (MVAR vs LSTM side-by-side)
    - Normalized R^2(t)/R^2(t_0) (relative decay comparison)
    - Relative L^1, L^2, L^inf error vs time
    - Mass conservation (true vs predicted)
    - KDE density snapshots      (true vs MVAR vs LSTM)
    - Spatial order parameter    (std rho vs time)

Cross-IC (regime-independent) plots:
    - sPOD singular value spectra + cumulative energy
    - POD vs sPOD eigenvalue decay contrast
    - Phase dynamics (shift trajectories, PSD, AR predictability)
    - Runtime comparison (training time, inference speed, params)

Output structure
----------------
    Analyses/
      cross_ic/                 <- SVD, phase dynamics, runtime
      IC_gaussian/
        group_CS_alignment/     <- r2, errors, mass, kde, spatial
        group_CS_attractive/
        ...
      IC_uniform/
        ...
      IC_two_clusters/
        ...
      IC_ring/
        ...

Usage
-----
    python plot_pipeline.py [--data_dir oscar_output] [--output_dir Analyses]
                            [--experiments EXP1 EXP2 ...]
                            [--ics gaussian uniform ring two_clusters]
                            [--groups CS_alignment VS_repulsive ...]
                            [--skip_kde]
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
# Regime grouping -- experiments batched by physics for clean visualization
# ============================================================================
#
# 54 systematic configs in 8 groups, ~6-9 members each.
# Members listed by experiment_name (must match oscar_output/ folder names).
#
# Constant-speed regimes:
#   CS_alignment   (6) -- low-force, alignment-dominated dynamics
#   CS_attractive  (5) -- strong Morse attraction (Ca >> Cr)
#   CS_repulsive   (7) -- strong Morse repulsion  (Cr >> Ca)
#   CS_collective  (9) -- organized patterns: swarms, mills, double rings
#
# Variable-speed (_VS) mirrors:
#   VS_alignment   (6) -- variable speed, alignment-dominated
#   VS_attractive  (5) -- variable speed, strong attraction
#   VS_repulsive   (7) -- variable speed, strong repulsion
#   VS_collective  (9) -- variable speed, organized patterns

REGIME_GROUPS = [
    {
        'key': 'CS_alignment',
        'label': 'Constant Speed \u2014 Alignment-Dominated',
        'members': [
            'NDYN01_crawl',        # slow flock, easy baseline
            'NDYN02_flock',        # coherent traveling cluster
            'NDYN04_gas',          # disordered gas, noise-dominated
            'NDYN08_pure_vicsek',  # pure Vicsek, no forces
            'NDYN09_longrange',    # long-range forces, global modes
            'NDYN13_chaos',        # extreme speed + noise, lower bound
        ],
    },
    {
        'key': 'CS_attractive',
        'label': 'Constant Speed \u2014 Attractive',
        'members': [
            'NDYN03_sprint',           # fast transport
            'NDYN05_blackhole',        # extreme attraction, singular spike
            'NDYN07_crystal',          # balanced Ca~Cr, crystal/ring
            'NDYN10_shortrange',       # short-range spike forces
            'NDYN11_noisy_collapse',   # strong attraction vs high noise
        ],
    },
    {
        'key': 'CS_repulsive',
        'label': 'Constant Speed \u2014 Repulsive',
        'members': [
            'NDYN06_supernova',        # explosive repulsion
            'NDYN12_fast_explosion',   # fast speed + strong repulsion
            'DO_EC01_esccol_C2_l3',    # escape column Cr=2
            'DO_EC02_esccol_C3_l05',   # escape column Cr=3
            'DO_ES01_escsym_C3_l09',   # escape symmetric Cr=3
            'DO_EU01_escuns_C2_l2',    # escape unstable Cr=2
            'DO_EU02_escuns_C3_l3',    # escape unstable Cr=3
        ],
    },
    {
        'key': 'CS_collective',
        'label': 'Constant Speed \u2014 Collective Patterns',
        'members': [
            'DO_CS01_swarm_C01_l05',   # swarm, weak
            'DO_CS02_swarm_C05_l3',    # swarm, medium
            'DO_CS03_swarm_C09_l3',    # swarm, strong
            'DO_DM01_dmill_C09_l05',   # double mill
            'DO_DR01_dring_C01_l01',   # double ring, weak
            'DO_DR02_dring_C09_l09',   # double ring, strong
            'DO_SM01_mill_C05_l01',    # single mill, weak
            'DO_SM02_mill_C3_l01',     # single mill, strong
            'DO_SM03_mill_C2_l05',     # single mill, medium
        ],
    },
    {
        'key': 'VS_alignment',
        'label': 'Variable Speed \u2014 Alignment-Dominated',
        'members': [
            'NDYN01_crawl_VS',
            'NDYN02_flock_VS',
            'NDYN04_gas_VS',
            'NDYN09_longrange_VS',
            'NDYN13_chaos_VS',
            'NDYN14_varspeed',         # dedicated variable-speed regime
        ],
    },
    {
        'key': 'VS_attractive',
        'label': 'Variable Speed \u2014 Attractive',
        'members': [
            'NDYN03_sprint_VS',
            'NDYN05_blackhole_VS',
            'NDYN07_crystal_VS',
            'NDYN10_shortrange_VS',
            'NDYN11_noisy_collapse_VS',
        ],
    },
    {
        'key': 'VS_repulsive',
        'label': 'Variable Speed \u2014 Repulsive',
        'members': [
            'NDYN06_supernova_VS',
            'NDYN12_fast_explosion_VS',
            'DO_EC01_esccol_C2_l3_VS',
            'DO_EC02_esccol_C3_l05_VS',
            'DO_ES01_escsym_C3_l09_VS',
            'DO_EU01_escuns_C2_l2_VS',
            'DO_EU02_escuns_C3_l3_VS',
        ],
    },
    {
        'key': 'VS_collective',
        'label': 'Variable Speed \u2014 Collective Patterns',
        'members': [
            'DO_CS01_swarm_C01_l05_VS',
            'DO_CS02_swarm_C05_l3_VS',
            'DO_CS03_swarm_C09_l3_VS',
            'DO_DM01_dmill_C09_l05_VS',
            'DO_DR01_dring_C01_l01_VS',
            'DO_DR02_dring_C09_l09_VS',
            'DO_SM01_mill_C05_l01_VS',
            'DO_SM02_mill_C3_l01_VS',
            'DO_SM03_mill_C2_l05_VS',
        ],
    },
]

# IC types in the order generated by generate_test_configs
IC_NAMES = ['gaussian', 'uniform', 'ring', 'two_clusters']

IC_DISPLAY = {
    'gaussian': 'Gaussian',
    'uniform': 'Uniform',
    'ring': 'Ring',
    'two_clusters': 'Double Gaussian',
}

# ============================================================================
# Color and label helpers
# ============================================================================


def _color(i, n):
    """Pick a color from tab10/tab20."""
    cmap = plt.cm.get_cmap('tab10' if n <= 10 else 'tab20')
    return cmap(i / max(n - 1, 1))


def _short(name):
    """Shorten experiment name for legend labels."""
    for prefix in ['NDYN', 'DO_']:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


# ============================================================================
# Data loaders
# ============================================================================

def load_singular_values(exp_dir):
    """Load aligned (sPOD) singular values."""
    for subdir in ['rom_common', 'mvar']:
        path = Path(exp_dir) / subdir / 'pod_basis.npz'
        if path.exists():
            data = np.load(path)
            return data['all_singular_values']
    return None


def load_unaligned_singular_values(exp_dir):
    """Load unaligned (standard POD) singular values."""
    for subdir in ['rom_common', 'mvar']:
        path = Path(exp_dir) / subdir / 'pod_basis_unaligned.npz'
        if path.exists():
            data = np.load(path)
            return data['all_singular_values']
    return None


def load_shift_data(exp_dir):
    """Load shift alignment data (reference field and per-run shifts)."""
    for subdir in ['rom_common', 'mvar']:
        path = Path(exp_dir) / subdir / 'shift_align_data.npz'
        if path.exists():
            data = np.load(path)
            return {'ref': data['ref'], 'shifts': data['shifts']}
    return None


def load_r2_single_test(exp_dir, test_idx, model='mvar'):
    """Load R^2 vs time for a single test run and model."""
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
    """Load (rho_true, rho_pred, times) for a single test run.

    Returns aligned arrays where rho_pred starts at forecast_start_idx.
    """
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


# ============================================================================
# Error computation helpers
# ============================================================================

def compute_error_timeseries(rho_true, rho_pred):
    """Compute per-timestep relative L1, L2, Linf errors and total mass."""
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
        'rel_e1': np.array(e1),
        'rel_e2': np.array(e2),
        'rel_einf': np.array(einf),
        'mass_true': np.array(mass_true),
        'mass_pred': np.array(mass_pred),
    }


def compute_spatial_order_timeseries(rho):
    """std(rho) per timestep -- measures spatial non-uniformity."""
    return np.array([np.std(rho[t]) for t in range(rho.shape[0])])


# ============================================================================
# Discovery
# ============================================================================

def discover_experiments(data_dir, experiment_filter=None):
    """Find all valid experiment folders.  Returns {name: Path}."""
    data_dir = Path(data_dir)
    experiments = {}
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        if experiment_filter and d.name not in experiment_filter:
            continue
        has_pod = (d / 'rom_common' / 'pod_basis.npz').exists() or \
                  (d / 'mvar' / 'pod_basis.npz').exists()
        has_test = (d / 'test').exists()
        if has_pod and has_test:
            experiments[d.name] = d
    return experiments


def detect_models(exp_dir):
    """Return list of available model names for an experiment."""
    models = []
    p = Path(exp_dir)
    if (p / 'MVAR' / 'test_results.csv').exists() or \
       (p / 'mvar' / 'mvar_model.npz').exists():
        models.append('mvar')
    if (p / 'LSTM').exists():
        try:
            if any((p / 'LSTM').iterdir()):
                models.append('lstm')
        except StopIteration:
            pass
    return models


def get_ic_test_idx_map(exp_dir):
    """Determine which test index corresponds to which IC type.

    Reads test/metadata.json.  Falls back to the standard ordering
    from generate_test_configs: gaussian(0), uniform(1), ring(2), two_clusters(3).
    """
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
    """Fallback IC -> test_idx mapping (generate_test_configs order)."""
    return {'gaussian': 0, 'uniform': 1, 'ring': 2, 'two_clusters': 3}


# ============================================================================
# Shared figure-save helper
# ============================================================================

def _save_fig(fig, output_dir, stem):
    """Save figure as both PDF and PNG, then close."""
    for ext in ['pdf', 'png']:
        fig.savefig(
            output_dir / f'{stem}.{ext}',
            bbox_inches='tight',
            dpi=200 if ext == 'png' else None,
        )
    plt.close(fig)


# ============================================================================
# PLOT 1: SVD spectra -- sPOD and POD vs sPOD  (cross-IC)
# ============================================================================

def plot_svd_spectra(experiments, output_dir, max_modes=40):
    """Singular value decay and cumulative energy for all experiments."""
    exp_list = list(experiments.values())
    n = len(exp_list)

    # Figure 1: Per-experiment sPOD spectra
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
    ax1.set_title('Singular Value Decay (sPOD \u2014 aligned)')
    ax1.legend(fontsize=5, ncol=4)
    ax1.grid(True, alpha=0.3)
    ax2.axhline(0.99, color='gray', ls=':', alpha=0.5, label='99 %')
    ax2.axhline(0.999, color='gray', ls='--', alpha=0.5, label='99.9 %')
    ax2.set_xlabel('Number of modes $d$')
    ax2.set_ylabel('Cumulative energy')
    ax2.set_title('Cumulative Energy (sPOD \u2014 aligned)')
    ax2.legend(fontsize=5, ncol=4)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.85, 1.005)
    fig.suptitle('sPOD Spectra Across Experiments', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig, output_dir, 'svd_spectra_comparison')
    print(f"  Saved: svd_spectra_comparison.pdf")

    # Figure 2: POD vs sPOD contrast
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
    if not has_unaligned:
        subtitle += ' (no unaligned data yet)'
    fig2.suptitle(subtitle, fontsize=13, fontweight='bold')
    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig2, output_dir, 'pod_vs_spod_contrast')
    print(f"  Saved: pod_vs_spod_contrast.pdf")


# ============================================================================
# PLOT 2: R^2 degradation  (per IC x per group, MVAR vs LSTM)
# ============================================================================

def plot_r2_degradation_group(group_exps, group_label, ic_name,
                              ic_maps, output_dir):
    """R^2 over forecast horizon -- left panel MVAR, right panel LSTM."""
    n = len(group_exps)
    if n == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    for mi, model in enumerate(['mvar', 'lstm']):
        ax = axes[mi]
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
        ax.set_title(model.upper())
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.05)
    fig.suptitle(f'{group_label} \u2014 $R^2$ Degradation ({IC_DISPLAY[ic_name]} IC)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig, output_dir, 'r2_degradation')


# ============================================================================
# PLOT 2b: Normalized R^2(t) / R^2(t_0)
# ============================================================================

def plot_normalized_r2_group(group_exps, group_label, ic_name,
                              ic_maps, output_dir):
    """Normalized R^2 so all curves start at 1.0 -- shows relative decay."""
    n = len(group_exps)
    if n == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    for mi, model in enumerate(['mvar', 'lstm']):
        ax = axes[mi]
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
        ax.set_title(model.upper())
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
# PLOT 3: Error norms  (L^1, L^2, L^inf)  per IC x per group
# ============================================================================

def plot_error_norms_group(group_exps, group_label, ic_name,
                            ic_maps, output_dir):
    """Relative error norms vs time -- one figure per norm, MVAR vs LSTM."""
    NORMS = [
        ('L2',   'rel_e2',   r'Relative $L^2$ Error'),
        ('L1',   'rel_e1',   r'Relative $L^1$ Error'),
        ('Linf', 'rel_einf', r'Relative $L^\infty$ Error'),
    ]
    n = len(group_exps)
    if n == 0:
        return

    # Pre-load errors to avoid redundant I/O
    cache = {}  # exp_name -> {model: (errs_dict, times)}
    for exp_name, exp_dir in group_exps:
        tidx = ic_maps.get(exp_name, {}).get(ic_name)
        if tidx is None:
            continue
        cache[exp_name] = {}
        for model in ['mvar', 'lstm']:
            rho_t, rho_p, times = load_density_pair(exp_dir, tidx, model)
            if rho_t is not None:
                cache[exp_name][model] = (compute_error_timeseries(rho_t, rho_p), times)

    for norm_tag, key, title_str in NORMS:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
        for mi, model in enumerate(['mvar', 'lstm']):
            ax = axes[mi]
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
            ax.set_title(model.upper())
            ax.legend(fontsize=6, ncol=2)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f'{group_label} \u2014 {title_str} ({IC_DISPLAY[ic_name]} IC)',
                     fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        _save_fig(fig, output_dir, f'error_{norm_tag}')


# ============================================================================
# PLOT 4: Mass conservation  per IC x per group
# ============================================================================

def plot_mass_conservation_group(group_exps, group_label, ic_name,
                                  ic_maps, output_dir):
    """Predicted vs true total mass over time -- MVAR and LSTM panels."""
    n = len(group_exps)
    if n == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for mi, model in enumerate(['mvar', 'lstm']):
        ax = axes[mi]
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
        ax.set_title(model.upper())
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f'{group_label} \u2014 Mass Conservation ({IC_DISPLAY[ic_name]} IC)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig, output_dir, 'mass_conservation')


# ============================================================================
# PLOT 5: KDE density snapshots  (true vs MVAR vs LSTM)
# ============================================================================

def plot_kde_snapshots_group(group_exps, group_label, ic_name,
                              ic_maps, output_dir, n_frames=4):
    """Density snapshot grid: each experiment row shows True | MVAR | LSTM
    at n_frames evenly-spaced forecast times."""
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
    n_col = n_frames * 3  # true, mvar, lstm columns

    fig, axes = plt.subplots(n_exp, n_col,
                             figsize=(2.5 * n_col, 2.2 * n_exp),
                             squeeze=False)

    for ei, (exp_name, exp_dir, tidx) in enumerate(valid):
        dt = np.load(Path(exp_dir) / 'test' / f'test_{tidx:03d}' / 'density_true.npz')
        rho_true_full = dt['rho']
        times_full = dt['times']
        T_total = rho_true_full.shape[0]
        # Sample frames from the latter half (forecast region)
        start_f = T_total // 4
        frame_idxs = np.linspace(start_f, T_total - 1, n_frames, dtype=int)

        preds = {}
        for model in ['mvar', 'lstm']:
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

            # MVAR prediction
            ax = axes[ei, n_frames + fi]
            if 'mvar' in preds:
                rho_p, times_p = preds['mvar']
                p_idx = np.argmin(np.abs(times_p - t_val))
                if p_idx < len(rho_p):
                    ax.imshow(rho_p[p_idx], origin='lower', cmap='hot', aspect='auto')
            ax.axis('off')
            if ei == 0:
                ax.set_title(f'MVAR\nt={t_val:.1f}s', fontsize=7)

            # LSTM prediction
            ax = axes[ei, 2 * n_frames + fi]
            if 'lstm' in preds:
                rho_p, times_p = preds['lstm']
                p_idx = np.argmin(np.abs(times_p - t_val))
                if p_idx < len(rho_p):
                    ax.imshow(rho_p[p_idx], origin='lower', cmap='hot', aspect='auto')
            ax.axis('off')
            if ei == 0:
                ax.set_title(f'LSTM\nt={t_val:.1f}s', fontsize=7)

    fig.suptitle(f'{group_label} \u2014 Density Snapshots ({IC_DISPLAY[ic_name]} IC)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, output_dir, 'kde_snapshots')


# ============================================================================
# PLOT 6: Spatial order parameter  per IC x per group
# ============================================================================

def plot_spatial_order_group(group_exps, group_label, ic_name,
                              ic_maps, output_dir):
    """Spatial order (std rho) vs time -- solid=true, dashed=predicted."""
    n = len(group_exps)
    if n == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    for mi, model in enumerate(['mvar', 'lstm']):
        ax = axes[mi]
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
        ax.set_title(f'{model.upper()} (solid=true, dashed=pred)')
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f'{group_label} \u2014 Spatial Order ({IC_DISPLAY[ic_name]} IC)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig, output_dir, 'spatial_order')


# ============================================================================
# PLOT 7: Phase dynamics  (cross-IC)
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

        # (a) trajectory
        mean_s = shifts_3d.mean(axis=0)
        ax_traj.plot(mean_s[:, 1], mean_s[:, 0], '-', color=color,
                     linewidth=1.5, label=label, alpha=0.8)

        # (b) PSD
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

        # (c) AR predictability
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

        # (d) phase variance
        ax_var.bar(idx, np.var(shifts_3d, axis=0).mean(),
                   color=color, alpha=0.7, label=label)

        # (e) autocorrelation
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

    fig.suptitle('Phase Dynamics Analysis', fontsize=14, fontweight='bold')
    _save_fig(fig, output_dir, 'phase_dynamics_analysis')
    print(f"  Saved: phase_dynamics_analysis.pdf")


# ============================================================================
# PLOT 8: Runtime comparison  (cross-IC)
# ============================================================================

def plot_runtime_comparison(experiments, output_dir):
    """Bar chart of training time, inference speed, parameter count."""
    records = []
    for exp_name, exp_dir in experiments.items():
        for model in ['mvar', 'lstm']:
            profile_path = exp_dir / model.upper() / 'runtime_profile.json'
            if not profile_path.exists():
                continue
            with open(profile_path) as f:
                prof = json.load(f)
            records.append({
                'experiment': _short(exp_name),
                'model': model.upper(),
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
        ('training_s', 'Training Time (s)'),
        ('inference_us', 'Inference (\u00b5s / step)'),
        ('params', 'Parameters'),
    ]
    for ax, (col, ylabel) in zip(axes, metrics):
        pivot = df.pivot_table(index='experiment', columns='model',
                               values=col, aggfunc='mean')
        pivot.plot.bar(ax=ax, rot=45, alpha=0.8)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    fig.suptitle('Runtime Comparison Across Experiments',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig, output_dir, 'runtime_comparison')
    print(f"  Saved: runtime_comparison.pdf")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='IC-Stratified Cross-Experiment Analysis Pipeline')
    parser.add_argument('--data_dir', type=str, default='oscar_output',
                        help='Base directory with experiment outputs')
    parser.add_argument('--output_dir', type=str, default='Analyses',
                        help='Output directory for all analysis plots')
    parser.add_argument('--experiments', nargs='*', default=None,
                        help='Only include these experiment names')
    parser.add_argument('--ics', nargs='*', default=None,
                        help='IC types to analyze (default: all 4)')
    parser.add_argument('--groups', nargs='*', default=None,
                        help='Regime group keys (default: all 8)')
    parser.add_argument('--skip_kde', action='store_true',
                        help='Skip KDE snapshot grids (large figures)')
    parser.add_argument('--skip_phase', action='store_true',
                        help='Skip phase dynamics (slow on many experiments)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = _time.time()

    print("=" * 80)
    print("IC-STRATIFIED CROSS-EXPERIMENT ANALYSIS PIPELINE")
    print("=" * 80)

    # ---- Discover experiments ----
    experiments = discover_experiments(data_dir, args.experiments)
    if not experiments:
        print(f"\n  No valid experiments found in {data_dir}/")
        return
    print(f"\nFound {len(experiments)} experiments in {data_dir}/")

    # ---- Build IC -> test_idx mapping per experiment ----
    ic_maps = {}
    for exp_name, exp_dir in experiments.items():
        ic_maps[exp_name] = get_ic_test_idx_map(exp_dir)

    # ---- Filter ICs and groups ----
    ic_types = args.ics if args.ics else IC_NAMES
    active_groups = REGIME_GROUPS
    if args.groups:
        active_groups = [g for g in REGIME_GROUPS if g['key'] in args.groups]
    print(f"IC types:  {', '.join(IC_DISPLAY.get(ic, ic) for ic in ic_types)}")
    print(f"Groups:    {len(active_groups)}")
    print(f"Output:    {output_dir}/")

    # ==================================================================
    # Cross-IC plots  (SVD, phase dynamics, runtime)
    # ==================================================================
    cross_dir = output_dir / 'cross_ic'
    cross_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("1. SVD Spectra & Cumulative Energy (cross-IC)")
    print("=" * 60)
    plot_svd_spectra(experiments, cross_dir)

    if not args.skip_phase:
        print(f"\n{'=' * 60}")
        print("2. Phase Dynamics Analysis (cross-IC)")
        print("=" * 60)
        plot_phase_dynamics(experiments, cross_dir)

    print(f"\n{'=' * 60}")
    print("3. Runtime Comparison (cross-IC)")
    print("=" * 60)
    plot_runtime_comparison(experiments, cross_dir)

    # ==================================================================
    # Per-IC  x  Per-Group plots
    # ==================================================================
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

            # Resolve to existing experiments only
            group_exps = [
                (m, experiments[m]) for m in gmembers if m in experiments
            ]
            if not group_exps:
                continue

            gdir = ic_dir / f'group_{gkey}'
            gdir.mkdir(parents=True, exist_ok=True)

            print(f"\n  {glabel}  ({len(group_exps)} experiments)")

            # R^2 degradation
            print(f"    \u2022 R\u00b2 degradation \u2026")
            plot_r2_degradation_group(group_exps, glabel, ic_name,
                                     ic_maps, gdir)
            total_generated += 1

            # Normalized R^2
            print(f"    \u2022 Normalized R\u00b2 \u2026")
            plot_normalized_r2_group(group_exps, glabel, ic_name,
                                    ic_maps, gdir)
            total_generated += 1

            # Error norms (3 figures)
            print(f"    \u2022 Error norms (L1, L2, Linf) \u2026")
            plot_error_norms_group(group_exps, glabel, ic_name,
                                  ic_maps, gdir)
            total_generated += 3

            # Mass conservation
            print(f"    \u2022 Mass conservation \u2026")
            plot_mass_conservation_group(group_exps, glabel, ic_name,
                                        ic_maps, gdir)
            total_generated += 1

            # KDE snapshots
            if not args.skip_kde:
                print(f"    \u2022 KDE density snapshots \u2026")
                plot_kde_snapshots_group(group_exps, glabel, ic_name,
                                        ic_maps, gdir)
                total_generated += 1

            # Spatial order
            print(f"    \u2022 Spatial order \u2026")
            plot_spatial_order_group(group_exps, glabel, ic_name,
                                    ic_maps, gdir)
            total_generated += 1

    # ==================================================================
    # Summary
    # ==================================================================
    elapsed = _time.time() - t0
    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\n  Output directory:  {output_dir}/")
    print(f"  Cross-IC plots:    {output_dir}/cross_ic/")
    for ic in ic_types:
        n_grp = sum(
            1 for g in active_groups
            if any(m in experiments for m in g['members'])
        )
        print(f"  {IC_DISPLAY[ic]:15s}  -> {output_dir}/IC_{ic}/  "
              f"({n_grp} groups)")
    print(f"\n  Figures generated:   ~{total_generated}")
    print(f"  Wall-clock time:     {elapsed / 60:.1f} min")
    print("=" * 80)


if __name__ == '__main__':
    main()
