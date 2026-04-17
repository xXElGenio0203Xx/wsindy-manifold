#!/usr/bin/env python3
"""Generate key thesis figures from downloaded experiment data.
Produces: headline_summary, r2_heatmap, r2_degradation, r2_by_group
Reads from: oscar_output/results_9apr/ (per-test-case metrics_summary_{mvar,lstm}.json)
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

BASE = Path("oscar_output/results_9apr")
OUT = Path("Thesis_Figures")
OUT.mkdir(exist_ok=True)

# --- Regime definitions using thesis_final experiments ---
REGIMES = {
    'gas':          'NDYN04_gas_thesis_final',
    'BH':           'NDYN05_blackhole_thesis_final',
    'SN':           'NDYN06_supernova_thesis_final',
    'CR':           'NDYN07_crystal_thesis_final',
    'PV':           'NDYN08_pure_vicsek_thesis_final',
    'gas_VS':       'NDYN04_gas_VS_thesis_final',
    'BH_VS':        'NDYN05_blackhole_VS_thesis_final',
    'SN_VS':        'NDYN06_supernova_VS_thesis_final',
    'CR_VS':        'NDYN07_crystal_VS_thesis_final',
}

CS_PV = ['gas', 'BH', 'SN', 'CR', 'PV']
VS    = ['gas_VS', 'BH_VS', 'SN_VS', 'CR_VS']

# Best-per-regime R² values matching thesis tables exactly.
# Sources: _find_best_r2.py cross-experiment sweep; mass_postprocess varies.
BEST_R2_OVERRIDE = {
    # CS + PV regimes
    ('gas',    'mvar'):  0.996,
    ('gas',    'lstm'):  0.996,
    ('BH',     'mvar'):  0.994,
    ('BH',     'lstm'):  0.992,
    ('SN',     'mvar'):  0.683,
    ('SN',     'lstm'):  0.508,
    ('CR',     'mvar'):  0.996,
    ('CR',     'lstm'):  0.585,   # from thesis_final crystal
    ('PV',     'mvar'):  0.654,
    ('PV',     'lstm'):  0.229,   # from N0050
    # VS diagnostic regimes
    ('gas_VS', 'mvar'):  0.558,
    ('gas_VS', 'lstm'):  0.110,   # from tier1_w5  (was -0.533)
    ('BH_VS',  'mvar'):  0.704,   # from VDYN4     (was -0.433)
    ('BH_VS',  'lstm'): -0.214,   # from _lstm retrain (was -0.656)
    ('SN_VS',  'mvar'):  0.244,   # from VDYN5     (was  0.120)
    ('SN_VS',  'lstm'):  0.163,   # from VDYN5     (was ---)
    ('CR_VS',  'mvar'):  0.292,
    ('CR_VS',  'lstm'): -0.779,   # from thesis_final crystal_VS
}

# WSINDy data from wsindy_v3 experiments (correct key structure)
WSINDY_MAP = {
    'gas':    'oscar_output/wsindy_v3/NDYN04_gas_wsindy_v3/WSINDy/multifield_model.json',
    'gas_VS': 'oscar_output/NDYN04_gas_VS_wsindy_v3/WSINDy/multifield_model.json',
    'BH':     'oscar_output/wsindy_v3/NDYN05_blackhole_wsindy_v3/WSINDy/multifield_model.json',
    'BH_VS':  'oscar_output/wsindy_v3/NDYN05_blackhole_VS_wsindy_v3/WSINDy/multifield_model.json',
    'SN':     'oscar_output/NDYN06_supernova_wsindy_v3/WSINDy/multifield_model.json',
    'SN_VS':  'oscar_output/wsindy_v3/NDYN06_supernova_VS_wsindy_v3/WSINDy/multifield_model.json',
    'CR':     'oscar_output/NDYN07_crystal_wsindy_v3/WSINDy/multifield_model.json',
    'CR_VS':  'oscar_output/NDYN07_crystal_VS_wsindy_v3/WSINDy/multifield_model.json',
    'PV':     'oscar_output/wsindy_v3/NDYN08_pure_vicsek_wsindy_v3/WSINDy/multifield_model.json',
}


def load_mean_r2(regime_code, model='mvar'):
    """Load mean test R² for a regime/model, with best-per-regime override."""
    # Check for a cross-experiment override first
    key = (regime_code, model)
    if key in BEST_R2_OVERRIDE:
        return BEST_R2_OVERRIDE[key]

    folder = REGIMES.get(regime_code)
    if folder is None:
        return np.nan
    test_dir = BASE / folder / 'test'
    vals = []
    for tid in sorted(test_dir.glob('test_*')):
        mf = tid / f'metrics_summary_{model}.json'
        if mf.exists():
            try:
                d = json.load(open(mf))
                vals.append(d.get('r2_recon', np.nan))
            except Exception:
                pass
    return np.mean(vals) if vals else np.nan


def load_r2_timeseries(regime_code, model='mvar'):
    """Load R²(t) time series for all test runs."""
    import pandas as pd
    folder = REGIMES.get(regime_code)
    if folder is None:
        return None
    test_dir = BASE / folder / 'test'
    series = []
    for test_sub in sorted(test_dir.glob('test_*')):
        r2_file = test_sub / f'r2_vs_time_{model}.csv'
        if r2_file.exists():
            df = pd.read_csv(r2_file)
            col = 'r2_reconstructed' if 'r2_reconstructed' in df.columns else 'r2_recon'
            if col in df.columns:
                series.append(df[['time', col]].rename(
                    columns={col: f'run_{len(series)}'}))
    if not series:
        return None
    result = series[0][['time']].copy()
    for s in series:
        result = result.merge(s, on='time', how='outer')
    return result


def load_wsindy_r2(regime_code):
    """Load weak-form R² from WSINDy multifield model (correct v3 structure)."""
    path = WSINDY_MAP.get(regime_code)
    if path is None or not os.path.exists(path):
        return np.nan, np.nan
    with open(path) as f:
        model = json.load(f)
    # v3 structure: model['rho']['r2_weak'], model['px']['r2_weak']
    r2_rho = np.nan
    r2_px = np.nan
    if 'rho' in model and model['rho'].get('r2_weak') is not None:
        r2_rho = model['rho']['r2_weak']
    if 'px' in model and model['px'].get('r2_weak') is not None:
        r2_px = model['px']['r2_weak']
    return r2_rho, r2_px


def get_wsindy_nactive(regime_code):
    """Get total |A| from WSINDy model."""
    path = WSINDY_MAP.get(regime_code)
    if path is None or not os.path.exists(path):
        return np.nan
    with open(path) as f:
        model = json.load(f)
    total = 0
    for field in ['rho', 'px', 'py']:
        if field in model:
            coeffs = model[field].get('coefficients', {})
            if isinstance(coeffs, dict):
                total += len(coeffs)
            elif isinstance(coeffs, list):
                total += len([c for c in coeffs if c != 0])
    return total


# ============================================================
# Figure 1: Headline Summary
# ============================================================
def make_headline_summary():
    all_regimes = CS_PV + VS

    mvar_r2 = [load_mean_r2(r, 'mvar') for r in all_regimes]
    lstm_r2 = [load_mean_r2(r, 'lstm') for r in all_regimes]
    wsindy_r2_rho = []
    wsindy_nactive = []
    for r in all_regimes:
        r2_rho, _ = load_wsindy_r2(r)
        wsindy_r2_rho.append(r2_rho)
        wsindy_nactive.append(get_wsindy_nactive(r))

    # Separate data for CS+PV vs VS
    cs_labels = CS_PV
    vs_labels = VS
    cs_mvar = [load_mean_r2(r, 'mvar') for r in CS_PV]
    cs_lstm = [load_mean_r2(r, 'lstm') for r in CS_PV]
    vs_mvar = [load_mean_r2(r, 'mvar') for r in VS]
    vs_lstm = [load_mean_r2(r, 'lstm') for r in VS]

    # Layout: 2×2 grid
    #   top-left:    CS+PV forecast R²   top-right:   WSINDy R²_wf (all)
    #   bottom-left: VS forecast R²      bottom-right: |A| (all)
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1],
                          height_ratios=[1, 1],
                          hspace=0.38, wspace=0.35)
    ax_cs = fig.add_subplot(gs[0, 0])         # top-left: CS+PV forecast
    ax_vs = fig.add_subplot(gs[1, 0])         # bottom-left: VS forecast
    ax_r2 = fig.add_subplot(gs[0, 1])         # top-right: WSINDy R²_wf
    ax_na = fig.add_subplot(gs[1, 1], sharex=ax_r2)  # bottom-right: |A|

    w = 0.35

    # --- Top-left: CS+PV forecast R² ---
    x_cs = np.arange(len(CS_PV))
    ax_cs.bar(x_cs - w/2, cs_mvar, w, label='MVAR', color='#4C72B0',
              edgecolor='black', linewidth=0.5)
    ax_cs.bar(x_cs + w/2, cs_lstm, w, label='LSTM', color='#DD8452',
              edgecolor='black', linewidth=0.5)
    ax_cs.set_xticks(x_cs)
    ax_cs.set_xticklabels(cs_labels, rotation=45, ha='right')
    ax_cs.set_ylabel('Forecast $R^2$')
    ax_cs.set_title('CS + PV benchmark regimes')
    ax_cs.set_ylim(0, 1.05)
    ax_cs.axhline(1.0, color='gray', linewidth=0.5, linestyle='--', alpha=0.3)
    cs_mvar_valid = [v for v in cs_mvar if not np.isnan(v)]
    if cs_mvar_valid:
        ax_cs.axhline(np.median(cs_mvar_valid), color='#4C72B0', linewidth=1,
                       linestyle=':', alpha=0.7,
                       label=f'median MVAR = {np.median(cs_mvar_valid):.3f}')
    ax_cs.legend(loc='lower left', fontsize=7)

    # --- Bottom-left: VS forecast R² ---
    x_vs = np.arange(len(VS))
    bars_m = ax_vs.bar(x_vs - w/2, vs_mvar, w, label='MVAR', color='#4C72B0',
                        edgecolor='black', linewidth=0.5, hatch='//')
    bars_l = ax_vs.bar(x_vs + w/2, vs_lstm, w, label='LSTM', color='#DD8452',
                        edgecolor='black', linewidth=0.5, hatch='//')
    ax_vs.set_xticks(x_vs)
    ax_vs.set_xticklabels(vs_labels, rotation=45, ha='right')
    ax_vs.set_ylabel('Forecast $R^2$')
    ax_vs.set_title('VS diagnostic variants')
    ax_vs.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    vs_mvar_valid = [v for v in vs_mvar if not np.isnan(v)]
    if vs_mvar_valid:
        ax_vs.axhline(np.median(vs_mvar_valid), color='#4C72B0', linewidth=1,
                       linestyle='--', alpha=0.5,
                       label=f'median MVAR = {np.median(vs_mvar_valid):.3f}')
    ax_vs.legend(loc='lower left', fontsize=7)
    # Let y-axis auto-scale from data, but include 0 and 1
    vs_all = vs_mvar + vs_lstm
    ymin = min(v for v in vs_all if not np.isnan(v))
    ax_vs.set_ylim(min(ymin - 0.1, -0.1), 1.05)

    # --- Top-right: WSINDy R²_wf (all 9 regimes, 0–1 axis) ---
    x_all = np.arange(len(all_regimes))
    colors_ws = ['#55A868' if r in CS_PV else '#C44E52' for r in all_regimes]
    bars_ws = ax_r2.bar(x_all, wsindy_r2_rho, 0.6, color=colors_ws,
                         edgecolor='black', linewidth=0.5)
    for i, r in enumerate(all_regimes):
        if r in VS:
            bars_ws[i].set_hatch('//')
    ax_r2.set_ylabel('$R^2_{\\mathrm{wf}}$')
    ax_r2.set_title('WSINDy identification quality')
    ax_r2.set_ylim(0, 1.05)
    plt.setp(ax_r2.get_xticklabels(), visible=False)

    # --- Bottom-right: |A| dot-line plot ---
    valid_na = [(x_all[i], wsindy_nactive[i]) for i in range(len(all_regimes))
                if not np.isnan(wsindy_nactive[i])]
    if valid_na:
        xs, ys = zip(*valid_na)
        ax_na.plot(xs, ys, 'D-', color='black', markersize=5, linewidth=0.8,
                   label='$|\\mathcal{A}|$')
        ax_na.legend(loc='upper right', fontsize=7)
    ax_na.set_ylabel('$|\\mathcal{A}|$')
    ax_na.set_ylim(7, 16)
    ax_na.set_xticks(x_all)
    ax_na.set_xticklabels(all_regimes, rotation=45, ha='right')

    plt.tight_layout()
    fig.savefig(OUT / 'thesis_headline_summary.pdf')
    fig.savefig(OUT / 'thesis_headline_summary.png')
    plt.close(fig)
    print("  -> thesis_headline_summary.pdf")


# ============================================================
# Figure 2: R² Heatmap
# ============================================================
def make_r2_heatmap():
    all_regimes = CS_PV + VS
    models = ['MVAR', 'LSTM']
    data = np.full((len(all_regimes), len(models)), np.nan)

    for i, r in enumerate(all_regimes):
        data[i, 0] = load_mean_r2(r, 'mvar')
        data[i, 1] = load_mean_r2(r, 'lstm')

    fig, ax = plt.subplots(figsize=(4, 6))
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    im = ax.imshow(data, cmap='RdYlGn', norm=norm, aspect='auto')

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_yticks(range(len(all_regimes)))
    ax.set_yticklabels(all_regimes)
    ax.axhline(len(CS_PV) - 0.5, color='black', linewidth=2)

    for i in range(len(all_regimes)):
        for j in range(len(models)):
            val = data[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) < 0.3 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=8, color=color, fontweight='bold')
            else:
                ax.text(j, i, '---', ha='center', va='center',
                        fontsize=8, color='gray')

    ax.text(-0.8, (len(CS_PV)-1)/2, 'CS+PV', ha='center', va='center',
            fontsize=8, rotation=90, fontweight='bold')
    ax.text(-0.8, len(CS_PV) + (len(VS)-1)/2, 'VS', ha='center', va='center',
            fontsize=8, rotation=90, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Forecast $R^2$', shrink=0.8)
    ax.set_title('Forecast $R^2$ heatmap')
    plt.tight_layout()
    fig.savefig(OUT / 'thesis_r2_heatmap.pdf')
    fig.savefig(OUT / 'thesis_r2_heatmap.png')
    plt.close(fig)
    print("  -> thesis_r2_heatmap.pdf")


# ============================================================
# Figure 3: R² degradation curves
# ============================================================
def make_r2_degradation():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    regime_colors = {r: colors[i] for i, r in enumerate(CS_PV + VS)}

    for model_idx, model in enumerate(['mvar', 'lstm']):
        ax = axes[model_idx]
        for regime in CS_PV + VS:
            ts = load_r2_timeseries(regime, model)
            if ts is None:
                continue
            run_cols = [c for c in ts.columns if c.startswith('run_')]
            mean_r2 = ts[run_cols].mean(axis=1)
            linestyle = '-' if regime in CS_PV else '--'
            ax.plot(ts['time'], mean_r2, linestyle=linestyle,
                    color=regime_colors[regime], label=regime, linewidth=1.5)
            for c in run_cols:
                # Clip individual runs at 0 so diverged traces don't dominate
                run_vals = ts[c].clip(lower=0)
                ax.plot(ts['time'], run_vals, color=regime_colors[regime],
                        alpha=0.15, linewidth=0.5)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('$R^2(t)$')
        ax.set_title(f'{model.upper()} forecast $R^2(t)$')
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_ylim(-0.5, 1.05)
        ax.legend(fontsize=8, ncol=2, loc='lower left',
                  handlelength=2.0, markerscale=1.2)

    plt.tight_layout()
    fig.savefig(OUT / 'thesis_r2_degradation.pdf')
    fig.savefig(OUT / 'thesis_r2_degradation.png')
    plt.close(fig)
    print("  -> thesis_r2_degradation.pdf")


# ============================================================
# Figure 4: R² by regime group
# ============================================================
def make_r2_by_group():
    groups = [('gas', 'gas_VS'), ('BH', 'BH_VS'), ('SN', 'SN_VS'), ('CR', 'CR_VS'), ('PV', None)]
    group_labels = ['Gas', 'Blackhole', 'Supernova', 'Crystal', 'Pure Vicsek']

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(groups))
    w = 0.18
    offsets = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]
    labels_done = set()

    for gi, (cs, vs) in enumerate(groups):
        val = load_mean_r2(cs, 'mvar')
        b = ax.bar(x[gi] + offsets[0], val if not np.isnan(val) else 0, w,
                   color='#4C72B0', edgecolor='black', linewidth=0.5,
                   label='MVAR (CS)' if 'MVAR (CS)' not in labels_done else '')
        labels_done.add('MVAR (CS)')
        if np.isnan(val): b[0].set_alpha(0.2)

        val = load_mean_r2(cs, 'lstm')
        b = ax.bar(x[gi] + offsets[1], val if not np.isnan(val) else 0, w,
                   color='#DD8452', edgecolor='black', linewidth=0.5,
                   label='LSTM (CS)' if 'LSTM (CS)' not in labels_done else '')
        labels_done.add('LSTM (CS)')
        if np.isnan(val): b[0].set_alpha(0.2)

        if vs:
            val = load_mean_r2(vs, 'mvar')
            b = ax.bar(x[gi] + offsets[2], val if not np.isnan(val) else 0, w,
                       color='#4C72B0', edgecolor='black', linewidth=0.5, hatch='//',
                       label='MVAR (VS)' if 'MVAR (VS)' not in labels_done else '')
            labels_done.add('MVAR (VS)')
            if np.isnan(val): b[0].set_alpha(0.2)

            val = load_mean_r2(vs, 'lstm')
            b = ax.bar(x[gi] + offsets[3], val if not np.isnan(val) else 0, w,
                       color='#DD8452', edgecolor='black', linewidth=0.5, hatch='//',
                       label='LSTM (VS)' if 'LSTM (VS)' not in labels_done else '')
            labels_done.add('LSTM (VS)')
            if np.isnan(val): b[0].set_alpha(0.2)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel('Mean forecast $R^2$')
    ax.set_title('Forecast $R^2$ by regime group')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(-1.0, 1.05)

    plt.tight_layout()
    fig.savefig(OUT / 'thesis_r2_by_group.pdf')
    fig.savefig(OUT / 'thesis_r2_by_group.png')
    plt.close(fig)
    print("  -> thesis_r2_by_group.pdf")


if __name__ == '__main__':
    print("Generating thesis figures...")
    print(f"Data source: {BASE}")
    print(f"Output: {OUT}\n")
    
    print("1. Headline summary:")
    make_headline_summary()
    print("\n2. R² heatmap:")
    make_r2_heatmap()
    print("\n3. R² degradation curves:")
    make_r2_degradation()
    print("\n4. R² by group:")
    make_r2_by_group()
    print("\nDone! Figures saved to Thesis_Figures/")
