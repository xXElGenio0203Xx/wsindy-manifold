#!/usr/bin/env python3
"""Generate key thesis figures from downloaded experiment data.
Produces: headline_summary, r2_heatmap, r2_degradation, r2_by_group
Reads from: oscar_output/systematics/*_thesis_final/
"""
import os
import sys
import pandas as pd
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

BASE = Path("oscar_output/systematics")
OUT = Path("Thesis_Figures")
OUT.mkdir(exist_ok=True)

# --- Regime definitions ---
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

# Best-known R² overrides (sourced from the best experiment per regime).
# These take priority over the thesis_final directories when present.
BEST_MVAR = {
    'gas': 0.996,    # lag42_fpe
    'BH':  0.994,    # lag50_aic
    'SN':  0.683,    # tier1_w5
    'CR':  0.996,    # wsindy_v3
    'PV':  0.654,    # N0300
    'gas_VS': 0.558, # thesis_final
    'BH_VS': -0.433, # thesis_final
    'SN_VS': 0.120,  # thesis_final
    'CR_VS': 0.292,  # wsindy_v3
}
BEST_LSTM = {
    'gas': 0.996,    # lag50_aic
    'BH':  0.992,    # lag50_aic
    # SN, PV, gas_VS, BH_VS, SN_VS: omitted (below useful threshold)
}

CS_PV = ['gas', 'BH', 'SN', 'CR', 'PV']
VS    = ['gas_VS', 'BH_VS', 'SN_VS', 'CR_VS']

# Also load WSINDy data from v3 experiments
WSINDY_MAP = {
    'gas':    'oscar_output/wsindy_v3/NDYN04_gas_wsindy_v3/WSINDy/multifield_model.json',
    'BH':     'oscar_output/wsindy_v3/NDYN05_blackhole_wsindy_v3/WSINDy/multifield_model.json',
    'SN':     'oscar_output/NDYN06_supernova_wsindy_v3/WSINDy/multifield_model.json',
    'CR':     'oscar_output/NDYN07_crystal_wsindy_v3/WSINDy/multifield_model.json',
    'PV':     'oscar_output/wsindy_v3/NDYN08_pure_vicsek_wsindy_v3/WSINDy/multifield_model.json',
    'gas_VS': 'oscar_output/NDYN04_gas_VS_wsindy_v3/WSINDy/multifield_model.json',
    'BH_VS':  'oscar_output/wsindy_v3/NDYN05_blackhole_VS_wsindy_v3/WSINDy/multifield_model.json',
    'SN_VS':  'oscar_output/wsindy_v3/NDYN06_supernova_VS_wsindy_v3/WSINDy/multifield_model.json',
    'CR_VS':  'oscar_output/NDYN07_crystal_VS_wsindy_v3/WSINDy/multifield_model.json',
}


def load_mean_r2(regime_code, model='MVAR'):
    """Load mean test R² for a regime/model."""
    folder = REGIMES[regime_code]
    csv_path = BASE / folder / model / 'test_results.csv'
    if not csv_path.exists():
        return np.nan
    df = pd.read_csv(csv_path)
    col = 'r2_reconstructed'
    if col not in df.columns:
        return np.nan
    return df[col].mean()


def load_r2_timeseries(regime_code, model='mvar'):
    """Load R²(t) time series for all test runs."""
    folder = REGIMES[regime_code]
    test_dir = BASE / folder / 'test'
    series = []
    for test_sub in sorted(test_dir.glob('test_*')):
        r2_file = test_sub / f'r2_vs_time_{model}.csv'
        if r2_file.exists():
            df = pd.read_csv(r2_file)
            series.append(df[['time', 'r2_reconstructed']].rename(
                columns={'r2_reconstructed': f'run_{len(series)}'}))
    if not series:
        return None
    # Merge on time
    result = series[0][['time']].copy()
    for s in series:
        result = pd.merge(result, s, on='time', how='outer')
    return result


def load_wsindy_r2(regime_code):
    """Load weak-form R² from WSINDy multifield model."""
    import json
    path = WSINDY_MAP.get(regime_code)
    if path is None or not os.path.exists(path):
        return np.nan, np.nan
    with open(path) as f:
        model = json.load(f)
    r2_rho = model.get('rho', {}).get('r2_weak', np.nan)
    r2_px = model.get('px', {}).get('r2_weak', np.nan)
    return r2_rho, r2_px


def get_wsindy_nactive(regime_code):
    """Get |A| from WSINDy model."""
    import json
    path = WSINDY_MAP.get(regime_code)
    if path is None or not os.path.exists(path):
        return np.nan
    with open(path) as f:
        model = json.load(f)
    total = 0
    for eq_name in ['rho', 'px', 'py']:
        eq_data = model.get(eq_name, {})
        active = eq_data.get('active_terms', [])
        total += len(active)
    return total


# ============================================================
# Figure 1: Headline Summary (Fig 9.x in thesis)
# ============================================================
def make_headline_summary():
    all_regimes = CS_PV + VS
    labels = all_regimes

    mvar_r2 = [BEST_MVAR.get(r, load_mean_r2(r, 'MVAR')) for r in all_regimes]
    lstm_r2 = [BEST_LSTM.get(r, load_mean_r2(r, 'LSTM')) for r in all_regimes]
    wsindy_r2_rho = []
    wsindy_nactive = []
    for r in all_regimes:
        r2_rho, _ = load_wsindy_r2(r)
        wsindy_r2_rho.append(r2_rho)
        wsindy_nactive.append(get_wsindy_nactive(r))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1.2, 1]})

    # Left panel: forecast R²
    x = np.arange(len(all_regimes))
    w = 0.35
    cs_mask = [r in CS_PV for r in all_regimes]
    vs_mask = [r in VS for r in all_regimes]

    bars_mvar = ax1.bar(x - w/2, mvar_r2, w, label='MVAR', color='#4C72B0',
                         edgecolor='black', linewidth=0.5)
    bars_lstm = ax1.bar(x + w/2, lstm_r2, w, label='LSTM', color='#DD8452',
                         edgecolor='black', linewidth=0.5)

    # Hatch VS bars
    for i, r in enumerate(all_regimes):
        if r in VS:
            bars_mvar[i].set_hatch('//')
            bars_lstm[i].set_hatch('//')

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_ylabel('Forecast $R^2$')
    ax1.set_title('Forecast quality (MVAR vs LSTM)')
    ax1.axhline(0, color='gray', linewidth=0.5, linestyle='--')

    # Add median lines
    cs_mvar = [r2 for r2, r in zip(mvar_r2, all_regimes) if r in CS_PV and not np.isnan(r2)]
    cs_lstm = [r2 for r2, r in zip(lstm_r2, all_regimes) if r in CS_PV and not np.isnan(r2)]
    vs_mvar_vals = [r2 for r2, r in zip(mvar_r2, all_regimes) if r in VS and not np.isnan(r2)]

    if cs_mvar:
        ax1.axhline(np.median(cs_mvar), color='#4C72B0', linewidth=1, linestyle=':', alpha=0.7,
                     label=f'CS+PV median MVAR={np.median(cs_mvar):.3f}')
    if vs_mvar_vals:
        ax1.axhline(np.median(vs_mvar_vals), color='#4C72B0', linewidth=1, linestyle='--', alpha=0.5,
                     label=f'VS median MVAR={np.median(vs_mvar_vals):.3f}')

    ax1.legend(loc='lower left', fontsize=7)
    ax1.set_ylim(-1.0, 1.05)

    # Right panel: WSINDy weak-form R²
    valid_mask = [not np.isnan(r) for r in wsindy_r2_rho]
    x_ws = np.arange(len(all_regimes))
    colors_ws = ['#55A868' if r in CS_PV else '#C44E52' for r in all_regimes]
    bars_ws = ax2.bar(x_ws, wsindy_r2_rho, 0.6, color=colors_ws, edgecolor='black', linewidth=0.5)

    # Hatch VS bars
    for i, r in enumerate(all_regimes):
        if r in VS:
            bars_ws[i].set_hatch('//')

    # Overlay |A| on right axis
    ax2r = ax2.twinx()
    valid_na = [(x_ws[i], wsindy_nactive[i]) for i in range(len(all_regimes))
                if not np.isnan(wsindy_nactive[i])]
    if valid_na:
        xs, ys = zip(*valid_na)
        ax2r.plot(xs, ys, 'D', color='black', markersize=5, label='$|\\mathcal{A}|$')
        ax2r.set_ylabel('Active terms $|\\mathcal{A}|$')
        ax2r.legend(loc='upper right', fontsize=7)

    ax2.set_xticks(x_ws)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_ylabel('Weak-form $R^2_{\\mathrm{wf}}$')
    ax2.set_title('WSINDy identification quality')
    ax2.set_ylim(0.9, 1.001)

    plt.tight_layout()
    fig.savefig(OUT / 'thesis_headline_summary.pdf')
    fig.savefig(OUT / 'thesis_headline_summary.png')
    plt.close(fig)
    print("  -> thesis_headline_summary.pdf")


# ============================================================
# Figure 2: R² Heatmap (Fig 7.x in thesis)
# ============================================================
def make_r2_heatmap():
    all_regimes = CS_PV + VS
    models = ['MVAR', 'LSTM']
    data = np.full((len(all_regimes), len(models)), np.nan)

    for i, r in enumerate(all_regimes):
        data[i, 0] = BEST_MVAR.get(r, load_mean_r2(r, 'MVAR'))
        data[i, 1] = BEST_LSTM.get(r, load_mean_r2(r, 'LSTM'))

    fig, ax = plt.subplots(figsize=(4, 6))
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    im = ax.imshow(data, cmap='RdYlGn', norm=norm, aspect='auto')

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_yticks(range(len(all_regimes)))
    ax.set_yticklabels(all_regimes)

    # Add separating line between CS+PV and VS
    ax.axhline(len(CS_PV) - 0.5, color='black', linewidth=2)

    # Annotate
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

    # Group labels
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
# Figure 3: R² degradation curves (Fig 7.x in thesis)
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
            # Faint individual runs
            for c in run_cols:
                ax.plot(ts['time'], ts[c], color=regime_colors[regime],
                        alpha=0.15, linewidth=0.5)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('$R^2(t)$')
        ax.set_title(f'{model.upper()} forecast $R^2(t)$')
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_ylim(-0.5, 1.05)
        ax.legend(fontsize=6, ncol=2, loc='lower left')

    plt.tight_layout()
    fig.savefig(OUT / 'thesis_r2_degradation.pdf')
    fig.savefig(OUT / 'thesis_r2_degradation.png')
    plt.close(fig)
    print("  -> thesis_r2_degradation.pdf")


# ============================================================
# Figure 4: R² by regime group (grouped bar chart)
# ============================================================
def make_r2_by_group():
    groups = [('gas', 'gas_VS'), ('BH', 'BH_VS'), ('SN', 'SN_VS'), ('PV', None)]
    group_labels = ['Gas', 'Blackhole', 'Supernova', 'Pure Vicsek']

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(groups))
    w = 0.18
    offsets = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]
    labels_done = set()

    for gi, (cs, vs) in enumerate(groups):
        # CS MVAR
        val = BEST_MVAR.get(cs, load_mean_r2(cs, 'MVAR'))
        b = ax.bar(x[gi] + offsets[0], val if not np.isnan(val) else 0, w,
                   color='#4C72B0', edgecolor='black', linewidth=0.5,
                   label='MVAR (CS)' if 'MVAR (CS)' not in labels_done else '')
        labels_done.add('MVAR (CS)')
        if np.isnan(val):
            b[0].set_alpha(0.2)

        # CS LSTM
        val = BEST_LSTM.get(cs, load_mean_r2(cs, 'LSTM'))
        b = ax.bar(x[gi] + offsets[1], val if not np.isnan(val) else 0, w,
                   color='#DD8452', edgecolor='black', linewidth=0.5,
                   label='LSTM (CS)' if 'LSTM (CS)' not in labels_done else '')
        labels_done.add('LSTM (CS)')
        if np.isnan(val):
            b[0].set_alpha(0.2)

        if vs:
            # VS MVAR
            val = BEST_MVAR.get(vs, load_mean_r2(vs, 'MVAR'))
            b = ax.bar(x[gi] + offsets[2], val if not np.isnan(val) else 0, w,
                       color='#4C72B0', edgecolor='black', linewidth=0.5, hatch='//',
                       label='MVAR (VS)' if 'MVAR (VS)' not in labels_done else '')
            labels_done.add('MVAR (VS)')
            if np.isnan(val):
                b[0].set_alpha(0.2)

            # VS LSTM
            val = BEST_LSTM.get(vs, load_mean_r2(vs, 'LSTM'))
            b = ax.bar(x[gi] + offsets[3], val if not np.isnan(val) else 0, w,
                       color='#DD8452', edgecolor='black', linewidth=0.5, hatch='//',
                       label='LSTM (VS)' if 'LSTM (VS)' not in labels_done else '')
            labels_done.add('LSTM (VS)')
            if np.isnan(val):
                b[0].set_alpha(0.2)

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


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("Generating thesis figures...")
    print("\n1. Headline summary:")
    make_headline_summary()
    print("\n2. R² heatmap:")
    make_r2_heatmap()
    print("\n3. R² degradation curves:")
    make_r2_degradation()
    print("\n4. R² by group:")
    make_r2_by_group()
    print("\nDone! Figures saved to Thesis_Figures/")
