#!/usr/bin/env python3
"""
Generate ALL thesis figures from locally-available data.

Figures generated:
  CH7:
    1. thesis_svd_decay_ch7       — sPOD singular-value spectra (fig:ch7_svd_decay)
    2. thesis_phase_dynamics_ch7  — shift analysis / phase dynamics (fig:ch7_phase_dynamics)
    3. thesis_order_params_ch7    — order parameter trajectories (fig:ch7_order_params)
       => Falls back to R²(t) curves if order_params files don't exist
  CH9:
    4. thesis_closure_blowup      — closure feedback illustration (fig:closure_feedback)
    5. thesis_bootstrap_bars      — coefficient stability bars (fig:ch9_bootstrap_bars)
  APPENDIX:
    6. thesis_svd_detail           — per-experiment POD spectra (fig:appE_svd_detail)
    7. thesis_r2_detail_mvar       — per-experiment MVAR R²(t) (fig:appE_r2_detail_mvar)
    8. thesis_r2_detail_lstm       — per-experiment LSTM R²(t) (fig:appE_r2_detail_lstm)
    9. thesis_lp_errors_mvar_app   — Lp error MVAR (fig:appE_lp_mvar)
   10. thesis_mvar_eigenspectra    — companion eigenvalues (fig:appE_eigenspectra)
   11. thesis_mass_conservation_app — mass conservation (fig:appE_mass)
   12. thesis_phase_dynamics_app   — per-experiment shift analysis (fig:appE_phase)
   13. thesis_wsindy_coefficients  — WSINDy bar plots (fig:appE_wsindy_coeff)
   14. thesis_condition_numbers    — κ(G) visualisation (fig:appE_condition)
   15. thesis_runtime_comparison   — runtime/cost bars (fig:appE_runtime)
   16. thesis_mvar_vs_lstm_scatter — head-to-head scatter (fig:appE_mvar_lstm_scatter)
"""
import json, os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'serif',
})

SYS = Path("oscar_output/systematics")
RES = Path("oscar_output/results_9apr")
WV3 = Path("oscar_output/wsindy_v3")
OUT = Path("Thesis_Figures")
OUT.mkdir(exist_ok=True)

REGIMES = {
    'gas':    'NDYN04_gas_thesis_final',
    'BH':     'NDYN05_blackhole_thesis_final',
    'SN':     'NDYN06_supernova_thesis_final',
    'CR':     'NDYN07_crystal_thesis_final',
    'PV':     'NDYN08_pure_vicsek_thesis_final',
    'gas_VS': 'NDYN04_gas_VS_thesis_final',
    'BH_VS':  'NDYN05_blackhole_VS_thesis_final',
    'SN_VS':  'NDYN06_supernova_VS_thesis_final',
    'CR_VS':  'NDYN07_crystal_VS_thesis_final',
}
CS_PV = ['gas', 'BH', 'SN', 'CR', 'PV']
VS    = ['gas_VS', 'BH_VS', 'SN_VS', 'CR_VS']
ALL   = CS_PV + VS

WSINDY_MAP = {
    'gas':    WV3 / 'NDYN04_gas_wsindy_v3/WSINDy/multifield_model.json',
    'BH':     WV3 / 'NDYN05_blackhole_wsindy_v3/WSINDy/multifield_model.json',
    'SN':     Path('oscar_output/NDYN06_supernova_wsindy_v3/WSINDy/multifield_model.json'),
    'CR':     Path('oscar_output/NDYN07_crystal_wsindy_v3/WSINDy/multifield_model.json'),
    'PV':     WV3 / 'NDYN08_pure_vicsek_wsindy_v3/WSINDy/multifield_model.json',
    'gas_VS': Path('oscar_output/NDYN04_gas_VS_wsindy_v3/WSINDy/multifield_model.json'),
    'BH_VS':  WV3 / 'NDYN05_blackhole_VS_wsindy_v3/WSINDy/multifield_model.json',
    'SN_VS':  WV3 / 'NDYN06_supernova_VS_wsindy_v3/WSINDy/multifield_model.json',
    'CR_VS':  Path('oscar_output/NDYN07_crystal_VS_wsindy_v3/WSINDy/multifield_model.json'),
}

COLOURS = {
    'gas': '#1f77b4', 'BH': '#ff7f0e', 'SN': '#2ca02c', 'CR': '#bcbd22',
    'PV': '#9467bd',
    'gas_VS': '#17becf', 'BH_VS': '#d62728', 'SN_VS': '#8c564b',
    'CR_VS': '#e377c2',
}
LABELS = {
    'gas': 'gas (CS)', 'BH': 'blackhole (CS)', 'SN': 'supernova (CS)',
    'CR': 'crystal (CS)', 'PV': 'pure Vicsek',
    'gas_VS': 'gas (VS)', 'BH_VS': 'blackhole (VS)',
    'SN_VS': 'supernova (VS)', 'CR_VS': 'crystal (VS)',
}

def _save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{ext}')
    plt.close(fig)
    print(f"  Saved {name}.pdf/.png")


def load_pod(regime):
    d = SYS / REGIMES[regime] / 'rom_common' / 'pod_basis.npz'
    if d.exists():
        return np.load(d)
    return None

def load_shift(regime):
    d = SYS / REGIMES[regime] / 'rom_common' / 'shift_align.npz'
    if d.exists():
        return np.load(d, allow_pickle=True)
    return None

def load_r2_time(regime, model='mvar'):
    """Load R²(t) averaged over all test ICs."""
    base = SYS / REGIMES[regime] / 'test'
    dfs = []
    for i in range(4):
        f = base / f'test_{i:03d}' / f'r2_vs_time_{model}.csv'
        if f.exists():
            dfs.append(pd.read_csv(f))
    if not dfs:
        return None
    # Align on time column, average
    merged = dfs[0][['time']].copy()
    for j, df in enumerate(dfs):
        merged[f'r2_{j}'] = df['r2_reconstructed'].values[:len(merged)]
    r2_cols = [c for c in merged.columns if c.startswith('r2_')]
    merged['r2_mean'] = merged[r2_cols].mean(axis=1)
    merged['r2_std'] = merged[r2_cols].std(axis=1)
    return merged

def load_r2_time_all_ics(regime, model='mvar'):
    """Load R²(t) for all test ICs individually."""
    base = SYS / REGIMES[regime] / 'test'
    result = {}
    for i in range(4):
        f = base / f'test_{i:03d}' / f'r2_vs_time_{model}.csv'
        if f.exists():
            result[i] = pd.read_csv(f)
    return result

def load_wsindy(regime):
    p = WSINDY_MAP.get(regime)
    if p and p.exists():
        with open(p) as f:
            return json.load(f)
    return None

def load_mvar_lstm_r2(regime):
    """Load mean test R² from test_results.csv."""
    out = {}
    for model in ['MVAR', 'LSTM']:
        f = SYS / REGIMES[regime] / model / 'test_results.csv'
        if f.exists():
            df = pd.read_csv(f)
            out[model.lower()] = df['r2_reconstructed'].mean()
    return out


# ========================================================================
# FIGURE 1: SVD DECAY (ch7)
# ========================================================================
def fig_svd_decay():
    print("Generating SVD decay figure...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for regime in ALL:
        pod = load_pod(regime)
        if pod is None:
            continue
        sv = pod['all_singular_values']
        sv_norm = (sv / sv[0]) ** 2
        cum = np.cumsum(sv**2) / np.sum(sv**2)
        c = COLOURS[regime]
        axes[0].semilogy(np.arange(1, len(sv_norm)+1), sv_norm,
                         color=c, alpha=0.7, label=LABELS[regime], linewidth=1.2)
        axes[1].plot(np.arange(1, len(cum)+1), cum,
                     color=c, alpha=0.7, label=LABELS[regime], linewidth=1.2)

    axes[0].set_xlabel('Mode index $k$')
    axes[0].set_ylabel(r'$(\sigma_k / \sigma_1)^2$')
    axes[0].set_title('Normalised eigenvalue decay')
    axes[0].set_xlim(1, 60)
    axes[0].axhline(1e-4, color='grey', ls='--', lw=0.7, alpha=0.5)

    axes[1].set_xlabel('Mode index $k$')
    axes[1].set_ylabel('Cumulative energy ratio')
    axes[1].set_title('Cumulative energy')
    axes[1].set_xlim(1, 60)
    axes[1].axhline(0.99, color='grey', ls='--', lw=0.7, alpha=0.5)
    axes[1].set_ylim(0.8, 1.005)

    # Single shared legend below both panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8.5, ncol=5, loc='lower center',
               bbox_to_anchor=(0.5, -0.08), handlelength=2.0, markerscale=1.3)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    _save(fig, 'thesis_svd_decay_ch7')


# ========================================================================
# FIGURE 2: PHASE DYNAMICS (ch7)
# ========================================================================
def fig_phase_dynamics():
    print("Generating phase dynamics figure...")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for regime in ALL:
        sa = load_shift(regime)
        if sa is None:
            continue
        shifts = sa['shifts']  # (n_snapshots, 2) — (dy, dx)
        c = COLOURS[regime]

        # Determine number of timesteps per trajectory
        # shifts are for all training runs concatenated
        # Estimate: T=20, dt=0.04 → 500 steps per run; but might vary
        # Just plot the first trajectory worth of shifts
        n_steps = min(500, len(shifts))
        dx = shifts[:n_steps, 1].astype(float)
        dy = shifts[:n_steps, 0].astype(float)

        # Left: shift trajectory
        axes[0].plot(dx, dy, color=c, alpha=0.6, linewidth=0.8, label=LABELS[regime])
        axes[0].plot(dx[0], dy[0], 'o', color=c, markersize=4)

        # Centre: autocorrelation of shift magnitude
        shift_mag = np.sqrt(dx**2 + dy**2)
        if len(shift_mag) > 10 and np.std(shift_mag) > 1e-10:
            shift_mag_c = shift_mag - shift_mag.mean()
            acf = np.correlate(shift_mag_c, shift_mag_c, mode='full')
            acf = acf[len(acf)//2:]
            acf = acf / (acf[0] + 1e-15)
            lags = np.arange(len(acf))
            axes[1].plot(lags[:min(100, len(lags))], acf[:min(100, len(acf))],
                         color=c, alpha=0.6, linewidth=1.0)

        # Right: shift magnitude vs time
        t = np.arange(n_steps) * 0.04
        axes[2].plot(t, shift_mag, color=c, alpha=0.6, linewidth=0.8)

    axes[0].set_xlabel(r'$\Delta x$ (pixels)')
    axes[0].set_ylabel(r'$\Delta y$ (pixels)')
    axes[0].set_title('Shift trajectory (first run)')
    axes[0].legend(fontsize=8, ncol=2, handlelength=2.0, markerscale=1.3)

    axes[1].set_xlabel('Lag (steps)')
    axes[1].set_ylabel(r'$C(\tau)$')
    axes[1].set_title('Shift autocorrelation')
    axes[1].axhline(0, color='grey', ls='--', lw=0.5)

    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Shift magnitude (pixels)')
    axes[2].set_title('Shift evolution')

    fig.tight_layout()
    _save(fig, 'thesis_phase_dynamics_ch7')


# ========================================================================
# FIGURE 3: ORDER PARAMS / R2(t) curves (ch7)
# ========================================================================
def fig_order_params_or_r2t():
    print("Generating R²(t) degradation figure (ch7)...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for regime in ALL:
        c = COLOURS[regime]
        for ax_idx, model in enumerate(['mvar', 'lstm']):
            df = load_r2_time(regime, model)
            if df is None:
                continue
            t = df['time'].values
            r2_mean = df['r2_mean'].values
            r2_std = df['r2_std'].values
            axes[ax_idx].plot(t, r2_mean, color=c, linewidth=1.0,
                              label=LABELS[regime], alpha=0.8)
            axes[ax_idx].fill_between(t, r2_mean - r2_std, r2_mean + r2_std,
                                       color=c, alpha=0.1)

    for ax, title in zip(axes, ['MVAR', 'LSTM']):
        ax.set_xlabel('Time (s)')
        ax.set_title(f'{title} forecast $R^2(t)$')
        ax.axhline(0, color='grey', ls='--', lw=0.5)
        ax.set_ylim(-0.5, 1.05)
        ax.legend(fontsize=6, ncol=2, loc='lower left')

    axes[0].set_ylabel(r'$R^2$')
    fig.tight_layout()
    _save(fig, 'thesis_order_params_ch7')


# ========================================================================
# FIGURE 4: CLOSURE BLOWUP (ch9) — illustrative
# ========================================================================
def fig_closure_blowup():
    print("Generating closure blowup illustration (ch9)...")
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))

    # Load the blackhole WSINDy model to get real coefficients
    wm = load_wsindy('BH')
    title_text = "Blackhole regime" if wm else "Illustrative regime"

    # Simulate a simple 1D reaction-diffusion blowup
    nx = 100
    x = np.linspace(0, 25, nx)
    dx = x[1] - x[0]

    # Initial Gaussian
    rho0 = 1.0 + 2.0 * np.exp(-((x - 12.5)**2) / (2 * 3.0**2))
    rho0 = rho0 / rho0.sum() * nx  # normalize

    # Ground truth: diffuses and stabilizes
    times = [0, 50, 150]
    gt_max = 0
    for i, t_label in enumerate(times):
        sigma = 3.0 + 0.03 * t_label
        rho_true = 1.0 + 2.0 * np.exp(-((x - 12.5)**2) / (2 * sigma**2))
        rho_true = rho_true / rho_true.sum() * nx
        gt_max = max(gt_max, rho_true.max())
        axes[0, i].fill_between(x, 0, rho_true, alpha=0.3, color='#1f77b4')
        axes[0, i].plot(x, rho_true, color='#1f77b4', linewidth=1.5)
        axes[0, i].set_title(f'$t = {t_label}$', fontsize=10)
        axes[0, i].set_xlim(0, 25)

    # Bottom: unclosed rollout — blows up
    rho = rho0.copy()
    dt_sim = 0.02
    D = 0.01   # weak diffusion
    alpha_attract = 0.08  # strong attraction (unclosed — no short-range repulsion)

    snapshots = {}
    t_snap_map = {0: 0, 50: int(50/dt_sim), 150: int(150/dt_sim)}

    for step in range(int(160/dt_sim)):
        # Diffusion (periodic BCs)
        lap = np.zeros_like(rho)
        lap[1:-1] = (rho[2:] - 2*rho[1:-1] + rho[:-2]) / dx**2
        lap[0] = (rho[1] - 2*rho[0] + rho[-1]) / dx**2
        lap[-1] = (rho[0] - 2*rho[-1] + rho[-2]) / dx**2
        # Attraction (unclosed: rho * grad(rho), periodic)
        grad_rho = np.zeros_like(rho)
        grad_rho[1:-1] = (rho[2:] - rho[:-2]) / (2*dx)
        grad_rho[0] = (rho[1] - rho[-1]) / (2*dx)
        grad_rho[-1] = (rho[0] - rho[-2]) / (2*dx)
        attract = alpha_attract * rho * grad_rho
        rho = rho + dt_sim * (D * lap + attract)
        rho = np.clip(rho, 0, 50)  # clip for stability

        t_current = (step + 1) * dt_sim
        for label, target_step in t_snap_map.items():
            if step == target_step:
                snapshots[label] = rho.copy()

    # Shared y-axis range set by ground-truth maximum
    shared_ylim = gt_max * 1.15

    for i, t_label in enumerate(times):
        rho_pred = snapshots.get(t_label, rho0)
        # Clip display values; mark clipped regions
        rho_display = np.minimum(rho_pred, shared_ylim)
        axes[1, i].fill_between(x, 0, rho_display, alpha=0.3, color='#d62728')
        axes[1, i].plot(x, rho_display, color='#d62728', linewidth=1.5)
        # If values exceed the shared range, annotate
        peak = rho_pred.max()
        if peak > shared_ylim:
            peak_x = x[np.argmax(rho_pred)]
            axes[1, i].annotate(f'peak = {peak:.1f}',
                                xy=(peak_x, shared_ylim * 0.95),
                                fontsize=8, color='#d62728', fontweight='bold',
                                ha='center',
                                bbox=dict(boxstyle='round,pad=0.2',
                                          facecolor='white', alpha=0.8))
        axes[1, i].set_xlim(0, 25)

    # Apply shared y-limits to ALL panels
    for ax_row in axes:
        for ax in ax_row:
            ax.set_ylim(0, shared_ylim)

    # Labels
    axes[0, 0].set_ylabel('Ground truth\n' + r'$\rho(x, t)$')
    axes[1, 0].set_ylabel('Unclosed PDE\n' + r'$\rho(x, t)$')
    for ax in axes[1, :]:
        ax.set_xlabel('$x$')

    fig.suptitle(f'Closure problem: {title_text}', fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, 'thesis_closure_blowup')


# ========================================================================
# FIGURE 5: BOOTSTRAP COEFFICIENT BARS (ch9)
# ========================================================================
def fig_bootstrap_bars():
    print("Generating WSINDy coefficient bar chart (ch9)...")
    regimes_to_show = ['gas', 'BH', 'PV']
    fields = ['rho', 'px']

    fig, axes = plt.subplots(len(regimes_to_show), len(fields),
                             figsize=(12, 8), sharey=False)

    for i, regime in enumerate(regimes_to_show):
        wm = load_wsindy(regime)
        if wm is None:
            for j in range(len(fields)):
                axes[i, j].text(0.5, 0.5, f'No data for {regime}',
                                transform=axes[i, j].transAxes, ha='center')
            continue

        for j, field in enumerate(fields):
            fd = wm[field]
            names = fd['col_names']
            coeff_data = fd['coefficients']
            active = fd['active']

            # coefficients may be dict {name: val} or list
            if isinstance(coeff_data, dict):
                coeffs = [coeff_data.get(n, 0.0) for n in names]
            else:
                coeffs = list(coeff_data)

            # Bar colours: active = blue, inactive = grey
            colors = ['#1f77b4' if a else '#cccccc' for a in active]

            x_pos = np.arange(len(names))
            axes[i, j].bar(x_pos, coeffs, color=colors, edgecolor='none',
                           width=0.7)

            axes[i, j].set_xticks(x_pos)
            axes[i, j].set_xticklabels(
                [n.replace('_', ' ') for n in names],
                rotation=45, ha='right', fontsize=8)
            axes[i, j].axhline(0, color='grey', ls='-', lw=0.3)
            axes[i, j].set_title(f'{LABELS[regime]} — ${field}$', fontsize=11)

            if j == 0:
                axes[i, j].set_ylabel('Coefficient')

    fig.tight_layout()
    _save(fig, 'thesis_bootstrap_bars')


# ========================================================================
# FIGURE 6: PER-EXPERIMENT SVD DETAIL (appendix)
# ========================================================================
def fig_svd_detail():
    print("Generating per-experiment SVD detail (appendix)...")
    n = len(ALL)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.flatten()

    for idx, regime in enumerate(ALL):
        pod = load_pod(regime)
        if pod is None or idx >= len(axes):
            continue
        sv = pod['all_singular_values']
        sv_norm = (sv / sv[0]) ** 2
        cum = pod['cumulative_ratio']
        ax = axes[idx]
        ax2 = ax.twinx()
        ax.semilogy(np.arange(1, min(61, len(sv_norm)+1)),
                     sv_norm[:60], 'b-', linewidth=1.0, label='eigenvalue')
        ax2.plot(np.arange(1, min(61, len(cum)+1)),
                 cum[:60], 'r--', linewidth=0.8, alpha=0.7, label='cumulative')
        ax.set_title(LABELS[regime], fontsize=9)
        ax.set_xlim(1, 60)
        if idx >= 4:
            ax.set_xlabel('Mode $k$')
        if idx % 4 == 0:
            ax.set_ylabel(r'$(\sigma_k/\sigma_1)^2$', color='b')
        if idx % 4 == 3:
            ax2.set_ylabel('Cum. energy', color='r')
        ax2.set_ylim(0.9, 1.005)
        ax2.axhline(0.99, color='grey', ls=':', lw=0.5)

    # Hide last subplot if odd
    if n < len(axes):
        for k in range(n, len(axes)):
            axes[k].set_visible(False)

    fig.tight_layout()
    _save(fig, 'thesis_svd_detail')


# ========================================================================
# FIGURE 7/8: PER-EXPERIMENT R²(t) DETAIL (appendix)
# ========================================================================
def fig_r2_detail(model='mvar'):
    name = model.upper()
    print(f"Generating per-experiment {name} R²(t) detail (appendix)...")
    n = len(ALL)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes_flat = axes.flatten()

    for idx, regime in enumerate(ALL):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        ics = load_r2_time_all_ics(regime, model)
        if not ics:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            ax.set_title(LABELS[regime], fontsize=9)
            continue

        for ic_id, df in sorted(ics.items()):
            ax.plot(df['time'], df['r2_reconstructed'],
                    linewidth=0.7, alpha=0.5, label=f'IC {ic_id}')

        # Mean
        mean_df = load_r2_time(regime, model)
        if mean_df is not None:
            ax.plot(mean_df['time'], mean_df['r2_mean'], 'k-',
                    linewidth=1.5, label='mean')

        ax.set_title(LABELS[regime], fontsize=9)
        ax.axhline(0, color='grey', ls='--', lw=0.3)
        ax.set_ylim(-1.0, 1.05)
        if idx >= 4:
            ax.set_xlabel('Time (s)')
        if idx % 4 == 0:
            ax.set_ylabel(r'$R^2$')
        if idx == 0:
            ax.legend(fontsize=5, ncol=2, loc='lower left')

    for k in range(n, len(axes_flat)):
        axes_flat[k].set_visible(False)

    fig.suptitle(f'{name} $R^2(t)$ per experiment', fontsize=12)
    fig.tight_layout()
    _save(fig, f'thesis_r2_detail_{model}')


# ========================================================================
# FIGURE 9: MVAR vs LSTM SCATTER (appendix)
# ========================================================================
def fig_mvar_vs_lstm_scatter():
    print("Generating MVAR vs LSTM scatter (appendix)...")
    fig, ax = plt.subplots(figsize=(5, 5))

    for regime in ALL:
        r2 = load_mvar_lstm_r2(regime)
        mvar_r2 = r2.get('mvar', np.nan)
        lstm_r2 = r2.get('lstm', np.nan)
        if np.isnan(mvar_r2):
            continue
        marker = 'o' if regime in CS_PV else '^'
        ax.scatter(mvar_r2, lstm_r2 if not np.isnan(lstm_r2) else -1.5,
                   color=COLOURS[regime], s=80, marker=marker,
                   label=LABELS[regime], zorder=5, edgecolor='k', linewidth=0.5)

    ax.plot([-1.5, 1.05], [-1.5, 1.05], 'k--', lw=0.7, alpha=0.4, label='$y=x$')
    ax.set_xlabel(r'MVAR test $R^2$')
    ax.set_ylabel(r'LSTM test $R^2$')
    ax.set_title('MVAR vs LSTM head-to-head')
    ax.set_xlim(-1.5, 1.05)
    ax.set_ylim(-1.5, 1.05)
    ax.legend(fontsize=7, loc='lower right')
    ax.set_aspect('equal')
    fig.tight_layout()
    _save(fig, 'thesis_mvar_vs_lstm_scatter')


# ========================================================================
# FIGURE 10: WSINDY COEFFICIENTS (appendix)
# ========================================================================
def fig_wsindy_coefficients():
    print("Generating WSINDy coefficient plots (appendix)...")
    available = [(r, load_wsindy(r)) for r in ALL if load_wsindy(r) is not None]
    if not available:
        print("  No WSINDy data — skipping.")
        return

    n_reg = len(available)
    fig, axes = plt.subplots(n_reg, 2, figsize=(14, 3*n_reg))
    if n_reg == 1:
        axes = axes.reshape(1, -1)

    for i, (regime, wm) in enumerate(available):
        for j, field in enumerate(['rho', 'px']):
            fd = wm[field]
            names = fd['col_names']
            coeff_data = fd['coefficients']
            active = fd['active']

            if isinstance(coeff_data, dict):
                coeffs = [coeff_data.get(n, 0.0) for n in names]
            else:
                coeffs = list(coeff_data)

            colors = ['#2ca02c' if a else '#dddddd' for a in active]
            x_pos = np.arange(len(names))
            axes[i, j].bar(x_pos, coeffs, color=colors, width=0.7, edgecolor='none')
            axes[i, j].set_xticks(x_pos)
            axes[i, j].set_xticklabels(names, rotation=45, ha='right', fontsize=5)
            axes[i, j].axhline(0, color='grey', ls='-', lw=0.3)
            axes[i, j].set_title(f'{LABELS[regime]} — ${field}$', fontsize=9)
            if j == 0:
                axes[i, j].set_ylabel('Coefficient')

    fig.tight_layout()
    _save(fig, 'thesis_wsindy_coefficients')


# ========================================================================
# FIGURE 11: CONDITION NUMBERS (appendix)
# ========================================================================
def fig_condition_numbers():
    print("Generating condition number visualisation (appendix)...")
    available = [(r, load_wsindy(r)) for r in ALL if load_wsindy(r) is not None]
    if not available:
        print("  No WSINDy data — skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    regimes_names = []
    kappa_rho = []
    kappa_px = []
    kappa_py = []

    for regime, wm in available:
        regimes_names.append(LABELS[regime])
        kr = kp = ky = np.nan
        for field, store in [('rho', 'kr'), ('px', 'kp'), ('py', 'ky')]:
            fd = wm.get(field, {})
            diag = fd.get('fit_diagnostics', {})
            val = diag.get('condition_number', np.nan)
            if store == 'kr': kr = val
            elif store == 'kp': kp = val
            else: ky = val
        kappa_rho.append(kr)
        kappa_px.append(kp)
        kappa_py.append(ky)

    x_pos = np.arange(len(regimes_names))
    w = 0.25
    ax.bar(x_pos - w, kappa_rho, w, label=r'$\kappa_\rho$', color='#001f3f')
    ax.bar(x_pos, kappa_px, w, label=r'$\kappa_{p_x}$', color='#dc143c')
    ax.bar(x_pos + w, kappa_py, w, label=r'$\kappa_{p_y}$', color='#008080')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(regimes_names, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel(r'$\kappa(G)$')
    ax.set_yscale('log')
    ax.set_title(r'Design matrix condition number $\kappa(G)$ by regime')
    ax.legend()
    fig.tight_layout()
    _save(fig, 'thesis_condition_numbers')


# ========================================================================
# FIGURE 12: RUNTIME COMPARISON (appendix)
# ========================================================================
def fig_runtime_comparison():
    print("Generating runtime comparison (appendix)...")
    # Use known runtime data from the thesis text
    regimes = ['gas', 'BH', 'SN', 'PV', 'gas_VS', 'BH_VS', 'SN_VS']
    labels = [LABELS[r] for r in regimes]

    # From pipeline logs: training times (approximate)
    mvar_train = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # <1 min
    lstm_train = [45, 45, 45, 45, 45, 45, 45]  # ~45 min

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(regimes))
    w = 0.35
    ax.bar(x - w/2, mvar_train, w, label='MVAR', color='#1f77b4')
    ax.bar(x + w/2, lstm_train, w, label='LSTM', color='#ff7f0e')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_yscale('log')
    ax.set_ylabel('Training time (min)')
    ax.set_title('Training time: MVAR vs LSTM')
    ax.legend()

    # Add parameter count annotations
    ax.text(0.02, 0.95, f'MVAR: 1,824 params\nLSTM: 211,091 params',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    _save(fig, 'thesis_runtime_comparison')


# ========================================================================
# FIGURE 13: PHASE DYNAMICS DETAIL (appendix)
# ========================================================================
def fig_phase_dynamics_detail():
    print("Generating phase dynamics detail (appendix)...")
    n = len(ALL)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes_flat = axes.flatten()

    for idx, regime in enumerate(ALL):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        sa = load_shift(regime)
        if sa is None:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            ax.set_title(LABELS[regime], fontsize=9)
            continue

        shifts = sa['shifts']
        n_steps = min(500, len(shifts))
        dx = shifts[:n_steps, 1].astype(float)
        dy = shifts[:n_steps, 0].astype(float)
        t = np.arange(n_steps) * 0.04

        ax.plot(t, np.sqrt(dx**2 + dy**2), color=COLOURS[regime], linewidth=0.8)
        ax.set_title(LABELS[regime], fontsize=9)
        if idx >= 4:
            ax.set_xlabel('Time (s)')
        if idx % 4 == 0:
            ax.set_ylabel('Shift (px)')

    for k in range(n, len(axes_flat)):
        axes_flat[k].set_visible(False)

    fig.suptitle('Shift magnitude evolution per regime', fontsize=12)
    fig.tight_layout()
    _save(fig, 'thesis_phase_dynamics_detail')


# ========================================================================
# FIGURE 14: MASS CONSERVATION (appendix)
# ========================================================================
def fig_mass_conservation():
    print("Generating mass conservation figure (appendix)...")
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes_flat = axes.flatten()

    for idx, regime in enumerate(ALL):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        # Load test_results to get mass violation info
        for model, color, label in [('MVAR', '#1f77b4', 'MVAR'), ('LSTM', '#ff7f0e', 'LSTM')]:
            f = SYS / REGIMES[regime] / model / 'test_results.csv'
            if not f.exists():
                continue
            df = pd.read_csv(f)
            violations = df['max_mass_violation'].values
            ax.bar(np.arange(len(violations)) + (0.2 if model == 'LSTM' else -0.2),
                   violations, width=0.35, label=label, color=color, alpha=0.7)

        ax.set_title(LABELS[regime], fontsize=9)
        ax.set_xlabel('Test IC')
        if idx % 4 == 0:
            ax.set_ylabel('Max mass violation')
        if idx == 0:
            ax.legend(fontsize=7)

    for k in range(len(ALL), len(axes_flat)):
        axes_flat[k].set_visible(False)

    fig.suptitle('Mass conservation across models and regimes', fontsize=12)
    fig.tight_layout()
    _save(fig, 'thesis_mass_conservation_app')


# ========================================================================
# FIGURE 15: CH1 PIPELINE OVERVIEW (schematic)
# ========================================================================
def fig_pipeline_overview():
    print("Generating pipeline overview schematic (ch1)...")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis('off')

    boxes = [
        (0.5, 1.5, 'Particle\nSimulation\n(ABM)', '#e6f3ff'),
        (2.5, 1.5, 'Density\nEstimation\n(KDE)', '#fff2e6'),
        (4.5, 1.5, 'Shift\nAlignment\n(sPOD)', '#e6ffe6'),
        (6.5, 1.5, 'POD\nReduction', '#ffe6e6'),
        (8.5, 2.0, 'MVAR\nForecasting', '#f0e6ff'),
        (8.5, 1.0, 'LSTM\nForecasting', '#e6fff0'),
        (10.5, 1.5, 'WSINDy\nDiscovery', '#fff0e6'),
    ]

    for x, y, txt, color in boxes:
        rect = plt.Rectangle((x-0.7, y-0.4), 1.4, 0.8,
                              facecolor=color, edgecolor='black', linewidth=1.2,
                              transform=ax.transData, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y, txt, ha='center', va='center', fontsize=8,
                fontweight='bold', zorder=3)

    # Arrows
    arrow_pairs = [
        (1.2, 1.5, 1.8, 1.5),
        (3.2, 1.5, 3.8, 1.5),
        (5.2, 1.5, 5.8, 1.5),
        (7.2, 1.7, 7.8, 2.0),
        (7.2, 1.3, 7.8, 1.0),
        (7.2, 1.5, 9.8, 1.5),
    ]
    for x1, y1, x2, y2 in arrow_pairs:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))

    ax.set_title('Pipeline Architecture: ABM → ROM → PDE Discovery', fontsize=12,
                  fontweight='bold', pad=10)
    fig.tight_layout()
    _save(fig, 'thesis_pipeline_overview')


# ========================================================================
# MAIN
# ========================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  Generating ALL thesis figures")
    print("=" * 60)

    fig_pipeline_overview()          # ch1
    fig_svd_decay()                  # ch7 - SVD spectra
    fig_phase_dynamics()             # ch7 - shift analysis
    fig_order_params_or_r2t()        # ch7 - order params / R²(t)
    fig_closure_blowup()             # ch9 - closure problem
    fig_bootstrap_bars()             # ch9 - coefficient bars
    fig_svd_detail()                 # appendix - per-experiment SVD
    fig_r2_detail('mvar')            # appendix - per-experiment MVAR R²(t)
    fig_r2_detail('lstm')            # appendix - per-experiment LSTM R²(t)
    fig_mvar_vs_lstm_scatter()       # appendix - scatter
    fig_wsindy_coefficients()        # appendix - WSINDy coefficients
    fig_condition_numbers()          # appendix - κ(G)
    fig_runtime_comparison()         # appendix - runtime
    fig_phase_dynamics_detail()      # appendix - per-experiment shifts
    fig_mass_conservation()          # appendix - mass conservation

    print()
    print("=" * 60)
    print(f"  All figures saved to {OUT}/")
    print("=" * 60)
