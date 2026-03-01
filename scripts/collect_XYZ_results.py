#!/usr/bin/env python3
"""
Suite XYZ Results Collector & Figure Generator
===============================================

Collects results from Suites X (cross-regime), Y (mid-horizon), and Z (shift-aligned)
and generates the three thesis figures plus a LaTeX-ready results table.

Run AFTER downloading outputs from OSCAR:
    python scripts/collect_XYZ_results.py [--root /path/to/workspace]

Outputs (in oscar_output/XYZ_analysis/):
    results_table.csv          — all metrics, one row per experiment
    results_table.tex          — LaTeX booktabs table
    fig1_cross_regime_bar.png  — R² raw vs √ρ+simplex across regimes (X suite)
    fig2_knee_lite_curves.png  — R² vs horizon per regime (X+Y combined)
    fig3_phase_vs_structure.png — shift-aligned diagnostics (Z suite)
    thesis_paragraphs.txt      — fill-in-the-blank thesis text

=== OFFICIAL PIPELINE DEFINITIONS ===

  Raw baseline:
    POD on ρ  →  MVAR(p, d)  →  clamp C2  →  no mass projection
    (mass already approximately conserved in raw density space)

  Best pipeline (√ρ + simplex):
    POD on u = √ρ  →  MVAR(p, d)  →  inverse map ρ̂ = û²
    →  simplex L₂ projection to {ρ ≥ 0, Σρ = M₀}
    where M₀ = Σ ρ_true(t = forecast_start)

=== REGIME PARAMETERS ===

  V1  (control):  speed=1.5, Ca=0.8, R=2.5, η=0.2, d=19, p=5, α=1e-4
  V3.3 (moderate): speed=3.0, Ca=2.0, R=4.0, η=0.15, d=20, p=3, α=10.0
  V3.4 (extreme):  speed=5.0, Ca=3.0, R=5.0, η=0.1,  d=10, p=3, α=10.0

=== HORIZON → test_sim.T MAPPING ===

  H37:   V1 → T=5.0s (no test_sim), V3.3/V3.4 → T=6.0s (no test_sim)
  H100:  all → T=12.6s  [(100+5)*0.12]
  H162:  V1 → T=20.04s, V3.3/V3.4 → T=19.80s

  ROM_dt = 0.12s  (dt=0.04 × subsample=3)
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# EXPERIMENT REGISTRY
# ─────────────────────────────────────────────────────────────

EXPERIMENTS = {
    # Suite X: cross-regime (H37 + H162)
    'X1':  dict(name='X1_V1_raw_H37',            regime='V1',   pipeline='raw',          H=37),
    'X2':  dict(name='X2_V1_sqrtSimplex_H37',    regime='V1',   pipeline='sqrt+simplex', H=37),
    'X3':  dict(name='X3_V1_raw_H162',           regime='V1',   pipeline='raw',          H=162),
    'X4':  dict(name='X4_V1_sqrtSimplex_H162',   regime='V1',   pipeline='sqrt+simplex', H=162),
    'X5':  dict(name='X5_V33_raw_H37',           regime='V3.3', pipeline='raw',          H=37),
    'X6':  dict(name='X6_V33_sqrtSimplex_H37',   regime='V3.3', pipeline='sqrt+simplex', H=37),
    'X7':  dict(name='X7_V33_raw_H162',          regime='V3.3', pipeline='raw',          H=162),
    'X8':  dict(name='X8_V33_sqrtSimplex_H162',  regime='V3.3', pipeline='sqrt+simplex', H=162),
    'X9':  dict(name='X9_V34_raw_H37',           regime='V3.4', pipeline='raw',          H=37),
    'X10': dict(name='X10_V34_sqrtSimplex_H37',  regime='V3.4', pipeline='sqrt+simplex', H=37),
    'X11': dict(name='X11_V34_raw_H162',         regime='V3.4', pipeline='raw',          H=162),
    'X12': dict(name='X12_V34_sqrtSimplex_H162', regime='V3.4', pipeline='sqrt+simplex', H=162),
    # Suite Y: mid-horizon H100
    'Y1':  dict(name='Y1_V1_raw_H100',           regime='V1',   pipeline='raw',          H=100),
    'Y2':  dict(name='Y2_V1_sqrtSimplex_H100',   regime='V1',   pipeline='sqrt+simplex', H=100),
    'Y3':  dict(name='Y3_V33_raw_H100',          regime='V3.3', pipeline='raw',          H=100),
    'Y4':  dict(name='Y4_V33_sqrtSimplex_H100',  regime='V3.3', pipeline='sqrt+simplex', H=100),
    'Y5':  dict(name='Y5_V34_raw_H100',          regime='V3.4', pipeline='raw',          H=100),
    'Y6':  dict(name='Y6_V34_sqrtSimplex_H100',  regime='V3.4', pipeline='sqrt+simplex', H=100),
}

# Suite Z: shift-aligned (read from separate output dirs)
Z_EXPERIMENTS = {
    'Z1': dict(base='X3_V1_raw_H162',            regime='V1',   pipeline='raw'),
    'Z2': dict(base='X4_V1_sqrtSimplex_H162',    regime='V1',   pipeline='sqrt+simplex'),
    'Z3': dict(base='X7_V33_raw_H162',           regime='V3.3', pipeline='raw'),
    'Z4': dict(base='X8_V33_sqrtSimplex_H162',   regime='V3.3', pipeline='sqrt+simplex'),
    'Z5': dict(base='X11_V34_raw_H162',          regime='V3.4', pipeline='raw'),
    'Z6': dict(base='X12_V34_sqrtSimplex_H162',  regime='V3.4', pipeline='sqrt+simplex'),
}

ROM_DT = 0.12  # seconds per forecast step
REGIME_LABELS = {'V1': 'V1 (control)', 'V3.3': 'V3.3 (moderate)', 'V3.4': 'V3.4 (extreme)'}
REGIME_ORDER = ['V1', 'V3.3', 'V3.4']


# ─────────────────────────────────────────────────────────────
# DATA COLLECTION
# ─────────────────────────────────────────────────────────────

def collect_experiment(exp_name, root):
    """Read test_results.csv and summary.json for one experiment."""
    base = os.path.join(root, 'oscar_output', exp_name)
    
    # test_results.csv has per-run metrics
    csv_path = os.path.join(base, 'test', 'test_results.csv')
    if not os.path.exists(csv_path):
        return None
    
    df = pd.read_csv(csv_path)
    
    # summary.json has spectral radius under summary['mvar']['spectral_radius']
    summary_path = os.path.join(base, 'summary.json')
    spectral = np.nan
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            s = json.load(f)
        mvar = s.get('mvar', {})
        spectral = mvar.get('spectral_radius_after', mvar.get('spectral_radius', np.nan))
    
    metrics = {
        'R2_rollout':       df['r2_reconstructed'].mean(),
        'R2_rollout_std':   df['r2_reconstructed'].std(),
        'R2_1step':         df['r2_1step'].mean(),
        'R2_latent':        df['r2_latent'].mean(),
        'R2_pod':           df['r2_pod'].mean(),
        'neg_pct':          df['negativity_frac'].mean(),   # already stored as % (0-100)
        'mass_err_pct':     df['max_mass_violation'].mean() * 100,  # stored as fraction → %
        'rho_spectral':     spectral,
        'n_tests':          len(df),
    }
    return metrics


def collect_Z_experiment(z_key, root):
    """Read shift_aligned_summary.json for one Z experiment."""
    z_info = Z_EXPERIMENTS[z_key]
    
    # Find the Z output directory
    # Naming pattern from suite_Z_shift_aligned.py
    regime_map = {'V1': 'V1', 'V3.3': 'V33', 'V3.4': 'V34'}
    pipe_map = {'raw': 'raw', 'sqrt+simplex': 'sqrtsimplex'}
    r_tag = regime_map[z_info['regime']]
    p_tag = pipe_map[z_info['pipeline']]
    z_dir_name = f"{z_key}_shiftAlign_{r_tag}_{p_tag}_H162"
    
    summary_path = os.path.join(root, 'oscar_output', z_dir_name, 'shift_aligned_summary.json')
    if not os.path.exists(summary_path):
        return None
    
    with open(summary_path) as f:
        return json.load(f)


def collect_all(root):
    """Collect all X/Y experiment metrics."""
    rows = []
    for key, info in EXPERIMENTS.items():
        m = collect_experiment(info['name'], root)
        if m is None:
            print(f"  ⚠ {info['name']}: not found")
            continue
        row = {
            'suite': key[0],
            'key': key,
            'name': info['name'],
            'regime': info['regime'],
            'pipeline': info['pipeline'],
            'H': info['H'],
            'time_s': info['H'] * ROM_DT,
        }
        row.update(m)
        rows.append(row)
        print(f"  ✓ {info['name']}: R²={m['R2_rollout']:+.4f}, neg={m['neg_pct']:.1f}%, "
              f"mass_err={m['mass_err_pct']:.2f}%, ρ_spec={m['rho_spectral']:.4f}")
    
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# FIGURE 1: Cross-Regime Bar Plot (Suite X)
# ─────────────────────────────────────────────────────────────

def fig1_cross_regime_bar(df, out_dir):
    """
    For each regime × horizon: two bars (raw vs sqrt+simplex).
    X-axis: regime groups, sub-grouped by horizon.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    
    for ax_idx, H in enumerate([37, 162]):
        ax = axes[ax_idx]
        sub = df[(df['H'] == H) & (df['suite'] == 'X')]
        
        x_positions = np.arange(len(REGIME_ORDER))
        width = 0.35
        
        raw_vals = []
        sqrt_vals = []
        raw_errs = []
        sqrt_errs = []
        
        for regime in REGIME_ORDER:
            raw_row = sub[(sub['regime'] == regime) & (sub['pipeline'] == 'raw')]
            sqrt_row = sub[(sub['regime'] == regime) & (sub['pipeline'] == 'sqrt+simplex')]
            
            raw_vals.append(raw_row['R2_rollout'].values[0] if len(raw_row) else np.nan)
            sqrt_vals.append(sqrt_row['R2_rollout'].values[0] if len(sqrt_row) else np.nan)
            raw_errs.append(raw_row['R2_rollout_std'].values[0] if len(raw_row) else 0)
            sqrt_errs.append(sqrt_row['R2_rollout_std'].values[0] if len(sqrt_row) else 0)
        
        bars1 = ax.bar(x_positions - width/2, raw_vals, width, yerr=raw_errs,
                       label='Raw', color='#4C72B0', alpha=0.85, capsize=3)
        bars2 = ax.bar(x_positions + width/2, sqrt_vals, width, yerr=sqrt_errs,
                       label='√ρ + simplex', color='#DD8452', alpha=0.85, capsize=3)
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels([REGIME_LABELS[r] for r in REGIME_ORDER], fontsize=10)
        ax.set_title(f'H={H} ({H * ROM_DT:.1f}s)', fontsize=13, fontweight='bold')
        ax.set_ylabel('$R^2_{\\rm rollout}$' if ax_idx == 0 else '', fontsize=12)
        ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Annotate ΔR²
        for i in range(len(REGIME_ORDER)):
            delta = sqrt_vals[i] - raw_vals[i]
            y_pos = max(raw_vals[i], sqrt_vals[i]) + max(raw_errs[i], sqrt_errs[i]) + 0.02
            if not np.isnan(delta):
                ax.text(x_positions[i], y_pos, f'Δ={delta:+.3f}',
                        ha='center', va='bottom', fontsize=9, color='#333333')
    
    fig.suptitle('Figure 1: Cross-Regime Validation — Raw vs √ρ+Simplex',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, 'fig1_cross_regime_bar.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────
# FIGURE 2: Knee-Lite Horizon Curves (X + Y combined)
# ─────────────────────────────────────────────────────────────

def fig2_knee_lite_curves(df, out_dir):
    """
    For each regime: R² vs time with raw and sqrt+simplex curves.
    Three points: H37, H100, H162.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    
    colors_raw = {'V1': '#4C72B0', 'V3.3': '#4C72B0', 'V3.4': '#4C72B0'}
    colors_sqrt = {'V1': '#DD8452', 'V3.3': '#DD8452', 'V3.4': '#DD8452'}
    
    for ax_idx, regime in enumerate(REGIME_ORDER):
        ax = axes[ax_idx]
        sub = df[df['regime'] == regime].sort_values('H')
        
        for pipeline, color, marker, ls in [
            ('raw', '#4C72B0', 's', '--'),
            ('sqrt+simplex', '#DD8452', 'o', '-'),
        ]:
            psub = sub[sub['pipeline'] == pipeline].sort_values('H')
            if len(psub) == 0:
                continue
            ax.plot(psub['time_s'], psub['R2_rollout'], marker=marker, linewidth=2,
                    linestyle=ls, color=color, markersize=8, label=pipeline,
                    markeredgecolor='white', markeredgewidth=1)
            # Error band
            ax.fill_between(psub['time_s'],
                           psub['R2_rollout'] - psub['R2_rollout_std'],
                           psub['R2_rollout'] + psub['R2_rollout_std'],
                           alpha=0.15, color=color)
        
        ax.set_xlabel('Forecast time (s)', fontsize=11)
        ax.set_ylabel('$R^2_{\\rm rollout}$' if ax_idx == 0 else '', fontsize=12)
        ax.set_title(REGIME_LABELS[regime], fontsize=13, fontweight='bold')
        ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
        ax.legend(fontsize=10, loc='lower left')
        ax.grid(alpha=0.3)
        
        # Mark horizon labels
        for _, row in psub.iterrows():
            if row['pipeline'] == 'sqrt+simplex':
                ax.annotate(f'H{int(row["H"])}', (row['time_s'], row['R2_rollout']),
                           textcoords="offset points", xytext=(8, 8),
                           fontsize=8, color='#666666')
    
    fig.suptitle('Figure 2: Horizon Decay Curves per Regime',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, 'fig2_knee_lite_curves.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────
# FIGURE 3: Phase Drift vs Structural Distortion (Z suite)
# ─────────────────────────────────────────────────────────────

def fig3_phase_vs_structure(z_data, out_dir):
    """
    Bar plot of phase_drift_pct per regime × pipeline at H162.
    Plus R²_raw vs R²_SA scatter.
    """
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: phase drift % bars
    labels = []
    pcts = []
    colors = []
    color_map = {'raw': '#4C72B0', 'sqrt+simplex': '#DD8452'}
    
    for z_key in ['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6']:
        zd = z_data.get(z_key)
        if zd is None:
            continue
        info = Z_EXPERIMENTS[z_key]
        label = f"{info['regime']}\n{info['pipeline']}"
        labels.append(label)
        pcts.append(zd['phase_drift_pct_mean'])
        colors.append(color_map[info['pipeline']])
    
    if labels:
        x = np.arange(len(labels))
        ax1.bar(x, pcts, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=9)
        ax1.set_ylabel('Phase drift %', fontsize=12)
        ax1.set_title('Phase Drift at H162', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Annotate values
        for i, pct in enumerate(pcts):
            ax1.text(i, pct + 0.3, f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Right: R² raw vs R² shift-aligned (scatter)
    for z_key in ['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6']:
        zd = z_data.get(z_key)
        if zd is None:
            continue
        info = Z_EXPERIMENTS[z_key]
        marker = 'o' if info['pipeline'] == 'sqrt+simplex' else 's'
        color = color_map[info['pipeline']]
        ax2.scatter(zd['R2_raw_mean'], zd['R2_SA_mean'],
                   marker=marker, s=120, c=color, edgecolors='white',
                   linewidth=1.5, zorder=5)
        ax2.annotate(f"{info['regime']}", (zd['R2_raw_mean'], zd['R2_SA_mean']),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)
    
    # Diagonal reference (SA = raw, i.e., no phase drift)
    lims = ax2.get_xlim()
    ax2.plot([-1, 1], [-1, 1], 'k--', alpha=0.3, linewidth=1)
    ax2.set_xlabel('$R^2_{\\rm raw}$', fontsize=12)
    ax2.set_ylabel('$R^2_{\\rm shift\\text{-}aligned}$', fontsize=12)
    ax2.set_title('Raw vs Shift-Aligned R² at H162', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#4C72B0', markersize=10, label='Raw'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#DD8452', markersize=10, label='√ρ+simplex'),
    ]
    ax2.legend(handles=legend_elements, fontsize=10)
    
    fig.suptitle('Figure 3: Transport Drift vs Structural Distortion at H162',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, 'fig3_phase_vs_structure.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────
# TABLES
# ─────────────────────────────────────────────────────────────

def generate_tables(df, z_data, out_dir):
    """Generate CSV and LaTeX results tables."""
    
    # Merge Z data into df where applicable (H162 only)
    sa_col = []
    phase_col = []
    for _, row in df.iterrows():
        sa, phase = np.nan, np.nan
        if row['H'] == 162:
            for z_key, z_info in Z_EXPERIMENTS.items():
                if z_info['regime'] == row['regime'] and z_info['pipeline'] == row['pipeline']:
                    zd = z_data.get(z_key)
                    if zd:
                        sa = zd['R2_SA_mean']
                        phase = zd['phase_drift_pct_mean']
        sa_col.append(sa)
        phase_col.append(phase)
    
    df = df.copy()
    df['SA_R2'] = sa_col
    df['phase_drift_pct'] = phase_col
    
    # Save CSV
    cols = ['key', 'regime', 'pipeline', 'H', 'time_s',
            'R2_rollout', 'R2_rollout_std', 'R2_1step', 'SA_R2',
            'rho_spectral', 'mass_err_pct', 'neg_pct', 'phase_drift_pct', 'n_tests']
    csv_path = os.path.join(out_dir, 'results_table.csv')
    df[cols].to_csv(csv_path, index=False, float_format='%.4f')
    print(f"  → {csv_path}")
    
    # LaTeX table
    tex_path = os.path.join(out_dir, 'results_table.tex')
    with open(tex_path, 'w') as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Cross-regime validation: raw vs $\\sqrt{\\rho}$+simplex pipeline.}\n")
        f.write("\\label{tab:cross_regime}\n")
        f.write("\\begin{tabular}{llcrrrrr}\n\\toprule\n")
        f.write("Regime & Pipeline & $H$ & $R^2_{\\text{rollout}}$ & $R^2_{\\text{1-step}}$ "
                "& $R^2_{\\text{SA}}$ & mass err\\% & neg\\% \\\\\n\\midrule\n")
        
        for regime in REGIME_ORDER:
            rsub = df[df['regime'] == regime].sort_values(['H', 'pipeline'])
            for _, row in rsub.iterrows():
                sa_str = f"{row['SA_R2']:.3f}" if not np.isnan(row['SA_R2']) else "---"
                f.write(f"{regime} & {row['pipeline']} & {int(row['H'])} & "
                        f"{row['R2_rollout']:+.3f} & {row['R2_1step']:.3f} & "
                        f"{sa_str} & {row['mass_err_pct']:.1f} & {row['neg_pct']:.1f} \\\\\n")
            f.write("\\midrule\n")
        
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  → {tex_path}")
    
    return df


# ─────────────────────────────────────────────────────────────
# THESIS PARAGRAPH TEMPLATES
# ─────────────────────────────────────────────────────────────

def generate_thesis_paragraphs(df, z_data, out_dir):
    """Write fill-in-the-blank thesis paragraphs with actual numbers."""
    
    path = os.path.join(out_dir, 'thesis_paragraphs.txt')
    with open(path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("THESIS PARAGRAPH TEMPLATES — auto-generated from Suite X/Y/Z results\n")
        f.write("=" * 80 + "\n\n")
        
        # --- Section 1: Pipeline definition ---
        f.write("--- §X.X Pipeline Definition ---\n\n")
        f.write(
            "We compare two density-field ROM pipelines. The \\emph{raw} baseline applies\n"
            "proper orthogonal decomposition (POD) directly to the density field $\\rho$,\n"
            "fits a multivariate autoregressive model MVAR($p$) to the latent coefficients,\n"
            "and clamps negative predictions to zero with mass renormalization (``clamp C2'').\n"
            "The \\emph{best} pipeline instead decomposes $u = \\sqrt{\\rho}$, fits an identical\n"
            "MVAR model in the transformed latent space, maps back via $\\hat\\rho = \\hat u^2$,\n"
            "and applies a simplex $L_2$ projection~\\cite{duchi2008} to enforce\n"
            "$\\hat\\rho \\ge 0$ and $\\sum \\hat\\rho = M_0$, where $M_0 = \\sum \\rho_{\\rm true}(t_{\\rm start})$\n"
            "is the conserved total mass at forecast onset.\n\n"
        )
        
        # --- Section 2: Cross-regime results ---
        f.write("--- §X.X Cross-Regime Validation (Table/Fig 1) ---\n\n")
        
        for regime in REGIME_ORDER:
            rsub = df[df['regime'] == regime]
            for H in [37, 162]:
                raw = rsub[(rsub['pipeline'] == 'raw') & (rsub['H'] == H)]
                sqr = rsub[(rsub['pipeline'] == 'sqrt+simplex') & (rsub['H'] == H)]
                if len(raw) == 0 or len(sqr) == 0:
                    continue
                r_r2 = raw['R2_rollout'].values[0]
                s_r2 = sqr['R2_rollout'].values[0]
                delta = s_r2 - r_r2
                f.write(f"  {regime} H{H}: raw R²={r_r2:+.3f}, √ρ+S R²={s_r2:+.3f}, Δ={delta:+.3f}\n")
        
        f.write(
            "\nAcross all three regimes and both forecast horizons, the $\\sqrt{\\rho}$+simplex\n"
            "pipeline consistently [improves/matches] the raw baseline.\n"
            "At the short horizon ($H=37$, $\\approx 4.4$\\,s), the gain is\n"
            "[FILL: Δ range], while at $H=162$ ($\\approx 19$\\,s) the improvement is\n"
            "[FILL: Δ range]. The most [dramatic/modest] improvement occurs in the\n"
            "[V1/V3.3/V3.4] regime, where [FILL: interpretation].\n\n"
        )
        
        # --- Section 3: Horizon decay ---
        f.write("--- §X.X Horizon Decay (Fig 2) ---\n\n")
        f.write(
            "Figure~\\ref{fig:knee_lite} shows the rollout $R^2$ as a function of forecast\n"
            "horizon for each regime. In all three cases, both pipelines display monotonic\n"
            "decay as the horizon increases from $H=37$ to $H=162$. The $\\sqrt{\\rho}$+simplex\n"
            "curve lies [above/near] the raw curve at every point, with the gap\n"
            "[widening/narrowing/stable] at longer horizons.\n"
            "The crossover horizon (where $R^2$ drops below zero) occurs at approximately\n"
            "[FILL: H value] for raw and [FILL: H value] for $\\sqrt{\\rho}$+simplex in the\n"
            "[FILL: regime] regime.\n\n"
        )
        
        # --- Section 4: Phase vs structure ---
        f.write("--- §X.X Transport Drift vs Structural Distortion (Fig 3, Z suite) ---\n\n")
        
        for z_key in ['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6']:
            zd = z_data.get(z_key)
            if zd is None:
                f.write(f"  {z_key}: [PENDING — Z suite not yet downloaded]\n")
                continue
            info = Z_EXPERIMENTS[z_key]
            f.write(f"  {z_key} ({info['regime']} {info['pipeline']}): "
                    f"R²_raw={zd['R2_raw_mean']:+.4f}, R²_SA={zd['R2_SA_mean']:+.4f}, "
                    f"phase_drift={zd['phase_drift_pct_mean']:.1f}%\n")
        
        f.write(
            "\nAt the long horizon $H=162$, shift-aligned $R^2$ provides only a\n"
            "[modest/negligible] improvement over raw $R^2$ (Δ$R^2 \\approx$ [FILL]),\n"
            "with phase drift accounting for [FILL]\\% of the total error.\n"
            "This confirms that the dominant source of long-horizon degradation is\n"
            "[structural distortion of the density field / transport/phase drift],\n"
            "consistent across all three physics regimes.\n"
            "The $\\sqrt{\\rho}$ transform [does/does not] alter this balance, suggesting\n"
            "that the transform's benefit comes from [FILL: mechanism].\n\n"
        )
        
        f.write("=" * 80 + "\n")
        f.write("END OF TEMPLATES\n")
    
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Collect Suite X/Y/Z results and generate figures')
    parser.add_argument('--root', default=None, help='Workspace root')
    args = parser.parse_args()
    
    root = args.root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, 'oscar_output', 'XYZ_analysis')
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 70)
    print("Collecting Suite X/Y Results")
    print("=" * 70)
    df = collect_all(root)
    
    if len(df) == 0:
        print("\nNo results found! Download from OSCAR first.")
        sys.exit(1)
    
    print(f"\nCollected {len(df)} experiments")
    
    print("\n" + "=" * 70)
    print("Collecting Suite Z Results (shift-aligned)")
    print("=" * 70)
    z_data = {}
    for z_key in Z_EXPERIMENTS:
        zd = collect_Z_experiment(z_key, root)
        if zd:
            z_data[z_key] = zd
            print(f"  ✓ {z_key}: SA_R²={zd['R2_SA_mean']:+.4f}, phase={zd['phase_drift_pct_mean']:.1f}%")
        else:
            print(f"  ⚠ {z_key}: not found")
    
    print("\n" + "=" * 70)
    print("Generating Tables")
    print("=" * 70)
    df_full = generate_tables(df, z_data, out_dir)
    
    print("\n" + "=" * 70)
    print("Generating Figures")
    print("=" * 70)
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        fig1_cross_regime_bar(df, out_dir)
        fig2_knee_lite_curves(df, out_dir)
        if z_data:
            fig3_phase_vs_structure(z_data, out_dir)
        else:
            print("  ⚠ Skipping Fig 3 (no Z data yet)")
    except ImportError:
        print("  ⚠ matplotlib not available — skipping figures")
    
    print("\n" + "=" * 70)
    print("Generating Thesis Paragraphs")
    print("=" * 70)
    generate_thesis_paragraphs(df_full, z_data, out_dir)
    
    print("\n" + "=" * 70)
    print("DONE")
    print(f"All outputs in: {out_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
