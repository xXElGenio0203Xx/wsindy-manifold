#!/usr/bin/env python3
"""
Cross-Suite R² Comparison Figure
=================================
Generates thesis-quality figures comparing density R², latent R², and 1-step R²
across all experiment suites: DYN1-7, LST4/7, XABL1-8, DEG1.

Outputs:
  artifacts/thesis_figures/cross_suite_r2_comparison.pdf
  artifacts/thesis_figures/density_vs_latent_scatter.pdf
  artifacts/thesis_figures/xabl_ablation_bar.pdf
"""
import csv
import statistics
import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# Style
# ============================================================================
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 9,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

OUT_DIR = Path('artifacts/thesis_figures')
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE = Path('oscar_output')

# ============================================================================
# Data Collection
# ============================================================================

def read_csv_r2(csv_path):
    """Read r2_reconstructed, r2_latent, r2_1step from a test_results.csv."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    r2d, r2l, r2s = [], [], []
    for r in rows:
        r2d.append(float(r['r2_reconstructed']))
        r2l.append(float(r['r2_latent']))
        v = r.get('r2_1step', '')
        if v and v.lower() != 'nan':
            try:
                r2s.append(float(v))
            except ValueError:
                pass
    return r2d, r2l, r2s

def stats(vals):
    m = statistics.mean(vals) if vals else float('nan')
    s = statistics.stdev(vals) if len(vals) > 1 else 0
    return m, s


# --- DYN suite ---
dyn_names = [
    ('DYN1', 'DYN1_gentle_v2', 'test'),
    ('DYN2', 'DYN2_hypervelocity_v2', 'test'),
    ('DYN3', 'DYN3_hypernoisy_v2', 'test'),
    ('DYN4', 'DYN4_blackhole_v2', 'test'),
    ('DYN5', 'DYN5_supernova', 'test'),
    ('DYN6', 'DYN6_varspeed_v2', 'test'),
    ('DYN7', 'DYN7_pure_vicsek', 'test'),
]

# --- LST suite ---
lst_names = [
    ('LST4-MVAR', 'LST4_sqrt_simplex_align_h64_L2', 'MVAR'),
    ('LST4-LSTM', 'LST4_sqrt_simplex_align_h64_L2', 'LSTM'),
    ('LST7-MVAR', 'LST7_raw_none_align_h128_L2', 'MVAR'),
    ('LST7-LSTM', 'LST7_raw_none_align_h128_L2', 'LSTM'),
]

# --- DEG1 ---
deg_names = [
    ('DEG1', 'DEG1_long_horizon_200s', 'MVAR'),
]

all_data = {}

for label, exp, sub in dyn_names + lst_names + deg_names:
    csv_path = BASE / exp / sub / 'test_results.csv'
    if not csv_path.exists():
        csv_path = BASE / exp / 'test' / 'test_results.csv'
    if not csv_path.exists():
        print(f"  SKIP {label}: {csv_path} not found")
        continue
    r2d, r2l, r2s = read_csv_r2(csv_path)
    all_data[label] = {
        'density': stats(r2d),
        'latent': stats(r2l),
        '1step': stats(r2s),
        'density_raw': r2d,
        'latent_raw': r2l,
        'suite': label.split('-')[0].rstrip('0123456789') if '-' in label else label[:3],
    }

# --- XABL (hardcoded from Oscar extraction) ---
xabl_results = {
    'XABL1': {'density': (0.9706, 0.0155), 'latent': (0.7071, 0.0932), 'label': 'raw/none/noScale'},
    'XABL2': {'density': (0.9082, 0.0323), 'latent': (0.0387, 0.0143), 'label': 'raw/none/scale'},
    'XABL3': {'density': (0.9706, 0.0155), 'latent': (0.7071, 0.0932), 'label': 'raw/simplex/noScale'},
    'XABL4': {'density': (0.9086, 0.0323), 'latent': (0.0387, 0.0143), 'label': 'raw/simplex/scale'},
    'XABL5': {'density': (0.9694, 0.0154), 'latent': (0.5821, 0.1049), 'label': 'sqrt/none/noScale'},
    'XABL6': {'density': (0.9033, 0.0309), 'latent': (0.0148, 0.0073), 'label': 'sqrt/none/scale'},
    'XABL7': {'density': (0.9704, 0.0154), 'latent': (0.5821, 0.1049), 'label': 'sqrt/simplex/noScale'},
    'XABL8': {'density': (0.9045, 0.0308), 'latent': (0.0148, 0.0073), 'label': 'sqrt/simplex/scale'},
}

# ============================================================================
# FIGURE 1: Cross-Suite R² Comparison (grouped bar chart)
# ============================================================================
print("\n[Figure 1] Cross-suite R² comparison...")

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

# --- Panel A: DYN + LST + DEG1 ---
ax = axes[0]
labels_a = ['DYN1', 'DYN2', 'DYN3', 'DYN4', 'DYN5', 'DYN6', 'DYN7',
            '', 'LST4-MVAR', 'LST4-LSTM', 'LST7-MVAR', 'LST7-LSTM',
            '', 'DEG1']
x_pos = []
pos = 0
for l in labels_a:
    if l == '':
        pos += 0.5
    else:
        x_pos.append(pos)
        pos += 1

labels_clean = [l for l in labels_a if l != '']
density_means = [all_data[l]['density'][0] if l in all_data else 0 for l in labels_clean]
density_stds = [all_data[l]['density'][1] if l in all_data else 0 for l in labels_clean]
latent_means = [all_data[l]['latent'][0] if l in all_data else 0 for l in labels_clean]
latent_stds = [all_data[l]['latent'][1] if l in all_data else 0 for l in labels_clean]

w = 0.35
x = np.array(x_pos)
bars1 = ax.bar(x - w/2, density_means, w, yerr=density_stds, label='Density $R^2$',
               color='#4C72B0', capsize=3, alpha=0.85, edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + w/2, latent_means, w, yerr=latent_stds, label='Latent $R^2$',
               color='#DD8452', capsize=3, alpha=0.85, edgecolor='white', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(labels_clean, rotation=30, ha='right')
ax.set_ylabel('$R^2$')
ax.set_title('(a) Dynamics & LSTM Suites — Density vs Latent $R^2$')
ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')
ax.set_ylim(-0.6, 1.05)
ax.legend(loc='lower left')

# Annotate the problematic ones
for i, l in enumerate(labels_clean):
    if l in all_data:
        d_m = all_data[l]['density'][0]
        l_m = all_data[l]['latent'][0]
        if abs(d_m - l_m) > 0.5:
            ax.annotate('!', (x[i], max(d_m, l_m) + 0.06), ha='center', fontsize=14, color='red', fontweight='bold')

# --- Panel B: XABL ablation ---
ax2 = axes[1]
xabl_labels = [f"XABL{i}" for i in range(1, 9)]
xabl_sublabels = [xabl_results[l]['label'] for l in xabl_labels]
xd = [xabl_results[l]['density'][0] for l in xabl_labels]
xd_err = [xabl_results[l]['density'][1] for l in xabl_labels]
xl = [xabl_results[l]['latent'][0] for l in xabl_labels]
xl_err = [xabl_results[l]['latent'][1] for l in xabl_labels]

x2 = np.arange(len(xabl_labels))
bars3 = ax2.bar(x2 - w/2, xd, w, yerr=xd_err, label='Density $R^2$',
                color='#4C72B0', capsize=3, alpha=0.85, edgecolor='white', linewidth=0.5)
bars4 = ax2.bar(x2 + w/2, xl, w, yerr=xl_err, label='Latent $R^2$',
                color='#DD8452', capsize=3, alpha=0.85, edgecolor='white', linewidth=0.5)

# Color even-numbered (scaled) bars differently
for i in [1, 3, 5, 7]:  # XABL2,4,6,8 = scaled
    bars3[i].set_facecolor('#C44E52')
    bars3[i].set_alpha(0.7)
    bars4[i].set_facecolor('#937860')
    bars4[i].set_alpha(0.7)

ax2.set_xticks(x2)
ax2.set_xticklabels([f"{xabl_labels[i]}\n{xabl_sublabels[i]}" for i in range(8)],
                     rotation=0, ha='center', fontsize=8)
ax2.set_ylabel('$R^2$')
ax2.set_title('(b) XABL Ablation Suite — Effect of Transform, Simplex, and Spectral Scaling')
ax2.set_ylim(-0.05, 1.05)
ax2.legend(loc='lower left')

# Add bracket annotations
ax2.annotate('spectral scaling ON', xy=(1, 0.03), xytext=(1, -0.03),
             fontsize=8, color='#C44E52', ha='center', style='italic')

plt.tight_layout()
fig.savefig(OUT_DIR / 'cross_suite_r2_comparison.pdf')
fig.savefig(OUT_DIR / 'cross_suite_r2_comparison.png')
print(f"  Saved: {OUT_DIR / 'cross_suite_r2_comparison.pdf'}")

# ============================================================================
# FIGURE 2: Density R² vs Latent R² scatter
# ============================================================================
print("\n[Figure 2] Density vs Latent scatter...")

fig2, ax3 = plt.subplots(figsize=(8, 7))

# Plot each test run as a point, colored by suite
suite_colors = {
    'DYN': '#4C72B0',
    'LST': '#55A868',
    'DEG': '#C44E52',
}
suite_markers = {
    'DYN': 'o',
    'LST': 's',
    'DEG': '^',
}

for label, data in all_data.items():
    suite = data['suite']
    color = suite_colors.get(suite, 'gray')
    marker = suite_markers.get(suite, 'o')
    ax3.scatter(data['density_raw'], data['latent_raw'],
                c=color, marker=marker, alpha=0.4, s=30, edgecolors='none')
    # Plot mean as larger marker with border
    dm, ds = data['density']
    lm, ls = data['latent']
    ax3.scatter([dm], [lm], c=color, marker=marker, s=120,
                edgecolors='black', linewidths=1.0, zorder=5)
    ax3.annotate(label, (dm, lm), textcoords='offset points',
                 xytext=(6, 4), fontsize=7, color=color, fontweight='bold')

# Add XABL means
for label, data in xabl_results.items():
    dm = data['density'][0]
    lm = data['latent'][0]
    is_scaled = label in ['XABL2', 'XABL4', 'XABL6', 'XABL8']
    color = '#C44E52' if is_scaled else '#8172B2'
    ax3.scatter([dm], [lm], c=color, marker='D', s=80,
                edgecolors='black', linewidths=0.8, zorder=5, alpha=0.8)
    ax3.annotate(label, (dm, lm), textcoords='offset points',
                 xytext=(5, 3), fontsize=6, color=color)

# Reference lines
ax3.plot([0, 1], [0, 1], '--', color='gray', alpha=0.4, linewidth=1, label='$R^2_{density} = R^2_{latent}$')
ax3.axhline(y=0, color='lightgray', linewidth=0.5)
ax3.axvline(x=0, color='lightgray', linewidth=0.5)

# Danger zone annotation
ax3.fill_between([0.85, 1.0], -0.5, 0.2, alpha=0.08, color='red')
ax3.text(0.92, -0.15, 'Misleading\nDensity $R^2$', ha='center', fontsize=8,
         color='red', alpha=0.6, style='italic')

ax3.set_xlabel('Density $R^2$ (reconstructed)')
ax3.set_ylabel('Latent $R^2$')
ax3.set_title('Density $R^2$ vs Latent $R^2$ — Metric Reliability')
ax3.set_xlim(-0.5, 1.05)
ax3.set_ylim(-0.55, 1.05)
ax3.legend(loc='upper left')
ax3.set_aspect('equal')

plt.tight_layout()
fig2.savefig(OUT_DIR / 'density_vs_latent_scatter.pdf')
fig2.savefig(OUT_DIR / 'density_vs_latent_scatter.png')
print(f"  Saved: {OUT_DIR / 'density_vs_latent_scatter.pdf'}")

# ============================================================================
# FIGURE 3: XABL ablation factor decomposition
# ============================================================================
print("\n[Figure 3] XABL factor decomposition...")

fig3, (ax4, ax5) = plt.subplots(1, 2, figsize=(12, 5))

# Factor effects on DENSITY R²
factors = ['sqrt transform', 'simplex postproc', 'spectral scaling']
# Effect = mean(with factor ON) - mean(with factor OFF)
# sqrt: XABL5-8 vs XABL1-4
# simplex: XABL3,4,7,8 vs XABL1,2,5,6
# scaling: XABL2,4,6,8 vs XABL1,3,5,7

def xabl_r2(label, metric='density'):
    return xabl_results[label][metric][0]

# Density effects
eff_sqrt_d = np.mean([xabl_r2(f'XABL{i}') for i in [5,6,7,8]]) - np.mean([xabl_r2(f'XABL{i}') for i in [1,2,3,4]])
eff_simplex_d = np.mean([xabl_r2(f'XABL{i}') for i in [3,4,7,8]]) - np.mean([xabl_r2(f'XABL{i}') for i in [1,2,5,6]])
eff_scaling_d = np.mean([xabl_r2(f'XABL{i}') for i in [2,4,6,8]]) - np.mean([xabl_r2(f'XABL{i}') for i in [1,3,5,7]])

# Latent effects
eff_sqrt_l = np.mean([xabl_r2(f'XABL{i}', 'latent') for i in [5,6,7,8]]) - np.mean([xabl_r2(f'XABL{i}', 'latent') for i in [1,2,3,4]])
eff_simplex_l = np.mean([xabl_r2(f'XABL{i}', 'latent') for i in [3,4,7,8]]) - np.mean([xabl_r2(f'XABL{i}', 'latent') for i in [1,2,5,6]])
eff_scaling_l = np.mean([xabl_r2(f'XABL{i}', 'latent') for i in [2,4,6,8]]) - np.mean([xabl_r2(f'XABL{i}', 'latent') for i in [1,3,5,7]])

density_effects = [eff_sqrt_d, eff_simplex_d, eff_scaling_d]
latent_effects = [eff_sqrt_l, eff_simplex_l, eff_scaling_l]

colors_d = ['#55A868' if e >= 0 else '#C44E52' for e in density_effects]
colors_l = ['#55A868' if e >= 0 else '#C44E52' for e in latent_effects]

x3 = np.arange(len(factors))
ax4.barh(x3, density_effects, color=colors_d, alpha=0.8, edgecolor='white')
ax4.set_yticks(x3)
ax4.set_yticklabels(factors)
ax4.set_xlabel('$\\Delta R^2$ (density)')
ax4.set_title('Factor Effects on Density $R^2$')
ax4.axvline(x=0, color='black', linewidth=0.8)
for i, v in enumerate(density_effects):
    ax4.text(v + 0.002 * np.sign(v), i, f'{v:+.4f}', va='center', fontsize=10)

ax5.barh(x3, latent_effects, color=colors_l, alpha=0.8, edgecolor='white')
ax5.set_yticks(x3)
ax5.set_yticklabels(factors)
ax5.set_xlabel('$\\Delta R^2$ (latent)')
ax5.set_title('Factor Effects on Latent $R^2$')
ax5.axvline(x=0, color='black', linewidth=0.8)
for i, v in enumerate(latent_effects):
    ax5.text(v + 0.02 * np.sign(v), i, f'{v:+.4f}', va='center', fontsize=10)

plt.tight_layout()
fig3.savefig(OUT_DIR / 'xabl_factor_decomposition.pdf')
fig3.savefig(OUT_DIR / 'xabl_factor_decomposition.png')
print(f"  Saved: {OUT_DIR / 'xabl_factor_decomposition.pdf'}")

# ============================================================================
# Summary Table (printed)
# ============================================================================
print("\n" + "=" * 100)
print("COMPLETE R² SUMMARY TABLE")
print("=" * 100)
print(f"{'Experiment':<22s} {'Model':<6s} {'Density R²':>14s} {'Latent R²':>14s} {'1-step R²':>12s} {'Gap':>8s} {'Status':>10s}")
print("-" * 100)

for label, exp, sub in dyn_names + lst_names + deg_names:
    if label not in all_data:
        continue
    d = all_data[label]
    dm, ds = d['density']
    lm, ls = d['latent']
    sm, ss = d['1step']
    gap = dm - lm
    # Status: OK if both high, WARN if density high but latent low
    if dm > 0.9 and lm > 0.5:
        status = "OK"
    elif dm > 0.8 and lm > 0.3:
        status = "MODERATE"
    elif dm > 0.8 and lm < 0.2:
        status = "MISLEADING"
    else:
        status = "POOR"
    model = sub if sub in ['MVAR', 'LSTM'] else 'MVAR'
    d_str = f"{dm:.4f}±{ds:.4f}"
    l_str = f"{lm:.4f}±{ls:.4f}"
    s_str = f"{sm:.4f}" if not (sm != sm) else "N/A"
    print(f'{label:<22s} {model:<6s} {d_str:>14s} {l_str:>14s} {s_str:>12s} {gap:>+8.3f} {status:>10s}')

print("-" * 100)
for label, data in sorted(xabl_results.items()):
    dm = data['density'][0]
    ds = data['density'][1]
    lm = data['latent'][0]
    ls = data['latent'][1]
    gap = dm - lm
    status = "MISLEADING" if (dm > 0.8 and lm < 0.2) else ("OK" if lm > 0.5 else "MODERATE")
    d_str = f"{dm:.4f}±{ds:.4f}"
    l_str = f"{lm:.4f}±{ls:.4f}"
    print(f'{label + " " + data["label"]:<22s} {"MVAR":<6s} {d_str:>14s} {l_str:>14s} {"N/A":>12s} {gap:>+8.3f} {status:>10s}')

print("\nDone.")
