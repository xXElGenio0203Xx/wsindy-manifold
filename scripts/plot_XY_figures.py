#!/usr/bin/env python3
"""
Quick thesis figures from hardcoded Suite X/Y results.
Run locally — no OSCAR download needed.

  python scripts/plot_XY_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import os

OUT = Path("artifacts/thesis_figures")
OUT.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────
# DATA  (26 test runs each, mean ± std of R²_rollout)
# ──────────────────────────────────────────────────────────

DATA = {
    # (regime, pipeline, H): (mean_R2, std_R2)
    ('V1',   'raw',          37):  (+0.4923, 0.1335),
    ('V1',   '√ρ+simplex',   37):  (+0.5624, 0.1394),
    ('V1',   'raw',         100):  (+0.0966, 0.1054),
    ('V1',   '√ρ+simplex',  100):  (+0.1526, 0.0971),
    ('V1',   'raw',         162):  (+0.0186, 0.0607),
    ('V1',   '√ρ+simplex',  162):  (+0.0673, 0.0609),

    ('V3.3', 'raw',          37):  (+0.1495, 0.0947),
    ('V3.3', '√ρ+simplex',   37):  (+0.1973, 0.0914),
    ('V3.3', 'raw',         100):  (+0.0155, 0.0572),
    ('V3.3', '√ρ+simplex',  100):  (+0.0638, 0.0453),
    ('V3.3', 'raw',         162):  (-0.0184, 0.0456),
    ('V3.3', '√ρ+simplex',  162):  (+0.0341, 0.0300),

    ('V3.4', 'raw',          37):  (+0.0796, 0.0653),
    ('V3.4', '√ρ+simplex',   37):  (+0.1074, 0.0601),
    ('V3.4', 'raw',         100):  (+0.0191, 0.0385),
    ('V3.4', '√ρ+simplex',  100):  (+0.0402, 0.0299),
    ('V3.4', 'raw',         162):  (+0.0088, 0.0246),
    ('V3.4', '√ρ+simplex',  162):  (+0.0245, 0.0186),
}

ROM_DT = 0.12
REGIMES = ['V1', 'V3.3', 'V3.4']
REGIME_LABELS = {'V1': 'V1 (control)', 'V3.3': 'V3.3 (moderate)', 'V3.4': 'V3.4 (extreme)'}
HORIZONS = [37, 100, 162]
COLOR_RAW  = '#4C72B0'
COLOR_SQRT = '#DD8452'


# ──────────────────────────────────────────────────────────
# FIGURE 1: Cross-regime bar plot at H37
# ──────────────────────────────────────────────────────────

def fig1():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(REGIMES))
    w = 0.32

    raw_m  = [DATA[(r, 'raw', 37)][0] for r in REGIMES]
    raw_s  = [DATA[(r, 'raw', 37)][1] for r in REGIMES]
    sqrt_m = [DATA[(r, '√ρ+simplex', 37)][0] for r in REGIMES]
    sqrt_s = [DATA[(r, '√ρ+simplex', 37)][1] for r in REGIMES]

    ax.bar(x - w/2, raw_m,  w, yerr=raw_s,  label='Raw baseline',
           color=COLOR_RAW, alpha=0.85, capsize=4, edgecolor='white', linewidth=0.8)
    ax.bar(x + w/2, sqrt_m, w, yerr=sqrt_s, label='√ρ + simplex',
           color=COLOR_SQRT, alpha=0.85, capsize=4, edgecolor='white', linewidth=0.8)

    # Delta annotations
    for i in range(len(REGIMES)):
        delta = sqrt_m[i] - raw_m[i]
        y = max(raw_m[i] + raw_s[i], sqrt_m[i] + sqrt_s[i]) + 0.025
        ax.text(x[i], y, f'Δ = {delta:+.3f}', ha='center', fontsize=10,
                fontweight='bold', color='#333')

    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABELS[r] for r in REGIMES], fontsize=11)
    ax.set_ylabel('$R^2_{\\rm rollout}$  (26 test runs)', fontsize=12)
    ax.set_title('Short-Horizon Forecast Accuracy ($H{=}37$, $\\approx$4.4 s)',
                 fontsize=13, fontweight='bold')
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=-0.05)
    fig.tight_layout()
    path = OUT / 'fig1_cross_regime_H37.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    fig.savefig(OUT / 'fig1_cross_regime_H37.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"✓ {path}")


# ──────────────────────────────────────────────────────────
# FIGURE 2: Knee-lite horizon curves per regime
# ──────────────────────────────────────────────────────────

def fig2():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for ax_i, regime in enumerate(REGIMES):
        ax = axes[ax_i]

        for pipe, color, marker, ls, zorder in [
            ('raw',         COLOR_RAW,  's', '--', 3),
            ('√ρ+simplex',  COLOR_SQRT, 'o', '-',  4),
        ]:
            hs, means, stds = [], [], []
            for H in HORIZONS:
                key = (regime, pipe, H)
                if key in DATA:
                    hs.append(H * ROM_DT)
                    means.append(DATA[key][0])
                    stds.append(DATA[key][1])

            if not hs:
                continue
            hs, means, stds = np.array(hs), np.array(means), np.array(stds)

            ax.plot(hs, means, marker=marker, ls=ls, lw=2.2, color=color,
                    ms=8, label=pipe, zorder=zorder,
                    markeredgecolor='white', markeredgewidth=1.2)
            ax.fill_between(hs, means - stds, means + stds,
                            alpha=0.13, color=color, zorder=2)

            # Label each point
            for h, m in zip(hs, means):
                H_int = int(round(h / ROM_DT))
                ax.annotate(f'H{H_int}\n{m:+.3f}',
                            (h, m), textcoords='offset points',
                            xytext=(10, 8 if pipe == '√ρ+simplex' else -16),
                            fontsize=7.5, color=color, ha='left')

        ax.axhline(0, color='gray', lw=0.8, ls='--')
        ax.set_xlabel('Forecast time (s)', fontsize=11)
        if ax_i == 0:
            ax.set_ylabel('$R^2_{\\rm rollout}$', fontsize=12)
        ax.set_title(REGIME_LABELS[regime], fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(alpha=0.3)

    fig.suptitle('Horizon Decay: Raw vs √ρ+Simplex across Regimes',
                 fontsize=14, fontweight='bold', y=1.03)
    fig.tight_layout()
    path = OUT / 'fig2_knee_lite_curves.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    fig.savefig(OUT / 'fig2_knee_lite_curves.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"✓ {path}")


# ──────────────────────────────────────────────────────────
# TABLE: Final results skeleton (CSV + formatted console)
# ──────────────────────────────────────────────────────────

def table():
    header = f"{'Regime':<8} | {'H':>3} | {'Raw R²':>10} | {'√ρ+S R²':>10} | {'Δ':>8} | {'neg% raw':>9} | {'neg% √ρ':>9}"
    sep = '-' * len(header)
    lines = [header, sep]

    # Neg% lookup (from earlier results)
    NEG = {
        ('V1',   'raw',  37): 28.0, ('V1',   '√ρ+simplex',  37): 18.8,
        ('V1',   'raw', 100): 23.0, ('V1',   '√ρ+simplex', 100): 19.1,
        ('V1',   'raw', 162): 20.3, ('V1',   '√ρ+simplex', 162): 19.2,
        ('V3.3', 'raw',  37): 23.7, ('V3.3', '√ρ+simplex',  37): 17.1,
        ('V3.3', 'raw', 100): 19.7, ('V3.3', '√ρ+simplex', 100): 14.9,
        ('V3.3', 'raw', 162): 17.0, ('V3.3', '√ρ+simplex', 162): 12.8,
        ('V3.4', 'raw',  37): 16.6, ('V3.4', '√ρ+simplex',  37): 13.6,
        ('V3.4', 'raw', 100):  8.7, ('V3.4', '√ρ+simplex', 100):  7.1,
        ('V3.4', 'raw', 162):  5.5, ('V3.4', '√ρ+simplex', 162):  4.5,
    }

    csv_rows = ['regime,H,raw_R2,raw_std,sqrt_R2,sqrt_std,delta,neg_raw,neg_sqrt']
    for regime in REGIMES:
        for H in HORIZONS:
            raw_key  = (regime, 'raw', H)
            sqrt_key = (regime, '√ρ+simplex', H)
            if raw_key in DATA and sqrt_key in DATA:
                rm, rs = DATA[raw_key]
                sm, ss = DATA[sqrt_key]
                delta = sm - rm
                nr = NEG.get(raw_key, float('nan'))
                ns = NEG.get(sqrt_key, float('nan'))
                lines.append(f"{regime:<8} | {H:>3} | {rm:+.4f}±{rs:.3f} | {sm:+.4f}±{ss:.3f} | {delta:+.4f} | {nr:>8.1f}% | {ns:>8.1f}%")
                csv_rows.append(f"{regime},{H},{rm:.4f},{rs:.4f},{sm:.4f},{ss:.4f},{delta:.4f},{nr:.1f},{ns:.1f}")
            else:
                # Missing data (Y5/Y6 still running)
                if raw_key in DATA:
                    rm, rs = DATA[raw_key]
                    lines.append(f"{regime:<8} | {H:>3} | {rm:+.4f}±{rs:.3f} | {'pending':>10} | {'---':>8} |           |          ")
                elif sqrt_key in DATA:
                    sm, ss = DATA[sqrt_key]
                    lines.append(f"{regime:<8} | {H:>3} | {'pending':>10} | {sm:+.4f}±{ss:.3f} | {'---':>8} |           |          ")
                else:
                    lines.append(f"{regime:<8} | {H:>3} | {'pending':>10} | {'pending':>10} | {'---':>8} |           |          ")
                csv_rows.append(f"{regime},{H},,,,,,,")

    table_str = '\n'.join(lines)
    print('\n' + table_str + '\n')

    csv_path = OUT / 'results_table.csv'
    csv_path.write_text('\n'.join(csv_rows) + '\n')
    print(f"✓ {csv_path}")

    txt_path = OUT / 'results_table.txt'
    txt_path.write_text(table_str + '\n')
    print(f"✓ {txt_path}")


# ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("Generating thesis figures from completed Suite X/Y results")
    print("=" * 60)
    fig1()
    fig2()
    table()
    print("\nDone. Fill V3.4 H100 once Y5/Y6 finish.")
