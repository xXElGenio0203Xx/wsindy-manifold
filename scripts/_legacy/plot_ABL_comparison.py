#!/usr/bin/env python3
"""Analyze and plot ABL suite results."""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXPS = [
    ("ABL1", "raw",  "none",    "noAlign", "ABL1_N200_raw_none_noAlign_H300"),
    ("ABL2", "raw",  "none",    "align",   "ABL2_N200_raw_none_align_H300"),
    ("ABL3", "raw",  "simplex", "noAlign", "ABL3_N200_raw_simplex_noAlign_H300"),
    ("ABL4", "raw",  "simplex", "align",   "ABL4_N200_raw_simplex_align_H300"),
    ("ABL5", "sqrt", "none",    "noAlign", "ABL5_N200_sqrt_none_noAlign_H300"),
    ("ABL6", "sqrt", "none",    "align",   "ABL6_N200_sqrt_none_align_H300"),
    ("ABL7", "sqrt", "simplex", "noAlign", "ABL7_N200_sqrt_simplex_noAlign_H300"),
    ("ABL8", "sqrt", "simplex", "align",   "ABL8_N200_sqrt_simplex_align_H300"),
]


def load_results():
    results = {}
    for short, transform, mass_pp, align, full_name in EXPS:
        csv_path = os.path.join(ROOT, "oscar_output", full_name, "test", "test_results.csv")
        r2_recon, r2_lat, r2_1step, r2_pod, neg, massviol = [], [], [], [], [], []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                r2_recon.append(float(row['r2_reconstructed']))
                r2_lat.append(float(row['r2_latent']))
                r2_1step.append(float(row['r2_1step']))
                r2_pod.append(float(row['r2_pod']))
                neg.append(float(row['negativity_frac']))
                massviol.append(float(row['max_mass_violation']))

        results[short] = {
            'r2_recon_mean': np.mean(r2_recon),
            'r2_recon_std': np.std(r2_recon),
            'r2_recon_all': r2_recon,
            'r2_lat_mean': np.mean(r2_lat),
            'r2_1step_mean': np.mean(r2_1step),
            'r2_pod_mean': np.mean(r2_pod),
            'neg_mean': np.mean(neg),
            'mass_viol_mean': np.mean(massviol),
            'transform': transform,
            'mass_pp': mass_pp,
            'align': align,
            'n_tests': len(r2_recon),
        }
    return results


def print_table(results):
    print(f"{'Exp':<6} {'Xform':<6} {'Mass':<8} {'Align':<8} | "
          f"{'R2_recon':>9} {'R2_lat':>8} {'R2_1step':>9} {'R2_POD':>7} {'neg%':>6} {'massV':>8}")
    print("-" * 85)
    for short, _, _, _, _ in EXPS:
        r = results[short]
        print(f"{short:<6} {r['transform']:<6} {r['mass_pp']:<8} {r['align']:<8} | "
              f"{r['r2_recon_mean']:>+8.4f} {r['r2_lat_mean']:>+8.4f} "
              f"{r['r2_1step_mean']:>+8.4f} {r['r2_pod_mean']:>7.4f} "
              f"{r['neg_mean']:>5.1f}% {r['mass_viol_mean']:>7.4f}")

    print("\n=== FACTOR EFFECTS ON R2_recon ===")
    for factor_name, vals in [("transform", ["raw", "sqrt"]),
                               ("mass_pp", ["none", "simplex"]),
                               ("align", ["noAlign", "align"])]:
        groups = {v: [] for v in vals}
        for short, info in results.items():
            groups[info[factor_name]].append(info['r2_recon_mean'])
        means = {v: np.mean(groups[v]) for v in vals}
        print(f"  {factor_name}: {vals[0]}={means[vals[0]]:+.4f}  "
              f"{vals[1]}={means[vals[1]]:+.4f}  "
              f"(delta={means[vals[1]] - means[vals[0]]:+.4f})")


def plot_comparison(results):
    out_dir = os.path.join(ROOT, "oscar_output", "ABL_comparison")
    os.makedirs(out_dir, exist_ok=True)

    # ── Plot 1: Bar chart of R2_recon for all 8 configs ──
    fig, ax = plt.subplots(figsize=(14, 6))

    labels = []
    r2_means = []
    r2_stds = []
    colors = []
    hatches = []

    color_map = {
        ('raw', 'none'): '#e74c3c',
        ('raw', 'simplex'): '#e67e22',
        ('sqrt', 'none'): '#3498db',
        ('sqrt', 'simplex'): '#2ecc71',
    }
    hatch_map = {'noAlign': '', 'align': '///'}

    for short, transform, mass_pp, align, _ in EXPS:
        r = results[short]
        label = f"{short}\n{transform}+{mass_pp}\n{'align' if align == 'align' else 'no align'}"
        labels.append(label)
        r2_means.append(r['r2_recon_mean'])
        r2_stds.append(r['r2_recon_std'])
        colors.append(color_map[(transform, mass_pp)])
        hatches.append(hatch_map[align])

    x = np.arange(len(labels))
    bars = ax.bar(x, r2_means, yerr=r2_stds, capsize=5, color=colors,
                  edgecolor='black', linewidth=0.8)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('R$^2$ (density reconstruction)', fontsize=13)
    ax.set_title('ABL Suite: R$^2$ Comparison\n(2x2x2 factorial: transform x mass_projection x shift_alignment)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value annotations
    for i, (m, s) in enumerate(zip(r2_means, r2_stds)):
        y_pos = m + s + 0.02 if m >= 0 else m - s - 0.04
        ax.text(i, y_pos, f'{m:+.3f}', ha='center', va='bottom' if m >= 0 else 'top',
                fontsize=10, fontweight='bold')

    # Legend for color coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='raw + none'),
        Patch(facecolor='#e67e22', label='raw + simplex'),
        Patch(facecolor='#3498db', label='sqrt + none'),
        Patch(facecolor='#2ecc71', label='sqrt + simplex'),
        Patch(facecolor='white', edgecolor='black', hatch='///', label='shift-aligned'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'ABL_r2_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"  -> Saved {out_dir}/ABL_r2_comparison.png")
    plt.close(fig)

    # ── Plot 2: Grouped bar — factor effects ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    factor_configs = [
        ("Density Transform", "transform", ["raw", "sqrt"], ['#e74c3c', '#3498db']),
        ("Mass Projection", "mass_pp", ["none", "simplex"], ['#95a5a6', '#2ecc71']),
        ("Shift Alignment", "align", ["noAlign", "align"], ['#95a5a6', '#9b59b6']),
    ]

    for ax, (title, key, vals, cols) in zip(axes, factor_configs):
        groups = {v: [] for v in vals}
        for short, info in results.items():
            groups[info[key]].append(info['r2_recon_mean'])

        means = [np.mean(groups[v]) for v in vals]
        stds = [np.std(groups[v]) for v in vals]

        bars = ax.bar(vals, means, yerr=stds, capsize=5, color=cols,
                      edgecolor='black', linewidth=0.8, width=0.5)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean R$^2$', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 0.02, f'{m:+.3f}', ha='center', fontsize=11, fontweight='bold')

        delta = means[1] - means[0]
        ax.set_xlabel(f'Effect: {delta:+.3f}', fontsize=11, color='green' if delta > 0 else 'red')

    fig.suptitle('ABL Suite: Factor Effects on R$^2$', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'ABL_factor_effects.png'), dpi=150, bbox_inches='tight')
    print(f"  -> Saved {out_dir}/ABL_factor_effects.png")
    plt.close(fig)

    # ── Plot 3: Multi-metric comparison for top configs ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Sort by R2_recon
    sorted_exps = sorted(results.items(), key=lambda x: x[1]['r2_recon_mean'], reverse=True)

    metrics = [
        ('r2_recon_mean', 'R$^2$ Density Recon', 'R$^2$'),
        ('r2_lat_mean', 'R$^2$ Latent', 'R$^2$'),
        ('neg_mean', 'Negativity %', '%'),
    ]

    for ax, (metric_key, title, ylabel) in zip(axes, metrics):
        names = [s for s, _ in sorted_exps]
        vals = [r[metric_key] for _, r in sorted_exps]
        clrs = [color_map[(r['transform'], r['mass_pp'])] for _, r in sorted_exps]
        ax.barh(range(len(names)), vals, color=clrs, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()

        for i, v in enumerate(vals):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

    fig.suptitle('ABL Suite: All Metrics (sorted by R$^2$ recon)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'ABL_multi_metric.png'), dpi=150, bbox_inches='tight')
    print(f"  -> Saved {out_dir}/ABL_multi_metric.png")
    plt.close(fig)


if __name__ == '__main__':
    results = load_results()
    print_table(results)
    plot_comparison(results)
    print("\nDone.")
