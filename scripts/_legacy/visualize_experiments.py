#!/usr/bin/env python3
"""
Visualize CF-series and S2-series experiments.

For each experiment, generates per test case (best + worst by R²):
  1. Side-by-side density video (truth vs pred, with R² strip)
  2. Snapshot comparison grid (6 time-points)
  3. Frame-wise R² curve

Also generates aggregate comparison plots across experiments.

Usage:
  PYTHONPATH=src python scripts/visualize_experiments.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from rectsim.rom_video_utils import (
    make_truth_vs_pred_density_video,
    make_density_snapshot_comparison,
)

# ── Experiment registry ──
EXPERIMENTS = {
    'CF8_longPrefix_sqrtSimplex_H37': {
        'label': 'CF8 MVAR (√ρ+simplex, H37)',
        'short': 'CF8 MVAR',
        'model': 'mvar',
        'color': '#1f77b4',
    },
    'CF8_LSTM_longPrefix_sqrtSimplex_H37': {
        'label': 'CF8 LSTM (√ρ+simplex, H37)',
        'short': 'CF8 LSTM',
        'model': 'lstm',
        'color': '#ff7f0e',
    },
    'CF10_longPrefix_sqrtSimplex_kstep4_H37': {
        'label': 'CF10 k-step MVAR (k=4, H37)',
        'short': 'CF10 k=4',
        'model': 'mvar',
        'color': '#2ca02c',
    },
    'S2a_v1_sqrtD19_p5_H100_eta02_aligned': {
        'label': 'S2a Aligned (η=0.2, H100)',
        'short': 'S2a η=0.2',
        'model': 'mvar',
        'color': '#d62728',
    },
    'S2b_v1_sqrtD19_p5_H100_eta0_aligned': {
        'label': 'S2b Aligned (η=0, H100)',
        'short': 'S2b η=0',
        'model': 'mvar',
        'color': '#9467bd',
    },
}


def load_test_case(test_dir: Path, model_type: str):
    """Load truth + prediction for one test case."""
    true_data = np.load(test_dir / 'density_true.npz')
    rho_true_full = true_data['rho']        # (T_full, 48, 48)
    times_full = true_data['times']         # (T_full,)

    pred_file = test_dir / f'density_pred_{model_type}.npz'
    if not pred_file.exists():
        pred_file = test_dir / 'density_pred.npz'
    pred_data = np.load(pred_file)
    rho_pred = pred_data['rho']             # (T_pred, 48, 48) — includes prefix
    times_pred = pred_data['times']
    fc_idx = int(pred_data['forecast_start_idx'])

    # Extract forecast portion only
    rho_pred_fc = rho_pred[fc_idx:]
    times_fc = times_pred[fc_idx:]
    T_fc = len(rho_pred_fc)

    # Align truth to same forecast window
    # Find matching start time in full truth
    t_start = times_fc[0]
    idx_start = np.argmin(np.abs(times_full - t_start))
    # Subsample truth to match pred time steps
    rho_true_fc = np.zeros_like(rho_pred_fc)
    for i, t in enumerate(times_fc):
        idx = np.argmin(np.abs(times_full - t))
        rho_true_fc[i] = rho_true_full[idx]

    # Load metrics if available
    metrics = {}
    metrics_file = test_dir / 'metrics_summary.json'
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)

    # Load r2 vs time
    r2_vs_time = None
    r2_file = test_dir / 'r2_vs_time.csv'
    if r2_file.exists():
        r2_vs_time = np.genfromtxt(r2_file, delimiter=',', names=True)

    return {
        'rho_true': rho_true_fc,
        'rho_pred': rho_pred_fc,
        'times': times_fc,
        'T_fc': T_fc,
        'fc_idx': fc_idx,
        'metrics': metrics,
        'r2_vs_time': r2_vs_time,
    }


def compute_r2_per_frame(rho_true, rho_pred):
    """Compute R² for each frame."""
    T = rho_true.shape[0]
    r2 = np.zeros(T)
    for t in range(T):
        y = rho_true[t].ravel()
        yhat = rho_pred[t].ravel()
        ss_res = np.sum((y - yhat)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2[t] = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
    return r2


def visualize_single_case(data, viz_dir, exp_name, case_name, model_label):
    """Generate all visualizations for one test case."""
    rho_true = data['rho_true']
    rho_pred = data['rho_pred']
    times = data['times']
    T = data['T_fc']
    metrics = data['metrics']

    r2_pod = metrics.get('r2_pod', np.nan)
    r2_lat = metrics.get('r2_latent', np.nan)

    viz_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Side-by-side video ──
    print(f"    Video: truth vs pred...")
    r2_frames = compute_r2_per_frame(rho_true, rho_pred)

    # Build video with R² strip
    make_truth_vs_pred_video_with_r2(
        rho_true, rho_pred, r2_frames, times,
        viz_dir / f'{case_name}_truth_vs_pred.mp4',
        title=f'{model_label} — {case_name} (R²_pod={r2_pod:+.3f})',
        fps=12,
    )

    # ── 2. Snapshot comparison ──
    print(f"    Snapshots...")
    n_snaps = min(6, T)
    snap_idx = np.linspace(0, T - 1, n_snaps, dtype=int)
    make_density_snapshot_comparison(
        rho_true, rho_pred,
        list(snap_idx), times,
        viz_dir / f'{case_name}_snapshots.png',
        title=f'{model_label} — {case_name} (R²_pod={r2_pod:+.3f})',
        cmap='hot',
    )

    # ── 3. R² + error curves ──
    print(f"    R² curve...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(times, r2_frames, 'b-', lw=1.5, label='Frame R²')
    ax1.axhline(r2_pod, color='r', ls='--', lw=1.5,
                label=f'Overall R²_pod = {r2_pod:+.3f}')
    ax1.axhline(0, color='gray', ls=':', lw=0.5)
    ax1.fill_between(times, r2_frames, alpha=0.12, color='blue')
    ax1.set_ylabel('R²', fontsize=12, fontweight='bold')
    ax1.set_title(f'{model_label} — {case_name}', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_ylim(min(r2_frames.min() - 0.05, -0.2), 1.05)
    ax1.grid(True, alpha=0.3)

    # Mass conservation
    mass_true = np.sum(rho_true.reshape(T, -1), axis=1)
    mass_pred = np.sum(rho_pred.reshape(T, -1), axis=1)
    mass_err = np.abs(mass_true - mass_pred) / (mass_true + 1e-10) * 100

    ax2.plot(times, mass_err, 'orange', lw=1.5, label='Mass error (%)')
    ax2.set_ylabel('Mass Error (%)', fontsize=11)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(viz_dir / f'{case_name}_r2_curve.png', dpi=150, bbox_inches='tight')
    plt.close()


def make_truth_vs_pred_video_with_r2(
    rho_true, rho_pred, r2_frames, times, out_path, title='', fps=12,
):
    """Side-by-side video with R² strip at bottom."""
    import imageio

    T, Ny, Nx = rho_true.shape
    vmin = min(rho_true.min(), rho_pred.min())
    vmax = max(rho_true.max(), rho_pred.max())

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], hspace=0.25, wspace=0.05)

    ax_true = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[0, 1])
    ax_r2 = fig.add_subplot(gs[1, :])

    im_true = ax_true.imshow(rho_true[0], cmap='hot', vmin=vmin, vmax=vmax,
                              origin='lower', aspect='equal')
    ax_true.set_title('Ground Truth', fontsize=12)

    im_pred = ax_pred.imshow(rho_pred[0], cmap='hot', vmin=vmin, vmax=vmax,
                              origin='lower', aspect='equal')
    ax_pred.set_title('ROM Prediction', fontsize=12)

    fig.colorbar(im_true, ax=[ax_true, ax_pred], fraction=0.046, pad=0.04,
                 label='Density')

    ax_r2.plot(times, r2_frames, 'b-', lw=1.5, alpha=0.7)
    ax_r2.axhline(0, color='gray', ls=':', lw=0.5)
    ax_r2.set_ylim(min(r2_frames.min() - 0.05, -0.3), 1.05)
    ax_r2.set_ylabel('R²', fontsize=11)
    ax_r2.set_xlabel('Time (s)', fontsize=11)
    ax_r2.grid(True, alpha=0.3)
    vline = ax_r2.axvline(times[0], color='red', lw=2)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    time_text = fig.text(0.5, 0.92, f't={times[0]:.2f}s  R²={r2_frames[0]:+.3f}',
                         ha='center', fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    frames = []
    for t in range(T):
        im_true.set_data(rho_true[t])
        im_pred.set_data(rho_pred[t])
        vline.set_xdata([times[t], times[t]])
        time_text.set_text(f't={times[t]:.2f}s  R²={r2_frames[t]:+.3f}')

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(buf[:, :, :3].copy())

    plt.close(fig)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=fps, codec='libx264',
                    quality=8, pixelformat='yuv420p')
    sz = out_path.stat().st_size / 1024 / 1024
    print(f"      -> {out_path.name} ({sz:.1f} MB)")


def aggregate_comparison(all_results, viz_root):
    """Cross-experiment comparison plots."""
    viz_root.mkdir(parents=True, exist_ok=True)

    # ── 1. Box plot of R² across test cases ──
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = []
    data = []
    colors = []
    for exp_name, info in EXPERIMENTS.items():
        if exp_name not in all_results:
            continue
        r2s = [r['metrics'].get('r2_pod', np.nan) for r in all_results[exp_name]]
        r2s = [x for x in r2s if not np.isnan(x)]
        if r2s:
            data.append(r2s)
            labels.append(info['short'])
            colors.append(info['color'])

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    for med in bp['medians']:
        med.set_color('black')
        med.set_linewidth(2)

    ax.axhline(0, color='gray', ls=':', lw=0.5)
    ax.set_ylabel('R² (density)', fontsize=12, fontweight='bold')
    ax.set_title('Forecast Accuracy Across Test Cases', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_root / 'comparison_r2_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison_r2_boxplot.png")

    # ── 2. Mean R² curve over time (first 3 test cases) ──
    fig, ax = plt.subplots(figsize=(12, 6))
    for exp_name, info in EXPERIMENTS.items():
        if exp_name not in all_results:
            continue
        cases = all_results[exp_name][:5]  # use up to 5 cases
        # Collect R² curves, find common length
        curves = []
        time_ref = None
        for c in cases:
            r2f = compute_r2_per_frame(c['rho_true'], c['rho_pred'])
            curves.append(r2f)
            if time_ref is None or len(c['times']) < len(time_ref):
                time_ref = c['times']

        min_len = min(len(c) for c in curves)
        curves_arr = np.array([c[:min_len] for c in curves])
        mean_r2 = np.mean(curves_arr, axis=0)
        std_r2 = np.std(curves_arr, axis=0)
        t = time_ref[:min_len]

        ax.plot(t, mean_r2, lw=2, color=info['color'], label=info['short'])
        ax.fill_between(t, mean_r2 - std_r2, mean_r2 + std_r2,
                        alpha=0.15, color=info['color'])

    ax.axhline(0, color='gray', ls=':', lw=0.5)
    ax.set_xlabel('Forecast Time (s)', fontsize=12)
    ax.set_ylabel('R² (frame-wise, mean ± 1σ)', fontsize=12)
    ax.set_title('R² Decay Over Forecast Horizon', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 1.05)
    plt.tight_layout()
    plt.savefig(viz_root / 'comparison_r2_vs_time.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison_r2_vs_time.png")

    # ── 3. Summary table ──
    print("\n  ┌─────────────────────────────────┬─────────┬─────────┬─────────┐")
    print("  │ Experiment                      │ med R²  │ mean R² │  n_test │")
    print("  ├─────────────────────────────────┼─────────┼─────────┼─────────┤")
    for exp_name, info in EXPERIMENTS.items():
        if exp_name not in all_results:
            continue
        r2s = [r['metrics'].get('r2_pod', np.nan) for r in all_results[exp_name]]
        r2s = [x for x in r2s if not np.isnan(x)]
        med = np.median(r2s) if r2s else np.nan
        mean = np.mean(r2s) if r2s else np.nan
        print(f"  │ {info['short']:31s} │ {med:+.4f} │ {mean:+.4f} │ {len(r2s):7d} │")
    print("  └─────────────────────────────────┴─────────┴─────────┴─────────┘")


def main():
    oscar_out = ROOT / 'oscar_output'
    viz_root = ROOT / 'oscar_output' / 'visualizations'

    all_results = {}

    for exp_name, info in EXPERIMENTS.items():
        exp_dir = oscar_out / exp_name
        test_dir = exp_dir / 'test'
        if not test_dir.exists():
            print(f"SKIP {exp_name}: no test/ directory")
            continue

        # Discover test cases
        test_cases = sorted([d for d in test_dir.iterdir() if d.is_dir() and d.name.startswith('test_')])
        n_cases = len(test_cases)
        print(f"\n{'='*70}")
        print(f"  {info['label']}  ({n_cases} test cases)")
        print(f"{'='*70}")

        # Load all test cases and their metrics
        cases_data = []
        for tc in test_cases:
            try:
                d = load_test_case(tc, info['model'])
                d['case_name'] = tc.name
                cases_data.append(d)
            except Exception as e:
                print(f"  WARN: {tc.name}: {e}")

        all_results[exp_name] = cases_data

        # Sort by R² — pick best and worst
        valid = [c for c in cases_data if not np.isnan(c['metrics'].get('r2_pod', np.nan))]
        if not valid:
            print("  No valid test cases found.")
            continue

        valid.sort(key=lambda c: c['metrics'].get('r2_pod', -999))
        best = valid[-1]
        worst = valid[0]
        median_idx = len(valid) // 2
        median_case = valid[median_idx]

        r2_vals = [c['metrics'].get('r2_pod', np.nan) for c in valid]
        print(f"  R² range: [{min(r2_vals):+.4f}, {max(r2_vals):+.4f}], "
              f"median={np.median(r2_vals):+.4f}, n={len(valid)}")

        # Visualize best, worst, median
        exp_viz = viz_root / exp_name
        for tag, case in [('best', best), ('worst', worst), ('median', median_case)]:
            r2 = case['metrics'].get('r2_pod', np.nan)
            print(f"\n  [{tag.upper()}] {case['case_name']} (R²_pod={r2:+.4f})")
            visualize_single_case(
                case, exp_viz,
                exp_name, f"{tag}_{case['case_name']}",
                f"{info['short']} ({tag})",
            )

    # ── Aggregate comparison ──
    print(f"\n{'='*70}")
    print(f"  AGGREGATE COMPARISON")
    print(f"{'='*70}")
    aggregate_comparison(all_results, viz_root)

    # ── Summary ──
    total_files = list(viz_root.rglob('*'))
    total_files = [f for f in total_files if f.is_file()]
    print(f"\n  Done! {len(total_files)} files in {viz_root}")


if __name__ == '__main__':
    main()
