#!/usr/bin/env python3
"""
FINAL HORIZON CURVE — raw baseline vs √ρ+simplex
=================================================
Computes at each of 5 horizons (H37, H60, H100, H162, H312):
  1. raw baseline R²
  2. √ρ uncorrected R²
  3. √ρ + simplex projection R²
  4. Shift-aligned R² for both raw and √ρ+simplex (phase-drift diagnostic)

Also outputs mass error curves and per-test detail CSVs.

The shift-aligned R² slides the prediction ±S pixels in each axis
and takes the best-matching alignment. This separates:
  - phase drift (spatial translation of features) → shift fixes this
  - structural distortion (wrong shape/amplitude) → shift can't fix this
"""

import numpy as np
import os
import json
import csv
from pathlib import Path
from datetime import datetime
from itertools import product


# ─────────────────────────────────────────────────────────────
# SIMPLEX PROJECTION (offset form, L2-optimal)
# ─────────────────────────────────────────────────────────────

def project_simplex(rho_flat: np.ndarray, M0: float) -> np.ndarray:
    """
    Euclidean projection onto {ρ ≥ 0, Σρ = M₀}.
    Duchi et al. (2008) O(n log n) algorithm.
    """
    n = len(rho_flat)
    mu = np.sort(rho_flat)[::-1]
    cumsum = np.cumsum(mu)
    arange = np.arange(1, n + 1, dtype=np.float64)
    test = mu - (cumsum - M0) / arange
    rho_max = np.max(np.where(test > 0)[0]) if np.any(test > 0) else 0
    theta = (cumsum[rho_max] - M0) / (rho_max + 1)
    return np.maximum(rho_flat - theta, 0.0)


# ─────────────────────────────────────────────────────────────
# SHIFT-ALIGNED R²
# ─────────────────────────────────────────────────────────────

def shift_aligned_r2(rho_true: np.ndarray, rho_pred: np.ndarray,
                     max_shift: int = 3) -> float:
    """
    Compute best R² over all integer pixel shifts ±max_shift in x and y.
    Uses periodic (wraparound) shifts since domain is periodic.

    Args:
        rho_true: (T, Ny, Nx) ground truth
        rho_pred: (T, Ny, Nx) prediction
        max_shift: max pixels to shift in each direction

    Returns:
        Best R² over all (2*max_shift+1)² shifts
    """
    T, Ny, Nx = rho_true.shape
    ss_tot = np.sum((rho_true - rho_true.mean()) ** 2)
    if ss_tot < 1e-15:
        return 0.0

    best_r2 = -np.inf
    shifts = range(-max_shift, max_shift + 1)

    for dy, dx in product(shifts, shifts):
        pred_shifted = np.roll(np.roll(rho_pred, dy, axis=1), dx, axis=2)
        ss_res = np.sum((rho_true - pred_shifted) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        if r2 > best_r2:
            best_r2 = r2

    return float(best_r2)


# ─────────────────────────────────────────────────────────────
# EXPERIMENT CONFIGURATION
# ─────────────────────────────────────────────────────────────

HORIZONS = [37, 60, 100, 162, 312]

SQRT_EXPERIMENTS = {
    37:  'K1_v1_sqrtD19_p5_H37',
    60:  'K2_v1_sqrtD19_p5_H60',
    100: 'K3_v1_sqrtD19_p5_H100',
    162: 'K4_v1_sqrtD19_p5_H162',
    312: 'K5_v1_sqrtD19_p5_H312',
}

RAW_EXPERIMENTS = {
    37:  'K6_v1_rawD19_p5_H37',
    60:  'K7_v1_rawD19_p5_H60',
    100: 'K8_v1_rawD19_p5_H100',
    162: 'K9_v1_rawD19_p5_H162',
    312: 'K10_v1_rawD19_p5_H312',
}

ROM_DT = 0.12  # seconds per forecast step


# ─────────────────────────────────────────────────────────────
# EVALUATE ONE EXPERIMENT
# ─────────────────────────────────────────────────────────────

def evaluate_experiment(exp_name: str, root: str, apply_simplex: bool,
                        compute_shift: bool = True, max_shift: int = 3):
    """
    Evaluate one experiment. Returns dict of aggregate metrics.

    If apply_simplex=True, applies simplex projection to each forecast frame.
    If compute_shift=True, also computes shift-aligned R².
    """
    test_dir = os.path.join(root, 'oscar_output', exp_name, 'test')
    test_runs = sorted([
        d for d in os.listdir(test_dir)
        if d.startswith('test_') and os.path.isdir(os.path.join(test_dir, d))
    ])

    r2_list = []
    r2_shift_list = []
    mass_err_list = []
    neg_pct_list = []

    for run_name in test_runs:
        run_dir = os.path.join(test_dir, run_name)
        d_pred = np.load(os.path.join(run_dir, 'density_pred_mvar.npz'))
        d_true = np.load(os.path.join(run_dir, 'density_true.npz'))

        rho_pred = d_pred['rho'].copy()
        rho_true = d_true['rho']
        fsi = int(d_pred['forecast_start_idx'])

        M0 = float(rho_true[max(fsi - 1, 0)].sum())

        # Forecast portion
        n_forecast = rho_pred.shape[0] - fsi
        true_f = rho_true[fsi:fsi + n_forecast]
        pred_f = rho_pred[fsi:fsi + n_forecast]
        L = min(true_f.shape[0], pred_f.shape[0])
        true_f = true_f[:L]
        pred_f = pred_f[:L]

        if L == 0:
            continue

        # Apply simplex projection if requested
        if apply_simplex:
            spatial_shape = pred_f.shape[1:]
            for t in range(L):
                pred_f[t] = project_simplex(pred_f[t].ravel(), M0).reshape(spatial_shape)

        # Overall R²
        ss_tot = np.sum((true_f - true_f.mean()) ** 2)
        if ss_tot < 1e-15:
            continue
        r2 = 1.0 - np.sum((true_f - pred_f) ** 2) / ss_tot
        r2_list.append(r2)

        # Mass error at final frame
        mass_err = abs(pred_f[-1].sum() - M0) / M0 * 100.0
        mass_err_list.append(mass_err)

        # Negative fraction at final frame
        neg_pct = (pred_f[-1] < 0).mean() * 100.0
        neg_pct_list.append(neg_pct)

        # Shift-aligned R²
        if compute_shift:
            r2_sa = shift_aligned_r2(true_f, pred_f, max_shift=max_shift)
            r2_shift_list.append(r2_sa)

    return {
        'experiment': exp_name,
        'n_tests': len(r2_list),
        'r2_mean': float(np.mean(r2_list)),
        'r2_std': float(np.std(r2_list)),
        'r2_shift_mean': float(np.mean(r2_shift_list)) if r2_shift_list else None,
        'r2_shift_std': float(np.std(r2_shift_list)) if r2_shift_list else None,
        'mass_err_mean': float(np.mean(mass_err_list)),
        'neg_pct_mean': float(np.mean(neg_pct_list)),
    }


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, 'oscar_output', 'final_horizon_curve')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("  FINAL HORIZON CURVE: raw baseline vs √ρ+simplex")
    print("  + Shift-aligned R² (max_shift=3 pixels)")
    print("=" * 80)

    results = []

    for H in HORIZONS:
        forecast_time = H * ROM_DT
        print(f"\n── H={H} ({forecast_time:.1f}s) ──")

        # 1. Raw baseline
        print(f"  Evaluating raw baseline ({RAW_EXPERIMENTS[H]})...")
        r_raw = evaluate_experiment(RAW_EXPERIMENTS[H], root,
                                    apply_simplex=False, compute_shift=True)
        r_raw['pipeline'] = 'raw'
        r_raw['horizon'] = H
        r_raw['forecast_time_s'] = forecast_time
        results.append(r_raw)
        print(f"    R²={r_raw['r2_mean']:+.4f}  R²_shift={r_raw['r2_shift_mean']:+.4f}  "
              f"mass_err={r_raw['mass_err_mean']:.1f}%")

        # 2. √ρ uncorrected
        print(f"  Evaluating √ρ uncorrected ({SQRT_EXPERIMENTS[H]})...")
        r_sqrt = evaluate_experiment(SQRT_EXPERIMENTS[H], root,
                                     apply_simplex=False, compute_shift=True)
        r_sqrt['pipeline'] = 'sqrt_uncorrected'
        r_sqrt['horizon'] = H
        r_sqrt['forecast_time_s'] = forecast_time
        results.append(r_sqrt)
        print(f"    R²={r_sqrt['r2_mean']:+.4f}  R²_shift={r_sqrt['r2_shift_mean']:+.4f}  "
              f"mass_err={r_sqrt['mass_err_mean']:.1f}%")

        # 3. √ρ + simplex
        print(f"  Evaluating √ρ + simplex ({SQRT_EXPERIMENTS[H]})...")
        r_simplex = evaluate_experiment(SQRT_EXPERIMENTS[H], root,
                                        apply_simplex=True, compute_shift=True)
        r_simplex['pipeline'] = 'sqrt_simplex'
        r_simplex['horizon'] = H
        r_simplex['forecast_time_s'] = forecast_time
        results.append(r_simplex)
        print(f"    R²={r_simplex['r2_mean']:+.4f}  R²_shift={r_simplex['r2_shift_mean']:+.4f}  "
              f"mass_err={r_simplex['mass_err_mean']:.1f}%")

    # ── Final comparison table ──
    print("\n\n" + "=" * 100)
    print("  FINAL HORIZON CURVE — COMPARISON TABLE")
    print("=" * 100)
    header = (f"{'H':>5s} {'Time':>6s} │ {'R²_raw':>8s} {'R²_√ρ':>8s} {'R²_√ρ+S':>8s} │ "
              f"{'SA_raw':>8s} {'SA_√ρ+S':>8s} │ {'SA−R²_raw':>9s} {'SA−R²_√ρ+S':>11s} │ "
              f"{'Merr_raw':>8s} {'Merr_√ρ+S':>9s}")
    print(header)
    print("─" * 100)

    for H in HORIZONS:
        r_raw = next(r for r in results if r['horizon'] == H and r['pipeline'] == 'raw')
        r_sqrt = next(r for r in results if r['horizon'] == H and r['pipeline'] == 'sqrt_uncorrected')
        r_simp = next(r for r in results if r['horizon'] == H and r['pipeline'] == 'sqrt_simplex')

        sa_gap_raw = r_raw['r2_shift_mean'] - r_raw['r2_mean']
        sa_gap_simp = r_simp['r2_shift_mean'] - r_simp['r2_mean']

        print(
            f"  {H:>3d} {H*ROM_DT:>5.1f}s │ "
            f"{r_raw['r2_mean']:>+7.4f} {r_sqrt['r2_mean']:>+7.4f} {r_simp['r2_mean']:>+7.4f} │ "
            f"{r_raw['r2_shift_mean']:>+7.4f} {r_simp['r2_shift_mean']:>+7.4f} │ "
            f"{sa_gap_raw:>+8.4f} {sa_gap_simp:>+10.4f} │ "
            f"{r_raw['mass_err_mean']:>7.1f}% {r_simp['mass_err_mean']:>8.1f}%"
        )

    # ── Phase drift diagnostic ──
    print("\n\n" + "=" * 80)
    print("  DIAGNOSTIC: Phase drift vs structural distortion")
    print("=" * 80)
    print("  SA−R² gap = shift-aligned R² minus standard R²")
    print("  Large gap → decay is dominated by phase drift (good: features exist but are displaced)")
    print("  Small gap → decay is dominated by structural distortion (bad: features are wrong)")
    print()

    for H in HORIZONS:
        r_raw = next(r for r in results if r['horizon'] == H and r['pipeline'] == 'raw')
        r_simp = next(r for r in results if r['horizon'] == H and r['pipeline'] == 'sqrt_simplex')

        gap_raw = r_raw['r2_shift_mean'] - r_raw['r2_mean']
        gap_simp = r_simp['r2_shift_mean'] - r_simp['r2_mean']
        # What fraction of the R² loss (from 1.0) is recoverable by shifting?
        loss_raw = max(1.0 - r_raw['r2_mean'], 1e-8)
        loss_simp = max(1.0 - r_simp['r2_mean'], 1e-8)
        recov_raw = gap_raw / loss_raw * 100
        recov_simp = gap_simp / loss_simp * 100

        print(f"  H={H:>3d}: raw gap={gap_raw:+.4f} ({recov_raw:.0f}% of loss recoverable)  "
              f"√ρ+S gap={gap_simp:+.4f} ({recov_simp:.0f}% of loss recoverable)")

    # ── Save results ──
    with open(os.path.join(out_dir, 'horizon_curve_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(out_dir, 'horizon_curve.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'horizon', 'forecast_time_s', 'pipeline',
            'r2_mean', 'r2_std', 'r2_shift_mean', 'r2_shift_std',
            'mass_err_mean', 'neg_pct_mean', 'n_tests'
        ])
        for r in results:
            writer.writerow([
                r['horizon'], r['forecast_time_s'], r['pipeline'],
                f"{r['r2_mean']:.6f}", f"{r['r2_std']:.6f}",
                f"{r['r2_shift_mean']:.6f}" if r['r2_shift_mean'] is not None else '',
                f"{r['r2_shift_std']:.6f}" if r['r2_shift_std'] is not None else '',
                f"{r['mass_err_mean']:.4f}", f"{r['neg_pct_mean']:.4f}",
                r['n_tests']
            ])

    print(f"\n  → Results saved to {out_dir}/")

    # ── Generate plots ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Plot 1: Horizon curve (R² vs forecast time)
        fig, ax = plt.subplots(figsize=(10, 6))
        times = [H * ROM_DT for H in HORIZONS]

        for pipeline, label, color, marker, ls in [
            ('raw', 'raw baseline', '#e74c3c', 'o', '-'),
            ('sqrt_uncorrected', '√ρ (uncorrected)', '#95a5a6', 's', '--'),
            ('sqrt_simplex', '√ρ + simplex', '#2ecc71', '^', '-'),
        ]:
            r2s = [next(r for r in results if r['horizon'] == H and r['pipeline'] == pipeline)['r2_mean']
                   for H in HORIZONS]
            stds = [next(r for r in results if r['horizon'] == H and r['pipeline'] == pipeline)['r2_std']
                    for H in HORIZONS]
            ax.errorbar(times, r2s, yerr=stds, label=label, color=color,
                       marker=marker, linewidth=2, markersize=8, capsize=4, linestyle=ls)

        ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.set_xlabel('Forecast horizon (s)', fontsize=13)
        ax.set_ylabel('R² (mean ± std over 26 tests)', fontsize=13)
        ax.set_title('Final Horizon Curve: raw vs √ρ+simplex', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'horizon_curve_r2.png'), dpi=150, bbox_inches='tight')
        print(f"  → Plot: {out_dir}/horizon_curve_r2.png")
        plt.close(fig)

        # Plot 2: Shift-aligned R² comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: R² and SA-R² for raw
        ax = axes[0]
        r2_raw = [next(r for r in results if r['horizon'] == H and r['pipeline'] == 'raw')['r2_mean']
                  for H in HORIZONS]
        sa_raw = [next(r for r in results if r['horizon'] == H and r['pipeline'] == 'raw')['r2_shift_mean']
                  for H in HORIZONS]
        ax.plot(times, r2_raw, 'o-', color='#e74c3c', linewidth=2, markersize=8, label='R² (standard)')
        ax.plot(times, sa_raw, 's--', color='#e74c3c', linewidth=2, markersize=8, label='R² (shift-aligned)', alpha=0.7)
        ax.fill_between(times, r2_raw, sa_raw, alpha=0.15, color='#e74c3c', label='phase drift gap')
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.set_xlabel('Forecast horizon (s)', fontsize=12)
        ax.set_ylabel('R²', fontsize=12)
        ax.set_title('raw baseline', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Right: R² and SA-R² for √ρ+simplex
        ax = axes[1]
        r2_simp = [next(r for r in results if r['horizon'] == H and r['pipeline'] == 'sqrt_simplex')['r2_mean']
                   for H in HORIZONS]
        sa_simp = [next(r for r in results if r['horizon'] == H and r['pipeline'] == 'sqrt_simplex')['r2_shift_mean']
                   for H in HORIZONS]
        ax.plot(times, r2_simp, 'o-', color='#2ecc71', linewidth=2, markersize=8, label='R² (standard)')
        ax.plot(times, sa_simp, 's--', color='#2ecc71', linewidth=2, markersize=8, label='R² (shift-aligned)', alpha=0.7)
        ax.fill_between(times, r2_simp, sa_simp, alpha=0.15, color='#2ecc71', label='phase drift gap')
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.set_xlabel('Forecast horizon (s)', fontsize=12)
        ax.set_title('√ρ + simplex', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        fig.suptitle('Shift-Aligned R² Diagnostic: Phase Drift vs Structural Distortion',
                     fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'shift_aligned_diagnostic.png'), dpi=150, bbox_inches='tight')
        print(f"  → Plot: {out_dir}/shift_aligned_diagnostic.png")
        plt.close(fig)

        # Plot 3: Mass error curve
        fig, ax = plt.subplots(figsize=(10, 5))
        for pipeline, label, color, marker in [
            ('raw', 'raw baseline', '#e74c3c', 'o'),
            ('sqrt_uncorrected', '√ρ (uncorrected)', '#95a5a6', 's'),
            ('sqrt_simplex', '√ρ + simplex', '#2ecc71', '^'),
        ]:
            merr = [next(r for r in results if r['horizon'] == H and r['pipeline'] == pipeline)['mass_err_mean']
                    for H in HORIZONS]
            ax.plot(times, merr, marker=marker, color=color, linewidth=2, markersize=8, label=label)

        ax.set_xlabel('Forecast horizon (s)', fontsize=13)
        ax.set_ylabel('Mass error % (final frame)', fontsize=13)
        ax.set_title('Mass Conservation: raw vs √ρ+simplex', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'mass_error_curve.png'), dpi=150, bbox_inches='tight')
        print(f"  → Plot: {out_dir}/mass_error_curve.png")
        plt.close(fig)

    except ImportError:
        print("  (matplotlib not available, skipping plots)")

    return results


if __name__ == '__main__':
    main()
