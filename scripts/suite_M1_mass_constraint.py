#!/usr/bin/env python3
"""
SUITE M1 — MASS CONSTRAINT STRATEGIES (EVAL-ONLY)
==================================================
Re-evaluates existing K4/K5 predictions with three mass-correction strategies:

  1. scale_to_M0:           ρ* = ρ̂ · (M₀ / Σρ̂)
  2. offset_then_clamp:     ρ* = max(ρ̂ + c, 0)  where c solves Σmax(ρ̂+c,0)=M₀  (bisection)
  3. simplex_L2_projection: ρ* = argmin_{ρ≥0, Σρ=M₀} ‖ρ - ρ̂‖²  (Duchi et al. 2008)

Reads saved predictions from oscar_output/{base_experiment}/test/
Outputs per-experiment results to oscar_output/{M1_name}/

Usage:
    python scripts/suite_M1_mass_constraint.py
"""

import numpy as np
import os
import json
import csv
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────────────────────
# PROJECTION METHODS
# ─────────────────────────────────────────────────────────────

def project_scale(rho_flat: np.ndarray, M0: float) -> np.ndarray:
    """Global multiplicative scaling:  ρ* = ρ̂ · (M₀ / Σρ̂)"""
    m = rho_flat.sum()
    if m > 1e-15:
        return rho_flat * (M0 / m)
    else:
        # Degenerate: all zero → uniform
        return np.full_like(rho_flat, M0 / len(rho_flat))


def project_offset_clamp(rho_flat: np.ndarray, M0: float,
                         tol: float = 1e-8, max_iter: int = 200) -> np.ndarray:
    """
    Offset-then-clamp:  ρ* = max(ρ̂ + c, 0)  where c is found by bisection
    so that Σ ρ* = M₀.

    The function g(c) = Σ max(ρ̂ + c, 0) is continuous, piecewise-linear,
    and monotonically non-decreasing in c → bisection converges.
    """
    n = len(rho_flat)
    current_mass = rho_flat.sum()

    # If mass already matches (within tolerance), just clamp negatives
    if abs(current_mass - M0) < tol:
        return np.maximum(rho_flat, 0.0)

    def g(c):
        return np.maximum(rho_flat + c, 0.0).sum()

    # Bracket: need c_lo where g(c_lo) ≤ M0 and c_hi where g(c_hi) ≥ M0
    # If current mass < M0, we need c > 0 (add mass)
    # If current mass > M0, we need c < 0 (remove mass)

    # Lower bound: c = -max(rho) makes everything ≤ 0 → g = 0 ≤ M0
    c_lo = -rho_flat.max()
    # Upper bound: c = M0 (if all were 0, each pixel becomes c, sum = n*c)
    # We want g(c_hi) ≥ M0 → c_hi = M0/n works if rho_flat ≥ 0
    # More robust: c_hi such that sum(rho + c_hi) ≥ M0 → c_hi = (M0 - sum(rho))/n + max(|rho|)
    c_hi = max(M0 / n, rho_flat.max()) + abs(M0 - current_mass) / max(n, 1)

    # Ensure bracket
    while g(c_lo) > M0 + tol:
        c_lo *= 2
    while g(c_hi) < M0 - tol:
        c_hi *= 2

    # Bisection
    for _ in range(max_iter):
        c_mid = 0.5 * (c_lo + c_hi)
        val = g(c_mid)
        if abs(val - M0) < tol:
            break
        if val < M0:
            c_lo = c_mid
        else:
            c_hi = c_mid

    c_star = 0.5 * (c_lo + c_hi)
    return np.maximum(rho_flat + c_star, 0.0)


def project_simplex(rho_flat: np.ndarray, M0: float) -> np.ndarray:
    """
    Euclidean projection onto the simplex {ρ ≥ 0, Σρ = M₀}.

    This is the L2-nearest point in the nonneg+mass polytope.
    Algorithm: Duchi et al. (2008) "Efficient Projections onto the
    l1-Ball for Learning in High Dimensions", adapted to the simplex.

    1. Sort descending.
    2. Find ρ_max = max{j : rho_sorted[j] - (Σ_{i≤j} rho_sorted[i] - M0)/(j+1) > 0}
    3. θ = (Σ_{i ≤ ρ_max} rho_sorted[i] - M0) / (ρ_max + 1)
    4. ρ* = max(ρ̂ - θ, 0)
    """
    n = len(rho_flat)
    # Sort descending
    mu = np.sort(rho_flat)[::-1]
    cumsum = np.cumsum(mu)
    # Find rho_max: largest j s.t. mu[j] - (cumsum[j] - M0)/(j+1) > 0
    arange = np.arange(1, n + 1, dtype=np.float64)
    test = mu - (cumsum - M0) / arange
    rho_max = np.max(np.where(test > 0)[0]) if np.any(test > 0) else 0
    theta = (cumsum[rho_max] - M0) / (rho_max + 1)
    return np.maximum(rho_flat - theta, 0.0)


# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────

def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² over flattened arrays."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < 1e-15:
        return 0.0
    return 1.0 - ss_res / ss_tot


def compute_frame_metrics(rho_true: np.ndarray, rho_pred: np.ndarray, M0: float):
    """Per-frame R², mass error %, and negative fraction."""
    r2 = compute_r2(rho_true.ravel(), rho_pred.ravel())
    mass_err = abs(rho_pred.sum() - M0) / M0 * 100.0
    neg_frac = (rho_pred < 0).mean() * 100.0
    return r2, mass_err, neg_frac


# ─────────────────────────────────────────────────────────────
# EXPERIMENT DEFINITIONS
# ─────────────────────────────────────────────────────────────

EXPERIMENTS = [
    {
        'name': 'M1-1_eval_massFix_scale',
        'base': 'K4_v1_sqrtD19_p5_H162',
        'constraint': 'scale_to_M0',
    },
    {
        'name': 'M1-2_eval_massFix_offsetClamp',
        'base': 'K4_v1_sqrtD19_p5_H162',
        'constraint': 'offset_then_clamp_to_M0',
    },
    {
        'name': 'M1-3_eval_massFix_simplex',
        'base': 'K4_v1_sqrtD19_p5_H162',
        'constraint': 'simplex_L2_projection',
    },
    {
        'name': 'M1-4_eval_massFix_scale_H312',
        'base': 'K5_v1_sqrtD19_p5_H312',
        'constraint': 'scale_to_M0',
    },
    {
        'name': 'M1-5_eval_massFix_offsetClamp_H312',
        'base': 'K5_v1_sqrtD19_p5_H312',
        'constraint': 'offset_then_clamp_to_M0',
    },
    {
        'name': 'M1-6_eval_massFix_simplex_H312',
        'base': 'K5_v1_sqrtD19_p5_H312',
        'constraint': 'simplex_L2_projection',
    },
]


PROJECTORS = {
    'scale_to_M0':              project_scale,
    'offset_then_clamp_to_M0':  project_offset_clamp,
    'simplex_L2_projection':    project_simplex,
}


# ─────────────────────────────────────────────────────────────
# MAIN EVALUATION LOOP
# ─────────────────────────────────────────────────────────────

def run_experiment(exp_cfg: dict, root: str):
    name = exp_cfg['name']
    base = exp_cfg['base']
    constraint = exp_cfg['constraint']
    projector = PROJECTORS[constraint]

    base_test_dir = os.path.join(root, 'oscar_output', base, 'test')
    out_dir = os.path.join(root, 'oscar_output', name)
    os.makedirs(out_dir, exist_ok=True)

    test_runs = sorted([
        d for d in os.listdir(base_test_dir)
        if d.startswith('test_') and os.path.isdir(os.path.join(base_test_dir, d))
    ])

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  base={base}  constraint={constraint}  tests={len(test_runs)}")
    print(f"{'='*70}")

    # Accumulators for aggregate metrics
    all_r2_raw = []       # per-test overall R² (no correction)
    all_r2_fix = []       # per-test overall R² (with correction)
    all_mass_err_raw = [] # per-test final-frame mass error %
    all_mass_err_fix = []
    all_neg_raw = []
    all_neg_fix = []

    # For time-series curves (averaged over tests) — use list-of-arrays,
    # then stack at the end (all test runs have same n_forecast in practice).
    per_test_timeseries = []   # list of dicts with arrays
    per_test_results = []

    for run_name in test_runs:
        run_dir = os.path.join(base_test_dir, run_name)

        # Load data
        d_pred = np.load(os.path.join(run_dir, 'density_pred_mvar.npz'))
        d_true = np.load(os.path.join(run_dir, 'density_true.npz'))
        rho_pred = d_pred['rho'].copy()   # (T_pred, Ny, Nx)
        rho_true = d_true['rho']           # (T_true, Ny, Nx)
        fsi = int(d_pred['forecast_start_idx'])

        # M₀ from last teacher-forced ground-truth frame
        M0 = float(rho_true[max(fsi - 1, 0)].sum())

        # Extract forecast portion
        n_forecast = rho_pred.shape[0] - fsi
        true_f = rho_true[fsi:fsi + n_forecast]
        pred_f_raw = rho_pred[fsi:fsi + n_forecast]
        L = min(true_f.shape[0], pred_f_raw.shape[0])
        true_f = true_f[:L]
        pred_f_raw = pred_f_raw[:L]

        if L == 0:
            continue

        spatial_shape = pred_f_raw.shape[1:]  # (Ny, Nx)

        # Apply projection frame-by-frame
        pred_f_fix = np.empty_like(pred_f_raw)
        for t in range(L):
            flat = pred_f_raw[t].ravel()
            flat_fix = projector(flat, M0)
            pred_f_fix[t] = flat_fix.reshape(spatial_shape)

        # ── Per-frame metrics ──
        r2_t_raw = np.zeros(L)
        r2_t_fix = np.zeros(L)
        mass_t_raw = np.zeros(L)
        mass_t_fix = np.zeros(L)
        neg_t_raw = np.zeros(L)
        neg_t_fix = np.zeros(L)

        for t in range(L):
            r2_t_raw[t], mass_t_raw[t], neg_t_raw[t] = compute_frame_metrics(
                true_f[t], pred_f_raw[t], M0)
            r2_t_fix[t], mass_t_fix[t], neg_t_fix[t] = compute_frame_metrics(
                true_f[t], pred_f_fix[t], M0)

        # ── Overall R² (whole forecast flattened) ──
        ss_tot = np.sum((true_f - true_f.mean()) ** 2)
        if ss_tot < 1e-15:
            r2_overall_raw = 0.0
            r2_overall_fix = 0.0
        else:
            r2_overall_raw = 1.0 - np.sum((true_f - pred_f_raw) ** 2) / ss_tot
            r2_overall_fix = 1.0 - np.sum((true_f - pred_f_fix) ** 2) / ss_tot

        all_r2_raw.append(r2_overall_raw)
        all_r2_fix.append(r2_overall_fix)
        all_mass_err_raw.append(mass_t_raw[-1])
        all_mass_err_fix.append(mass_t_fix[-1])
        all_neg_raw.append(neg_t_raw[-1])
        all_neg_fix.append(neg_t_fix[-1])

        per_test_results.append({
            'test_run': run_name,
            'r2_raw': float(r2_overall_raw),
            'r2_fix': float(r2_overall_fix),
            'r2_delta': float(r2_overall_fix - r2_overall_raw),
            'mass_err_raw_final': float(mass_t_raw[-1]),
            'mass_err_fix_final': float(mass_t_fix[-1]),
            'neg_raw_final': float(neg_t_raw[-1]),
            'neg_fix_final': float(neg_t_fix[-1]),
            'M0': float(M0),
            'n_forecast_frames': int(L),
        })

        per_test_timeseries.append({
            'r2_raw': r2_t_raw, 'r2_fix': r2_t_fix,
            'mass_raw': mass_t_raw, 'mass_fix': mass_t_fix,
            'neg_raw': neg_t_raw, 'neg_fix': neg_t_fix,
        })

    n_tests = len(all_r2_raw)
    if n_tests == 0:
        print("  !! No valid test runs found")
        return

    # Average time-series (all tests should have same L, but handle edge cases)
    n_frames_max = max(len(ts['r2_raw']) for ts in per_test_timeseries)
    keys_ts = ['r2_raw', 'r2_fix', 'mass_raw', 'mass_fix', 'neg_raw', 'neg_fix']
    avg_ts = {k: np.zeros(n_frames_max) for k in keys_ts}
    count_ts = np.zeros(n_frames_max)
    for ts in per_test_timeseries:
        L = len(ts['r2_raw'])
        for k in keys_ts:
            avg_ts[k][:L] += ts[k]
        count_ts[:L] += 1
    for k in keys_ts:
        avg_ts[k] /= np.maximum(count_ts, 1)

    r2_vs_time_raw = avg_ts['r2_raw']
    r2_vs_time_fix = avg_ts['r2_fix']
    mass_vs_time_raw = avg_ts['mass_raw']
    mass_vs_time_fix = avg_ts['mass_fix']
    neg_vs_time_raw = avg_ts['neg_raw']
    neg_vs_time_fix = avg_ts['neg_fix']

    # ── Print summary ──
    r2_raw_mean = np.mean(all_r2_raw)
    r2_fix_mean = np.mean(all_r2_fix)
    delta = r2_fix_mean - r2_raw_mean
    sign = '+' if delta >= 0 else ''

    print(f"\n  R² raw:   {r2_raw_mean:.4f} ± {np.std(all_r2_raw):.4f}")
    print(f"  R² fix:   {r2_fix_mean:.4f} ± {np.std(all_r2_fix):.4f}")
    print(f"  Δ R²:     {sign}{delta:.4f}")
    print(f"  Mass err raw (final): {np.mean(all_mass_err_raw):.1f}%")
    print(f"  Mass err fix (final): {np.mean(all_mass_err_fix):.4f}%")
    print(f"  Neg% raw (final):     {np.mean(all_neg_raw):.2f}%")
    print(f"  Neg% fix (final):     {np.mean(all_neg_fix):.2f}%")
    improved = sum(1 for r, f in zip(all_r2_raw, all_r2_fix) if f > r)
    print(f"  Tests improved:       {improved}/{n_tests}")

    # ── Save outputs ──

    # 1. Summary JSON
    summary = {
        'experiment_name': name,
        'base_experiment': base,
        'constraint': constraint,
        'target_mass': 'M0_forecast_start',
        'n_tests': n_tests,
        'r2_raw_mean': float(r2_raw_mean),
        'r2_raw_std': float(np.std(all_r2_raw)),
        'r2_fix_mean': float(r2_fix_mean),
        'r2_fix_std': float(np.std(all_r2_fix)),
        'r2_delta': float(delta),
        'mass_err_raw_final_mean': float(np.mean(all_mass_err_raw)),
        'mass_err_fix_final_mean': float(np.mean(all_mass_err_fix)),
        'neg_pct_raw_final_mean': float(np.mean(all_neg_raw)),
        'neg_pct_fix_final_mean': float(np.mean(all_neg_fix)),
        'tests_improved': improved,
        'timestamp': datetime.now().isoformat(),
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # 2. Per-test results CSV
    with open(os.path.join(out_dir, 'per_test_results.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=per_test_results[0].keys())
        writer.writeheader()
        writer.writerows(per_test_results)

    # 3. Time-series CSV (averaged over tests)
    with open(os.path.join(out_dir, 'metrics_vs_time.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'frame', 'r2_raw', 'r2_fix',
            'mass_err_raw', 'mass_err_fix',
            'neg_pct_raw', 'neg_pct_fix'
        ])
        for t in range(n_frames_max):
            writer.writerow([
                t,
                f'{r2_vs_time_raw[t]:.6f}', f'{r2_vs_time_fix[t]:.6f}',
                f'{mass_vs_time_raw[t]:.4f}', f'{mass_vs_time_fix[t]:.4f}',
                f'{neg_vs_time_raw[t]:.4f}', f'{neg_vs_time_fix[t]:.4f}',
            ])

    print(f"  → Saved to {out_dir}/")
    return summary


def run_all(root: str):
    """Run all M1 experiments and produce a combined comparison table."""
    print("=" * 70)
    print("  SUITE M1 — MASS CONSTRAINT STRATEGIES (EVAL-ONLY)")
    print("=" * 70)

    summaries = []
    for exp_cfg in EXPERIMENTS:
        s = run_experiment(exp_cfg, root)
        if s:
            summaries.append(s)

    # ── Combined comparison table ──
    print("\n\n" + "=" * 90)
    print("  SUITE M1 — COMBINED RESULTS")
    print("=" * 90)
    header = f"{'Experiment':<40s} {'R²_raw':>8s} {'R²_fix':>8s} {'ΔR²':>8s} {'MassErr%':>9s} {'Neg%':>6s} {'Win':>5s}"
    print(header)
    print("-" * 90)

    for s in summaries:
        delta_str = f"{s['r2_delta']:+.4f}"
        print(
            f"  {s['experiment_name']:<38s} "
            f"{s['r2_raw_mean']:>7.4f} "
            f"{s['r2_fix_mean']:>7.4f} "
            f"{delta_str:>8s} "
            f"{s['mass_err_fix_final_mean']:>8.2f}% "
            f"{s['neg_pct_fix_final_mean']:>5.1f}% "
            f"{s['tests_improved']:>2d}/{s['n_tests']}"
        )

    # Save combined JSON
    combined_path = os.path.join(root, 'oscar_output', 'M1_combined_results.json')
    with open(combined_path, 'w') as f:
        json.dump(summaries, f, indent=2)
    print(f"\n  → Combined results saved to {combined_path}")


if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_all(root)
