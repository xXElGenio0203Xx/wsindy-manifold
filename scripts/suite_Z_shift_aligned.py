#!/usr/bin/env python3
"""
Suite Z — Shift-Aligned Diagnostics (eval-only on existing H162 outputs).

For each test run in a base experiment folder, computes:
  1. Per-timestep best shift (dx_t, dy_t) over ±max_shift pixels
  2. Shift-aligned R² (SA_R2) — best global shift over forecast
  3. Per-timestep shift-aligned R² trajectory
  4. Phase drift percentage: how much of the R² gap is explained by shift

Usage:
    python scripts/suite_Z_shift_aligned.py --base X3_V1_raw_H162 [--max-shift 3]
    
Reads from:  oscar_output/<base>/test/test_XXX/density_{pred_mvar,true}.npz
Writes to:   oscar_output/<Z_name>/  (JSON + CSV results)
"""

import argparse
import json
import os
import sys

import numpy as np
from itertools import product


# ─────────────────────────────────────────────────────────────
# SHIFT-ALIGNED COMPUTATION — per-timestep version
# ─────────────────────────────────────────────────────────────

def best_shift_per_timestep(rho_true, rho_pred, max_shift=3):
    """
    For each forecast timestep, find the (dx, dy) that minimizes MSE.
    
    Returns:
        dx_t: (T,) best x-shift per timestep
        dy_t: (T,) best y-shift per timestep
        r2_per_t_aligned: (T,) per-timestep R² at best shift
        r2_per_t_raw: (T,) per-timestep R² with no shift
    """
    T = rho_true.shape[0]
    shifts = range(-max_shift, max_shift + 1)
    
    dx_t = np.zeros(T, dtype=int)
    dy_t = np.zeros(T, dtype=int)
    r2_per_t_aligned = np.zeros(T)
    r2_per_t_raw = np.zeros(T)
    
    for t in range(T):
        frame_true = rho_true[t]
        frame_pred = rho_pred[t]
        ss_tot = np.sum((frame_true - frame_true.mean()) ** 2)
        
        if ss_tot < 1e-15:
            continue
        
        # Raw R² (no shift)
        ss_res_raw = np.sum((frame_true - frame_pred) ** 2)
        r2_per_t_raw[t] = 1.0 - ss_res_raw / ss_tot
        
        # Best shift
        best_r2 = -np.inf
        best_dx, best_dy = 0, 0
        for dy, dx in product(shifts, shifts):
            pred_shifted = np.roll(np.roll(frame_pred, dy, axis=0), dx, axis=1)
            ss_res = np.sum((frame_true - pred_shifted) ** 2)
            r2 = 1.0 - ss_res / ss_tot
            if r2 > best_r2:
                best_r2 = r2
                best_dx, best_dy = dx, dy
        
        dx_t[t] = best_dx
        dy_t[t] = best_dy
        r2_per_t_aligned[t] = best_r2
    
    return dx_t, dy_t, r2_per_t_aligned, r2_per_t_raw


def shift_aligned_r2_global(rho_true, rho_pred, max_shift=3):
    """
    Compute best GLOBAL R² over all integer pixel shifts (single shift for
    entire forecast, not per-timestep).
    
    Returns: (best_r2, best_dx, best_dy)
    """
    ss_tot = np.sum((rho_true - rho_true.mean()) ** 2)
    if ss_tot < 1e-15:
        return 0.0, 0, 0
    
    best_r2 = -np.inf
    best_dx, best_dy = 0, 0
    shifts = range(-max_shift, max_shift + 1)
    
    for dy, dx in product(shifts, shifts):
        pred_shifted = np.roll(np.roll(rho_pred, dy, axis=1), dx, axis=2)
        ss_res = np.sum((rho_true - pred_shifted) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        if r2 > best_r2:
            best_r2 = r2
            best_dx, best_dy = dx, dy
    
    return float(best_r2), int(best_dx), int(best_dy)


# ─────────────────────────────────────────────────────────────
# EVALUATE ONE TEST RUN
# ─────────────────────────────────────────────────────────────

def evaluate_run(run_dir, max_shift=3):
    """Load predictions + truth from one test run, compute shift diagnostics."""
    pred_path = os.path.join(run_dir, 'density_pred_mvar.npz')
    true_path = os.path.join(run_dir, 'density_true.npz')
    
    if not os.path.exists(pred_path) or not os.path.exists(true_path):
        return None
    
    d_pred = np.load(pred_path)
    d_true = np.load(true_path)
    
    rho_pred = d_pred['rho']
    rho_true = d_true['rho']
    fsi = int(d_pred['forecast_start_idx'])
    
    # Forecast portion only
    n_forecast = rho_pred.shape[0] - fsi
    true_f = rho_true[fsi:fsi + n_forecast]
    pred_f = rho_pred[fsi:fsi + n_forecast]
    L = min(true_f.shape[0], pred_f.shape[0])
    true_f = true_f[:L]
    pred_f = pred_f[:L]
    
    if L < 2:
        return None
    
    # Raw R² (global, no shift)
    ss_tot = np.sum((true_f - true_f.mean()) ** 2)
    if ss_tot < 1e-15:
        return None
    r2_raw = float(1.0 - np.sum((true_f - pred_f) ** 2) / ss_tot)
    
    # Global shift-aligned R²
    r2_sa_global, best_dx_global, best_dy_global = shift_aligned_r2_global(
        true_f, pred_f, max_shift=max_shift
    )
    
    # Per-timestep shift diagnostics
    dx_t, dy_t, r2_per_t_aligned, r2_per_t_raw = best_shift_per_timestep(
        true_f, pred_f, max_shift=max_shift
    )
    
    # Phase drift percent: fraction of R² deficit explained by shift alignment
    # phase_drift_% = (SA_R2 - raw_R2) / (1 - raw_R2) * 100
    # Interpretation: if raw R²=-0.1 and SA R²=+0.1, denominator=1.1, num=0.2 → 18%
    denom = max(1.0 - r2_raw, 1e-10)
    phase_drift_pct = (r2_sa_global - r2_raw) / denom * 100.0
    
    # Cumulative drift magnitude
    drift_magnitude = np.sqrt(dx_t.astype(float)**2 + dy_t.astype(float)**2)
    
    return {
        'n_forecast_steps': int(L),
        'forecast_start_idx': int(fsi),
        'r2_raw': r2_raw,
        'r2_sa_global': r2_sa_global,
        'best_dx_global': best_dx_global,
        'best_dy_global': best_dy_global,
        'phase_drift_pct': float(phase_drift_pct),
        'dx_t': dx_t.tolist(),
        'dy_t': dy_t.tolist(),
        'r2_per_t_raw': r2_per_t_raw.tolist(),
        'r2_per_t_aligned': r2_per_t_aligned.tolist(),
        'drift_magnitude_mean': float(drift_magnitude.mean()),
        'drift_magnitude_max': float(drift_magnitude.max()),
    }


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

EXPERIMENT_MAP = {
    'Z1': ('X3_V1_raw_H162',            'V1',   'raw'),
    'Z2': ('X4_V1_sqrtSimplex_H162',    'V1',   'sqrt+simplex'),
    'Z3': ('X7_V33_raw_H162',           'V3.3', 'raw'),
    'Z4': ('X8_V33_sqrtSimplex_H162',   'V3.3', 'sqrt+simplex'),
    'Z5': ('X11_V34_raw_H162',          'V3.4', 'raw'),
    'Z6': ('X12_V34_sqrtSimplex_H162',  'V3.4', 'sqrt+simplex'),
}


def main():
    parser = argparse.ArgumentParser(description='Suite Z: Shift-aligned diagnostics')
    parser.add_argument('--base', required=True,
                        help='Base experiment name (e.g. X3_V1_raw_H162)')
    parser.add_argument('--max-shift', type=int, default=3,
                        help='Max pixel shift in each direction (default: 3)')
    parser.add_argument('--root', default=None,
                        help='Workspace root (default: auto-detect)')
    args = parser.parse_args()
    
    root = args.root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Find which Z experiment this is
    z_name = None
    regime = None
    pipeline = None
    for zk, (base_name, reg, pipe) in EXPERIMENT_MAP.items():
        if base_name == args.base:
            z_name = zk
            regime = reg
            pipeline = pipe
            break
    
    if z_name is None:
        # Allow arbitrary base names
        z_name = f"Z_custom_{args.base}"
        regime = "unknown"
        pipeline = "unknown"
    
    full_z_name = f"{z_name}_shiftAlign_{regime.replace('.', '')}_{pipeline.replace('+', '')}_H162"
    
    test_dir = os.path.join(root, 'oscar_output', args.base, 'test')
    if not os.path.isdir(test_dir):
        print(f"ERROR: Test directory not found: {test_dir}")
        sys.exit(1)
    
    out_dir = os.path.join(root, 'oscar_output', full_z_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # Find test runs
    test_runs = sorted([
        d for d in os.listdir(test_dir)
        if d.startswith('test_') and os.path.isdir(os.path.join(test_dir, d))
    ])
    
    print(f"{'='*70}")
    print(f"Suite Z: Shift-Aligned Diagnostics")
    print(f"  Base experiment: {args.base}")
    print(f"  Z label:         {full_z_name}")
    print(f"  Regime:          {regime}")
    print(f"  Pipeline:        {pipeline}")
    print(f"  Max shift:       ±{args.max_shift} pixels")
    print(f"  Test runs:       {len(test_runs)}")
    print(f"{'='*70}")
    
    # Evaluate all runs
    all_results = []
    for i, run_name in enumerate(test_runs):
        run_dir = os.path.join(test_dir, run_name)
        result = evaluate_run(run_dir, max_shift=args.max_shift)
        if result is None:
            print(f"  [{i+1:3d}/{len(test_runs)}] {run_name}: SKIPPED (missing data)")
            continue
        
        result['run_name'] = run_name
        all_results.append(result)
        
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1:3d}/{len(test_runs)}] {run_name}: "
                  f"R²_raw={result['r2_raw']:+.4f}  "
                  f"R²_SA={result['r2_sa_global']:+.4f}  "
                  f"phase%={result['phase_drift_pct']:.1f}%  "
                  f"shift=({result['best_dx_global']},{result['best_dy_global']})")
    
    if not all_results:
        print("ERROR: No valid test runs found!")
        sys.exit(1)
    
    # ── Aggregate metrics ──
    r2_raws = [r['r2_raw'] for r in all_results]
    r2_sas = [r['r2_sa_global'] for r in all_results]
    phase_pcts = [r['phase_drift_pct'] for r in all_results]
    drift_mags = [r['drift_magnitude_mean'] for r in all_results]
    
    summary = {
        'z_name': full_z_name,
        'base_experiment': args.base,
        'regime': regime,
        'pipeline': pipeline,
        'max_shift': args.max_shift,
        'n_runs': len(all_results),
        'R2_raw_mean': float(np.mean(r2_raws)),
        'R2_raw_std': float(np.std(r2_raws)),
        'R2_SA_mean': float(np.mean(r2_sas)),
        'R2_SA_std': float(np.std(r2_sas)),
        'delta_R2_SA': float(np.mean(r2_sas) - np.mean(r2_raws)),
        'phase_drift_pct_mean': float(np.mean(phase_pcts)),
        'phase_drift_pct_std': float(np.std(phase_pcts)),
        'drift_magnitude_mean': float(np.mean(drift_mags)),
    }
    
    print(f"\n{'─'*70}")
    print(f"AGGREGATE ({len(all_results)} runs):")
    print(f"  R²_raw:           {summary['R2_raw_mean']:+.4f} ± {summary['R2_raw_std']:.4f}")
    print(f"  R²_SA (global):   {summary['R2_SA_mean']:+.4f} ± {summary['R2_SA_std']:.4f}")
    print(f"  ΔR² (SA – raw):   {summary['delta_R2_SA']:+.4f}")
    print(f"  Phase drift %:    {summary['phase_drift_pct_mean']:.1f}% ± {summary['phase_drift_pct_std']:.1f}%")
    print(f"  Mean drift mag:   {summary['drift_magnitude_mean']:.2f} pixels")
    print(f"{'─'*70}")
    
    # ── Save outputs ──
    
    # 1. Summary JSON
    summary_path = os.path.join(out_dir, 'shift_aligned_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  → Summary:    {summary_path}")
    
    # 2. Per-run results JSON
    per_run_path = os.path.join(out_dir, 'shift_aligned_per_run.json')
    with open(per_run_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  → Per-run:    {per_run_path}")
    
    # 3. Drift sequence CSV (averaged across runs)
    # Find the max forecast length
    max_len = max(r['n_forecast_steps'] for r in all_results)
    
    # Aggregate dx_t, dy_t, r2 per timestep (pad with NaN)
    dx_all = np.full((len(all_results), max_len), np.nan)
    dy_all = np.full((len(all_results), max_len), np.nan)
    r2_raw_all = np.full((len(all_results), max_len), np.nan)
    r2_sa_all = np.full((len(all_results), max_len), np.nan)
    
    for i, r in enumerate(all_results):
        n = r['n_forecast_steps']
        dx_all[i, :n] = r['dx_t']
        dy_all[i, :n] = r['dy_t']
        r2_raw_all[i, :n] = r['r2_per_t_raw']
        r2_sa_all[i, :n] = r['r2_per_t_aligned']
    
    csv_path = os.path.join(out_dir, 'drift_timeseries.csv')
    with open(csv_path, 'w') as f:
        f.write('step,time_s,dx_mean,dy_mean,drift_mag_mean,r2_raw_mean,r2_sa_mean,n_valid\n')
        ROM_DT = 0.12
        for t in range(max_len):
            valid = ~np.isnan(dx_all[:, t])
            n_valid = valid.sum()
            if n_valid == 0:
                continue
            dx_m = np.nanmean(dx_all[:, t])
            dy_m = np.nanmean(dy_all[:, t])
            mag_m = np.nanmean(np.sqrt(dx_all[:, t]**2 + dy_all[:, t]**2))
            r2r_m = np.nanmean(r2_raw_all[:, t])
            r2s_m = np.nanmean(r2_sa_all[:, t])
            f.write(f'{t},{t*ROM_DT:.2f},{dx_m:.3f},{dy_m:.3f},{mag_m:.3f},'
                    f'{r2r_m:.6f},{r2s_m:.6f},{n_valid}\n')
    print(f"  → Drift CSV:  {csv_path}")
    
    print(f"\nDone. {len(all_results)} runs evaluated.")
    return summary


if __name__ == '__main__':
    main()
