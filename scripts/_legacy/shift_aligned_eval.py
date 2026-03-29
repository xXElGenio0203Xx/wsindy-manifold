#!/usr/bin/env python3
"""
Shift-Aligned Evaluation (K11-K14)
===================================
Post-hoc evaluation that finds the optimal periodic shift between
predicted and true density fields on the 48×48 periodic domain.

At long horizons, MVAR may predict the correct spatial pattern
(cluster shape, density magnitude) but with a phase/position error
due to accumulated drift. This script quantifies:

  R2_shift_aligned = R²(shifted_pred, true)
  best_shift = (dx, dy) in grid cells
  shift_magnitude = sqrt(dx² + dy²) × (Lx/nx)  in physical units

Uses circular cross-correlation (FFT-based) for efficient search
over all integer shifts on the periodic grid.

Usage:
  python scripts/shift_aligned_eval.py --source_exp K4_v1_sqrtD19_p5_H162 --tag K11
  python scripts/shift_aligned_eval.py --source_exp K5_v1_sqrtD19_p5_H312 --tag K12
  python scripts/shift_aligned_eval.py --source_exp K9_v1_rawD19_p5_H162  --tag K13
  python scripts/shift_aligned_eval.py --source_exp K10_v1_rawD19_p5_H312 --tag K14
"""

import argparse
import json
import numpy as np
from pathlib import Path
from scipy.signal import fftconvolve
import pandas as pd


def circular_cross_correlation_2d(true_field, pred_field):
    """
    Compute 2D circular cross-correlation using FFT.
    
    For periodic domains, we want to find the shift (dx, dy) that maximizes
    the overlap between pred and true. This is equivalent to finding the
    argmax of the circular cross-correlation.
    
    Parameters
    ----------
    true_field : ndarray, shape (ny, nx)
    pred_field : ndarray, shape (ny, nx)
    
    Returns
    -------
    cross_corr : ndarray, shape (ny, nx)
        Cross-correlation for each shift. Peak location = best shift.
    """
    # FFT-based circular cross-correlation
    # corr(dx,dy) = IFFT(FFT(true) * conj(FFT(pred)))
    ft_true = np.fft.fft2(true_field)
    ft_pred = np.fft.fft2(pred_field)
    cross_corr = np.real(np.fft.ifft2(ft_true * np.conj(ft_pred)))
    return cross_corr


def find_best_shift(true_field, pred_field):
    """
    Find the integer grid shift (dy, dx) that maximizes agreement
    between pred and true on a periodic domain.
    
    Returns
    -------
    best_dy, best_dx : int
        Shift in grid cells (applied as np.roll)
    """
    cc = circular_cross_correlation_2d(true_field, pred_field)
    ny, nx = cc.shape
    best_idx = np.unravel_index(np.argmax(cc), cc.shape)
    best_dy, best_dx = best_idx
    
    # Convert to centered shifts: if shift > half-grid, it's a negative shift
    if best_dy > ny // 2:
        best_dy -= ny
    if best_dx > nx // 2:
        best_dx -= nx
    
    return best_dy, best_dx


def apply_periodic_shift(field, dy, dx):
    """Apply periodic (circular) shift to a 2D field."""
    return np.roll(np.roll(field, dy, axis=0), dx, axis=1)


def r2_score(true, pred):
    """Compute R² between flattened arrays."""
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    if ss_tot < 1e-15:
        return 0.0
    return 1.0 - ss_res / ss_tot


def evaluate_shift_aligned(source_dir, Lx=15.0, Ly=15.0):
    """
    Run shift-aligned evaluation on all test runs in source_dir.
    
    Parameters
    ----------
    source_dir : Path
        Path to experiment output (oscar_output/<experiment_name>)
    Lx, Ly : float
        Physical domain size for converting grid shifts to physical units
    
    Returns
    -------
    results : list of dict
        Per-test-run results
    summary : dict
        Aggregate statistics
    """
    test_dir = source_dir / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"No test directory found: {test_dir}")
    
    # Find all test runs
    test_runs = sorted([d for d in test_dir.iterdir() if d.is_dir() and d.name.startswith("test_")])
    
    if not test_runs:
        raise FileNotFoundError(f"No test_XXX directories found in {test_dir}")
    
    results = []
    
    for test_run_dir in test_runs:
        test_name = test_run_dir.name
        
        # Load true and predicted densities
        true_file = test_run_dir / "density_true.npz"
        pred_file = test_run_dir / "density_pred.npz"
        
        if not true_file.exists() or not pred_file.exists():
            print(f"  ⚠ Skipping {test_name}: missing density files")
            continue
        
        true_data = np.load(true_file)
        pred_data = np.load(pred_file)
        
        rho_true = true_data['rho']  # (T, ny, nx)
        rho_pred = pred_data['rho']  # (T, ny, nx)
        
        # Get forecast start index
        forecast_start_idx = int(pred_data.get('forecast_start_idx', 5))
        
        # Truncate to matching length
        T_min = min(rho_true.shape[0], rho_pred.shape[0])
        rho_true = rho_true[:T_min]
        rho_pred = rho_pred[:T_min]
        
        # Only evaluate forecast portion (after conditioning window)
        rho_true_fc = rho_true[forecast_start_idx:]
        rho_pred_fc = rho_pred[forecast_start_idx:]
        
        ny, nx = rho_true_fc.shape[1], rho_true_fc.shape[2]
        T_fc = rho_true_fc.shape[0]
        
        if T_fc == 0:
            continue
        
        # ---- Original R² (no shift) ----
        r2_original = r2_score(rho_true_fc.ravel(), rho_pred_fc.ravel())
        
        # ---- Per-frame best shift ----
        shifts_dy = np.zeros(T_fc, dtype=int)
        shifts_dx = np.zeros(T_fc, dtype=int)
        r2_per_frame_shifted = np.zeros(T_fc)
        r2_per_frame_original = np.zeros(T_fc)
        
        for t in range(T_fc):
            dy, dx = find_best_shift(rho_true_fc[t], rho_pred_fc[t])
            shifts_dy[t] = dy
            shifts_dx[t] = dx
            
            shifted_pred = apply_periodic_shift(rho_pred_fc[t], dy, dx)
            r2_per_frame_shifted[t] = r2_score(rho_true_fc[t].ravel(), shifted_pred.ravel())
            r2_per_frame_original[t] = r2_score(rho_true_fc[t].ravel(), rho_pred_fc[t].ravel())
        
        # ---- Global best shift (single shift for entire trajectory) ----
        # Use time-averaged cross-correlation
        cc_sum = np.zeros((ny, nx))
        for t in range(T_fc):
            cc_sum += circular_cross_correlation_2d(rho_true_fc[t], rho_pred_fc[t])
        
        best_idx = np.unravel_index(np.argmax(cc_sum), cc_sum.shape)
        global_dy, global_dx = best_idx
        if global_dy > ny // 2:
            global_dy -= ny
        if global_dx > nx // 2:
            global_dx -= nx
        
        # Apply global shift and compute R²
        rho_pred_global_shifted = np.array([
            apply_periodic_shift(rho_pred_fc[t], global_dy, global_dx)
            for t in range(T_fc)
        ])
        r2_global_shifted = r2_score(rho_true_fc.ravel(), rho_pred_global_shifted.ravel())
        
        # ---- Per-frame shifted, then aggregate R² ----
        rho_pred_perframe_shifted = np.array([
            apply_periodic_shift(rho_pred_fc[t], shifts_dy[t], shifts_dx[t])
            for t in range(T_fc)
        ])
        r2_perframe_shifted_agg = r2_score(rho_true_fc.ravel(), rho_pred_perframe_shifted.ravel())
        
        # Physical shift magnitudes
        dx_phys = Lx / nx
        dy_phys = Ly / ny
        shift_magnitudes = np.sqrt((shifts_dx * dx_phys)**2 + (shifts_dy * dy_phys)**2)
        global_shift_mag = np.sqrt((global_dx * dx_phys)**2 + (global_dy * dy_phys)**2)
        
        result = {
            'test_run': test_name,
            'r2_original': float(r2_original),
            'r2_global_shifted': float(r2_global_shifted),
            'r2_perframe_shifted': float(r2_perframe_shifted_agg),
            'global_shift_dy': int(global_dy),
            'global_shift_dx': int(global_dx),
            'global_shift_magnitude_phys': float(global_shift_mag),
            'mean_perframe_shift_magnitude_phys': float(np.mean(shift_magnitudes)),
            'max_perframe_shift_magnitude_phys': float(np.max(shift_magnitudes)),
            'r2_perframe_shifted_mean': float(np.mean(r2_per_frame_shifted)),
            'r2_perframe_original_mean': float(np.mean(r2_per_frame_original)),
            'n_forecast_steps': int(T_fc),
        }
        results.append(result)
        
        # Save per-frame details
        frame_df = pd.DataFrame({
            'frame': np.arange(T_fc),
            'shift_dy': shifts_dy,
            'shift_dx': shifts_dx,
            'shift_mag_phys': shift_magnitudes,
            'r2_original': r2_per_frame_original,
            'r2_shifted': r2_per_frame_shifted,
        })
        frame_df.to_csv(test_run_dir / "shift_aligned_per_frame.csv", index=False)
    
    # ---- Aggregate summary ----
    if results:
        r2_orig_arr = [r['r2_original'] for r in results]
        r2_glob_arr = [r['r2_global_shifted'] for r in results]
        r2_pf_arr = [r['r2_perframe_shifted'] for r in results]
        shift_arr = [r['global_shift_magnitude_phys'] for r in results]
        
        summary = {
            'n_test_runs': len(results),
            'r2_original_mean': float(np.mean(r2_orig_arr)),
            'r2_original_std': float(np.std(r2_orig_arr)),
            'r2_global_shifted_mean': float(np.mean(r2_glob_arr)),
            'r2_global_shifted_std': float(np.std(r2_glob_arr)),
            'r2_perframe_shifted_mean': float(np.mean(r2_pf_arr)),
            'r2_perframe_shifted_std': float(np.std(r2_pf_arr)),
            'r2_improvement_global': float(np.mean(r2_glob_arr) - np.mean(r2_orig_arr)),
            'r2_improvement_perframe': float(np.mean(r2_pf_arr) - np.mean(r2_orig_arr)),
            'mean_global_shift_phys': float(np.mean(shift_arr)),
            'max_global_shift_phys': float(np.max(shift_arr)),
        }
    else:
        summary = {'error': 'No valid test runs found'}
    
    return results, summary


def main():
    parser = argparse.ArgumentParser(description="Shift-Aligned Evaluation (Suite KNEE K11-K14)")
    parser.add_argument("--source_exp", required=True,
                        help="Source experiment name (e.g. K4_v1_sqrtD19_p5_H162)")
    parser.add_argument("--tag", required=True,
                        help="Output tag (e.g. K11, K12, K13, K14)")
    parser.add_argument("--base_dir", default="oscar_output",
                        help="Base output directory")
    parser.add_argument("--Lx", type=float, default=15.0)
    parser.add_argument("--Ly", type=float, default=15.0)
    args = parser.parse_args()
    
    source_dir = Path(args.base_dir) / args.source_exp
    if not source_dir.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        return 1
    
    print(f"=" * 60)
    print(f"Shift-Aligned Evaluation: {args.tag}")
    print(f"Source experiment: {args.source_exp}")
    print(f"Source directory: {source_dir}")
    print(f"=" * 60)
    
    results, summary = evaluate_shift_aligned(source_dir, Lx=args.Lx, Ly=args.Ly)
    
    # Save results
    output_dir = Path(args.base_dir) / f"{args.tag}_shift_aligned_{args.source_exp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "shift_aligned_results.json", 'w') as f:
        json.dump({'summary': summary, 'per_run': results}, f, indent=2)
    
    # Also save summary to source experiment directory
    with open(source_dir / f"shift_aligned_summary_{args.tag}.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {args.tag} (shift-aligned eval of {args.source_exp})")
    print(f"{'='*60}")
    print(f"  N test runs:           {summary.get('n_test_runs', 0)}")
    print(f"  R² original (mean):    {summary.get('r2_original_mean', 'N/A'):.4f}")
    print(f"  R² global-shifted:     {summary.get('r2_global_shifted_mean', 'N/A'):.4f}")
    print(f"  R² per-frame-shifted:  {summary.get('r2_perframe_shifted_mean', 'N/A'):.4f}")
    print(f"  Improvement (global):  {summary.get('r2_improvement_global', 'N/A'):+.4f}")
    print(f"  Improvement (per-frm): {summary.get('r2_improvement_perframe', 'N/A'):+.4f}")
    print(f"  Mean shift magnitude:  {summary.get('mean_global_shift_phys', 'N/A'):.3f} phys units")
    print(f"  Max shift magnitude:   {summary.get('max_global_shift_phys', 'N/A'):.3f} phys units")
    print(f"\nResults saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())
