#!/usr/bin/env python3
"""Verify that R² metrics in AL experiments are computed on FORECAST only, not conditioning."""
import numpy as np
import json
import sys
from pathlib import Path

BASE = Path("oscar_output")

# Find AL experiments that have density data
al_dirs = sorted(BASE.glob("AL[0-9]*"))
print(f"Found {len(al_dirs)} AL experiments\n")

for exp_dir in al_dirs:
    exp_name = exp_dir.name
    test_dir = exp_dir / "test"
    if not test_dir.exists():
        print(f"SKIP {exp_name}: no test dir")
        continue
    
    # Find test runs with density data
    test_runs_with_data = []
    for run_dir in sorted(test_dir.glob("test_*")):
        pred_f = run_dir / "density_pred_mvar.npz"
        if not pred_f.exists():
            pred_f = run_dir / "density_pred.npz"
        true_f = run_dir / "density_true.npz"
        if pred_f.exists() and true_f.exists():
            test_runs_with_data.append((run_dir, pred_f, true_f))
    
    if not test_runs_with_data:
        # Just show metrics from json
        test0 = test_dir / "test_000" / "metrics_summary.json"
        if test0.exists():
            with open(test0) as f:
                m = json.load(f)
            print(f"{'='*80}")
            print(f"EXPERIMENT: {exp_name} (no density data, metrics only)")
            print(f"  r2_recon={m['r2_recon']:.4f}  r2_latent={m['r2_latent']:.4f}")
            print(f"  r2_pod={m['r2_pod']:.4f}  r2_1step={m['r2_1step']:.4f}")
            print(f"  shift_align={m.get('shift_align', 'N/A')}")
            print()
        continue
    
    print(f"{'='*80}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"  {len(test_runs_with_data)} test runs have density data")
    print()
    
    for run_dir, pred_f, true_f in test_runs_with_data[:3]:  # Check up to 3
        run_name = run_dir.name
        
        pred = np.load(pred_f)
        truth = np.load(true_f)
        with open(run_dir / "metrics_summary.json") as f:
            metrics = json.load(f)
        
        rho_pred = pred['rho']
        rho_true = truth['rho']
        fsi = int(pred['forecast_start_idx']) if 'forecast_start_idx' in pred else 0
        times_pred = pred['times']
        times_true = truth['times']
        
        print(f"  --- {run_name} ---")
        print(f"  pred shape: {rho_pred.shape}  times: [{times_pred[0]:.2f} .. {times_pred[-1]:.2f}]")
        print(f"  true shape: {rho_true.shape}  times: [{times_true[0]:.2f} .. {times_true[-1]:.2f}]")
        print(f"  forecast_start_idx: {fsi} ({fsi}/{rho_pred.shape[0]} = {fsi/rho_pred.shape[0]*100:.0f}% conditioning)")
        
        # Temporal alignment
        dt_true = float(times_true[1] - times_true[0])
        dt_pred = float(times_pred[1] - times_pred[0])
        rom_sub = max(1, round(dt_pred / dt_true))
        print(f"  dt_true={dt_true:.4f}, dt_pred={dt_pred:.4f}, rom_subsample={rom_sub}")
        
        # Subsample truth to pred resolution
        truth_sub = rho_true[::rom_sub]
        
        # ---- R² on forecast only ----
        forecast_pred = rho_pred[fsi:]
        forecast_truth = truth_sub[fsi:fsi+len(forecast_pred)]
        L = min(len(forecast_pred), len(forecast_truth))
        forecast_pred = forecast_pred[:L]
        forecast_truth = forecast_truth[:L]
        
        ss_res_fc = np.sum((forecast_truth.flatten() - forecast_pred[:L].flatten())**2)
        ss_tot_fc = np.sum((forecast_truth.flatten() - forecast_truth.flatten().mean())**2)
        r2_forecast = 1 - ss_res_fc / ss_tot_fc if ss_tot_fc > 0 else float('nan')
        
        # ---- R² on full (incl conditioning) ----
        full_truth = truth_sub[:len(rho_pred)]
        L2 = min(len(rho_pred), len(full_truth))
        ss_res_full = np.sum((full_truth[:L2].flatten() - rho_pred[:L2].flatten())**2)
        ss_tot_full = np.sum((full_truth[:L2].flatten() - full_truth[:L2].flatten().mean())**2)
        r2_full = 1 - ss_res_full / ss_tot_full if ss_tot_full > 0 else float('nan')
        
        print(f"  MANUAL R² (forecast-only, {L} frames): {r2_forecast:.6f}")
        print(f"  MANUAL R² (full+cond, {L2} frames):    {r2_full:.6f}")
        print(f"  REPORTED r2_recon:                      {metrics['r2_recon']:.6f}")
        
        diff_fc = abs(metrics['r2_recon'] - r2_forecast)
        diff_full = abs(metrics['r2_recon'] - r2_full)
        print(f"  |reported - manual_forecast| = {diff_fc:.8f}")
        print(f"  |reported - manual_full|     = {diff_full:.8f}")
        if diff_fc < diff_full:
            print(f"  >>> MATCHES FORECAST-ONLY ✓ (legit)")
        else:
            print(f"  >>> MATCHES FULL (includes conditioning!) ✗ WARNING")
        
        # Check conditioning window: POD truth vs raw truth?
        if fsi > 0:
            cond_pred = rho_pred[:fsi]
            cond_truth = truth_sub[:fsi]
            L3 = min(len(cond_pred), len(cond_truth))
            if L3 > 0:
                diff_cond = np.abs(cond_pred[:L3] - cond_truth[:L3]).mean()
                max_diff = np.abs(cond_pred[:L3] - cond_truth[:L3]).max()
                print(f"  Conditioning diff: mean={diff_cond:.6f}, max={max_diff:.6f}")
                if diff_cond < 1e-6:
                    print(f"    => RAW truth (exact)")
                elif diff_cond < 1.0:
                    print(f"    => POD-reconstructed truth (small diff)")
                else:
                    print(f"    => Large differences in conditioning window")
        
        # Check: is the forecast actually different from truth?
        fc_diff = np.abs(forecast_pred - forecast_truth[:len(forecast_pred)]).mean()
        print(f"  Mean |forecast_pred - truth|: {fc_diff:.6f}")
        
        # Also check r2_pod — this should be the POD ceiling (no MVAR needed)
        print(f"  r2_pod (POD ceiling): {metrics['r2_pod']:.4f}")
        print(f"  r2_1step (teacher-forced): {metrics['r2_1step']:.4f}")
        print(f"  r2_latent: {metrics['r2_latent']:.4f}")
        print()

# Summary of what each metric means  
print("="*80)
print("METRIC DEFINITIONS (from test_evaluator.py):")
print("="*80)
print("""
r2_recon (R² reconstructed / R² rollout):
  Compare MVAR-forecast density (lifted back to physical) vs ground truth density.
  Computed on FORECAST region only (t > T_train). This is the main metric.
  Pipeline: latent IC → MVAR autoregressive forecast → inverse standardize → POD lift → inverse density transform.

r2_latent (R² latent):
  Compare MVAR-forecast latent coefficients vs true latent coefficients.
  Computed in the RAW (un-standardized) latent space, FORECAST only.
  
r2_pod (R² POD ceiling):
  Project TRUE latent coefficients through POD reconstruction, compare to TRUE density.
  This measures the BEST possible R² with this number of POD modes. No MVAR involved.
  If r2_pod < 1, it means we're losing info in the POD truncation.
  
r2_1step (R² one-step teacher-forced):
  At each forecast time, feed TRUE latent history to MVAR, predict 1 step ahead.
  This isolates single-step accuracy from autoregressive drift.
  
r2_kstep_density:
  Same as r2_recon but periodically reset to true latent every k steps.

NOTES:
- All R² metrics use forecast region only (indices T_train onward)  
- The saved density_pred.npz contains conditioning + forecast (for video continuity)
- forecast_start_idx marks where conditioning ends and MVAR forecast begins
- The conditioning window IS the POD reconstruction of truth (not raw truth)
""")
