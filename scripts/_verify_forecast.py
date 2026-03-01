#!/usr/bin/env python3
"""Verify that R² metrics are computed on the FORECAST period only, not conditioning window."""
import numpy as np
import json
import csv
import sys
from pathlib import Path

EXPS = [
    "S2a_v1_sqrtD19_p5_H100_eta02_aligned",
    "S2b_v1_sqrtD19_p5_H100_eta0_aligned",
]
BASE = Path("oscar_output")
TEST_RUN = "test_000"

for exp_name in EXPS:
    exp_dir = BASE / exp_name
    run_dir = exp_dir / "test" / TEST_RUN
    if not run_dir.exists():
        print(f"SKIP {exp_name}: {run_dir} not found")
        continue

    print("=" * 70)
    print(f"EXPERIMENT: {exp_name}")
    print("=" * 70)

    # Load NPZs
    pred = np.load(run_dir / "density_pred.npz")
    truth = np.load(run_dir / "density_true.npz")

    pred_rho = pred["rho"]
    truth_rho = truth["rho"]
    pred_t = pred["times"]
    truth_t = truth["times"]
    fs_idx = int(pred["forecast_start_idx"])

    print(f"  pred  shape: {pred_rho.shape}   times: [{pred_t[0]:.2f} .. {pred_t[-1]:.2f}]")
    print(f"  truth shape: {truth_rho.shape}  times: [{truth_t[0]:.2f} .. {truth_t[-1]:.2f}]")
    print(f"  forecast_start_idx: {fs_idx}")
    print(f"  Conditioning window: {fs_idx} frames (indices 0..{fs_idx-1})")
    print(f"  Forecast region:     {pred_rho.shape[0] - fs_idx} frames (indices {fs_idx}..{pred_rho.shape[0]-1})")
    print(f"  dt_pred:  {pred_t[1]-pred_t[0]:.4f}")
    print(f"  dt_truth: {truth_t[1]-truth_t[0]:.4f}")
    print()

    # Extract forecast-only prediction
    forecast_pred = pred_rho[fs_idx:]
    forecast_times = pred_t[fs_idx:]

    # Match forecast times to truth times
    truth_indices = []
    for t in forecast_times:
        idx = np.argmin(np.abs(truth_t - t))
        truth_indices.append(idx)
    forecast_truth = truth_rho[truth_indices]

    print(f"  Forecast truth indices: [{truth_indices[0]}..{truth_indices[-1]}] of {truth_rho.shape[0]} total")
    print(f"  Forecast time range: [{forecast_times[0]:.2f} .. {forecast_times[-1]:.2f}]")
    print()

    # Manual R² on forecast only
    ss_res = np.sum((forecast_truth.flatten() - forecast_pred.flatten()) ** 2)
    ss_tot = np.sum((forecast_truth.flatten() - forecast_truth.flatten().mean()) ** 2)
    r2_forecast = 1 - ss_res / ss_tot
    print(f"  MANUAL R² (forecast only, {len(forecast_pred)} frames): {r2_forecast:.6f}")

    # Manual R² on FULL prediction (including conditioning = POD-reconstructed truth)
    # Map all pred times to truth
    all_truth_idx = []
    for t in pred_t:
        idx = np.argmin(np.abs(truth_t - t))
        all_truth_idx.append(idx)
    full_truth = truth_rho[all_truth_idx]
    ss_res_full = np.sum((full_truth.flatten() - pred_rho.flatten()) ** 2)
    ss_tot_full = np.sum((full_truth.flatten() - full_truth.flatten().mean()) ** 2)
    r2_full = 1 - ss_res_full / ss_tot_full
    print(f"  MANUAL R² (full incl. cond, {len(pred_rho)} frames):   {r2_full:.6f}")

    # Load reported metrics
    with open(run_dir / "metrics_summary.json") as f:
        metrics = json.load(f)
    r2_reported = metrics.get("r2_recon", "N/A")
    print(f"  REPORTED r2_recon:                                    {r2_reported}")
    print()

    # Which one matches?
    if isinstance(r2_reported, float):
        diff_forecast = abs(r2_reported - r2_forecast)
        diff_full = abs(r2_reported - r2_full)
        print(f"  |reported - manual_forecast| = {diff_forecast:.8f}")
        print(f"  |reported - manual_full|     = {diff_full:.8f}")
        if diff_forecast < diff_full:
            print(f"  >>> MATCHES FORECAST-ONLY ✓  (it's legit)")
        else:
            print(f"  >>> MATCHES FULL (incl. conditioning) ✗  WARNING: R² includes conditioning window!")
    print()

    # Check conditioning window: is it POD-reconstructed truth or raw truth?
    cond_pred = pred_rho[:fs_idx]
    cond_truth = truth_rho[all_truth_idx[:fs_idx]]
    cond_diff = np.abs(cond_pred - cond_truth).mean()
    cond_max = np.abs(cond_pred - cond_truth).max()
    print(f"  Conditioning check: mean|pred-truth| = {cond_diff:.6f}, max = {cond_max:.6f}")
    if cond_diff < 1e-6:
        print(f"    => Conditioning is RAW truth (perfect match)")
    elif cond_diff < 0.5:
        print(f"    => Conditioning is POD-reconstructed truth (small but nonzero diff)")
    else:
        print(f"    => Conditioning is NOT truth — something else going on")
    print()

    # R² vs time: check timestep range
    r2_csv = run_dir / "r2_vs_time.csv"
    if r2_csv.exists():
        with open(r2_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        cols = list(rows[0].keys()) if rows else []
        print(f"  r2_vs_time.csv: {len(rows)} rows, cols={cols}")
        if rows:
            print(f"    First: {rows[0]}")
            print(f"    Last:  {rows[-1]}")
            # Check if timesteps start from 0 (includes cond) or from forecast start
            step_col = [c for c in cols if 'step' in c.lower() or c == 't' or c == 'frame']
            if step_col:
                steps = [int(float(r[step_col[0]])) for r in rows]
                print(f"    Timestep range: {min(steps)} to {max(steps)} ({len(steps)} entries)")
                if min(steps) == 0:
                    print(f"    => Starts from 0 (may include conditioning)")
                else:
                    print(f"    => Starts from {min(steps)} (forecast-only)")

    print()
    print()

print("DONE.")
