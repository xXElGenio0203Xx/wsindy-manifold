#!/usr/bin/env python3
"""
Export and visualize the best CF8 trial (Trial 8, seed=8096, R²=+0.920).

Re-runs the exact trial to capture density arrays (rho_test, rho_pred),
then generates:
  1. Side-by-side MP4 video  (truth vs prediction)
  2. Snapshot comparison PNG  (5 key time steps)
  3. Frame-wise R² curve PNG
  4. Exported .npz with all arrays

Usage:
  PYTHONPATH=src python scripts/export_best_CF8.py
"""

import numpy as np
import sys
import time
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import everything from the suite script
import importlib.util
suite_path = Path(__file__).parent / 'suite_CONT_v1.py'
spec = importlib.util.spec_from_file_location('suite', suite_path)
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)

from rectsim.rom_video_utils import (
    make_truth_vs_pred_density_video,
    make_density_snapshot_comparison,
)


def main():
    ROOT = Path(__file__).parent.parent

    # ── CF8 config (exact match to EXPERIMENTS dict) ──
    cfg = suite.EXPERIMENTS['CF8_sqrtSimplex_singleIC_longPrefix_d19_p5_H37']
    seed = 8096   # trial 8: 7000 + 8*137
    trial = 8

    H        = cfg['H']           # 37
    T_train  = cfg['T_train']     # 72.0
    rom_d    = cfg['rom_d']       # 19
    lag      = cfg['lag']         # 5
    alpha    = cfg['alpha']       # 1e-2
    transform = cfg['transform']  # 'sqrt'
    do_simplex = cfg['simplex']   # True

    T_total = T_train + H * suite.ROM_DT + 1.0  # enough time
    ic_params = suite.make_ic_params(seed)

    out_dir = ROOT / 'oscar_output' / 'CF8_best_trial_export'
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  EXPORT BEST CF8 TRIAL")
    print(f"  Trial {trial}, seed={seed}")
    print(f"  IC: center=({ic_params['center'][0]:.2f}, {ic_params['center'][1]:.2f}), "
          f"sigma={ic_params['sigma']:.2f}")
    print(f"  Config: d={rom_d}, p={lag}, α={alpha}, T_train={T_train}s, H={H}")
    print(f"  Transform: √ρ + simplex")
    print("=" * 70)

    # ── 1. Simulate ──
    print("\n[1/6] Running simulation...")
    t0 = time.time()
    rho_full, times_full = suite.run_single_sim(T_total, seed, 'gaussian', ic_params)
    print(f"  Simulation done in {time.time()-t0:.1f}s  "
          f"({rho_full.shape[0]} frames, shape={rho_full.shape})")

    # ── 2. Subsample + split ──
    print("\n[2/6] Subsampling and splitting...")
    rho_sub = rho_full[::suite.ROM_SUBSAMPLE]
    T_train_frames = min(int(round(T_train / suite.ROM_DT)), rho_sub.shape[0])
    T_forecast = min(H, rho_sub.shape[0] - T_train_frames)

    rho_train = rho_sub[:T_train_frames]
    rho_test  = rho_sub[T_train_frames:T_train_frames + T_forecast]
    M0 = float(rho_train[-1].sum())  # mass reference

    print(f"  Train: {T_train_frames} frames  |  Test: {T_forecast} frames")
    print(f"  Mass M0 = {M0:.2f}")

    X_train = rho_train.reshape(T_train_frames, -1)

    # ── 3. POD ──
    print("\n[3/6] Building POD basis...")
    pod_data = suite.build_pod_from_data(X_train, rom_d=rom_d, transform=transform)
    print(f"  R_POD={pod_data['R_POD']}, "
          f"energy_captured={pod_data['energy_captured']:.4f}")

    # ── 4. Train MVAR ──
    print("\n[4/6] Training MVAR...")
    if transform == 'sqrt':
        X_work = np.sqrt(X_train + pod_data['density_transform_eps'])
    else:
        X_work = X_train.copy()
    z_train = (X_work - pod_data['X_mean']) @ pod_data['U_r']

    model, _, r2_train, diag = suite.train_mvar([z_train], lag=lag, alpha=alpha)
    print(f"  r2_train={r2_train:.4f}")
    print(f"  n_samples={diag['n_samples']}, n_features={diag['n_features']}, "
          f"ratio={diag['sample_feature_ratio']:.2f}")
    print(f"  condition_number={diag['condition_number']:.2e}")

    if diag['ABORT']:
        print("  !! ABORTED")
        return

    # ── 5. Forecast + lift ──
    print("\n[5/6] Forecasting...")
    z_init = z_train[-lag:]
    z_forecast = suite.mvar_forecast(model, z_init, lag, T_forecast)

    rho_pred_flat = suite.inverse_transform(z_forecast, pod_data)
    rho_pred = rho_pred_flat.reshape(T_forecast, suite.DENSITY_NY, suite.DENSITY_NX)

    if do_simplex:
        for t in range(T_forecast):
            rho_pred[t] = suite.project_simplex(
                rho_pred[t].ravel(), M0
            ).reshape(suite.DENSITY_NY, suite.DENSITY_NX)

    # ── Metrics ──
    ss_tot = np.sum((rho_test - rho_test.mean()) ** 2)
    r2_overall = 1.0 - np.sum((rho_test - rho_pred) ** 2) / ss_tot if ss_tot > 1e-12 else 0.0
    mass_err = abs(rho_pred[-1].sum() - M0) / M0 * 100 if M0 > 1e-12 else 0.0
    r2_frames = suite.compute_frame_r2(rho_test, rho_pred)

    print(f"  R² overall = {r2_overall:+.4f}")
    print(f"  Mass error = {mass_err:.2f}%")
    print(f"  Frame R²: first={r2_frames[0]:+.4f}, last={r2_frames[-1]:+.4f}")

    # ── 6. Export + Visualize ──
    print("\n[6/6] Exporting and visualizing...")

    # Time axis for the forecast window
    t_start = T_train_frames * suite.ROM_DT
    times_forecast = t_start + np.arange(T_forecast) * suite.ROM_DT

    # 6a. NPZ export
    npz_path = out_dir / 'CF8_best_trial.npz'
    np.savez_compressed(
        npz_path,
        rho_test=rho_test,
        rho_pred=rho_pred,
        rho_train=rho_train,
        times_forecast=times_forecast,
        r2_frames=r2_frames,
        r2_overall=r2_overall,
        mass_err=mass_err,
        seed=seed,
        ic_params_center=np.array(ic_params['center']),
        ic_params_sigma=ic_params['sigma'],
    )
    print(f"  ✓ Saved {npz_path}  ({npz_path.stat().st_size/1024:.0f} KB)")

    # 6b. Side-by-side video
    mp4_path = out_dir / 'CF8_best_truth_vs_pred.mp4'
    make_truth_vs_pred_density_video(
        rho_test,
        rho_pred,
        mp4_path,
        fps=10,
        title=f"CF8 Best Trial (seed={seed}, R²={r2_overall:+.3f})",
        times=times_forecast,
    )

    # 6c. Snapshot comparison
    n_snaps = min(5, T_forecast)
    snap_indices = np.linspace(0, T_forecast - 1, n_snaps, dtype=int)
    snap_path = out_dir / 'CF8_best_snapshots.png'
    make_density_snapshot_comparison(
        rho_test,
        rho_pred,
        list(snap_indices),
        times_forecast,
        snap_path,
        title=f"CF8 Best Trial: Density Snapshots (R²={r2_overall:+.3f})",
    )
    print(f"  ✓ Saved {snap_path}")

    # 6d. Frame-wise R² curve
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times_forecast, r2_frames, 'b-', lw=1.5, label='Frame R²')
    ax.axhline(r2_overall, color='r', ls='--', lw=1, label=f'Overall R²={r2_overall:+.3f}')
    ax.axhline(0, color='gray', ls=':', lw=0.5)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title(f'CF8 Best Trial — Frame-wise R² (seed={seed})', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(min(r2_frames.min() - 0.05, -0.1), 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    r2_curve_path = out_dir / 'CF8_best_r2_curve.png'
    plt.savefig(r2_curve_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {r2_curve_path}")

    # Summary JSON
    summary = {
        'experiment': 'CF8_sqrtSimplex_singleIC_longPrefix_d19_p5_H37',
        'trial': trial,
        'seed': seed,
        'ic_params': {'center': list(ic_params['center']), 'sigma': ic_params['sigma']},
        'r2_overall': float(r2_overall),
        'r2_first_frame': float(r2_frames[0]),
        'r2_last_frame': float(r2_frames[-1]),
        'mass_err_pct': float(mass_err),
        'r2_train': float(r2_train),
        'T_train_frames': T_train_frames,
        'T_forecast_frames': T_forecast,
        'rom_d': rom_d,
        'lag': lag,
        'alpha': alpha,
        'outputs': {
            'npz': str(npz_path.relative_to(ROOT)),
            'video': str(mp4_path.relative_to(ROOT)),
            'snapshots': str(snap_path.relative_to(ROOT)),
            'r2_curve': str(r2_curve_path.relative_to(ROOT)),
        },
    }
    summary_path = out_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Saved {summary_path}")

    print("\n" + "=" * 70)
    print("  DONE — All outputs in:", out_dir)
    print("=" * 70)


if __name__ == '__main__':
    main()
