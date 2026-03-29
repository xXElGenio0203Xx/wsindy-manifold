#!/usr/bin/env python3
"""
Reconstruction-Floor Diagnostic
================================

Decomposes the total error into:
  1. Conditioning error  (POD ↔ truth, no MVAR — this is the hard floor)
  2. Forecast error      (MVAR rollout ↔ truth)

And attributes the conditioning floor across three postprocessing pipelines:
  A. raw-density POD  (d=19, no transform, clamp negatives)
  B. √ρ POD           (d=19, sqrt transform, clamp negatives, NO simplex)
  C. √ρ POD + simplex (d=19, sqrt transform, simplex projection)

Outputs:
  - Console table of per-test-run and mean conditioning / forecast RMSE
  - CSV with time-resolved RMSE for both regimes
  - PNG plot showing conditioning vs forecast RMSE curves

Usage:
    python scripts/diagnose_reconstruction_floor.py \
        --run-dir oscar_output/CF8_longPrefix_sqrtSimplex_H37

    Optional:
        --n-test N          Override number of test runs (default: auto-detect)
        --out-dir PATH      Where to write outputs (default: <run-dir>/diagnostics)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_simplex(v_flat, mass_target):
    """Duchi et al. (2008) simplex projection: v >= 0, sum(v) = mass_target."""
    n = len(v_flat)
    if mass_target <= 0:
        return np.zeros_like(v_flat)
    u = np.sort(v_flat)[::-1]
    cssv = np.cumsum(u) - mass_target
    rho_idx = np.nonzero(u * np.arange(1, n + 1) > cssv)[0][-1]
    theta = cssv[rho_idx] / (rho_idx + 1.0)
    return np.maximum(v_flat - theta, 0.0)


def rmse_per_frame(pred, truth):
    """RMSE per frame, shape [T]."""
    diff = (pred - truth).reshape(pred.shape[0], -1)
    return np.sqrt(np.mean(diff ** 2, axis=1))


def r2_per_frame(pred, truth):
    """R² per frame, shape [T]."""
    truth_flat = truth.reshape(truth.shape[0], -1)
    pred_flat = pred.reshape(pred.shape[0], -1)
    ss_res = np.sum((truth_flat - pred_flat) ** 2, axis=1)
    ss_tot = np.sum((truth_flat - truth_flat.mean(axis=1, keepdims=True)) ** 2, axis=1)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-30)


# ---------------------------------------------------------------------------
# Core reconstruction functions
# ---------------------------------------------------------------------------

def reconstruct_pod(density_flat, U_r, X_mean, density_transform, eps):
    """
    Project → lift → inverse-transform — pure POD, no MVAR.
    Returns reconstructed density in ORIGINAL (raw ρ) space.
    """
    # Forward transform
    if density_transform == 'sqrt':
        transformed = np.sqrt(density_flat + eps)
    elif density_transform == 'log':
        transformed = np.log(density_flat + eps)
    else:
        transformed = density_flat.copy()

    # Center, project, lift
    centered = transformed - X_mean
    latent = centered @ U_r               # [T, d]
    recon_tf = latent @ U_r.T + X_mean    # [T, N_spatial]

    # Inverse transform
    if density_transform == 'sqrt':
        recon = np.maximum(recon_tf, 0.0) ** 2 - eps
    elif density_transform == 'log':
        recon = np.exp(recon_tf) - eps
    else:
        recon = recon_tf.copy()

    return recon


def postprocess_clamp(recon_flat):
    """C1-style: clamp negatives to 0, no renorm."""
    return np.maximum(recon_flat, 0.0)


def postprocess_c2(recon_flat):
    """C2-style: clamp negatives + mass renormalization per frame."""
    out = recon_flat.copy()
    for t in range(out.shape[0]):
        mass_before = out[t].sum()
        out[t] = np.maximum(out[t], 0.0)
        mass_after = out[t].sum()
        if mass_after > 0 and mass_before > 0:
            out[t] *= mass_before / mass_after
    return out


def postprocess_simplex(recon_flat, truth_flat, forecast_start_idx):
    """Simplex projection with M₀ from truth at forecast start."""
    out = recon_flat.copy()
    M0 = truth_flat[max(forecast_start_idx - 1, 0)].sum()
    for t in range(out.shape[0]):
        out[t] = _project_simplex(out[t], M0)
    return out


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def run_diagnostic(run_dir, n_test_override=None, out_dir=None):
    run_dir = Path(run_dir)
    rom_dir = run_dir / "rom_common"
    test_dir = run_dir / "test"

    if out_dir is None:
        out_dir = run_dir / "diagnostics"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load POD basis ----
    pod = np.load(rom_dir / "pod_basis.npz")
    U_r = pod['U']                              # (N_spatial, d)
    X_mean_sqrt = np.load(rom_dir / "X_train_mean.npy")  # mean in √ρ space

    # Read config to get params
    cfg_path = run_dir / "config_used.yaml"
    if cfg_path.exists():
        import yaml
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        rom_subsample = cfg.get('rom', {}).get('subsample', 3)
        density_transform_eps = cfg.get('rom', {}).get('density_transform_eps', 1e-10)
        d_modes = cfg.get('rom', {}).get('fixed_modes', 19)
        forecast_start_time = cfg.get('eval', {}).get('forecast_start', 72.0)
        sim_dt = cfg.get('sim', {}).get('dt', 0.04)
    else:
        rom_subsample = 3
        density_transform_eps = 1e-10
        d_modes = 19
        forecast_start_time = 72.0
        sim_dt = 0.04

    eps = density_transform_eps

    # ---- Build raw-density POD basis from available data ----
    # The saved POD is in √ρ space.  For the raw-density comparison we need
    # a raw-space POD.  Training density files may have been purged (OSCAR
    # export), so we build from whatever is available.
    print("Building raw-density POD basis for comparison ...")
    train_dir = run_dir / "train"
    train_runs_with_density = sorted([
        d for d in train_dir.iterdir()
        if d.name.startswith("train_") and (d / "density.npz").exists()
    ])

    if len(train_runs_with_density) >= 10:
        # Enough training data available
        source_label = f"{len(train_runs_with_density)} training runs"
        snap_sources = train_runs_with_density
        density_file = 'density.npz'
    else:
        # Fall back to test truth (sufficient for POD at d=19)
        print(f"  Only {len(train_runs_with_density)} train density files; "
              f"using test truth data instead")
        snap_sources = sorted([
            d for d in test_dir.iterdir()
            if d.name.startswith("test_") and (d / "density_true.npz").exists()
        ])
        source_label = f"{len(snap_sources)} test runs (proxy)"
        density_file = 'density_true.npz'

    raw_snapshots = []
    for tdir in snap_sources:
        rho = np.load(tdir / density_file)['rho']
        if rom_subsample > 1:
            rho = rho[::rom_subsample]
        raw_snapshots.append(rho.reshape(rho.shape[0], -1))
    raw_all = np.vstack(raw_snapshots)
    n_train = len(snap_sources)
    X_mean_raw = raw_all.mean(axis=0)
    raw_centered = raw_all - X_mean_raw
    U_raw_full, _, _ = np.linalg.svd(raw_centered.T, full_matrices=False)
    U_r_raw = U_raw_full[:, :d_modes]
    del raw_all, raw_centered, raw_snapshots  # free memory
    print(f"  ✓ Raw-density POD basis: d={d_modes}  from {source_label}  "
          f"(U shape {U_r_raw.shape})")

    # ---- Detect test runs ----
    test_runs = sorted([d for d in test_dir.iterdir()
                        if d.is_dir() and d.name.startswith("test_")])
    n_test = n_test_override if n_test_override else len(test_runs)
    test_runs = test_runs[:n_test]
    print(f"\nRunning diagnostic on {n_test} test runs ...\n")

    # ---- Per-run, per-frame metrics ----
    # We'll collect per-run summaries + accumulate frame curves
    summary_rows = []
    frame_curves = {
        'cond_rmse_raw': [],
        'cond_rmse_sqrt_clamp': [],
        'cond_rmse_sqrt_simplex': [],
        'forecast_rmse_sqrt_simplex': [],
        'cond_r2_raw': [],
        'cond_r2_sqrt_clamp': [],
        'cond_r2_sqrt_simplex': [],
        'forecast_r2_sqrt_simplex': [],
    }

    for idx, tdir in enumerate(test_runs):
        # Load truth (full resolution)
        truth_data = np.load(tdir / "density_true.npz")
        rho_true_full = truth_data['rho']       # (T_full, 48, 48)
        times_full = truth_data['times']

        # Subsample to ROM dt
        if rom_subsample > 1:
            rho_true = rho_true_full[::rom_subsample]
            times = times_full[::rom_subsample]
        else:
            rho_true = rho_true_full
            times = times_full

        T_rom = rho_true.shape[0]
        nx, ny = rho_true.shape[1], rho_true.shape[2]
        truth_flat = rho_true.reshape(T_rom, -1)

        # Forecast start index in ROM frames
        fsi = int(forecast_start_time / (sim_dt * rom_subsample))

        # ================================================================
        # A) Raw-density POD reconstruction (full trajectory)
        # ================================================================
        recon_raw = reconstruct_pod(truth_flat, U_r_raw, X_mean_raw,
                                     density_transform='raw', eps=0)
        recon_raw = postprocess_c2(recon_raw)  # clamp + renorm (standard)

        # ================================================================
        # B) √ρ POD reconstruction, clamp only (no simplex)
        # ================================================================
        recon_sqrt_clamp = reconstruct_pod(truth_flat, U_r, X_mean_sqrt,
                                            density_transform='sqrt', eps=eps)
        recon_sqrt_clamp = postprocess_c2(recon_sqrt_clamp)

        # ================================================================
        # C) √ρ POD reconstruction + simplex
        # ================================================================
        recon_sqrt_simplex = reconstruct_pod(truth_flat, U_r, X_mean_sqrt,
                                              density_transform='sqrt', eps=eps)
        recon_sqrt_simplex = postprocess_simplex(recon_sqrt_simplex, truth_flat, fsi)

        # ================================================================
        # D) MVAR forecast (load saved prediction)
        # ================================================================
        pred_data = np.load(tdir / "density_pred_mvar.npz")
        pred_rho = pred_data['rho']             # (642, 48, 48) — cond + forecast
        pred_fsi = int(pred_data['forecast_start_idx'])
        pred_flat = pred_rho.reshape(pred_rho.shape[0], -1)

        # ---- Frame-level RMSE & R² ----
        # Conditioning window: [0, fsi)
        cond_rmse_raw = rmse_per_frame(
            recon_raw[:fsi].reshape(-1, nx, ny), rho_true[:fsi])
        cond_rmse_sqrt_clamp = rmse_per_frame(
            recon_sqrt_clamp[:fsi].reshape(-1, nx, ny), rho_true[:fsi])
        cond_rmse_sqrt_simplex = rmse_per_frame(
            recon_sqrt_simplex[:fsi].reshape(-1, nx, ny), rho_true[:fsi])
        
        cond_r2_raw = r2_per_frame(
            recon_raw[:fsi].reshape(-1, nx, ny), rho_true[:fsi])
        cond_r2_sqrt_clamp = r2_per_frame(
            recon_sqrt_clamp[:fsi].reshape(-1, nx, ny), rho_true[:fsi])
        cond_r2_sqrt_simplex = r2_per_frame(
            recon_sqrt_simplex[:fsi].reshape(-1, nx, ny), rho_true[:fsi])

        # Forecast window: [fsi, end)
        forecast_pred = pred_flat[pred_fsi:]
        forecast_truth = rho_true[fsi:fsi + forecast_pred.shape[0]]
        H = forecast_truth.shape[0]
        forecast_rmse = rmse_per_frame(
            forecast_pred.reshape(-1, nx, ny), forecast_truth)
        forecast_r2 = r2_per_frame(
            forecast_pred.reshape(-1, nx, ny), forecast_truth)

        # Also get the conditioning-portion RMSE from the saved pred (POD-recon for cond window)
        # This is the actual pipeline conditioning error (matches variant C)
        cond_pred = pred_flat[:pred_fsi]
        cond_truth = rho_true[:fsi]
        cond_len = min(cond_pred.shape[0], cond_truth.shape[0])
        pipeline_cond_rmse = rmse_per_frame(
            cond_pred[:cond_len].reshape(-1, nx, ny), cond_truth[:cond_len])

        # Accumulate frame curves (use first run's length as reference)
        frame_curves['cond_rmse_raw'].append(cond_rmse_raw)
        frame_curves['cond_rmse_sqrt_clamp'].append(cond_rmse_sqrt_clamp)
        frame_curves['cond_rmse_sqrt_simplex'].append(cond_rmse_sqrt_simplex)
        frame_curves['forecast_rmse_sqrt_simplex'].append(forecast_rmse)
        frame_curves['cond_r2_raw'].append(cond_r2_raw)
        frame_curves['cond_r2_sqrt_clamp'].append(cond_r2_sqrt_clamp)
        frame_curves['cond_r2_sqrt_simplex'].append(cond_r2_sqrt_simplex)
        frame_curves['forecast_r2_sqrt_simplex'].append(forecast_r2)

        # Per-run summary
        row = {
            'test_idx': idx,
            'cond_rmse_raw': float(np.mean(cond_rmse_raw)),
            'cond_rmse_sqrt_clamp': float(np.mean(cond_rmse_sqrt_clamp)),
            'cond_rmse_sqrt_simplex': float(np.mean(cond_rmse_sqrt_simplex)),
            'forecast_rmse_sqrt_simplex': float(np.mean(forecast_rmse)),
            'cond_r2_raw': float(np.mean(cond_r2_raw)),
            'cond_r2_sqrt_clamp': float(np.mean(cond_r2_sqrt_clamp)),
            'cond_r2_sqrt_simplex': float(np.mean(cond_r2_sqrt_simplex)),
            'forecast_r2_sqrt_simplex': float(np.mean(forecast_r2)),
            'pipeline_cond_rmse': float(np.mean(pipeline_cond_rmse)),
            'H': H,
        }
        summary_rows.append(row)

        if idx < 3 or idx == n_test - 1:
            print(f"  test_{idx:03d}: cond_raw={row['cond_rmse_raw']:.4f}  "
                  f"cond_√ρ_clamp={row['cond_rmse_sqrt_clamp']:.4f}  "
                  f"cond_√ρ_simplex={row['cond_rmse_sqrt_simplex']:.4f}  "
                  f"forecast={row['forecast_rmse_sqrt_simplex']:.4f}")
        elif idx == 3:
            print("  ...")

    # ---- Aggregate ----
    print("\n" + "=" * 80)
    print("RECONSTRUCTION-FLOOR DIAGNOSTIC — SUMMARY")
    print("=" * 80)

    keys_of_interest = [
        ('cond_rmse_raw',            'Conditioning RMSE (raw-density POD d=19)'),
        ('cond_rmse_sqrt_clamp',     'Conditioning RMSE (√ρ POD d=19, C2 clamp)'),
        ('cond_rmse_sqrt_simplex',   'Conditioning RMSE (√ρ POD d=19, simplex)'),
        ('pipeline_cond_rmse',       'Conditioning RMSE (saved pipeline pred)'),
        ('forecast_rmse_sqrt_simplex', 'Forecast RMSE (MVAR √ρ+simplex)'),
    ]

    means = {}
    for key, label in keys_of_interest:
        vals = [r[key] for r in summary_rows]
        m = np.mean(vals)
        s = np.std(vals)
        means[key] = m
        print(f"  {label:50s}  {m:.4f} ± {s:.4f}")

    # Attribution
    print("\n--- Error Attribution ---")
    floor_raw = means['cond_rmse_raw']
    floor_sqrt = means['cond_rmse_sqrt_clamp']
    floor_simplex = means['cond_rmse_sqrt_simplex']
    forecast = means['forecast_rmse_sqrt_simplex']

    print(f"  Raw-density POD floor (truncation only):       {floor_raw:.4f}")
    print(f"  √ρ-transform adds:                            +{floor_sqrt - floor_raw:.4f}  "
          f"(transform nonlinearity)")
    print(f"  Simplex projection adds:                       +{floor_simplex - floor_sqrt:.4f}  "
          f"(postprocessing distortion)")
    print(f"  MVAR dynamics add:                             +{forecast - floor_simplex:.4f}  "
          f"(forecast degradation beyond floor)")
    print(f"\n  Total forecast RMSE:                           {forecast:.4f}")
    print(f"  Conditioning floor (√ρ+simplex):               {floor_simplex:.4f}")
    print(f"  Floor as % of forecast RMSE:                   "
          f"{floor_simplex / forecast * 100:.1f}%")

    # R² version
    print("\n--- R² Summary ---")
    r2_keys = [
        ('cond_r2_raw',              'Conditioning R² (raw-density POD)'),
        ('cond_r2_sqrt_clamp',       'Conditioning R² (√ρ POD, C2 clamp)'),
        ('cond_r2_sqrt_simplex',     'Conditioning R² (√ρ POD, simplex)'),
        ('forecast_r2_sqrt_simplex', 'Forecast R²     (MVAR √ρ+simplex)'),
    ]
    for key, label in r2_keys:
        vals = [r[key] for r in summary_rows]
        m = np.mean(vals)
        s = np.std(vals)
        print(f"  {label:50s}  {m:.4f} ± {s:.4f}")

    # ---- Save CSV with per-run summary ----
    import csv
    csv_path = out_dir / "reconstruction_floor_summary.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n✓ Per-run summary → {csv_path}")

    # ---- Save time-resolved RMSE (mean across runs) ----
    # Trim all curves to the shortest length
    def mean_curve(curves):
        min_len = min(len(c) for c in curves)
        stacked = np.array([c[:min_len] for c in curves])
        return stacked.mean(axis=0), stacked.std(axis=0)

    cond_len = min(len(c) for c in frame_curves['cond_rmse_raw'])
    fcast_len = min(len(c) for c in frame_curves['forecast_rmse_sqrt_simplex'])

    cond_times = np.arange(cond_len) * sim_dt * rom_subsample

    # Save time-resolved CSV
    csv_time_path = out_dir / "rmse_vs_time.csv"
    cond_raw_mean, cond_raw_std = mean_curve(frame_curves['cond_rmse_raw'])
    cond_sqrt_mean, cond_sqrt_std = mean_curve(frame_curves['cond_rmse_sqrt_clamp'])
    cond_simp_mean, cond_simp_std = mean_curve(frame_curves['cond_rmse_sqrt_simplex'])
    fcast_mean, fcast_std = mean_curve(frame_curves['forecast_rmse_sqrt_simplex'])

    with open(csv_time_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'time_s', 'regime',
                         'rmse_raw_mean', 'rmse_raw_std',
                         'rmse_sqrt_clamp_mean', 'rmse_sqrt_clamp_std',
                         'rmse_sqrt_simplex_mean', 'rmse_sqrt_simplex_std'])
        for i in range(cond_len):
            writer.writerow([i, f'{cond_times[i]:.3f}', 'conditioning',
                             f'{cond_raw_mean[i]:.6f}', f'{cond_raw_std[i]:.6f}',
                             f'{cond_sqrt_mean[i]:.6f}', f'{cond_sqrt_std[i]:.6f}',
                             f'{cond_simp_mean[i]:.6f}', f'{cond_simp_std[i]:.6f}'])

    fcast_times = forecast_start_time + np.arange(fcast_len) * sim_dt * rom_subsample
    csv_fcast_path = out_dir / "forecast_rmse_vs_time.csv"
    with open(csv_fcast_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'time_s', 'rmse_mean', 'rmse_std'])
        for i in range(fcast_len):
            writer.writerow([i, f'{fcast_times[i]:.3f}',
                             f'{fcast_mean[i]:.6f}', f'{fcast_std[i]:.6f}'])
    print(f"✓ Conditioning RMSE vs time → {csv_time_path}")
    print(f"✓ Forecast RMSE vs time → {csv_fcast_path}")

    # ---- Plot ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=False)

        # --- Top panel: Conditioning RMSE vs time ---
        ax = axes[0]
        ax.plot(cond_times, cond_raw_mean, label='raw-density POD (d=19)', alpha=0.9, lw=1.5)
        ax.fill_between(cond_times, cond_raw_mean - cond_raw_std,
                         cond_raw_mean + cond_raw_std, alpha=0.15)
        ax.plot(cond_times, cond_sqrt_mean, label='√ρ POD (d=19), C2 clamp', alpha=0.9, lw=1.5)
        ax.fill_between(cond_times, cond_sqrt_mean - cond_sqrt_std,
                         cond_sqrt_mean + cond_sqrt_std, alpha=0.15)
        ax.plot(cond_times, cond_simp_mean, label='√ρ POD (d=19), simplex', alpha=0.9, lw=1.5)
        ax.fill_between(cond_times, cond_simp_mean - cond_simp_std,
                         cond_simp_mean + cond_simp_std, alpha=0.15)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('RMSE (density units)')
        ax.set_title('Conditioning Error (POD reconstruction only — no MVAR)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # --- Bottom panel: Conditioning floor vs MVAR forecast ---
        ax2 = axes[1]
        # Show conditioning floor as horizontal band
        ax2.axhspan(cond_simp_mean.mean() - cond_simp_mean.std(),
                     cond_simp_mean.mean() + cond_simp_mean.std(),
                     alpha=0.2, color='gray', label='Conditioning floor (√ρ+simplex)')
        ax2.axhline(cond_simp_mean.mean(), color='gray', ls='--', lw=1)
        # Forecast RMSE
        ax2.plot(fcast_times, fcast_mean, 'r-', lw=2, label='MVAR forecast')
        ax2.fill_between(fcast_times, fcast_mean - fcast_std,
                          fcast_mean + fcast_std, color='red', alpha=0.15)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('RMSE (density units)')
        ax2.set_title('Forecast Error vs Conditioning Floor')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = out_dir / "reconstruction_floor.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Plot → {fig_path}")

    except ImportError:
        print("⚠️  matplotlib not available — skipping plot")

    # ---- Save JSON summary ----
    json_path = out_dir / "reconstruction_floor.json"
    summary = {
        'n_test': n_test,
        'n_train': n_train,
        'd_modes': d_modes,
        'rom_subsample': rom_subsample,
        'forecast_start': forecast_start_time,
        'sim_dt': sim_dt,
        'conditioning_frames': cond_len,
        'forecast_frames': fcast_len,
        'mean_conditioning_rmse_raw': float(means['cond_rmse_raw']),
        'mean_conditioning_rmse_sqrt_clamp': float(means['cond_rmse_sqrt_clamp']),
        'mean_conditioning_rmse_sqrt_simplex': float(means['cond_rmse_sqrt_simplex']),
        'mean_forecast_rmse': float(means['forecast_rmse_sqrt_simplex']),
        'floor_fraction_of_forecast': float(floor_simplex / forecast),
        'attribution': {
            'truncation_only': float(floor_raw),
            'transform_addition': float(floor_sqrt - floor_raw),
            'simplex_addition': float(floor_simplex - floor_sqrt),
            'dynamics_addition': float(forecast - floor_simplex),
        }
    }
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ JSON summary → {json_path}")

    return summary


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reconstruction-floor diagnostic")
    parser.add_argument('--run-dir', required=True, help="CF8 output directory")
    parser.add_argument('--n-test', type=int, default=None, help="Override n_test")
    parser.add_argument('--out-dir', type=str, default=None, help="Output directory")
    args = parser.parse_args()
    run_diagnostic(args.run_dir, n_test_override=args.n_test, out_dir=args.out_dir)
