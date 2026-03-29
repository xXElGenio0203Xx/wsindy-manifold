#!/usr/bin/env python3
"""
Run the full visualization pipeline on the best CF8 trial export.

Generates:
  1. side-by-side density video (truth vs pred) with error strip — 'hot' cmap
  2. density video (truth only)
  3. density video (pred only)
  4. error timeseries plot (L1/L2/Linf)
  5. error distribution histograms
  6. density-based order parameters (spatial order, mass conservation)
  7. snapshot comparison grid
  8. frame-wise R² curve

Usage:
  PYTHONPATH=src python scripts/run_viz_CF8.py
"""

import numpy as np
import sys
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from rectsim.legacy_functions import (
    side_by_side_video,
    save_video,
    plot_errors_timeseries,
)
from rectsim.rom_video_utils import (
    make_truth_vs_pred_density_video,
    make_density_snapshot_comparison,
)


def compute_frame_metrics(rho_true, rho_pred):
    """Compute per-frame error metrics matching the pipeline convention."""
    T = rho_true.shape[0]
    e1 = np.zeros(T)   # relative L1
    e2 = np.zeros(T)   # relative L2
    einf = np.zeros(T)  # relative Linf

    for t in range(T):
        true_flat = rho_true[t].ravel()
        pred_flat = rho_pred[t].ravel()
        diff = true_flat - pred_flat

        norm1_true = np.sum(np.abs(true_flat)) + 1e-12
        norm2_true = np.sqrt(np.sum(true_flat ** 2)) + 1e-12
        norminf_true = np.max(np.abs(true_flat)) + 1e-12

        e1[t] = np.sum(np.abs(diff)) / norm1_true
        e2[t] = np.sqrt(np.sum(diff ** 2)) / norm2_true
        einf[t] = np.max(np.abs(diff)) / norminf_true

    return {'e1': e1, 'e2': e2, 'einf': einf}


def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0


def main():
    ROOT = Path(__file__).parent.parent
    export_dir = ROOT / 'oscar_output' / 'CF8_best_trial_export'
    viz_dir = export_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)

    # ── Load exported data ──
    print("Loading CF8 best trial export...")
    data = np.load(export_dir / 'CF8_best_trial.npz')
    rho_test = data['rho_test']      # (37, 48, 48)
    rho_pred = data['rho_pred']      # (37, 48, 48)
    rho_train = data['rho_train']    # (600, 48, 48)
    times_fc = data['times_forecast']
    r2_frames_saved = data['r2_frames']
    r2_overall = float(data['r2_overall'])
    mass_err = float(data['mass_err'])
    seed = int(data['seed'])

    with open(export_dir / 'summary.json') as f:
        summary_data = json.load(f)

    T_forecast = rho_test.shape[0]
    print(f"  rho_test:  {rho_test.shape}")
    print(f"  rho_pred:  {rho_pred.shape}")
    print(f"  rho_train: {rho_train.shape}")
    print(f"  R²={r2_overall:+.4f}, mass_err={mass_err:.2f}%, seed={seed}")

    # ── Compute frame metrics ──
    print("\nComputing frame-level error metrics...")
    frame_metrics = compute_frame_metrics(rho_test, rho_pred)

    # Find tau_tol: first frame where relative L2 > 10%
    tau_tol = T_forecast
    for t in range(T_forecast):
        if frame_metrics['e2'][t] > 0.10:
            tau_tol = t
            break

    summary = {
        'r2': r2_overall,
        'median_e2': float(np.median(frame_metrics['e2'])),
        'tau_tol': tau_tol,
    }

    print(f"  median L2 error: {summary['median_e2']:.4f}")
    print(f"  tau_tol (frames >10% L2): {tau_tol}")
    print()

    # ═══════════════════════════════════════════════════════════
    # 1. Side-by-side density video (legacy pipeline — 'hot' cmap)
    # ═══════════════════════════════════════════════════════════
    print("[1/8] Side-by-side density video (hot cmap, with error strip)...")
    side_by_side_video(
        path=viz_dir,
        left_frames=rho_test,
        right_frames=rho_pred,
        lower_strip_timeseries=frame_metrics['e2'],
        name="density_truth_vs_pred",
        fps=8,
        cmap='hot',
        titles=('Ground Truth', 'MVAR-ROM Prediction'),
    )

    # ═══════════════════════════════════════════════════════════
    # 2. Truth-only density video
    # ═══════════════════════════════════════════════════════════
    print("[2/8] Ground truth density video...")
    save_video(
        path=viz_dir,
        frames=rho_test,
        fps=8,
        name="density_truth",
        cmap='hot',
        title=f'Ground Truth Density (seed={seed})',
    )

    # ═══════════════════════════════════════════════════════════
    # 3. Predicted-only density video
    # ═══════════════════════════════════════════════════════════
    print("[3/8] Predicted density video...")
    save_video(
        path=viz_dir,
        frames=rho_pred,
        fps=8,
        name="density_pred",
        cmap='hot',
        title=f'MVAR-ROM Predicted Density (R²={r2_overall:+.3f})',
    )

    # ═══════════════════════════════════════════════════════════
    # 4. Error timeseries plot
    # ═══════════════════════════════════════════════════════════
    print("[4/8] Error timeseries plot...")
    fig = plot_errors_timeseries(
        frame_metrics=frame_metrics,
        summary=summary,
        T0=0,
        save_path=viz_dir / "error_timeseries.png",
        title=f'CF8 Best Trial — Error Metrics (R²={r2_overall:+.3f}, seed={seed})',
    )
    plt.close('all')

    # ═══════════════════════════════════════════════════════════
    # 5. Error distribution histograms
    # ═══════════════════════════════════════════════════════════
    print("[5/8] Error distribution histograms...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Final-frame pixel-wise L1 error
    e1_final = np.abs(rho_test[-1] - rho_pred[-1])
    axes[0].hist(e1_final.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_xlabel('|ρ_true − ρ_pred|', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title(f'L1 Error at t={times_fc[-1]:.1f}s (final frame)', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Final-frame pixel-wise L2 error
    e2_final = (rho_test[-1] - rho_pred[-1]) ** 2
    axes[1].hist(e2_final.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_xlabel('(ρ_true − ρ_pred)²', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title(f'Squared Error at t={times_fc[-1]:.1f}s', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # Relative error
    rel_error_final = e1_final / (rho_test[-1] + 1e-10)
    axes[2].hist(rel_error_final.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[2].set_xlabel('|ρ_true − ρ_pred| / ρ_true', fontsize=11)
    axes[2].set_ylabel('Count', fontsize=11)
    axes[2].set_title(f'Relative Error at t={times_fc[-1]:.1f}s', fontsize=12)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f'CF8 Best Trial — Error Distributions (R²={r2_overall:+.3f})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(viz_dir / "error_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {viz_dir / 'error_distributions.png'}")

    # ═══════════════════════════════════════════════════════════
    # 6. Density-based order parameters
    # ═══════════════════════════════════════════════════════════
    print("[6/8] Order parameters (density-based)...")

    T = T_forecast
    spatial_order_true = np.std(rho_test.reshape(T, -1), axis=1)
    spatial_order_pred = np.std(rho_pred.reshape(T, -1), axis=1)
    mass_true = np.sum(rho_test.reshape(T, -1), axis=1)
    mass_pred = np.sum(rho_pred.reshape(T, -1), axis=1)
    mass_error_rel = np.abs(mass_true - mass_pred) / (mass_true + 1e-10)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Spatial order: σ(ρ) per frame
    axes[0].plot(times_fc, spatial_order_true, 'b-', lw=2.5, alpha=0.85, label='Ground Truth')
    axes[0].plot(times_fc, spatial_order_pred, 'r--', lw=2.5, alpha=0.85, label='MVAR Prediction')
    axes[0].set_ylabel('Spatial Order σ(ρ)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    axes[0].set_title(
        f'CF8 Best Trial — Density Order Parameters (R²={r2_overall:+.3f}, seed={seed})',
        fontsize=14, fontweight='bold')

    # Mass conservation
    axes[1].plot(times_fc, mass_true, 'b-', lw=2.5, alpha=0.85, label='Ground Truth')
    axes[1].plot(times_fc, mass_pred, 'r--', lw=2.5, alpha=0.85, label='MVAR Prediction')
    axes[1].set_ylabel('Total Mass Σρ', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11)

    # Mass error
    axes[2].plot(times_fc, mass_error_rel * 100, 'orange', lw=2, alpha=0.85)
    axes[2].set_ylabel('Mass Error (%)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    max_merr = np.max(mass_error_rel) * 100
    axes[2].set_title(f'Max mass error: {max_merr:.3f}%', fontsize=11)

    plt.tight_layout()
    plt.savefig(viz_dir / "order_parameters.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {viz_dir / 'order_parameters.png'}")

    # ═══════════════════════════════════════════════════════════
    # 7. Snapshot comparison (viridis, from rom_video_utils)
    # ═══════════════════════════════════════════════════════════
    print("[7/8] Snapshot comparison grid...")
    n_snaps = min(6, T_forecast)
    snap_idx = np.linspace(0, T_forecast - 1, n_snaps, dtype=int)

    make_density_snapshot_comparison(
        rho_test, rho_pred,
        list(snap_idx), times_fc,
        viz_dir / "snapshot_comparison.png",
        title=f'CF8 Best Trial — Density Snapshots (R²={r2_overall:+.3f})',
        cmap='hot',
    )
    print(f"  Saved {viz_dir / 'snapshot_comparison.png'}")

    # ═══════════════════════════════════════════════════════════
    # 8. Frame-wise R² curve (enhanced version)
    # ═══════════════════════════════════════════════════════════
    print("[8/8] Frame-wise R² curve...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})

    # R² per frame
    ax1.plot(times_fc, r2_frames_saved, 'b-', lw=2, label='Frame R²')
    ax1.axhline(r2_overall, color='r', ls='--', lw=1.5,
                label=f'Overall R² = {r2_overall:+.3f}')
    ax1.axhline(0, color='gray', ls=':', lw=0.5)
    ax1.fill_between(times_fc, r2_frames_saved, alpha=0.15, color='blue')
    ax1.set_ylabel('R²', fontsize=12, fontweight='bold')
    ax1.set_title(f'CF8 Best Trial — Frame-wise R² (seed={seed})', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.set_ylim(min(r2_frames_saved.min() - 0.05, -0.1), 1.05)
    ax1.grid(True, alpha=0.3)

    # Relative L2 error per frame
    ax2.plot(times_fc, frame_metrics['e2'] * 100, 'g-', lw=2, label='Rel. L² error (%)')
    ax2.axhline(10, color='r', ls=':', lw=1, label='10% threshold')
    ax2.set_ylabel('Rel. L² Error (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(viz_dir / "r2_and_error_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {viz_dir / 'r2_and_error_curve.png'}")

    # ── Summary ──
    outputs = sorted(viz_dir.glob('*'))
    print(f"\n{'='*70}")
    print(f"  VISUALIZATION PIPELINE COMPLETE")
    print(f"  Output directory: {viz_dir}")
    print(f"  Files generated: {len(outputs)}")
    for f in outputs:
        size = f.stat().st_size
        unit = 'KB' if size < 1024 * 1024 else 'MB'
        val = size / 1024 if unit == 'KB' else size / (1024 * 1024)
        print(f"    {f.name:40s}  ({val:.0f} {unit})")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
