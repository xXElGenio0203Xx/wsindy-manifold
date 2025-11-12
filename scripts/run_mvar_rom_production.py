#!/usr/bin/env python3
"""
Production MVAR-ROM evaluation script.

Loads simulation outputs from simulations/<sim_name>__<run_id>/
and generates MVAR-ROM forecasts in mvar_outputs/<sim_name>__<run_id>__<exp_name>/

Output structure:
- manifest.json
- config.yaml
- pod/ (Ud.npy, xbar.npy, energy_curve.npy, energy.png)
- model/ (A0.npy, Astack.npy, summary.json)
- forecast/ (latent_pred.npy, density_pred.npz, videos/)
- eval/ (metrics_over_time.csv, summary.json, plots/)
"""

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wsindy_manifold.io import (
    save_manifest, save_arrays, save_csv, 
    side_by_side_video, create_latest_symlink
)
from wsindy_manifold.pod import fit_pod, restrict, lift
from wsindy_manifold.standard_metrics import (
    rel_errors, r2_score, tolerance_horizon,
    compute_summary_metrics, check_mass_conservation
)
from wsindy_manifold.latent.mvar import fit_mvar, rollout


def convert_to_json_serializable(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    else:
        return str(obj)


@dataclass
class MVARROMConfig:
    """Configuration for MVAR-ROM evaluation."""
    # POD parameters
    pod_energy: float = 0.99
    pod_modes: Optional[int] = None  # Fixed mode count (overrides energy)
    
    # MVAR parameters
    mvar_order: int = 4
    ridge: float = 1e-6
    
    # Data split
    train_frac: float = 0.8  # 80% train - better than 90% for test set size
    
    # Evaluation
    tolerance_threshold: float = 0.10
    
    # Output
    save_videos: bool = True
    fps: int = 20


def load_simulation(sim_dir: Path) -> dict:
    """
    Load simulation outputs from standardized directory.
    
    Args:
        sim_dir: Path to simulation directory
        
    Returns:
        data: Dictionary with densities, metadata, etc.
    """
    print(f"Loading simulation: {sim_dir.name}")
    
    # Load manifest
    with open(sim_dir / "manifest.json", 'r') as f:
        manifest = json.load(f)
    
    # Load KDE density
    density_data = np.load(sim_dir / "density/kde.npz")
    densities = density_data['rho']  # (T, ny, nx)
    
    with open(sim_dir / "density/kde_meta.json", 'r') as f:
        kde_meta = json.load(f)
    
    T, ny, nx = densities.shape
    
    print(f"  ✓ Loaded density: shape={densities.shape}")
    print(f"  ✓ Grid: {nx}×{ny}, bandwidth={kde_meta['bandwidth']}")
    print(f"  ✓ Particles: {manifest['N']}, frames: {manifest['T_frames']}")
    
    return {
        'densities': densities,
        'nx': nx,
        'ny': ny,
        'kde_meta': kde_meta,
        'manifest': manifest,
        'sim_dir': sim_dir
    }


def plot_pod_energy(
    energy_curve: np.ndarray,
    d: int,
    save_path: Path
):
    """Plot POD energy spectrum."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    modes = np.arange(1, len(energy_curve) + 1)
    ax.plot(modes, energy_curve * 100, 'b-', linewidth=2)
    ax.axvline(d, color='r', linestyle='--', linewidth=2,
              label=f'd = {d} ({energy_curve[d-1]*100:.2f}% energy)')
    ax.axhline(99, color='k', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Number of Modes', fontsize=12)
    ax.set_ylabel('Cumulative Energy (%)', fontsize=12)
    ax.set_title('POD Energy Spectrum', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xlim([1, len(energy_curve)])
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved POD energy: {save_path}")


def plot_errors_timeseries(
    metrics_df: pd.DataFrame,
    T0: int,
    save_path: Path
):
    """Plot error timeseries."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    
    # L1 error
    ax = axes[0]
    ax.plot(metrics_df.index, metrics_df['e1'], 'b-', linewidth=2)
    ax.axhline(0.1, color='r', linestyle='--', alpha=0.5, label='10% threshold')
    ax.axvline(T0, color='k', linestyle=':', alpha=0.5, label='Train/Test split')
    ax.set_ylabel('Relative L¹ Error', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    ax.set_ylim([0, max(metrics_df['e1'].max() * 1.1, 0.2)])
    
    # L2 error
    ax = axes[1]
    ax.plot(metrics_df.index, metrics_df['e2'], 'g-', linewidth=2)
    ax.axhline(0.1, color='r', linestyle='--', alpha=0.5)
    ax.axvline(T0, color='k', linestyle=':', alpha=0.5)
    ax.set_ylabel('Relative L² Error', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(metrics_df['e2'].max() * 1.1, 0.2)])
    
    # Linf error
    ax = axes[2]
    ax.plot(metrics_df.index, metrics_df['einf'], 'm-', linewidth=2)
    ax.axhline(0.1, color='r', linestyle='--', alpha=0.5)
    ax.axvline(T0, color='k', linestyle=':', alpha=0.5)
    ax.set_ylabel('Relative L∞ Error', fontsize=11, fontweight='bold')
    ax.set_xlabel('Frame Index', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(metrics_df['einf'].max() * 1.1, 0.2)])
    
    plt.suptitle('MVAR-ROM Forecast Errors Over Time', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved errors timeseries: {save_path}")


def plot_snapshots(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    nx: int,
    ny: int,
    n_snapshots: int,
    save_path: Path
):
    """Plot truth/pred/diff snapshots grid."""
    T = X_true.shape[0]
    indices = np.linspace(0, T-1, n_snapshots, dtype=int)
    
    # Reshape to 2D
    true_2d = X_true.reshape(T, ny, nx)
    pred_2d = X_pred.reshape(T, ny, nx)
    
    fig, axes = plt.subplots(3, n_snapshots, figsize=(3*n_snapshots, 9))
    
    vmin = min(true_2d.min(), pred_2d.min())
    vmax = max(true_2d.max(), pred_2d.max())
    
    for col, t_idx in enumerate(indices):
        # Truth
        ax = axes[0, col]
        im = ax.imshow(true_2d[t_idx].T, origin='lower', cmap='viridis',
                      vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(f'Frame {t_idx}', fontsize=10)
        if col == 0:
            ax.set_ylabel('Ground Truth', fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Prediction
        ax = axes[1, col]
        ax.imshow(pred_2d[t_idx].T, origin='lower', cmap='viridis',
                 vmin=vmin, vmax=vmax, aspect='auto')
        if col == 0:
            ax.set_ylabel('MVAR-ROM', fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Difference
        ax = axes[2, col]
        diff = np.abs(pred_2d[t_idx] - true_2d[t_idx])
        im_diff = ax.imshow(diff.T, origin='lower', cmap='hot',
                           vmin=0, vmax=diff.max(), aspect='auto')
        if col == 0:
            ax.set_ylabel('|Difference|', fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Colorbars
        if col == n_snapshots - 1:
            plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)
            plt.colorbar(im_diff, ax=axes[2, col], fraction=0.046, pad=0.04)
    
    plt.suptitle('Density Snapshots: Truth | Prediction | Difference',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved snapshots: {save_path}")


def plot_latent_scatter(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    n_modes: int,
    save_path: Path
):
    """Plot true vs predicted latent modes."""
    d = min(n_modes, Y_true.shape[1])
    
    fig, axes = plt.subplots(1, d, figsize=(5*d, 4))
    if d == 1:
        axes = [axes]
    
    for i in range(d):
        ax = axes[i]
        ax.scatter(Y_true[:, i], Y_pred[:, i], alpha=0.5, s=10)
        
        # Diagonal
        lims = [min(Y_true[:, i].min(), Y_pred[:, i].min()),
                max(Y_true[:, i].max(), Y_pred[:, i].max())]
        ax.plot(lims, lims, 'r--', alpha=0.5, linewidth=2)
        
        # Correlation
        corr = np.corrcoef(Y_true[:, i], Y_pred[:, i])[0, 1]
        
        ax.set_xlabel(f'True y_{i+1}', fontsize=11)
        ax.set_ylabel(f'Predicted y_{i+1}', fontsize=11)
        ax.set_title(f'Mode {i+1} (ρ = {corr:.3f})', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.suptitle('MVAR-ROM: Latent Space Prediction Quality',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved latent scatter: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run MVAR-ROM evaluation on simulation outputs'
    )
    parser.add_argument('sim_dir', type=Path,
                       help='Simulation directory (simulations/<sim_name>__<run_id>/)')
    parser.add_argument('--exp-name', type=str, default=None,
                       help='Experiment name (default: mvar_pod_w<w>_ridge<λ>)')
    parser.add_argument('--out-root', type=Path, default=Path('mvar_outputs'),
                       help='Output root directory')
    parser.add_argument('--train-frac', type=float, default=0.8,
                       help='Training fraction (default: 0.8)')
    parser.add_argument('--pod-energy', type=float, default=0.99,
                       help='POD energy threshold (default: 0.99, optimal for MVAR)')
    parser.add_argument('--pod-modes', type=int, default=None,
                       help='Fixed number of POD modes (overrides --pod-energy)')
    parser.add_argument('--mvar-order', type=int, default=4,
                       help='MVAR lag order')
    parser.add_argument('--ridge', type=float, default=1e-6,
                       help='Ridge regularization')
    parser.add_argument('--no-videos', action='store_true',
                       help='Skip video generation')
    parser.add_argument('--fps', type=int, default=20,
                       help='Video frames per second (default: 20)')
    
    args = parser.parse_args()
    
    # Load simulation
    sim_data = load_simulation(args.sim_dir)
    densities = sim_data['densities']
    nx, ny = sim_data['nx'], sim_data['ny']
    n_c = nx * ny
    
    # Create config
    config = MVARROMConfig(
        pod_energy=args.pod_energy,
        pod_modes=args.pod_modes,
        mvar_order=args.mvar_order,
        ridge=args.ridge,
        train_frac=args.train_frac,
        save_videos=not args.no_videos,
        fps=args.fps
    )
    
    # Derive experiment name
    if args.exp_name:
        exp_name = args.exp_name
    else:
        exp_name = f"mvar_pod_w{config.mvar_order}_ridge{config.ridge:.0e}"
    
    # Create output directory
    sim_name = sim_data['manifest']['sim_name']
    run_id = sim_data['manifest']['run_id']
    out_dir = args.out_root / f"{sim_name}__{run_id}__{exp_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExperiment: {exp_name}")
    print(f"Output: {out_dir}\n")
    
    print("="*80)
    print("MVAR-ROM EVALUATION PIPELINE")
    print("="*80)
    
    # Flatten density fields
    T = densities.shape[0]
    X = densities.reshape(T, n_c)
    
    # Split train/test
    T0 = int(config.train_frac * T)
    T1 = T - T0
    
    X_train = X[:T0]
    X_test = X[T0:]
    
    print(f"\nData split: T0={T0}, T1={T1} ({config.train_frac*100:.0f}% train)\n")
    
    # ========================================================================
    # 1. POD
    # ========================================================================
    print("[1/7] Fitting POD...")
    t_start = time.time()
    Ud, xbar, d, energy_curve = fit_pod(X_train, energy=config.pod_energy, n_modes=config.pod_modes)
    t_pod = time.time() - t_start
    
    print(f"POD complete: d={d}, time={t_pod:.2f}s")
    print(f"Compression: {n_c} → {d} ({n_c/d:.1f}× reduction)")
    
    # Save POD artifacts
    pod_dir = out_dir / "pod"
    pod_dir.mkdir(exist_ok=True)
    
    save_arrays(pod_dir, Ud=Ud, xbar=xbar, energy_curve=energy_curve)
    plot_pod_energy(energy_curve, d, pod_dir / "energy.png")
    
    # ========================================================================
    # 2. Restrict to latent space
    # ========================================================================
    print("\n[2/7] Restricting to latent space...")
    Y_train = restrict(X_train, Ud, xbar)
    Y_test = restrict(X_test, Ud, xbar)
    
    print(f"Latent train: {Y_train.shape}")
    print(f"Latent test:  {Y_test.shape}")
    
    # ========================================================================
    # 3. Fit MVAR
    # ========================================================================
    print("\n[3/7] Fitting MVAR...")
    t_start = time.time()
    mvar_model = fit_mvar(Y_train, w=config.mvar_order, ridge_lambda=config.ridge)
    A0 = mvar_model['A0']
    A = mvar_model['A']
    t_mvar = time.time() - t_start
    
    print(f"MVAR complete: w={config.mvar_order}, λ={config.ridge:.0e}, time={t_mvar:.2f}s")
    
    # Save MVAR model
    model_dir = out_dir / "model"
    model_dir.mkdir(exist_ok=True)
    
    save_arrays(model_dir, A0=A0, Astack=A)
    
    model_summary = {
        'mvar_order': int(config.mvar_order),
        'ridge': float(config.ridge),
        'latent_dim': int(d),
        'train_samples': int(T0),
        'training_time_s': float(t_mvar)
    }
    
    with open(model_dir / "summary.json", 'w') as f:
        json.dump(model_summary, f, indent=2)
    print(f"✓ Saved model: {model_dir}/")
    
    # ========================================================================
    # 4. Forecast (closed-loop)
    # ========================================================================
    print("\n[4/7] Forecasting (closed-loop)...")
    
    # Seed with last w frames from train
    Y_seed = Y_train[-config.mvar_order:]
    
    t_start = time.time()
    Y_forecast = rollout(Y_seed, steps=T1, model=mvar_model)
    t_forecast = time.time() - t_start
    
    fps = T1 / t_forecast
    print(f"Forecast complete: {T1} steps in {t_forecast:.2f}s ({fps:.1f} FPS)")
    
    # Save forecast
    forecast_dir = out_dir / "forecast"
    forecast_dir.mkdir(exist_ok=True)
    
    save_arrays(forecast_dir, latent_seed=Y_seed, latent_pred=Y_forecast)
    
    # ========================================================================
    # 5. Lift back to physical space
    # ========================================================================
    print("\n[5/7] Lifting to physical space...")
    X_forecast = lift(Y_forecast, Ud, xbar, preserve_mass=True)
    
    # Reshape to 2D
    densities_true = X_test.reshape(T1, ny, nx)
    densities_pred = X_forecast.reshape(T1, ny, nx)
    
    # Save densities
    np.savez_compressed(forecast_dir / "density_true.npz", rho=densities_true)
    np.savez_compressed(forecast_dir / "density_pred.npz", rho=densities_pred)
    print(f"✓ Saved forecast densities: {forecast_dir}/")
    
    # ========================================================================
    # 6. Evaluate
    # ========================================================================
    print("\n[6/7] Evaluating...")
    
    # Frame-wise metrics
    frame_errors = rel_errors(X_forecast, X_test)
    
    metrics_df = pd.DataFrame({
        'e1': frame_errors['e1'],
        'e2': frame_errors['e2'],
        'einf': frame_errors['einf'],
        'rmse': frame_errors['rmse'],
        'mass_error': frame_errors['mass_error']
    })
    
    # Summary metrics
    summary = compute_summary_metrics(X_forecast, X_test, threshold=config.tolerance_threshold)
    
    # Add additional info
    summary['d'] = d
    summary['compression_ratio'] = n_c / d
    summary['mvar_order'] = config.mvar_order
    summary['ridge'] = config.ridge
    summary['train_frac'] = config.train_frac
    summary['forecast_fps'] = fps
    
    # Mass conservation
    mass_stats = check_mass_conservation(densities_pred, rtol=5e-3, verbose=True)
    summary['mass_drift_max'] = mass_stats['max_drift']
    summary['mass_conservation_ok'] = mass_stats['within_tolerance']
    
    # NaN check
    n_nans = int(np.isnan(X_forecast).sum())
    summary['nan_count'] = n_nans
    if n_nans > 0:
        print(f"⚠️  Warning: {n_nans} NaN values in forecast")
    
    print(f"\n{'-'*80}")
    print("Summary Metrics:")
    print(f"  R² = {summary['r2']:.4f}")
    print(f"  Median L² = {summary['median_e2']:.4f}")
    print(f"  P10 L² = {summary['p10_e2']:.4f}, P90 L² = {summary['p90_e2']:.4f}")
    print(f"  Tolerance horizon (10%) = {summary['tau_tol']} frames")
    print(f"  Mean mass error = {summary['mean_mass_error']:.6f}")
    print(f"  Max mass error = {summary['max_mass_error']:.6f}")
    print(f"{'-'*80}")
    
    # Save evaluation
    eval_dir = out_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    
    save_csv(eval_dir, metrics_df, "metrics_over_time")
    
    # Convert summary to JSON-serializable types
    summary_json = convert_to_json_serializable(summary)
    
    with open(eval_dir / "summary.json", 'w') as f:
        json.dump(summary_json, f, indent=2)
    print(f"✓ Saved evaluation: {eval_dir}/")
    
    # ========================================================================
    # 7. Plots and Videos
    # ========================================================================
    print("\n[7/7] Generating plots and videos...")
    
    # Plots
    plot_errors_timeseries(metrics_df, T0, eval_dir / "errors_timeseries.png")
    plot_snapshots(X_test, X_forecast, nx, ny, 6, eval_dir / "snapshots.png")
    plot_latent_scatter(Y_test, Y_forecast, min(3, d), eval_dir / "latent_scatter.png")
    
    # Videos
    if config.save_videos:
        videos_dir = forecast_dir / "videos"
        videos_dir.mkdir(exist_ok=True)
        
        from wsindy_manifold.io import save_video
        
        # True density
        save_video(
            videos_dir, densities_true, fps=config.fps,
            name="true_density", cmap="viridis",
            vmin=0, vmax=np.percentile(densities_true, 95),
            title="Ground Truth Density"
        )
        
        # Predicted density
        save_video(
            videos_dir, densities_pred, fps=config.fps,
            name="pred_density", cmap="viridis",
            vmin=0, vmax=np.percentile(densities_pred, 95),
            title="MVAR-ROM Predicted Density"
        )
        
        # Side-by-side comparison
        side_by_side_video(
            videos_dir,
            densities_true, densities_pred,
            lower_strip_timeseries=metrics_df['e2'].values,
            name="true_vs_pred_comparison",
            fps=config.fps,
            titles=("Ground Truth", "MVAR-ROM Prediction")
        )
    
    # Save manifest
    save_manifest(
        root=out_dir,
        sim_name=sim_name,
        config_path=str(args.sim_dir / "config.yaml"),
        simulator=f"mvar_rom_{exp_name}",
        seed=sim_data['manifest']['seed'],
        code_version="1.0.0",
        source_sim=str(args.sim_dir),
        T0=T0,
        T1=T1,
        d=d
    )
    
    # Save config
    config_dict = asdict(config)
    with open(out_dir / "config.yaml", 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    print(f"✓ Saved config: {out_dir}/config.yaml")
    
    # Create latest symlink
    print("\n[Symlink] Creating latest link...")
    create_latest_symlink(out_dir, args.out_root, f"{sim_name}__latest__{exp_name}")
    
    # Final summary
    print("\n" + "="*80)
    print("✓ MVAR-ROM EVALUATION COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {out_dir}")
    print(f"\nStructure:")
    print(f"  manifest.json")
    print(f"  config.yaml")
    print(f"  pod/")
    print(f"    Ud.npy               - POD basis ({n_c}, {d})")
    print(f"    xbar.npy             - Mean state ({n_c},)")
    print(f"    energy_curve.npy     - Cumulative energy")
    print(f"    energy.png           - Energy plot")
    print(f"  model/")
    print(f"    A0.npy               - MVAR bias ({d},)")
    print(f"    Astack.npy           - MVAR matrices ({config.mvar_order}, {d}, {d})")
    print(f"    summary.json         - Model metadata")
    print(f"  forecast/")
    print(f"    latent_seed.npy      - Initial latent state")
    print(f"    latent_pred.npy      - Predicted latent trajectory")
    print(f"    density_true.npz     - Ground truth density")
    print(f"    density_pred.npz     - Predicted density")
    
    if config.save_videos:
        print(f"    videos/")
        print(f"      true_density.mp4")
        print(f"      pred_density.mp4")
        print(f"      true_vs_pred_comparison.mp4")
    
    print(f"  eval/")
    print(f"    metrics_over_time.csv   - Frame-wise errors")
    print(f"    summary.json            - Aggregate metrics")
    print(f"    errors_timeseries.png")
    print(f"    snapshots.png")
    print(f"    latent_scatter.png")
    
    print(f"\nPerformance:")
    print(f"  Compression: {n_c} → {d} ({n_c/d:.1f}× reduction)")
    print(f"  POD time: {t_pod:.2f}s")
    print(f"  MVAR training: {t_mvar:.2f}s")
    print(f"  Forecast FPS: {fps:.1f}")
    
    print(f"\nForecast Quality:")
    print(f"  R² = {summary['r2']:.4f}")
    print(f"  Median L² = {summary['median_e2']:.4f}")
    print(f"  τ_tol (10%) = {summary['tau_tol']} frames")
    print(f"  Mass conservation: {'✓ PASS' if summary['mass_conservation_ok'] else '✗ FAIL'}")
    
    print(f"\nSymlink: {args.out_root}/{sim_name}__latest__{exp_name} -> {out_dir.name}")
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
