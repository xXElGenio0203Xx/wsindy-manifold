#!/usr/bin/env python3
"""
Apply pre-trained MVAR-ROM model to new simulation (cross-validation).

Tests generalization of a trained MVAR-ROM on simulation with:
- Same dynamics
- Different initial conditions
- Different random seed

Usage:
    python scripts/test_mvar_rom_generalization.py \\
        --model-dir mvar_outputs/trained_model/ \\
        --test-sim simulations/test_sim__latest/
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wsindy_manifold.io import (
    save_manifest, save_arrays, save_csv, create_latest_symlink,
    side_by_side_video
)
from wsindy_manifold.pod import restrict, lift
from wsindy_manifold.standard_metrics import (
    rel_errors, compute_summary_metrics, check_mass_conservation
)
from wsindy_manifold.latent.mvar import rollout


def load_trained_model(model_dir: Path) -> dict:
    """Load pre-trained POD and MVAR model."""
    print(f"Loading trained model: {model_dir.name}")
    
    # Load POD
    pod_dir = model_dir / "pod"
    Ud = np.load(pod_dir / "Ud.npy")
    xbar = np.load(pod_dir / "xbar.npy")
    
    # Load MVAR
    mvar_dir = model_dir / "model"
    A0 = np.load(mvar_dir / "A0.npy")
    A = np.load(mvar_dir / "Astack.npy")
    
    with open(mvar_dir / "summary.json", 'r') as f:
        model_info = json.load(f)
    
    mvar_model = {
        'A0': A0,
        'A': A,
        'w': model_info['mvar_order']
    }
    
    print(f"  ✓ POD: {Ud.shape[1]} modes")
    print(f"  ✓ MVAR: order {model_info['mvar_order']}, ridge λ={model_info['ridge']}")
    
    return {
        'Ud': Ud,
        'xbar': xbar,
        'mvar_model': mvar_model,
        'mvar_order': model_info['mvar_order'],
        'info': model_info
    }


def load_test_simulation(sim_dir: Path) -> dict:
    """Load test simulation."""
    print(f"\nLoading test simulation: {sim_dir.name}")
    
    with open(sim_dir / "manifest.json", 'r') as f:
        manifest = json.load(f)
    
    # Try different density file locations
    density_file = sim_dir / "density" / "kde.npz"
    if not density_file.exists():
        density_file = sim_dir / "density_fields.npz"
    
    with np.load(density_file) as data:
        densities = data['rho']
    
    # Load density metadata
    density_meta_file = sim_dir / "density" / "kde_meta.json"
    if density_meta_file.exists():
        with open(density_meta_file, 'r') as f:
            density_meta = json.load(f)
        nx, ny = density_meta['nx'], density_meta['ny']
    else:
        # Fallback: try manifest
        nx, ny = manifest.get('density', {}).get('nx', 50), manifest.get('density', {}).get('ny', 50)
    
    print(f"  ✓ Loaded density: shape={densities.shape}")
    print(f"  ✓ Grid: {nx}×{ny}")
    
    return {
        'densities': densities,
        'manifest': manifest,
        'nx': nx,
        'ny': ny
    }


def main():
    parser = argparse.ArgumentParser(
        description='Test MVAR-ROM generalization on new simulation'
    )
    parser.add_argument('--model-dir', type=Path, required=True,
                       help='Directory with trained MVAR-ROM model')
    parser.add_argument('--test-sim', type=Path, required=True,
                       help='Test simulation directory')
    parser.add_argument('--out-root', type=Path, default=Path('mvar_outputs'),
                       help='Output root directory')
    parser.add_argument('--skip-initial', type=int, default=100,
                       help='Skip first N frames (avoid initial transients)')
    parser.add_argument('--no-videos', action='store_true',
                       help='Skip video generation')
    parser.add_argument('--fps', type=int, default=20,
                       help='Video frames per second')
    
    args = parser.parse_args()
    
    # Load model and test data
    model = load_trained_model(args.model_dir)
    test_data = load_test_simulation(args.test_sim)
    
    Ud = model['Ud']
    xbar = model['xbar']
    mvar_model = model['mvar_model']
    mvar_order = model['mvar_order']
    
    densities = test_data['densities']
    nx, ny = test_data['nx'], test_data['ny']
    n_c = nx * ny
    
    # Create output directory
    model_name = args.model_dir.name
    test_name = test_data['manifest']['sim_name']
    test_run_id = test_data['manifest']['run_id']
    out_dir = args.out_root / f"{test_name}__{test_run_id}__generalization_from_{model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput: {out_dir}\n")
    print("="*80)
    print("MVAR-ROM GENERALIZATION TEST")
    print("="*80)
    
    # Prepare data
    T = densities.shape[0]
    X = densities.reshape(T, n_c)
    
    t_start_forecast = args.skip_initial
    T_forecast = T - t_start_forecast
    
    print(f"\nTest frames: T={T}")
    print(f"Skipping first {args.skip_initial} frames")
    print(f"Forecasting: t={t_start_forecast} to t={T} ({T_forecast} steps)\n")
    
    # ========================================================================
    # 1. Project to latent space using pre-trained POD
    # ========================================================================
    print("[1/4] Projecting to latent space (pre-trained POD)...")
    Y = restrict(X, Ud, xbar)
    d = Y.shape[1]
    print(f"Latent representation: {Y.shape}")
    
    # ========================================================================
    # 2. Forecast using pre-trained MVAR
    # ========================================================================
    print(f"\n[2/4] Forecasting with pre-trained MVAR (order {mvar_order})...")
    
    # Seed from test simulation
    Y_seed = Y[t_start_forecast:t_start_forecast + mvar_order]
    
    t_start = time.time()
    Y_forecast = rollout(Y_seed, steps=T_forecast, model=mvar_model)
    t_forecast = time.time() - t_start
    
    fps = T_forecast / t_forecast
    print(f"Forecast complete: {T_forecast} steps in {t_forecast:.2f}s ({fps:.1f} FPS)")
    
    # ========================================================================
    # 3. Lift to physical space
    # ========================================================================
    print("\n[3/4] Lifting to physical space...")
    X_forecast = lift(Y_forecast, Ud, xbar, preserve_mass=True)
    
    # Ground truth
    X_true_eval = X[t_start_forecast:]
    
    # Reshape
    densities_true = X_true_eval.reshape(T_forecast, ny, nx)
    densities_pred = X_forecast.reshape(T_forecast, ny, nx)
    
    # Save
    forecast_dir = out_dir / "forecast"
    forecast_dir.mkdir(exist_ok=True)
    save_arrays(forecast_dir, latent_seed=Y_seed, latent_pred=Y_forecast)
    np.savez_compressed(forecast_dir / "density_true.npz", rho=densities_true)
    np.savez_compressed(forecast_dir / "density_pred.npz", rho=densities_pred)
    
    # ========================================================================
    # 4. Evaluate
    # ========================================================================
    print("\n[4/4] Evaluating generalization...")
    
    # Frame-wise metrics
    frame_errors = rel_errors(X_forecast, X_true_eval)
    
    metrics_df = pd.DataFrame({
        'e1': frame_errors['e1'],
        'e2': frame_errors['e2'],
        'einf': frame_errors['einf'],
        'rmse': frame_errors['rmse'],
        'mass_error': frame_errors['mass_error']
    })
    
    # Summary metrics
    summary = compute_summary_metrics(X_forecast, X_true_eval, threshold=0.10)
    
    # Add metadata
    summary['d'] = d
    summary['compression_ratio'] = n_c / d
    summary['mvar_order'] = mvar_order
    summary['ridge'] = model['info']['ridge']
    summary['skip_initial'] = args.skip_initial
    summary['forecast_frames'] = T_forecast
    summary['forecast_fps'] = fps
    summary['trained_on'] = str(args.model_dir)
    summary['tested_on'] = test_name
    
    # Mass conservation
    mass_stats = check_mass_conservation(densities_pred, rtol=5e-3, verbose=True)
    summary['mass_drift_max'] = mass_stats['max_drift']
    summary['mass_conservation_ok'] = mass_stats['within_tolerance']
    
    # NaN check
    summary['nan_count'] = int(np.isnan(X_forecast).sum())
    
    print(f"\n{'-'*80}")
    print("GENERALIZATION METRICS:")
    print(f"  R² = {summary['r2']:.4f}")
    print(f"  Median L² = {summary['median_e2']:.4f}")
    print(f"  P10 L² = {summary['p10_e2']:.4f}, P90 L² = {summary['p90_e2']:.4f}")
    print(f"  Tolerance horizon (10%) = {summary['tau_tol']} frames")
    print(f"  Mean mass error = {summary['mean_mass_error']:.6f}")
    print(f"{'-'*80}")
    
    # Save evaluation
    eval_dir = out_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    
    save_csv(eval_dir, metrics_df, "metrics_over_time")
    
    with open(eval_dir / "summary.json", 'w') as f:
        # Convert to JSON-serializable
        summary_json = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                       for k, v in summary.items()}
        json.dump(summary_json, f, indent=2)
    
    print(f"\n✓ Saved evaluation: {eval_dir}/")
    
    # ========================================================================
    # 5. Generate Videos
    # ========================================================================
    if not args.no_videos:
        print("\n[5/5] Generating comparison videos...")
        videos_dir = forecast_dir / "videos"
        videos_dir.mkdir(exist_ok=True)
        
        # Side-by-side comparison
        from wsindy_manifold.io import side_by_side_video
        side_by_side_video(
            videos_dir,
            densities_true, densities_pred,
            lower_strip_timeseries=metrics_df['e2'].values,
            name="generalization_true_vs_pred",
            fps=args.fps,
            titles=("Ground Truth (Test)", "MVAR-ROM (Trained on Different IC)")
        )
        print(f"✓ Saved comparison video: {videos_dir}/generalization_true_vs_pred.mp4")
    
    # Save manifest
    save_manifest(
        root=out_dir,
        sim_name=f"{test_name}_generalization_test",
        config_path=str(args.test_sim / "config.yaml"),
        simulator="mvar_rom_generalization",
        seed=test_data['manifest']['seed'],
        code_version="1.0.0",
        source_sim=str(args.test_sim),
        trained_on=str(args.model_dir)
    )
    
    # Symlink
    create_latest_symlink(out_dir, args.out_root, 
                         f"{test_name}__latest__generalization")
    
    print("\n" + "="*80)
    print("✓ GENERALIZATION TEST COMPLETE")
    print("="*80)
    print(f"\nOutput: {out_dir}")
    print(f"\nModel trained on: {args.model_dir}")
    print(f"Tested on: {args.test_sim}")
    print(f"\nGeneralization R² = {summary['r2']:.4f}")
    print(f"(Compare to in-sample performance of training data)")


if __name__ == "__main__":
    main()
