#!/usr/bin/env python3
"""
Train MVAR-ROM on multiple simulations with different initial conditions.

This creates a model that generalizes across IC distributions by:
1. Aggregating density snapshots from multiple simulations
2. Fitting POD on combined dataset → IC-invariant basis
3. Fitting MVAR on combined latent trajectories → IC-invariant dynamics

Usage:
    python scripts/train_mvar_rom_ensemble.py \\
        --sim-dirs simulations/sim1 simulations/sim2 ... \\
        --pod-energy 0.995 \\
        --mvar-order 9 \\
        --exp-name ensemble_model
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wsindy_manifold.io import save_manifest, save_arrays
from wsindy_manifold.pod import fit_pod
from wsindy_manifold.latent.mvar import fit_mvar


def load_simulation_densities(sim_dir: Path) -> tuple:
    """Load density fields from simulation."""
    print(f"  Loading: {sim_dir.name}")
    
    # Load density
    density_file = sim_dir / "density" / "kde.npz"
    if not density_file.exists():
        density_file = sim_dir / "density_fields.npz"
    
    with np.load(density_file) as data:
        densities = data['rho']
    
    # Load metadata
    with open(sim_dir / "manifest.json", 'r') as f:
        manifest = json.load(f)
    
    density_meta_file = sim_dir / "density" / "kde_meta.json"
    if density_meta_file.exists():
        with open(density_meta_file, 'r') as f:
            density_meta = json.load(f)
        nx, ny = density_meta['nx'], density_meta['ny']
    else:
        nx, ny = 50, 50  # Default
    
    T = densities.shape[0]
    n_c = nx * ny
    X = densities.reshape(T, n_c)
    
    print(f"    ✓ {T} frames, {nx}×{ny} grid")
    
    return X, manifest['sim_name'], manifest['seed']


def main():
    parser = argparse.ArgumentParser(
        description='Train MVAR-ROM on ensemble of simulations with different ICs'
    )
    parser.add_argument('--sim-dirs', type=Path, nargs='+', required=True,
                       help='List of simulation directories')
    parser.add_argument('--pod-energy', type=float, default=0.995,
                       help='POD energy threshold')
    parser.add_argument('--mvar-order', type=int, default=9,
                       help='MVAR lag order')
    parser.add_argument('--ridge', type=float, default=1e-6,
                       help='Ridge regularization')
    parser.add_argument('--exp-name', type=str, default='ensemble_model',
                       help='Experiment name')
    parser.add_argument('--out-root', type=Path, default=Path('mvar_outputs'),
                       help='Output root directory')
    parser.add_argument('--skip-initial', type=int, default=100,
                       help='Skip first N frames from each simulation')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ENSEMBLE MVAR-ROM TRAINING")
    print("="*80)
    print(f"\nTraining on {len(args.sim_dirs)} simulations with different ICs\n")
    
    # ========================================================================
    # 1. Load and aggregate data
    # ========================================================================
    print("[1/4] Loading simulations...")
    
    X_list = []
    sim_info = []
    
    for sim_dir in args.sim_dirs:
        X, sim_name, seed = load_simulation_densities(sim_dir)
        
        # Skip initial transients
        if args.skip_initial > 0:
            X = X[args.skip_initial:]
            print(f"    (skipped first {args.skip_initial} frames)")
        
        X_list.append(X)
        sim_info.append({'name': sim_name, 'seed': seed, 'frames': X.shape[0]})
    
    # Stack all data
    X_combined = np.vstack(X_list)
    n_c = X_combined.shape[1]
    
    print(f"\n✓ Combined dataset: {X_combined.shape[0]} total frames")
    print(f"  {n_c} grid cells per frame")
    print(f"  Sources:")
    for info in sim_info:
        print(f"    - {info['name']} (seed {info['seed']}): {info['frames']} frames")
    
    # ========================================================================
    # 2. Fit POD on combined dataset
    # ========================================================================
    print(f"\n[2/4] Fitting POD on combined dataset...")
    t_start = time.time()
    Ud, xbar, d, energy_curve = fit_pod(X_combined, energy=args.pod_energy)
    t_pod = time.time() - t_start
    
    print(f"✓ POD complete: d={d} modes, time={t_pod:.2f}s")
    print(f"  Compression: {n_c} → {d} ({n_c/d:.1f}× reduction)")
    print(f"  This basis captures IC-invariant density patterns")
    
    # ========================================================================
    # 3. Project to latent space
    # ========================================================================
    print(f"\n[3/4] Projecting to latent space...")
    Y_combined = (X_combined - xbar) @ Ud
    
    print(f"✓ Latent representation: {Y_combined.shape}")
    
    # ========================================================================
    # 4. Fit MVAR on combined latent trajectories
    # ========================================================================
    print(f"\n[4/4] Fitting MVAR on combined latent data...")
    t_start = time.time()
    mvar_model = fit_mvar(Y_combined, w=args.mvar_order, ridge_lambda=args.ridge)
    t_mvar = time.time() - t_start
    
    A0 = mvar_model['A0']
    A = mvar_model['A']
    
    print(f"✓ MVAR complete: w={args.mvar_order}, λ={args.ridge:.0e}, time={t_mvar:.2f}s")
    print(f"  Model learns dynamics that work across all {len(args.sim_dirs)} ICs")
    
    # ========================================================================
    # 5. Save model
    # ========================================================================
    print(f"\n[5/5] Saving ensemble model...")
    
    out_dir = args.out_root / f"ensemble__{args.exp_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save POD
    pod_dir = out_dir / "pod"
    pod_dir.mkdir(exist_ok=True)
    save_arrays(pod_dir, Ud=Ud, xbar=xbar, energy_curve=energy_curve)
    
    # Save MVAR
    model_dir = out_dir / "model"
    model_dir.mkdir(exist_ok=True)
    save_arrays(model_dir, A0=A0, Astack=A)
    
    model_summary = {
        'mvar_order': int(args.mvar_order),
        'ridge': float(args.ridge),
        'latent_dim': int(d),
        'total_train_samples': int(X_combined.shape[0]),
        'n_simulations': len(args.sim_dirs),
        'training_sources': sim_info,
        'pod_energy': float(args.pod_energy),
        'skip_initial': int(args.skip_initial),
        'training_time_pod': float(t_pod),
        'training_time_mvar': float(t_mvar)
    }
    
    with open(model_dir / "summary.json", 'w') as f:
        json.dump(model_summary, f, indent=2)
    
    # Save manifest
    save_manifest(
        root=out_dir,
        sim_name=f"ensemble_{args.exp_name}",
        config_path="ensemble_training",
        simulator="mvar_rom_ensemble",
        seed=0,
        code_version="1.0.0",
        n_sources=len(args.sim_dirs)
    )
    
    print(f"✓ Saved model: {out_dir}/")
    
    print("\n" + "="*80)
    print("✓ ENSEMBLE MODEL TRAINING COMPLETE")
    print("="*80)
    print(f"\nOutput: {out_dir}")
    print(f"\nTrained on {len(args.sim_dirs)} simulations:")
    for info in sim_info:
        print(f"  - {info['name']} ({info['frames']} frames)")
    print(f"\nTotal training data: {X_combined.shape[0]} frames")
    print(f"POD modes: {d}")
    print(f"MVAR parameters: {d + args.mvar_order * d * d}")
    print(f"\nUse this model with test_mvar_rom_generalization.py to test on new ICs!")


if __name__ == "__main__":
    main()
