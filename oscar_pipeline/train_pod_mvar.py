#!/usr/bin/env python3
"""
Train global POD and MVAR models from generated training data.

This script runs after all training simulations are generated.
It loads all training densities, computes global POD, projects to latent space,
and trains an MVAR model.

Usage:
    python oscar_pipeline/train_pod_mvar.py \\
        --train_dir oscar_outputs/training \\
        --output_dir oscar_outputs/models
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm

# Import shared configuration
from oscar_pipeline.config import (
    BASE_CONFIG,
    IC_TYPES,
    N_TRAIN,
    TARGET_ENERGY,
    P_LAG,
    RIDGE_ALPHA,
    DENSITY_NX,
    DENSITY_NY,
)


def load_training_data(train_dir: Path, n_train: int = None):
    """Load all training densities and create index mapping."""
    print(f"\n📂 Loading training data from: {train_dir}")
    
    if n_train is None:
        n_train = N_TRAIN
    
    X_train_list = []
    index_map = []
    train_metadata = []
    
    for sim_id in tqdm(range(n_train), desc="Loading densities"):
        run_name = f"train_{sim_id:03d}"
        run_dir = train_dir / run_name
        
        # Load density
        density_path = run_dir / "density.npz"
        if not density_path.exists():
            raise FileNotFoundError(f"Missing density file: {density_path}")
        
        density_data = np.load(density_path)
        rho = density_data["rho"]  # (T, ny, nx)
        
        # Load metadata
        metadata_path = run_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Flatten density and add to training set
        T, ny, nx = rho.shape
        rho_flat = rho.reshape(T, ny * nx)
        X_train_list.append(rho_flat)
        
        # Build index mapping
        for t in range(T):
            index_map.append({
                "global_idx": len(index_map),
                "run_name": run_name,
                "ic_type": metadata["ic_type"],
                "time_idx": t,
            })
        
        train_metadata.append(metadata)
    
    # Stack all training data
    X_train = np.vstack(X_train_list)  # (N_total, n_space)
    
    print(f"✓ Loaded {N_TRAIN} training runs")
    print(f"✓ Training matrix: {X_train.shape}")
    print(f"   IC distribution: {dict((ic, sum(1 for m in train_metadata if m['ic_type'] == ic)) for ic in IC_TYPES)}")
    
    return X_train, index_map, train_metadata


def compute_pod(X_train, target_energy):
    """Compute global POD with dynamic mode selection."""
    print(f"\n🔬 Computing global POD...")
    
    # Subtract mean
    X_train_mean = X_train.mean(axis=0)
    X_centered = X_train - X_train_mean
    
    # SVD
    print("   Running SVD...")
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Determine R_POD based on energy threshold
    total_energy = np.sum(S**2)
    cumulative_energy = np.cumsum(S**2)
    cumulative_ratio = cumulative_energy / total_energy
    
    R_POD = np.argmax(cumulative_ratio >= target_energy) + 1
    actual_energy = cumulative_ratio[R_POD - 1]
    
    # Extract POD modes
    POD_modes = Vt[:R_POD, :].T  # (n_space, R_POD)
    
    print(f"✓ Determined R_POD = {R_POD} modes to achieve {actual_energy*100:.2f}% energy")
    print(f"   Compression: {DENSITY_NX * DENSITY_NY} → {R_POD} dims ({(1 - R_POD/(DENSITY_NX * DENSITY_NY))*100:.2f}% reduction)")
    
    return {
        "U": POD_modes,
        "S": S[:R_POD],
        "mean": X_train_mean,
        "energy": float(actual_energy),
        "R_POD": int(R_POD),
        "target_energy": float(target_energy),
        "total_energy": float(total_energy),
        "cumulative_ratio": cumulative_ratio,
    }


def project_to_latent(X_train, pod_model):
    """Project training data to latent space."""
    print(f"\n📊 Projecting to latent space...")
    
    X_centered = X_train - pod_model["mean"]
    Y_train = X_centered @ pod_model["U"]  # (N_total, R_POD)
    
    print(f"✓ Latent data: {Y_train.shape}")
    
    return Y_train


def fit_mvar(Y, p, alpha):
    """Fit MVAR model with ridge regularization."""
    print(f"\n🎯 Training MVAR model...")
    print(f"   Latent dimension: {Y.shape[1]}")
    print(f"   Lag order: {p}")
    print(f"   Ridge alpha: {alpha}")
    
    T, r = Y.shape
    
    # Build design matrix
    X_list = []
    Y_target_list = []
    
    for t in range(p, T):
        lags = []
        for lag in range(1, p+1):
            lags.append(Y[t-lag])
        X_list.append(np.concatenate(lags))
        Y_target_list.append(Y[t])
    
    X = np.array(X_list)  # (T-p, r*p)
    Y_target = np.array(Y_target_list)  # (T-p, r)
    
    # Ridge regression
    XtX = X.T @ X
    XtY = X.T @ Y_target
    A = np.linalg.solve(XtX + alpha * np.eye(X.shape[1]), XtY)  # (r*p, r)
    
    # Reshape to coefficient matrices
    A_matrices = []
    for i in range(p):
        A_matrices.append(A[i*r:(i+1)*r, :].T)  # (r, r)
    
    # Compute training metrics
    Y_pred = X @ A
    ss_res = np.sum((Y_target - Y_pred)**2)
    ss_tot = np.sum((Y_target - Y_target.mean(axis=0))**2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((Y_target - Y_pred)**2))
    
    print(f"✓ MVAR trained")
    print(f"   Training R²: {r2:.4f}")
    print(f"   Training RMSE: {rmse:.4f}")
    
    return {
        "A_matrices": np.array(A_matrices),  # (p, r, r)
        "p": int(p),
        "r": int(r),
        "alpha": float(alpha),
        "train_r2": float(r2),
        "train_rmse": float(rmse),
    }


def save_models(pod_model, mvar_model, train_metadata, index_map, output_dir: Path):
    """Save POD and MVAR models."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save POD model
    pod_path = output_dir / "pod_model.npz"
    np.savez(
        pod_path,
        U=pod_model["U"],
        S=pod_model["S"],
        mean=pod_model["mean"],
        energy=pod_model["energy"],
        R_POD=pod_model["R_POD"],
        target_energy=pod_model["target_energy"],
    )
    print(f"\n💾 Saved POD model: {pod_path}")
    
    # Save MVAR model
    mvar_path = output_dir / "mvar_model.npz"
    np.savez(
        mvar_path,
        A_matrices=mvar_model["A_matrices"],
        p=mvar_model["p"],
        r=mvar_model["r"],
        alpha=mvar_model["alpha"],
        train_r2=mvar_model["train_r2"],
        train_rmse=mvar_model["train_rmse"],
    )
    print(f"💾 Saved MVAR model: {mvar_path}")
    
    # Save training metadata
    metadata = {
        "N_train": N_TRAIN,
        "IC_types": IC_TYPES,
        "IC_distribution": {
            ic: sum(1 for m in train_metadata if m["ic_type"] == ic)
            for ic in IC_TYPES
        },
        "config": BASE_CONFIG,
        "density": {
            "nx": DENSITY_NX,
            "ny": DENSITY_NY,
        },
        "pod": {
            "R_POD": pod_model["R_POD"],
            "energy": pod_model["energy"],
            "target_energy": pod_model["target_energy"],
        },
        "mvar": {
            "p": mvar_model["p"],
            "alpha": mvar_model["alpha"],
            "train_r2": mvar_model["train_r2"],
            "train_rmse": mvar_model["train_rmse"],
        },
    }
    
    metadata_path = output_dir / "training_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Saved metadata: {metadata_path}")
    
    # Save index mapping
    import pandas as pd
    index_df = pd.DataFrame(index_map)
    index_path = output_dir / "index_mapping.csv"
    index_df.to_csv(index_path, index=False)
    print(f"💾 Saved index mapping: {index_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train global POD and MVAR models"
    )
    parser.add_argument(
        "--train_dir",
        type=Path,
        required=True,
        help="Directory with training simulations",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for models",
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=None,
        help="Number of training simulations (default: use config N_TRAIN)",
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("TRAINING POD + MVAR MODELS")
    print("="*80)
    
    # Load training data
    X_train, index_map, train_metadata = load_training_data(
        args.train_dir, 
        n_train=args.n_train
    )
    
    # Compute POD
    pod_model = compute_pod(X_train, TARGET_ENERGY)
    
    # Project to latent space
    Y_train = project_to_latent(X_train, pod_model)
    
    # Train MVAR
    mvar_model = fit_mvar(Y_train, P_LAG, RIDGE_ALPHA)
    
    # Save models
    save_models(pod_model, mvar_model, train_metadata, index_map, args.output_dir)
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
