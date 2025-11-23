#!/usr/bin/env python3
"""
Complete Oscar Pipeline - Heavy Computation
============================================

This script runs the FULL pipeline suitable for Oscar (or local testing):
1. Generate training simulations (trajectories + densities)
2. Train global POD + MVAR models
3. Generate test simulations (trajectories + densities + order params)
4. Run MVAR predictions on test data
5. Save all outputs in complete_pipeline structure

This generates ALL files needed for visualization. The lightweight 
run_complete_pipeline.py will then just load these and generate videos/plots.

Usage:
    # Full production run
    python oscar_pipeline/run_complete_oscar.py --output_dir oscar_outputs

    # Quick test
    python oscar_pipeline/run_complete_oscar.py --output_dir test_run \\
        --n_train 10 --n_test 4
"""

import argparse
import json
import sys
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm

# Import Oscar pipeline modules
from oscar_pipeline.config import (
    BASE_CONFIG,
    IC_TYPES,
    N_TRAIN,
    M_TEST as N_TEST,
    TARGET_ENERGY,
    P_LAG,
    RIDGE_ALPHA,
    DENSITY_NX,
    DENSITY_NY,
    DENSITY_BANDWIDTH,
)

# Import rectsim functions
from rectsim.vicsek_discrete import simulate_backend
from rectsim.legacy_functions import (
    kde_density_movie,
    compute_order_params,
    compute_frame_metrics,
    compute_summary_metrics,
)


def generate_simulation(config, seed, output_dir, run_name, compute_order=False):
    """Generate a single simulation with trajectory + density."""
    print(f"\n{'='*80}")
    print(f"Generating {run_name}")
    print(f"{'='*80}")
    
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Run simulation
    rng = np.random.default_rng(seed)
    out = simulate_backend(config, rng)
    
    # Save trajectory
    traj_path = run_dir / "trajectory.npz"
    np.savez(
        traj_path,
        positions=out["traj"],
        velocities=out["vel"],
        times=out["times"],
    )
    print(f"✓ Saved trajectory: {traj_path.name}")
    
    # Compute and save density
    rho, meta = kde_density_movie(
        out["traj"],
        config["sim"]["Lx"],
        config["sim"]["Ly"],
        nx=DENSITY_NX,
        ny=DENSITY_NY,
        bandwidth=DENSITY_BANDWIDTH,
    )
    
    # Create grid edges for saving
    x_edges = np.linspace(0.0, config["sim"]["Lx"], DENSITY_NX + 1)
    y_edges = np.linspace(0.0, config["sim"]["Ly"], DENSITY_NY + 1)
    
    density_path = run_dir / "density_true.npz"
    np.savez(
        density_path,
        rho=rho,
        times=out["times"],
        x_edges=x_edges,
        y_edges=y_edges,
        extent=meta["extent"],
    )
    print(f"✓ Saved density: {density_path.name}")
    
    # Save order parameters if requested
    if compute_order:
        T = out["vel"].shape[0]
        phi_list = []
        mean_speed_list = []
        speed_std_list = []
        
        for t in range(T):
            params = compute_order_params(out["vel"][t])
            phi_list.append(params["phi"])
            mean_speed_list.append(params["mean_speed"])
            speed_std_list.append(params["speed_std"])
        
        order_path = run_dir / "order_params.csv"
        import pandas as pd
        pd.DataFrame({
            "time": out["times"],
            "phi": phi_list,
            "mean_speed": mean_speed_list,
            "speed_std": speed_std_list,
        }).to_csv(order_path, index=False)
        print(f"✓ Saved order parameters: {order_path.name}")
    
    # Save metadata
    metadata_path = run_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump({
            "run_name": run_name,
            "seed": seed,
            "ic_type": config["ic"]["kind"],
            "config": config,
        }, f, indent=2)
    print(f"✓ Saved metadata: {metadata_path.name}")
    
    return rho, out["times"], meta


def train_pod_mvar(train_dir, n_train, output_dir):
    """Train global POD and MVAR models."""
    print(f"\n{'='*80}")
    print("TRAINING POD + MVAR MODELS")
    print(f"{'='*80}")
    
    # Load training densities
    print(f"\n📂 Loading {n_train} training simulations...")
    X_train_list = []
    index_map = []
    
    for sim_id in tqdm(range(n_train), desc="Loading densities"):
        run_name = f"train_{sim_id:03d}"
        density_path = train_dir / run_name / "density_true.npz"
        
        if not density_path.exists():
            print(f"⚠️  Missing: {density_path}")
            continue
            
        data = np.load(density_path)
        rho = data["rho"]  # (T, ny, nx)
        T, ny, nx = rho.shape
        
        # Flatten spatial dimensions
        rho_flat = rho.reshape(T, -1)  # (T, ny*nx)
        X_train_list.append(rho_flat)
        
        # Track which frames belong to which simulation
        for t in range(T):
            index_map.append({
                "run_name": run_name,
                "frame_idx": t,
                "sim_id": sim_id,
            })
    
    # Concatenate all training data
    X_train = np.vstack(X_train_list)  # (total_frames, n_c)
    print(f"✓ Training matrix: {X_train.shape}")
    
    # Compute POD
    print(f"\n🔬 Computing global POD (target energy={TARGET_ENERGY*100:.1f}%)...")
    U, s, Vt = np.linalg.svd(X_train - X_train.mean(axis=0), full_matrices=False)
    
    energy = np.cumsum(s**2) / np.sum(s**2)
    R_POD = np.searchsorted(energy, TARGET_ENERGY) + 1
    
    print(f"✓ R_POD = {R_POD} modes ({energy[R_POD-1]*100:.2f}% energy)")
    
    # Project to latent space
    print(f"\n📊 Projecting to latent space...")
    X_mean = X_train.mean(axis=0)
    X_centered = X_train - X_mean
    Y_train = X_centered @ Vt[:R_POD].T  # (total_frames, R_POD)
    print(f"✓ Latent data: {Y_train.shape}")
    
    # Train MVAR
    print(f"\n🎯 Training MVAR (p={P_LAG}, alpha={RIDGE_ALPHA})...")
    
    # Build lagged predictor matrix
    T_total = Y_train.shape[0]
    Y_hist = []
    Y_target = []
    
    for t in range(P_LAG, T_total):
        # Stack p previous steps: [y[t-p], y[t-p+1], ..., y[t-1]]
        y_lag = Y_train[t-P_LAG:t].flatten()
        Y_hist.append(y_lag)
        Y_target.append(Y_train[t])
    
    Y_hist = np.array(Y_hist)  # (T-p, p*R_POD)
    Y_target = np.array(Y_target)  # (T-p, R_POD)
    
    # Ridge regression
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
    ridge.fit(Y_hist, Y_target)
    
    # Reshape to MVAR coefficient matrices
    A_flat = ridge.coef_  # (R_POD, p*R_POD)
    A_matrices = []
    for i in range(P_LAG):
        A_i = A_flat[:, i*R_POD:(i+1)*R_POD].T  # (R_POD, R_POD)
        A_matrices.append(A_i)
    
    # Training metrics
    Y_pred = ridge.predict(Y_hist)
    train_r2 = 1 - np.sum((Y_target - Y_pred)**2) / np.sum((Y_target - Y_target.mean(axis=0))**2)
    train_rmse = np.sqrt(np.mean((Y_target - Y_pred)**2))
    
    print(f"✓ MVAR trained: R²={train_r2:.4f}, RMSE={train_rmse:.4f}")
    
    # Save models
    print(f"\n💾 Saving models to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # POD model
    pod_path = output_dir / "pod_model.npz"
    np.savez(
        pod_path,
        U=Vt[:R_POD].T,  # (n_c, R_POD) - basis vectors
        s=s[:R_POD],
        mean=X_mean,
        energy_retained=energy[R_POD-1],
        R_POD=R_POD,
    )
    print(f"✓ Saved POD: {pod_path.name}")
    
    # MVAR model
    mvar_path = output_dir / "mvar_model.npz"
    np.savez(
        mvar_path,
        A_matrices=np.array(A_matrices),  # (p, R_POD, R_POD)
        p=P_LAG,
        train_r2=train_r2,
        train_rmse=train_rmse,
    )
    print(f"✓ Saved MVAR: {mvar_path.name}")
    
    # Metadata
    meta_path = output_dir / "training_metadata.json"
    with open(meta_path, "w") as f:
        json.dump({
            "n_train": n_train,
            "R_POD": int(R_POD),
            "target_energy": TARGET_ENERGY,
            "actual_energy": float(energy[R_POD-1]),
            "p_lag": P_LAG,
            "ridge_alpha": RIDGE_ALPHA,
            "train_r2": float(train_r2),
            "train_rmse": float(train_rmse),
            "config": BASE_CONFIG,
        }, f, indent=2)
    print(f"✓ Saved metadata: {meta_path.name}")
    
    # Index mapping
    import pandas as pd
    index_path = output_dir / "index_mapping.csv"
    pd.DataFrame(index_map).to_csv(index_path, index=False)
    print(f"✓ Saved index mapping: {index_path.name}")
    
    return {
        "U": Vt[:R_POD].T,
        "mean": X_mean,
        "R_POD": R_POD,
        "A_matrices": A_matrices,
    }


def predict_and_save(test_dir, n_test, models_dir, output_dir):
    """Run predictions on all test simulations and save results."""
    print(f"\n{'='*80}")
    print("RUNNING PREDICTIONS ON TEST DATA")
    print(f"{'='*80}")
    
    # Load models
    print(f"\n📂 Loading models from {models_dir}...")
    pod_data = np.load(models_dir / "pod_model.npz")
    mvar_data = np.load(models_dir / "mvar_model.npz")
    
    U = pod_data["U"]  # (n_c, R_POD)
    X_mean = pod_data["mean"]
    A_matrices = [mvar_data["A_matrices"][i] for i in range(mvar_data["A_matrices"].shape[0])]
    p = len(A_matrices)
    R_POD = U.shape[1]
    
    print(f"✓ POD: {R_POD} modes")
    print(f"✓ MVAR: lag={p}")
    
    # Run predictions
    print(f"\n🔮 Making predictions on {n_test} test runs...")
    metrics_list = []
    
    for sim_id in tqdm(range(n_test), desc="Predictions"):
        run_name = f"test_{sim_id:03d}"
        run_dir = test_dir / run_name
        
        # Load true density
        density_path = run_dir / "density_true.npz"
        if not density_path.exists():
            continue
            
        data = np.load(density_path)
        rho_true = data["rho"]  # (T, ny, nx)
        T, ny, nx = rho_true.shape
        
        # Flatten
        rho_true_flat = rho_true.reshape(T, -1)  # (T, n_c)
        
        # Project to latent
        y_true = (rho_true_flat - X_mean) @ U  # (T, R_POD)
        
        # MVAR forecast
        y_init = [y_true[i] for i in range(p)]
        y_forecast = []
        
        for t in range(T - p):
            y_next = np.zeros(R_POD)
            for i, A in enumerate(A_matrices):
                y_next += A @ y_init[-(i+1)]
            y_forecast.append(y_next)
            y_init.append(y_next)
        
        y_forecast = np.array(y_forecast)  # (T-p, R_POD)
        
        # Reconstruct density
        rho_pred_flat = y_forecast @ U.T + X_mean  # (T-p, n_c)
        rho_pred = rho_pred_flat.reshape(T-p, ny, nx)
        
        # Pad with initial conditions
        rho_pred_full = np.vstack([rho_true[:p], rho_pred])
        
        # Save prediction
        pred_path = run_dir / "density_pred.npz"
        np.savez(
            pred_path,
            rho=rho_pred_full,
            times=data["times"],
            x_edges=data["x_edges"],
            y_edges=data["y_edges"],
            extent=data["extent"],
        )
        
        # Compute metrics
        frame_metrics = compute_frame_metrics(rho_true_flat, rho_pred_full.reshape(T, -1))
        summary = compute_summary_metrics(frame_metrics)
        
        # Load metadata to get IC type
        meta_path = run_dir / "metadata.json"
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        
        metrics_list.append({
            "run_name": run_name,
            "ic_type": metadata["ic_type"],
            "r2": summary["R2"],
            "median_e2": summary["median_e2"],
            "p10_e2": summary["p10_e2"],
            "p90_e2": summary["p90_e2"],
            "tau_tol": summary["tau_tol"],
            "mean_mass_error": summary["mean_mass_error"],
            "max_mass_error": summary["max_mass_error"],
        })
    
    # Save metrics
    import pandas as pd
    metrics_path = output_dir / "metrics_all_runs.csv"
    pd.DataFrame(metrics_list).to_csv(metrics_path, index=False)
    print(f"\n💾 Saved metrics: {metrics_path}")
    
    # Compute metrics by IC type
    df = pd.DataFrame(metrics_list)
    ic_stats = []
    
    for ic_type in IC_TYPES:
        ic_df = df[df["ic_type"] == ic_type]
        if len(ic_df) > 0:
            ic_stats.append({
                "ic_type": ic_type,
                "n_runs": len(ic_df),
                "mean_r2": ic_df["r2"].mean(),
                "std_r2": ic_df["r2"].std(),
                "median_r2": ic_df["r2"].median(),
                "mean_e2": ic_df["median_e2"].mean(),
                "std_e2": ic_df["median_e2"].std(),
            })
    
    ic_metrics_path = output_dir / "metrics_by_ic_type.csv"
    pd.DataFrame(ic_stats).to_csv(ic_metrics_path, index=False)
    print(f"💾 Saved IC metrics: {ic_metrics_path}")
    
    print(f"\n✓ Completed predictions for {n_test} test runs")


def main():
    parser = argparse.ArgumentParser(
        description="Run complete Oscar pipeline (heavy computation)"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/complete_pipeline"),
        help="Output directory for all results",
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=N_TRAIN,
        help=f"Number of training simulations (default: {N_TRAIN})",
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=N_TEST,
        help=f"Number of test simulations (default: {N_TEST})",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean output directory before starting",
    )
    
    args = parser.parse_args()
    
    # Setup directories
    if args.clean and args.output_dir.exists():
        print(f"\n🗑️  Cleaning: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    
    train_dir = args.output_dir / "train"
    test_dir = args.output_dir / "test"
    models_dir = args.output_dir / "mvar"
    
    for d in [train_dir, test_dir, models_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("COMPLETE OSCAR PIPELINE - HEAVY COMPUTATION")
    print("="*80)
    print(f"\nOutput: {args.output_dir}")
    print(f"Training sims: {args.n_train}")
    print(f"Test sims: {args.n_test}")
    
    # STEP 1: Generate training simulations
    print(f"\n{'='*80}")
    print("STEP 1: Generating Training Simulations")
    print(f"{'='*80}")
    
    for sim_id in tqdm(range(args.n_train), desc="Training sims"):
        ic_type = IC_TYPES[sim_id % len(IC_TYPES)]
        config = BASE_CONFIG.copy()
        config["ic"] = {"kind": ic_type}
        seed = 1000 + sim_id
        run_name = f"train_{sim_id:03d}"
        
        generate_simulation(config, seed, train_dir, run_name, compute_order=False)
    
    print(f"\n✓ Generated {args.n_train} training simulations")
    
    # STEP 2: Train POD + MVAR
    print(f"\n{'='*80}")
    print("STEP 2: Training POD + MVAR Models")
    print(f"{'='*80}")
    
    models = train_pod_mvar(train_dir, args.n_train, models_dir)
    
    # STEP 3: Generate test simulations
    print(f"\n{'='*80}")
    print("STEP 3: Generating Test Simulations")
    print(f"{'='*80}")
    
    # Stratified by IC type
    test_per_ic = args.n_test // len(IC_TYPES)
    sim_id = 0
    
    for ic_type in IC_TYPES:
        for i in range(test_per_ic):
            config = BASE_CONFIG.copy()
            config["ic"] = {"kind": ic_type}
            seed = 2000 + sim_id
            run_name = f"test_{sim_id:03d}"
            
            generate_simulation(config, seed, test_dir, run_name, compute_order=True)
            sim_id += 1
    
    print(f"\n✓ Generated {args.n_test} test simulations")
    
    # STEP 4: Run predictions
    predict_and_save(test_dir, args.n_test, models_dir, test_dir)
    
    print(f"\n{'='*80}")
    print("✅ OSCAR PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"\nAll outputs saved to: {args.output_dir}")
    print("\nGenerated files:")
    print(f"  • {args.n_train} training sims (trajectory, density, metadata)")
    print(f"  • {args.n_test} test sims (trajectory, density, order params, predictions, metadata)")
    print(f"  • POD + MVAR models")
    print(f"  • Metrics CSV files")
    print("\nNext step: Run visualization pipeline")
    print(f"  python run_complete_pipeline.py --load_from {args.output_dir}")


if __name__ == "__main__":
    main()
