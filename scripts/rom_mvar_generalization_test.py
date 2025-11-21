#!/usr/bin/env python3
"""ROM/MVAR Generalization Testing (Local Machine Only)

This script tests ROM/MVAR generalization on multiple ICs with different distributions:
1. Load trained ROM/MVAR model from Oscar exports
2. Generate predictions on new ICs (uniform and gaussian with 1-4 clusters)
3. Run ground truth simulations for comparison
4. Compute metrics for all predictions
5. Aggregate statistics: uniform vs gaussian (by cluster count)
6. Generate videos/plots only for best R² cases

Usage:
    # Test with default settings (10 uniform, 10 gaussian per cluster count)
    python scripts/rom_mvar_generalization_test.py \\
        --experiment vicsek_morse_test \\
        --config configs/vicsek_morse_test.yaml
    
    # Custom IC counts
    python scripts/rom_mvar_generalization_test.py \\
        --experiment vicsek_morse_test \\
        --config configs/vicsek_morse_test.yaml \\
        --num_uniform 20 \\
        --num_gaussian 15 \\
        --cluster_counts 1 2 3 4

Author: Maria
Date: November 2025
"""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rectsim.cli import _run_single
from rectsim.config import load_config
from rectsim.density import compute_density_grid
from rectsim.metrics import compute_timeseries
from rectsim.rom_mvar import (
    forecast_mvar,
    load_mvar_model,
    load_pod_model,
    project_to_pod,
)


def generate_uniform_ic(N: int, Lx: float, Ly: float, rng: np.random.Generator) -> np.ndarray:
    """Generate uniform random initial condition."""
    return rng.uniform(0, [Lx, Ly], size=(N, 2))


def generate_gaussian_ic(
    N: int, Lx: float, Ly: float, num_clusters: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate gaussian clustered initial condition."""
    # Place cluster centers randomly
    centers = rng.uniform([Lx * 0.2, Ly * 0.2], [Lx * 0.8, Ly * 0.8], size=(num_clusters, 2))
    
    # Assign particles to clusters
    cluster_ids = rng.choice(num_clusters, size=N)
    
    # Generate particles around cluster centers with std ~ Lx/10
    std = min(Lx, Ly) / 10
    positions = np.zeros((N, 2))
    for i in range(N):
        center = centers[cluster_ids[i]]
        positions[i] = center + rng.normal(0, std, size=2)
        # Wrap to domain
        positions[i, 0] = positions[i, 0] % Lx
        positions[i, 1] = positions[i, 1] % Ly
    
    return positions


def run_prediction_and_truth(
    cfg: dict,
    model_dir: Path,
    seed: int,
    ic_type: str,
    num_clusters: int | None,
    nx: int,
    ny: int,
    Lx: float,
    Ly: float,
    bandwidth: float,
) -> dict:
    """Run ROM/MVAR prediction and ground truth simulation for a single IC.
    
    Returns dict with:
        - density_true: (T, nx, ny)
        - density_pred: (T, nx, ny)
        - times: (T,)
        - metrics: DataFrame
        - summary: dict with R², RMSE, etc.
    """
    rng = np.random.default_rng(seed)
    
    # Load ROM/MVAR model
    print(f"    Loading model from {model_dir}...")
    pod_basis, pod_mean, pod_meta = load_pod_model(model_dir)
    mvar_params = load_mvar_model(model_dir)
    p = mvar_params["order"]
    
    # Generate initial condition
    N = cfg["sim"]["N"]
    if ic_type == "uniform":
        x0 = generate_uniform_ic(N, Lx, Ly, rng)
        print(f"    Generated uniform IC with seed={seed}")
    else:  # gaussian
        x0 = generate_gaussian_ic(N, Lx, Ly, num_clusters, rng)
        print(f"    Generated gaussian IC ({num_clusters} clusters) with seed={seed}")
    
    # Run ground truth simulation
    print("    Running ground truth simulation...")
    test_cfg = deepcopy(cfg)
    test_cfg["seed"] = seed
    test_cfg["initial_positions"] = x0.tolist()  # Override IC
    test_cfg["outputs"]["animate_traj"] = False
    test_cfg["outputs"]["animate_density"] = False
    test_cfg["outputs"]["plot_order_params"] = False
    
    sim_result = _run_single(test_cfg, ic_id=None, enable_videos=False, enable_order_plots=False)
    results = sim_result["results"]
    traj = results["traj"]
    times = results["times"]
    T = traj.shape[0]
    bc = test_cfg["sim"]["bc"]
    
    # Compute true density
    print("    Computing true density...")
    density_true = np.zeros((T, nx, ny))
    for t in range(T):
        rho, _, _ = compute_density_grid(traj[t], nx, ny, Lx, Ly, bandwidth=bandwidth, bc=bc)
        density_true[t] = rho
    
    # Project initial density to latent space
    print("    Projecting IC to latent space...")
    rho0, _, _ = compute_density_grid(traj[0], nx, ny, Lx, Ly, bandwidth=bandwidth, bc=bc)
    rho0_flat = rho0.flatten()
    a0 = project_to_pod(rho0_flat, pod_basis, pod_mean)
    
    # Need p initial latent states for MVAR
    # Use first p time steps from true trajectory
    a_init = np.zeros((p, a0.shape[0]))
    for i in range(min(p, T)):
        rho_i, _, _ = compute_density_grid(traj[i], nx, ny, Lx, Ly, bandwidth=bandwidth, bc=bc)
        a_init[i] = project_to_pod(rho_i.flatten(), pod_basis, pod_mean)
    
    # Forecast with MVAR
    print("    Forecasting with MVAR...")
    a_forecast = forecast_mvar(
        a_init=a_init,
        A_coeffs=mvar_params["A"],
        num_steps=T - p,
    )
    
    # Reconstruct density
    print("    Reconstructing density...")
    density_pred = np.zeros((T, nx, ny))
    for t in range(p):
        density_pred[t] = density_true[t]  # Use true density for first p steps
    
    for t in range(p, T):
        a_t = a_forecast[t - p]
        rho_pred_flat = a_t @ pod_basis.T + pod_mean
        rho_pred_flat = np.maximum(rho_pred_flat, 0)  # Enforce non-negativity
        density_pred[t] = rho_pred_flat.reshape(nx, ny)
    
    # Compute metrics
    print("    Computing metrics...")
    from rectsim.rom_mvar import compute_timeseries_metrics
    
    metrics_df = compute_timeseries_metrics(
        density_true=density_true,
        density_pred=density_pred,
        times=times,
        error_tolerance=0.1,
        error_metric="rmse",
    )
    
    # Summary statistics
    r2_mean = metrics_df["r2"].mean()
    rmse_mean = metrics_df["rmse"].mean()
    
    # Find tolerance horizon
    tau_idx = np.where(metrics_df["rmse"] > 0.1)[0]
    tau = tau_idx[0] if len(tau_idx) > 0 else len(times)
    
    summary = {
        "seed": seed,
        "ic_type": ic_type,
        "num_clusters": num_clusters if ic_type == "gaussian" else None,
        "r2_mean": float(r2_mean),
        "rmse_mean": float(rmse_mean),
        "tolerance_horizon": int(tau),
    }
    
    # Compute order parameters for visualization
    order_metrics = compute_timeseries(
        traj, results["vel"], times, Lx, Ly, bc
    )
    
    return {
        "density_true": density_true,
        "density_pred": density_pred,
        "times": times,
        "traj": traj,
        "vel": results["vel"],
        "metrics": metrics_df,
        "order_metrics": order_metrics,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--rom_root", default="rom_mvar", help="ROM root directory")
    parser.add_argument("--num_uniform", type=int, default=10, help="Number of uniform ICs to test")
    parser.add_argument("--num_gaussian", type=int, default=10, help="Number of gaussian ICs per cluster count")
    parser.add_argument("--cluster_counts", type=int, nargs="+", default=[1, 2, 3, 4], help="Cluster counts for gaussian ICs")
    parser.add_argument("--base_seed", type=int, default=2000, help="Base seed for IC generation")
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Setup paths
    rom_root = Path(args.rom_root)
    exp_dir = rom_root / args.experiment
    model_dir = exp_dir / "model"
    gen_test_dir = exp_dir / "generalization_test"
    gen_test_dir.mkdir(parents=True, exist_ok=True)
    
    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}")
        sys.exit(1)
    
    # Get grid parameters
    grid_cfg = cfg["outputs"]["grid_density"]
    nx = grid_cfg["nx"]
    ny = grid_cfg["ny"]
    bandwidth = grid_cfg["bandwidth"]
    Lx = cfg["sim"]["Lx"]
    Ly = cfg["sim"]["Ly"]
    
    print("=" * 70)
    print("ROM/MVAR GENERALIZATION TEST")
    print("=" * 70)
    print(f"Experiment: {args.experiment}")
    print(f"Uniform ICs: {args.num_uniform}")
    print(f"Gaussian ICs per cluster: {args.num_gaussian}")
    print(f"Cluster counts: {args.cluster_counts}")
    print("=" * 70)
    
    all_results = []
    
    # Test uniform ICs
    print("\n" + "=" * 70)
    print("TESTING UNIFORM ICs")
    print("=" * 70)
    
    for i in range(args.num_uniform):
        seed = args.base_seed + i
        print(f"\n[{i+1}/{args.num_uniform}] Uniform IC {seed}")
        
        result = run_prediction_and_truth(
            cfg, model_dir, seed, "uniform", None, nx, ny, Lx, Ly, bandwidth
        )
        
        # Save individual result
        ic_dir = gen_test_dir / f"uniform_ic_{seed:04d}"
        ic_dir.mkdir(exist_ok=True)
        
        np.savez(
            ic_dir / "densities.npz",
            density_true=result["density_true"],
            density_pred=result["density_pred"],
            times=result["times"],
            Lx=Lx,
            Ly=Ly,
        )
        result["metrics"].to_csv(ic_dir / "metrics.csv", index=False)
        
        with open(ic_dir / "summary.json", "w") as f:
            json.dump(result["summary"], f, indent=2)
        
        all_results.append(result)
        print(f"    ✓ R²={result['summary']['r2_mean']:.4f}, RMSE={result['summary']['rmse_mean']:.4f}")
    
    # Test gaussian ICs
    for num_clusters in args.cluster_counts:
        print("\n" + "=" * 70)
        print(f"TESTING GAUSSIAN ICs ({num_clusters} clusters)")
        print("=" * 70)
        
        for i in range(args.num_gaussian):
            seed = args.base_seed + 1000 * num_clusters + i
            print(f"\n[{i+1}/{args.num_gaussian}] Gaussian IC {seed} ({num_clusters} clusters)")
            
            result = run_prediction_and_truth(
                cfg, model_dir, seed, "gaussian", num_clusters, nx, ny, Lx, Ly, bandwidth
            )
            
            # Save individual result
            ic_dir = gen_test_dir / f"gaussian_{num_clusters}clust_ic_{seed:04d}"
            ic_dir.mkdir(exist_ok=True)
            
            np.savez(
                ic_dir / "densities.npz",
                density_true=result["density_true"],
                density_pred=result["density_pred"],
                times=result["times"],
                Lx=Lx,
                Ly=Ly,
            )
            result["metrics"].to_csv(ic_dir / "metrics.csv", index=False)
            result["order_metrics"].to_csv(ic_dir / "order_params.csv", index=False)
            
            with open(ic_dir / "summary.json", "w") as f:
                json.dump(result["summary"], f, indent=2)
            
            all_results.append(result)
            print(f"    ✓ R²={result['summary']['r2_mean']:.4f}, RMSE={result['summary']['rmse_mean']:.4f}")
    
    # Aggregate statistics
    print("\n" + "=" * 70)
    print("AGGREGATING STATISTICS")
    print("=" * 70)
    
    summaries_df = pd.DataFrame([r["summary"] for r in all_results])
    summaries_df.to_csv(gen_test_dir / "all_summaries.csv", index=False)
    
    # Compute aggregate stats
    stats = {}
    
    # Uniform stats
    uniform_results = summaries_df[summaries_df["ic_type"] == "uniform"]
    stats["uniform"] = {
        "count": len(uniform_results),
        "r2_mean": float(uniform_results["r2_mean"].mean()),
        "r2_std": float(uniform_results["r2_mean"].std()),
        "rmse_mean": float(uniform_results["rmse_mean"].mean()),
        "rmse_std": float(uniform_results["rmse_mean"].std()),
        "best_ic_seed": int(uniform_results.loc[uniform_results["r2_mean"].idxmax(), "seed"]),
        "best_ic_r2": float(uniform_results["r2_mean"].max()),
    }
    
    # Gaussian stats by cluster count
    stats["gaussian"] = {}
    for num_clusters in args.cluster_counts:
        gaussian_results = summaries_df[
            (summaries_df["ic_type"] == "gaussian") &
            (summaries_df["num_clusters"] == num_clusters)
        ]
        if len(gaussian_results) > 0:
            stats["gaussian"][f"{num_clusters}_clusters"] = {
                "count": len(gaussian_results),
                "r2_mean": float(gaussian_results["r2_mean"].mean()),
                "r2_std": float(gaussian_results["r2_mean"].std()),
                "rmse_mean": float(gaussian_results["rmse_mean"].mean()),
                "rmse_std": float(gaussian_results["rmse_mean"].std()),
                "best_ic_seed": int(gaussian_results.loc[gaussian_results["r2_mean"].idxmax(), "seed"]),
                "best_ic_r2": float(gaussian_results["r2_mean"].max()),
            }
    
    # Save aggregate stats
    with open(gen_test_dir / "aggregate_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print("\nAggregate Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Find best ICs for visualization
    best_uniform_seed = stats["uniform"]["best_ic_seed"]
    best_gaussian_seeds = {
        nc: stats["gaussian"][f"{nc}_clusters"]["best_ic_seed"]
        for nc in args.cluster_counts
    }
    
    print("\n" + "=" * 70)
    print("BEST ICs FOR VISUALIZATION")
    print("=" * 70)
    print(f"Best uniform IC: seed={best_uniform_seed}, R²={stats['uniform']['best_ic_r2']:.4f}")
    for nc in args.cluster_counts:
        seed = best_gaussian_seeds[nc]
        r2 = stats["gaussian"][f"{nc}_clusters"]["best_ic_r2"]
        print(f"Best {nc}-cluster gaussian IC: seed={seed}, R²={r2:.4f}")
    
    print("\n" + "=" * 70)
    print("GENERALIZATION TEST COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {gen_test_dir}")
    print("\nNext steps:")
    print("  1. Generate visualizations for best ICs:")
    print(f"     python scripts/rom_mvar_visualize_best.py --experiment {args.experiment}")
    print("  2. Review aggregate statistics:")
    print(f"     cat {gen_test_dir / 'aggregate_stats.json'}")


if __name__ == "__main__":
    main()
