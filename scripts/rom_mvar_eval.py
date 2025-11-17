#!/usr/bin/env python3
"""ROM/MVAR evaluation pipeline: test generalization on unseen ICs.

This script:
1. Loads trained POD + MVAR model from disk
2. Generates K_test simulations with unseen ICs
3. Projects initial conditions to latent space
4. Forecasts using MVAR and reconstructs density
5. Computes comprehensive metrics (R², RMSE, mass error, tolerance horizon τ)
6. Saves per-IC and aggregate metrics (NO videos by default for Oscar)

Usage:
    # Evaluate on Oscar (no videos)
    python scripts/rom_mvar_eval.py \\
        --experiment my_rom_experiment \\
        --config configs/rom_eval.yaml

    # Evaluate locally with plots
    python scripts/rom_mvar_eval.py \\
        --experiment my_rom_experiment \\
        --config configs/rom_eval.yaml \\
        rom.eval.generate_plots=true

Author: Maria
Date: November 2025
"""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from rectsim.cli import _run_single
from rectsim.config import load_config
from rectsim.density import compute_density_grid
from rectsim.rom_mvar import (
    ROMEvalConfig,
    compute_summary_metrics,
    compute_timeseries_metrics,
    forecast_mvar,
    load_mvar_model,
    load_pod_model,
    project_to_pod,
    reconstruct_from_pod,
    setup_rom_directories,
    setup_test_ic_directory,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained ROM/MVAR on unseen ICs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Name of trained ROM experiment",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file with simulation settings",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in key=value format",
    )
    
    args = parser.parse_args()
    
    # Parse overrides
    override_pairs = []
    for arg in args.overrides:
        if "=" not in arg:
            parser.error(f"Override must be key=value format, got: {arg}")
        key, value = arg.split("=", 1)
        import json as _json
        try:
            value = _json.loads(value)
        except _json.JSONDecodeError:
            pass
        override_pairs.append((key, value))
    
    # Load config
    cfg = load_config(args.config, overrides=override_pairs)
    
    # Extract ROM eval config
    rom_eval_dict = cfg.get("rom", {}).get("eval", {})
    eval_cfg = ROMEvalConfig(
        experiment_name=args.experiment,
        num_test_ics=rom_eval_dict.get("num_test_ics", 1),
        test_seeds=rom_eval_dict.get("test_seeds"),
        forecast_horizon=rom_eval_dict.get("forecast_horizon"),
        error_tolerance=rom_eval_dict.get("error_tolerance", 0.1),
        error_metric=rom_eval_dict.get("error_metric", "rmse"),
        generate_videos=rom_eval_dict.get("generate_videos", False),
        generate_plots=rom_eval_dict.get("generate_plots", False),
        rom_root=Path(rom_eval_dict.get("rom_root", "rom_mvar")),
    )
    
    print("=" * 70)
    print("ROM/MVAR EVALUATION PIPELINE")
    print("=" * 70)
    print(f"Experiment: {eval_cfg.experiment_name}")
    print(f"Test ICs: {eval_cfg.num_test_ics}")
    print(f"Generate videos: {eval_cfg.generate_videos}")
    print(f"Generate plots: {eval_cfg.generate_plots}")
    print("=" * 70)
    
    # Setup directories
    dirs = setup_rom_directories(eval_cfg.experiment_name, eval_cfg.rom_root)
    
    # Check that model exists
    model_dir = dirs["model"]
    if not (model_dir / "pod_basis.npz").exists():
        print(f"\nERROR: No trained model found at {model_dir}")
        print("Run rom_mvar_train.py first to train the model.")
        sys.exit(1)
    
    print(f"\n✓ Found trained model at {model_dir}\n")
    
    # ========================================================================
    # Step 1: Load trained POD + MVAR model
    # ========================================================================
    print("STEP 1: Loading trained model")
    print("-" * 70)
    
    pod_model = load_pod_model(model_dir)
    mvar_model = load_mvar_model(model_dir)
    
    print(f"\n✓ POD model loaded")
    print(f"  Latent dimension: {pod_model['latent_dim']}")
    print(f"  Grid: {pod_model['grid_info']['nx']}x{pod_model['grid_info']['ny']}")
    
    print(f"\n✓ MVAR model loaded")
    print(f"  Order: {mvar_model['order']}")
    print(f"  Latent dimension: {mvar_model['latent_dim']}")
    
    # Extract grid info
    nx = pod_model["grid_info"]["nx"]
    ny = pod_model["grid_info"]["ny"]
    Lx = pod_model["grid_info"]["Lx"]
    Ly = pod_model["grid_info"]["Ly"]
    dx = Lx / nx
    dy = Ly / ny
    
    # ========================================================================
    # Step 2: Evaluate on each test IC
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Evaluating on unseen test ICs")
    print("=" * 70)
    
    all_test_metrics = []
    
    for ic_idx, seed in enumerate(eval_cfg.test_seeds):
        print(f"\n[{ic_idx+1}/{eval_cfg.num_test_ics}] Test IC {ic_idx} (seed={seed})")
        print("-" * 70)
        
        # Create IC directory
        ic_dir = setup_test_ic_directory(dirs["test_ics"], ic_idx)
        
        # ----------------------------------------------------------------
        # 2a. Run ground truth simulation
        # ----------------------------------------------------------------
        print("\n  Running ground truth simulation...")
        test_cfg = deepcopy(cfg)
        test_cfg["seed"] = seed
        # Disable videos during simulation
        test_cfg["outputs"]["animate_traj"] = False
        test_cfg["outputs"]["animate_density"] = False
        test_cfg["outputs"]["plot_order_params"] = False
        
        sim_result = _run_single(test_cfg, ic_id=None, enable_videos=False, enable_order_plots=False)
        
        # Extract true density movie
        results = sim_result["results"]
        traj = results["traj"]
        times = results["times"]
        T = traj.shape[0]
        bc = results["sim"]["bc"]
        
        density_true_movie = np.zeros((T, nx, ny))
        for t in range(T):
            rho, _, _ = compute_density_grid(
                traj[t], nx, ny, Lx, Ly, bandwidth=0.5, bc=bc
            )
            density_true_movie[t] = rho
        
        print(f"    ✓ True density: {density_true_movie.shape}")
        
        # Save true density
        np.savez(
            ic_dir / "true_density.npz",
            density=density_true_movie,
            times=times,
            nx=nx,
            ny=ny,
            Lx=Lx,
            Ly=Ly,
        )
        
        # ----------------------------------------------------------------
        # 2b. MVAR forecast
        # ----------------------------------------------------------------
        print("\n  Forecasting with MVAR...")
        
        # Flatten true density for projection
        density_true_flat = density_true_movie.reshape(T, -1)
        
        # Project initial segment to latent space
        order = mvar_model["order"]
        z_init = project_to_pod(
            density_true_flat[:order],
            pod_model["mean_mode"],
            pod_model["pod_modes"],
        )
        
        # Forecast horizon
        horizon = eval_cfg.forecast_horizon if eval_cfg.forecast_horizon else (T - order)
        
        # Forecast latent trajectory
        z_forecast = forecast_mvar(
            z_init,
            mvar_model["A0"],
            mvar_model["A_coeffs"],
            horizon,
        )
        
        # Reconstruct density
        density_pred_flat = reconstruct_from_pod(
            z_forecast,
            pod_model["mean_mode"],
            pod_model["pod_modes"],
        )
        density_pred_movie = density_pred_flat.reshape(horizon, nx, ny)
        
        # Pad with initial states for alignment
        full_pred = np.vstack([
            density_true_flat[:order].reshape(order, nx, ny),
            density_pred_movie,
        ])
        
        print(f"    ✓ Predicted density: {full_pred.shape}")
        
        # Save predicted density
        np.savez(
            ic_dir / "pred_density.npz",
            density=full_pred,
            times=times[:full_pred.shape[0]],
            nx=nx,
            ny=ny,
            Lx=Lx,
            Ly=Ly,
        )
        
        # ----------------------------------------------------------------
        # 2c. Compute metrics
        # ----------------------------------------------------------------
        print("\n  Computing metrics...")
        
        # Trim to forecast length for fair comparison
        T_eval = min(full_pred.shape[0], density_true_movie.shape[0])
        density_true_eval = density_true_movie[:T_eval]
        density_pred_eval = full_pred[:T_eval]
        
        # Timeseries metrics
        metrics_df = compute_timeseries_metrics(
            density_true_eval,
            density_pred_eval,
            dx,
            dy,
        )
        metrics_df.to_csv(ic_dir / "metrics_timeseries.csv", index=False)
        
        # Summary metrics
        summary = compute_summary_metrics(
            metrics_df,
            eval_cfg.error_tolerance,
            eval_cfg.error_metric,
        )
        summary["ic_id"] = ic_idx
        summary["seed"] = seed
        summary["forecast_horizon"] = horizon
        
        with open(ic_dir / "metrics_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        all_test_metrics.append(summary)
        
        print(f"    ✓ R² (mean): {summary['r2_mean']:.4f}")
        print(f"    ✓ RMSE (mean): {summary['rmse_mean']:.4e}")
        print(f"    ✓ Tolerance horizon τ: {summary['tolerance_horizon']} steps")
        
        # ----------------------------------------------------------------
        # 2d. Optional: Generate plots (if enabled)
        # ----------------------------------------------------------------
        if eval_cfg.generate_plots:
            print("\n  Generating plots...")
            try:
                import matplotlib.pyplot as plt
                
                plots_dir = ic_dir / "plots"
                plots_dir.mkdir(exist_ok=True)
                
                # Error over time
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                axes[0, 0].plot(metrics_df["t"], metrics_df["r2"])
                axes[0, 0].set_ylabel("R²")
                axes[0, 0].set_title("Coefficient of Determination")
                axes[0, 0].grid(True, alpha=0.3)
                
                axes[0, 1].plot(metrics_df["t"], metrics_df["rmse"])
                axes[0, 1].axhline(eval_cfg.error_tolerance, color='r', linestyle='--', label='Tolerance')
                axes[0, 1].axvline(summary["tolerance_horizon"], color='g', linestyle=':', label=f'τ={summary["tolerance_horizon"]}')
                axes[0, 1].set_ylabel("RMSE")
                axes[0, 1].set_title("Root Mean Squared Error")
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                
                axes[1, 0].plot(metrics_df["t"], metrics_df["e2"])
                axes[1, 0].set_ylabel("Relative L² Error")
                axes[1, 0].set_xlabel("Time step")
                axes[1, 0].grid(True, alpha=0.3)
                
                axes[1, 1].plot(metrics_df["t"], metrics_df["mass_error"])
                axes[1, 1].set_ylabel("Absolute Mass Error")
                axes[1, 1].set_xlabel("Time step")
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(plots_dir / "error_timeseries.png", dpi=150)
                plt.close()
                
                print(f"    ✓ Saved plots to {plots_dir}")
            except Exception as e:
                print(f"    ✗ Plot generation failed: {e}")
    
    # ========================================================================
    # Step 3: Save aggregate metrics
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Computing aggregate metrics")
    print("-" * 70)
    
    # Save all test metrics
    all_metrics_df = pd.DataFrame(all_test_metrics)
    all_metrics_df.to_csv(dirs["aggregate_metrics"] / "all_test_metrics.csv", index=False)
    
    # Compute aggregate statistics
    aggregate = {
        "num_test_ics": eval_cfg.num_test_ics,
        "test_seeds": eval_cfg.test_seeds,
        "error_tolerance": eval_cfg.error_tolerance,
        "error_metric": eval_cfg.error_metric,
        "r2_mean": {
            "mean": float(all_metrics_df["r2_mean"].mean()),
            "std": float(all_metrics_df["r2_mean"].std()),
            "min": float(all_metrics_df["r2_mean"].min()),
            "max": float(all_metrics_df["r2_mean"].max()),
        },
        "rmse_mean": {
            "mean": float(all_metrics_df["rmse_mean"].mean()),
            "std": float(all_metrics_df["rmse_mean"].std()),
            "min": float(all_metrics_df["rmse_mean"].min()),
            "max": float(all_metrics_df["rmse_mean"].max()),
        },
        "tolerance_horizon": {
            "mean": float(all_metrics_df["tolerance_horizon"].mean()),
            "std": float(all_metrics_df["tolerance_horizon"].std()),
            "min": int(all_metrics_df["tolerance_horizon"].min()),
            "max": int(all_metrics_df["tolerance_horizon"].max()),
        },
    }
    
    with open(dirs["aggregate_metrics"] / "aggregate_summary.json", "w") as f:
        json.dump(aggregate, f, indent=2)
    
    print(f"\n✓ Aggregate metrics:")
    print(f"  R² (mean ± std): {aggregate['r2_mean']['mean']:.4f} ± {aggregate['r2_mean']['std']:.4f}")
    print(f"  RMSE (mean ± std): {aggregate['rmse_mean']['mean']:.4e} ± {aggregate['rmse_mean']['std']:.4e}")
    print(f"  τ (mean ± std): {aggregate['tolerance_horizon']['mean']:.1f} ± {aggregate['tolerance_horizon']['std']:.1f} steps")
    
    # ========================================================================
    # Done
    # ========================================================================
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {dirs['base']}")
    print("\nNext steps:")
    if not eval_cfg.generate_videos:
        print(f"  1. Generate videos (locally after rsync):")
        print(f"     python scripts/rom_mvar_visualize.py --experiment {eval_cfg.experiment_name}")
    print()


if __name__ == "__main__":
    main()
