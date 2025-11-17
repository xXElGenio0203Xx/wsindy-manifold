#!/usr/bin/env python3
"""Train and evaluate global POD + MVAR model on density field ensembles.

This script implements the complete EF-ROM pipeline:
1. Load density movies from simulation runs
2. Compute global POD basis across all data
3. Project density fields to latent space
4. Train MVAR model on latent trajectories with ridge regularization
5. Evaluate multi-step forecasts on held-out test segments
6. Save all outputs and visualizations

Usage
-----
python scripts/run_mvar_global.py \
    --sim_root outputs/simulations \
    --pattern "run_*/density.npz" \
    --out_root MVAR_outputs/global_run1 \
    --order 4 \
    --ridge 1e-6 \
    --train_frac 0.8 \
    --max_runs 20

Oscar HPC workflow:
1. Generate ensemble with: rectsim ensemble --config CONFIG
2. Train MVAR model: python scripts/run_mvar_global.py --sim_root simulations/...
3. Outputs saved to MVAR_outputs/global_run1/{pod, model, metrics, plots}

Author: Maria
Date: 2025
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add src to path for rectsim import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rectsim.mvar import (
    MVARModel,
    build_global_snapshot_matrix,
    compute_pod,
    evaluate_mvar_on_runs,
    fit_mvar_from_runs,
    load_density_movies,
    plot_pod_energy,
    project_to_pod,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate global POD + MVAR model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--sim_root",
        type=str,
        required=True,
        help="Root directory containing simulation run folders",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="run_*/density.npz",
        help="Glob pattern to match density.npz files",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="Output directory for all results",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=4,
        help="MVAR model order (number of lags)",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-6,
        help="Ridge regularization parameter",
    )
    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.8,
        help="Fraction of each run to use for training (0.0 to 1.0)",
    )
    parser.add_argument(
        "--max_runs",
        type=int,
        default=None,
        help="Maximum number of runs to load (for testing)",
    )
    parser.add_argument(
        "--energy_threshold",
        type=float,
        default=0.995,
        help="POD energy threshold for mode selection",
    )

    return parser.parse_args()


def setup_output_dirs(out_root: Path):
    """Create output directory structure."""
    subdirs = ["pod", "model", "metrics", "plots"]
    for subdir in subdirs:
        (out_root / subdir).mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()

    # Convert paths
    sim_root = Path(args.sim_root)
    out_root = Path(args.out_root)

    print("=" * 80)
    print("Global POD + MVAR Training Pipeline")
    print("=" * 80)
    print(f"Simulation root: {sim_root}")
    print(f"Output root:     {out_root}")
    print(f"MVAR order:      {args.order}")
    print(f"Ridge:           {args.ridge}")
    print(f"Train fraction:  {args.train_frac}")
    print(f"Energy threshold: {args.energy_threshold}")
    print()

    # Setup output directories
    setup_output_dirs(out_root)

    # ========================================================================
    # Step 1: Load density movies
    # ========================================================================
    print("[1/6] Loading density movies...")

    # Find all run directories
    density_files = list(sim_root.rglob(args.pattern))

    if not density_files:
        print(f"ERROR: No files matching pattern '{args.pattern}' in {sim_root}")
        sys.exit(1)

    run_dirs = [f.parent for f in density_files]

    # Limit runs if requested
    if args.max_runs is not None:
        run_dirs = run_dirs[: args.max_runs]

    print(f"Found {len(run_dirs)} run directories")

    # Load density data
    density_dict = load_density_movies(run_dirs)

    if not density_dict:
        print("ERROR: No density movies loaded")
        sys.exit(1)

    print(f"Successfully loaded {len(density_dict)} runs")

    # Print grid info
    first_rho = next(iter(density_dict.values()))["rho"]
    T_first, ny, nx = first_rho.shape
    print(f"Grid shape: ({ny}, {nx})")
    print(f"Time steps per run: {[v['rho'].shape[0] for v in list(density_dict.values())[:5]]}")
    print()

    # ========================================================================
    # Step 2: Build global snapshot matrix
    # ========================================================================
    print("[2/6] Building global snapshot matrix...")

    X, run_slices, global_mean_flat = build_global_snapshot_matrix(
        density_dict, subtract_mean=True
    )

    T_total, d = X.shape
    print(f"Snapshot matrix shape: ({T_total}, {d})")
    print(f"Total time steps: {T_total}")
    print()

    # ========================================================================
    # Step 3: Compute global POD basis
    # ========================================================================
    print("[3/6] Computing global POD basis...")

    pod_basis = compute_pod(
        X,
        r=None,
        energy_threshold=args.energy_threshold,
    )

    r = pod_basis["r"]
    energy_captured = pod_basis["energy"][r - 1]

    print(f"Number of POD modes: {r}")
    print(f"Energy captured: {energy_captured:.4f}")
    print()

    # Save POD basis
    np.save(out_root / "pod" / "Phi.npy", pod_basis["Phi"])
    np.save(out_root / "pod" / "S.npy", pod_basis["S"])
    np.save(out_root / "pod" / "mean.npy", global_mean_flat)

    with open(out_root / "pod" / "pod_info.json", "w") as f:
        json.dump(
            {
                "r": int(r),
                "energy_threshold": float(args.energy_threshold),
                "energy_captured": float(energy_captured),
                "ny": int(ny),
                "nx": int(nx),
                "d": int(d),
            },
            f,
            indent=2,
        )

    # Plot POD energy
    plot_pod_energy(
        pod_basis["S"],
        out_root / "plots" / "pod_energy.png",
        r_mark=r,
        energy_threshold=args.energy_threshold,
    )

    print(f"Saved POD basis to {out_root / 'pod'}")
    print()

    # ========================================================================
    # Step 4: Project to POD latent space
    # ========================================================================
    print("[4/6] Projecting density fields to POD latent space...")

    latent_dict = project_to_pod(
        density_dict,
        pod_basis["Phi"],
        global_mean_flat,
    )

    print(f"Projected {len(latent_dict)} runs to latent space")
    print()

    # ========================================================================
    # Step 5: Fit MVAR model
    # ========================================================================
    print("[5/6] Fitting MVAR model...")

    model, train_info = fit_mvar_from_runs(
        latent_dict,
        order=args.order,
        ridge=args.ridge,
        train_frac=args.train_frac,
    )

    print(f"MVAR model fitted:")
    print(f"  Order:          {model.order}")
    print(f"  Latent dim:     {model.latent_dim}")
    print(f"  Training samples: {train_info['total_samples']}")
    print()

    # Save MVAR model
    model.save(out_root / "model" / "mvar_model.npz")

    with open(out_root / "model" / "train_info.json", "w") as f:
        json.dump(train_info, f, indent=2)

    print(f"Saved MVAR model to {out_root / 'model'}")
    print()

    # ========================================================================
    # Step 6: Evaluate on test segments
    # ========================================================================
    print("[6/6] Evaluating MVAR model on held-out test segments...")

    results = evaluate_mvar_on_runs(
        model,
        latent_dict,
        density_dict,
        pod_basis,
        global_mean_flat,
        ny,
        nx,
        train_frac=args.train_frac,
    )

    # Save evaluation results
    with open(out_root / "metrics" / "mvar_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Create RMSE time series CSV
    rmse_data = []
    for run_name, run_results in results["per_run"].items():
        for t, rmse in enumerate(run_results["rmse_time_series"]):
            rmse_data.append(
                {
                    "run_name": run_name,
                    "time_step": t,
                    "rmse": rmse,
                }
            )

    if rmse_data:
        df_rmse = pd.DataFrame(rmse_data)
        df_rmse.to_csv(out_root / "metrics" / "rmse_time_series.csv", index=False)

    # Print summary statistics
    agg = results.get("aggregate", {})
    if agg:
        print("Aggregate evaluation metrics:")
        print(f"  Mean R²:        {agg.get('mean_R2', 0.0):.4f}")
        print(f"  Median R²:      {agg.get('median_R2', 0.0):.4f}")
        print(f"  Std R²:         {agg.get('std_R2', 0.0):.4f}")
        print(f"  Min/Max R²:     {agg.get('min_R2', 0.0):.4f} / {agg.get('max_R2', 0.0):.4f}")
        print(f"  Mean RMSE:      {agg.get('mean_rmse', 0.0):.6f}")
        print()

    # Plot RMSE time series
    if rmse_data:
        fig, ax = plt.subplots(figsize=(10, 6))

        for run_name, run_results in results["per_run"].items():
            rmse_series = run_results["rmse_time_series"]
            ax.plot(rmse_series, alpha=0.6, label=run_name)

        ax.set_xlabel("Forecast time step")
        ax.set_ylabel("RMSE")
        ax.set_title("MVAR Forecast RMSE Over Time")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

        fig.tight_layout()
        fig.savefig(out_root / "plots" / "rmse_time_series.png", dpi=150)
        plt.close(fig)

        print(f"Saved evaluation plots to {out_root / 'plots'}")
        print()

    print("=" * 80)
    print("Pipeline complete!")
    print(f"All outputs saved to: {out_root}")
    print("=" * 80)


if __name__ == "__main__":
    main()
