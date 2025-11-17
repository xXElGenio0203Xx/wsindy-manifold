#!/usr/bin/env python3
"""Evaluate MVAR forecasts on test runs.

This script implements Stage 4 of the ROM pipeline:
1. Load test run latent trajectories
2. Generate MVAR forecasts from initial conditions
3. Reconstruct density fields in physical space
4. Compute comprehensive metrics and visualizations

Usage
-----
python scripts/rom_evaluate.py \
    --experiment_name my_experiment \
    --no_videos  # Optional: skip video generation

Oscar workflow:
1. Generate ensemble: rectsim ensemble --config CONFIG
2. Build POD: python scripts/rom_build_pod.py --experiment_name EXP
3. Train MVAR: python scripts/rom_train_mvar.py --experiment_name EXP
4. Evaluate: python scripts/rom_evaluate.py --experiment_name EXP

Output structure:
    rom/<experiment_name>/mvar/forecast/
    ├── forecast_run_<id>.npz       # Forecast data
    ├── metrics_run_<id>.json       # Summary metrics
    ├── order_params_run_<id>.csv   # Order parameters
    ├── errors_time_run_<id>.png    # Error dashboard
    ├── order_params_run_<id>.png   # Order parameter plot
    ├── snapshot_grid_run_<id>.png  # Snapshot comparisons
    ├── density_true_run_<id>.mp4   # True density video
    ├── density_pred_run_<id>.mp4   # Predicted density video
    └── density_comparison_run_<id>.mp4  # Side-by-side video

Author: Maria
Date: November 2025
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rectsim.mvar import MVARModel, mvar_forecast, reconstruct_from_pod
from rectsim.rom_eval import (
    ROMConfig,
    check_mass_conservation,
    compare_order_parameters,
    compute_mass,
    compute_pointwise_errors,
    compute_r2_score,
    compute_summary_metrics,
    count_nans,
    create_comparison_video,
    create_density_video,
    get_forecast_split_indices,
    plot_errors_dashboard,
    plot_order_parameters,
    plot_snapshot_grid,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate MVAR forecasts on test runs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Unique experiment identifier",
    )
    parser.add_argument(
        "--rom_root",
        type=str,
        default="rom",
        help="Root directory for ROM outputs",
    )
    parser.add_argument(
        "--sim_root",
        type=str,
        required=True,
        help="Root directory containing original simulation runs",
    )
    parser.add_argument(
        "--no_videos",
        action="store_true",
        help="Skip video generation (faster)",
    )
    parser.add_argument(
        "--snapshot_times",
        type=int,
        nargs="+",
        default=None,
        help="Specific time indices for snapshots (default: start, middle, end)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    rom_root = Path(args.rom_root)
    sim_root = Path(args.sim_root)
    exp_dir = rom_root / args.experiment_name

    print("=" * 80)
    print("ROM Pipeline - Stage 4: Evaluate Forecasts")
    print("=" * 80)
    print(f"Experiment:   {args.experiment_name}")
    print(f"ROM root:     {rom_root}")
    print(f"Sim root:     {sim_root}")
    print(f"Generate videos: {not args.no_videos}")
    print()

    # Load configuration
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        print(f"ERROR: Configuration not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = ROMConfig.from_dict(json.load(f))

    if not config.test_runs:
        print("ERROR: No test runs specified in configuration")
        print("Please specify --test_runs when running rom_build_pod.py")
        sys.exit(1)

    print(f"Test runs: {config.test_runs}")
    print()

    # ========================================================================
    # Load POD basis
    # ========================================================================
    print("[1/5] Loading POD basis...")

    basis_path = exp_dir / "pod" / "basis.npz"
    if not basis_path.exists():
        print(f"ERROR: POD basis not found: {basis_path}")
        sys.exit(1)

    basis_data = np.load(basis_path)
    Phi = basis_data["Phi"]
    global_mean = basis_data["mean"]
    ny = int(basis_data["ny"])
    nx = int(basis_data["nx"])
    r = int(basis_data["r"])

    print(f"POD modes: {r}")
    print(f"Grid size: ({ny}, {nx})")
    print()

    # ========================================================================
    # Load MVAR model
    # ========================================================================
    print("[2/5] Loading MVAR model...")

    mvar_path = exp_dir / "mvar" / "mvar_model.npz"
    if not mvar_path.exists():
        print(f"ERROR: MVAR model not found: {mvar_path}")
        print("Run rom_train_mvar.py first!")
        sys.exit(1)

    model = MVARModel.load(mvar_path)

    print(f"MVAR order: {model.order}")
    print(f"Latent dim: {model.latent_dim}")
    print()

    # Create forecast directory
    forecast_dir = exp_dir / "mvar" / "forecast"
    forecast_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Process each test run
    # ========================================================================
    print(f"[3/5] Processing {len(config.test_runs)} test runs...")
    print()

    all_metrics = []

    for test_idx in config.test_runs:
        print(f"  Processing test run {test_idx}...")

        # --------------------------------------------------------------------
        # Load latent trajectory
        # --------------------------------------------------------------------
        latent_file = exp_dir / "latent" / f"run_{test_idx:04d}_latent.npz"
        if not latent_file.exists():
            print(f"    WARNING: Latent file not found: {latent_file}")
            continue

        latent_data = np.load(latent_file)
        Y_full = latent_data["Y"]
        times_full = latent_data["times"]
        T = Y_full.shape[0]

        # Split into train/forecast regions
        T_train, T_forecast = get_forecast_split_indices(T, config.train_frac)

        if T_forecast < model.order:
            print(f"    WARNING: Not enough forecast steps ({T_forecast} < {model.order})")
            continue

        # --------------------------------------------------------------------
        # Load true density
        # --------------------------------------------------------------------
        # Find density file for this run
        density_pattern = f"run_{test_idx:04d}/density.npz"
        density_files = list(sim_root.rglob(density_pattern))

        if not density_files:
            print(f"    WARNING: Density file not found for run {test_idx}")
            continue

        density_data = np.load(density_files[0])
        rho_true_full = density_data["rho"]

        # Ensure consistent time dimension
        if rho_true_full.shape[0] != T:
            print(f"    WARNING: Time dimension mismatch: density={rho_true_full.shape[0]}, latent={T}")
            continue

        # --------------------------------------------------------------------
        # Generate MVAR forecast
        # --------------------------------------------------------------------
        # Initial conditions: last 'order' states before forecast window
        Y_init = Y_full[T_train - model.order : T_train]

        # Forecast
        Y_pred = mvar_forecast(model, Y_init, steps=T_forecast)

        # Ground truth latent
        Y_true = Y_full[T_train:]

        # --------------------------------------------------------------------
        # Reconstruct in physical space
        # --------------------------------------------------------------------
        rho_pred = reconstruct_from_pod(Y_pred, Phi, global_mean, ny, nx)
        rho_true = rho_true_full[T_train:]

        times = times_full[T_train:]

        # --------------------------------------------------------------------
        # Compute metrics
        # --------------------------------------------------------------------
        # Pointwise errors
        errors = compute_pointwise_errors(rho_true, rho_pred)

        # Summary metrics
        summary = compute_summary_metrics(errors)

        # R² score
        r2 = compute_r2_score(rho_true, rho_pred)

        # Mass conservation
        mass_true = compute_mass(rho_true)
        mass_pred = compute_mass(rho_pred)
        mass_check = check_mass_conservation(mass_true, mass_pred)

        # NaN count
        nan_count = count_nans({
            "rho_true": rho_true,
            "rho_pred": rho_pred,
            "Y_pred": Y_pred,
        })

        # Order parameters
        order_df = compare_order_parameters(rho_true, rho_pred)

        # --------------------------------------------------------------------
        # Save forecast data
        # --------------------------------------------------------------------
        forecast_file = forecast_dir / f"forecast_run_{test_idx:04d}.npz"
        np.savez(
            forecast_file,
            density_true=rho_true,
            density_pred=rho_pred,
            Y_true=Y_true,
            Y_pred=Y_pred,
            times=times,
            errors_e1=errors["e1"],
            errors_e2=errors["e2"],
            errors_einf=errors["e_inf"],
            rmse=errors["rmse"],
            rmse_normalized=errors["rmse_normalized"],
            mass_true=mass_true,
            mass_pred=mass_pred,
            mass_error=mass_check["mass_error"],
        )

        # Save metrics
        metrics = {
            "run_id": int(test_idx),
            "T": int(T),
            "T_train": int(T_train),
            "T_forecast": int(T_forecast),
            "latent_dim": int(r),
            "compression_ratio": float((ny * nx) / r),
            "mvar_order": int(model.order),
            "ridge": float(model.ridge),
            "train_frac": float(config.train_frac),
            "r2": float(r2),
            **summary,
            "mass_drift_max": float(mass_check["mass_drift_max"]),
            "mass_conservation_ok": bool(mass_check["mass_conservation_ok"]),
            "nan_count": int(nan_count),
        }

        metrics_file = forecast_dir / f"metrics_run_{test_idx:04d}.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        all_metrics.append(metrics)

        # Save order parameters
        order_file = forecast_dir / f"order_params_run_{test_idx:04d}.csv"
        order_df.to_csv(order_file, index=False)

        # --------------------------------------------------------------------
        # Generate plots
        # --------------------------------------------------------------------
        # Error dashboard
        plot_errors_dashboard(
            times,
            errors,
            mass_check["mass_error"],
            forecast_dir / f"errors_time_run_{test_idx:04d}.png",
        )

        # Order parameters
        plot_order_parameters(
            order_df,
            forecast_dir / f"order_params_run_{test_idx:04d}.png",
        )

        # Snapshot grid
        snapshot_indices = args.snapshot_times
        if snapshot_indices is None:
            # Default: start, middle, end
            snapshot_indices = [0, T_forecast // 2, T_forecast - 1]

        plot_snapshot_grid(
            rho_true,
            rho_pred,
            times,
            snapshot_indices,
            forecast_dir / f"snapshot_grid_run_{test_idx:04d}.png",
        )

        # --------------------------------------------------------------------
        # Generate videos (optional)
        # --------------------------------------------------------------------
        if not args.no_videos:
            # Common colorbar range
            vmin = min(rho_true.min(), rho_pred.min())
            vmax = max(rho_true.max(), rho_pred.max())

            # True density
            create_density_video(
                rho_true,
                times,
                forecast_dir / f"density_true_run_{test_idx:04d}.mp4",
                title="True Density",
                vmin=vmin,
                vmax=vmax,
            )

            # Predicted density
            create_density_video(
                rho_pred,
                times,
                forecast_dir / f"density_pred_run_{test_idx:04d}.mp4",
                title="Predicted Density",
                vmin=vmin,
                vmax=vmax,
            )

            # Comparison
            create_comparison_video(
                rho_true,
                rho_pred,
                times,
                forecast_dir / f"density_comparison_run_{test_idx:04d}.mp4",
            )

        print(f"    R² = {r2:.4f}, RMSE = {summary['mean_rmse']:.6f}, Mass OK = {mass_check['mass_conservation_ok']}")

    print()

    # ========================================================================
    # Aggregate statistics
    # ========================================================================
    print("[4/5] Computing aggregate statistics...")

    if all_metrics:
        aggregate = {
            "n_test_runs": len(all_metrics),
            "mean_r2": float(np.mean([m["r2"] for m in all_metrics])),
            "median_r2": float(np.median([m["r2"] for m in all_metrics])),
            "std_r2": float(np.std([m["r2"] for m in all_metrics])),
            "mean_rmse": float(np.mean([m["mean_rmse"] for m in all_metrics])),
            "median_rmse": float(np.median([m["mean_rmse"] for m in all_metrics])),
            "mean_mass_drift": float(np.mean([m["mass_drift_max"] for m in all_metrics])),
            "max_mass_drift": float(np.max([m["mass_drift_max"] for m in all_metrics])),
            "all_mass_ok": all(m["mass_conservation_ok"] for m in all_metrics),
            "total_nans": sum(m["nan_count"] for m in all_metrics),
        }

        # Save aggregate metrics
        aggregate_file = forecast_dir / "aggregate_metrics.json"
        with open(aggregate_file, "w") as f:
            json.dump(aggregate, f, indent=2)

        print(f"  Mean R²:        {aggregate['mean_r2']:.4f}")
        print(f"  Median R²:      {aggregate['median_r2']:.4f}")
        print(f"  Mean RMSE:      {aggregate['mean_rmse']:.6f}")
        print(f"  Mass OK:        {aggregate['all_mass_ok']}")
        print()

    # ========================================================================
    # Summary
    # ========================================================================
    print("[5/5] Summary")
    print(f"  Processed {len(all_metrics)} test runs")
    print(f"  All outputs saved to: {forecast_dir}")
    print()

    print("=" * 80)
    print("Evaluation complete!")
    print(f"Results saved to: {forecast_dir}")
    print()
    print("Review outputs:")
    print(f"  - Individual metrics: {forecast_dir}/metrics_run_*.json")
    print(f"  - Aggregate metrics:  {forecast_dir}/aggregate_metrics.json")
    print(f"  - Plots:              {forecast_dir}/*.png")
    if not args.no_videos:
        print(f"  - Videos:             {forecast_dir}/*.mp4")
    print("=" * 80)


if __name__ == "__main__":
    main()
