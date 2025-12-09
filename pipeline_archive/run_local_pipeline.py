#!/usr/bin/env python3
"""
Local Pipeline - Prediction & Visualization

This script uses Oscar-generated data and models to:
1. Load POD + MVAR models
2. Load test simulation densities  
3. Run MVAR predictions in latent space
4. Reconstruct density predictions
5. Compute comprehensive metrics
6. Generate all visualizations (videos, plots, best runs)

NO simulation or training is performed locally - all heavy computation is done on Oscar.

Usage:
    python run_local_pipeline.py --oscar_dir oscar_outputs

Requirements:
    oscar_outputs/
    ├── training/          # Training sims (for reference only)
    ├── test/              # Test sims (truth data + order params)
    └── models/            # POD + MVAR models
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rectsim.legacy_functions import (
    trajectory_video,
    side_by_side_video,
    compute_frame_metrics,
    compute_summary_metrics,
    plot_errors_timeseries,
)


def load_models(model_dir: Path):
    """Load POD and MVAR models from Oscar outputs."""
    print(f"\n📂 Loading models from: {model_dir}")
    
    # Load POD model
    pod_path = model_dir / "pod_model.npz"
    pod_data = np.load(pod_path)
    pod_model = {
        "U": pod_data["U"],
        "S": pod_data["S"],
        "mean": pod_data["mean"],
        "energy": float(pod_data["energy"]),
        "R_POD": int(pod_data["R_POD"]),
    }
    print(f"✓ POD model: {pod_model['R_POD']} modes, {pod_model['energy']*100:.2f}% energy")
    
    # Load MVAR model
    mvar_path = model_dir / "mvar_model.npz"
    mvar_data = np.load(mvar_path)
    mvar_model = {
        "A_matrices": [mvar_data["A_matrices"][i] for i in range(len(mvar_data["A_matrices"]))],
        "p": int(mvar_data["p"]),
        "train_r2": float(mvar_data["train_r2"]),
    }
    print(f"✓ MVAR model: lag={mvar_model['p']}, train R²={mvar_model['train_r2']:.4f}")
    
    # Load metadata
    metadata_path = model_dir / "training_metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    return pod_model, mvar_model, metadata


def load_test_simulations(test_dir: Path):
    """Load all test simulation densities and metadata."""
    print(f"\n📂 Loading test simulations from: {test_dir}")
    
    test_data = []
    test_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir() and d.name.startswith("test_")])
    
    for run_dir in tqdm(test_dirs, desc="Loading test data"):
        # Load density
        density_path = run_dir / "density.npz"
        density_data = np.load(density_path)
        
        # Load trajectory
        traj_path = run_dir / "traj.npz"
        traj_data = np.load(traj_path, allow_pickle=True)
        
        # Load order parameters
        order_path = run_dir / "order_params.npz"
        order_data = np.load(order_path) if order_path.exists() else None
        
        # Load metadata
        metadata_path = run_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        test_data.append({
            "run_name": run_dir.name,
            "run_dir": run_dir,
            "rho_true": density_data["rho"],
            "times": density_data["times"],
            "x_edges": density_data["x_edges"],
            "y_edges": density_data["y_edges"],
            "extent": density_data["extent"],
            "trajectory": traj_data["positions"],
            "order_params": order_data if order_data else None,
            "metadata": metadata,
        })
    
    print(f"✓ Loaded {len(test_data)} test simulations")
    
    return test_data


def mvar_forecast(y_init, A_matrices, T_forecast):
    """Forecast latent state using MVAR model."""
    p = len(A_matrices)
    r = A_matrices[0].shape[0]
    
    # Initialize with history
    y_history = list(y_init)  # Last p steps
    y_forecast = []
    
    for t in range(T_forecast):
        # Predict next step: y[t] = sum(Ai * y[t-i])
        y_next = np.zeros(r)
        for i, A in enumerate(A_matrices):
            y_next += A @ y_history[-(i+1)]
        
        y_forecast.append(y_next)
        y_history.append(y_next)
    
    return np.array(y_forecast)


def predict_on_test_set(test_data, pod_model, mvar_model):
    """Run MVAR-ROM predictions on all test simulations."""
    print(f"\n🔮 Running predictions on {len(test_data)} test simulations...")
    
    predictions = []
    
    for data in tqdm(test_data, desc="Predictions"):
        rho_true = data["rho_true"]  # (T, ny, nx)
        T, ny, nx = rho_true.shape
        
        # Flatten density
        rho_true_flat = rho_true.reshape(T, ny * nx)
        
        # Project initial conditions to latent space
        p = mvar_model["p"]
        y_init = []
        for t in range(p):
            rho_centered = rho_true_flat[t] - pod_model["mean"]
            y_t = rho_centered @ pod_model["U"]
            y_init.append(y_t)
        
        # Forecast in latent space
        y_forecast = mvar_forecast(y_init, mvar_model["A_matrices"], T - p)
        
        # Reconstruct density
        rho_pred_flat = y_forecast @ pod_model["U"].T + pod_model["mean"]
        rho_pred = rho_pred_flat.reshape(T - p, ny, nx)
        
        # Pad initial conditions
        rho_pred_full = np.vstack([rho_true[:p], rho_pred])
        
        predictions.append({
            "run_name": data["run_name"],
            "rho_true": rho_true,
            "rho_pred": rho_pred_full,
            "trajectory": data["trajectory"],
            "times": data["times"],
            "order_params": data["order_params"],
            "metadata": data["metadata"],
            "extent": data["extent"],
        })
    
    print(f"✓ Predictions complete")
    
    return predictions


def compute_metrics(predictions, pod_model):
    """Compute comprehensive metrics for all predictions."""
    print(f"\n📊 Computing metrics...")
    
    all_metrics = []
    
    for pred in tqdm(predictions, desc="Computing metrics"):
        # Flatten densities
        T, ny, nx = pred["rho_true"].shape
        rho_true_flat = pred["rho_true"].reshape(T, ny * nx)
        rho_pred_flat = pred["rho_pred"].reshape(T, ny * nx)
        
        # Frame-wise metrics
        frame_metrics = compute_frame_metrics(rho_true_flat, rho_pred_flat)
        
        # Summary metrics
        summary = compute_summary_metrics(
            rho_true_flat,
            rho_pred_flat,
            pod_model["mean"],
            frame_metrics
        )
        summary["run_name"] = pred["run_name"]
        summary["ic_type"] = pred["metadata"]["ic_type"]
        
        # Store frame metrics
        pred["frame_metrics"] = frame_metrics
        
        all_metrics.append(summary)
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    print(f"✓ Metrics computed")
    print(f"   Mean R²: {metrics_df['r2'].mean():.4f} ± {metrics_df['r2'].std():.4f}")
    print(f"   Median L² error: {metrics_df['median_e2'].mean():.4f} ± {metrics_df['median_e2'].std():.4f}")
    
    return metrics_df


def generate_visualizations(predictions, metrics_df, output_dir: Path, config):
    """Generate all visualizations (videos, plots, best runs)."""
    print(f"\n🎬 Generating visualizations...")
    
    # Create output directories
    best_runs_dir = output_dir / "best_runs"
    plots_dir = output_dir / "plots"
    best_runs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Get IC types
    ic_types = metrics_df["ic_type"].unique()
    
    # Generate best run visualizations for each IC type
    for ic_type in tqdm(ic_types, desc="IC types"):
        ic_data = metrics_df[metrics_df["ic_type"] == ic_type]
        best_idx = ic_data["r2"].idxmax()
        best_run_name = ic_data.loc[best_idx, "run_name"]
        
        # Find prediction data
        best_pred = next(p for p in predictions if p["run_name"] == best_run_name)
        
        # Create IC-specific output directory
        ic_output_dir = best_runs_dir / ic_type
        ic_output_dir.mkdir(exist_ok=True)
        
        # Generate trajectory video
        trajectory_video(
            path=ic_output_dir,
            traj=best_pred["trajectory"],
            times=best_pred["times"],
            Lx=config["sim"]["Lx"],
            Ly=config["sim"]["Ly"],
            name="traj_truth",
            fps=20,
            title=f"Truth Trajectory: {ic_type.replace('_', ' ').title()}"
        )
        
        # Generate side-by-side density video
        side_by_side_video(
            path=ic_output_dir,
            left_frames=best_pred["rho_true"],
            right_frames=best_pred["rho_pred"],
            name="density_truth_vs_pred",
            fps=20,
            titles=("Ground Truth", "MVAR-ROM Prediction")
        )
        
        # Generate error plots
        error_plot_path = ic_output_dir / "error_time.png"
        plot_errors_timeseries(
            best_pred["frame_metrics"],
            metrics_df.loc[best_idx].to_dict(),
            save_path=error_plot_path,
            title=f"Error Metrics: {ic_type.replace('_', ' ').title()}"
        )
        
        # Generate error histogram
        error_hist_path = ic_output_dir / "error_hist.png"
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(best_pred["frame_metrics"]["e2"], bins=30, alpha=0.7, edgecolor="black")
        ax.set_xlabel("Relative L² Error")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Error Distribution: {ic_type.replace('_', ' ').title()}")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(error_hist_path, dpi=150)
        plt.close()
        
        # Generate order parameter plots
        if best_pred["order_params"] is not None:
            order_plot_path = ic_output_dir / "order_parameters.png"
            order_data = best_pred["order_params"]
            
            fig, axes = plt.subplots(3, 1, figsize=(10, 10))
            
            times = order_data["times"]
            axes[0].plot(times, order_data["phi"], "b-", linewidth=2)
            axes[0].set_ylabel("Polarization φ", fontsize=12)
            axes[0].grid(alpha=0.3)
            axes[0].set_title(f"Order Parameters: {ic_type.replace('_', ' ').title()}", fontsize=14)
            
            axes[1].plot(times, order_data["mean_speed"], "g-", linewidth=2)
            axes[1].set_ylabel("Mean Speed", fontsize=12)
            axes[1].grid(alpha=0.3)
            
            axes[2].plot(times, order_data["speed_std"], "r-", linewidth=2)
            axes[2].set_ylabel("Speed Std Dev", fontsize=12)
            axes[2].set_xlabel("Time (s)", fontsize=12)
            axes[2].grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(order_plot_path, dpi=150)
            plt.close()
    
    # Generate summary plots
    # (Add your existing summary plot generation code here)
    
    print(f"✓ Visualizations generated")
    print(f"   Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Local pipeline: Prediction & visualization using Oscar outputs"
    )
    parser.add_argument(
        "--oscar_dir",
        type=Path,
        required=True,
        help="Directory with Oscar outputs (training/, test/, models/)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/local_pipeline"),
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("LOCAL PIPELINE - PREDICTION & VISUALIZATION")
    print("="*80)
    
    # Load models
    model_dir = args.oscar_dir / "models"
    pod_model, mvar_model, metadata = load_models(model_dir)
    
    # Load test simulations
    test_dir = args.oscar_dir / "test"
    test_data = load_test_simulations(test_dir)
    
    # Run predictions
    predictions = predict_on_test_set(test_data, pod_model, mvar_model)
    
    # Compute metrics
    metrics_df = compute_metrics(predictions, pod_model)
    
    # Save metrics
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics_all_runs.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n💾 Saved metrics: {metrics_path}")
    
    # Generate visualizations
    generate_visualizations(predictions, metrics_df, args.output_dir, metadata["config"])
    
    print("\n" + "="*80)
    print("✅ LOCAL PIPELINE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
