#!/usr/bin/env python3
"""Generate best-run plots for ROM/MVAR evaluation.

This script:
1. Loads evaluation results (metrics + predictions)
2. Selects best simulation per IC type (max R²)
3. Generates error vs time plots
4. Generates order parameter plots

Usage:
    python scripts/rom_mvar_best_plots.py \\
        --eval_dir results/eval_unseen \\
        --unseen_root simulations_unseen \\
        --out_dir results/eval_unseen/best_plots

Author: Maria
Date: November 2025
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rectsim.rom_eval_metrics import SimulationMetrics
from rectsim.rom_eval_data import load_unseen_simulations
from rectsim.rom_eval_viz import (
    select_best_runs,
    make_best_run_error_plots,
    make_best_run_order_param_plots,
)


def load_predictions_from_npz(pred_dir: Path) -> dict:
    """Load prediction NPZ files organized by IC type.
    
    Parameters
    ----------
    pred_dir : Path
        Directory containing ic_type/sim_name.npz files.
        
    Returns
    -------
    predictions : dict
        Dictionary keyed by (ic_type, name) with prediction data.
    """
    predictions = {}
    
    if not pred_dir.exists():
        return predictions
    
    for ic_dir in pred_dir.iterdir():
        if not ic_dir.is_dir():
            continue
        
        ic_type = ic_dir.name
        
        for npz_file in ic_dir.glob("*.npz"):
            name = npz_file.stem  # sim_000, etc.
            
            data = np.load(npz_file)
            
            # Extract data
            predictions[(ic_type, name)] = {
                "density_pred": data["density_pred"],
                "density_true": data["density_true"],
                "latent_pred": data["latent_pred"],
                "latent_true": data["latent_true"],
                "times": data["times"],
                "T0": int(data["T0"]),
                "errors": {
                    "e1": data["e1"],
                    "e2": data["e2"],
                    "einf": data["einf"],
                    "rel_e2": data["rel_e2"],
                    "mass_error": data["mass_error"],
                }
            }
    
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Generate best-run plots for ROM/MVAR evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--eval_dir",
        type=Path,
        required=True,
        help="Evaluation directory with metrics_per_sim.json and predictions/",
    )
    
    parser.add_argument(
        "--unseen_root",
        type=Path,
        required=True,
        help="Root directory with test simulations (needed for trajectories)",
    )
    
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory (default: eval_dir/best_plots)",
    )
    
    parser.add_argument(
        "--metric",
        type=str,
        default="r2",
        help="Metric to select best run (default: r2)",
    )
    
    parser.add_argument(
        "--maximize",
        action="store_true",
        default=True,
        help="Maximize metric (default: True for R²)",
    )
    
    parser.add_argument(
        "--no-maximize",
        dest="maximize",
        action="store_false",
        help="Minimize metric (use for RMSE, error metrics)",
    )
    
    args = parser.parse_args()
    
    if args.out_dir is None:
        args.out_dir = args.eval_dir / "best_plots"
    
    print("=" * 70)
    print("ROM/MVAR Best Run Visualization")
    print("=" * 70)
    print(f"Evaluation dir: {args.eval_dir}")
    print(f"Simulations:    {args.unseen_root}")
    print(f"Output dir:     {args.out_dir}")
    print(f"Selection:      {'max' if args.maximize else 'min'} {args.metric}")
    print("=" * 70)
    print()
    
    # Load metrics
    metrics_file = args.eval_dir / "metrics_per_sim.json"
    print(f"Loading metrics from {metrics_file}...")
    
    if not metrics_file.exists():
        print(f"Error: {metrics_file} not found")
        print("Run rom_mvar_eval_unseen.py first with --save_predictions")
        return 1
    
    with open(metrics_file) as f:
        metrics_data = json.load(f)
    
    metrics_list = [SimulationMetrics(**m) for m in metrics_data]
    print(f"  Loaded {len(metrics_list)} simulation metrics")
    print()
    
    # Select best runs
    print(f"Selecting best runs per IC type (by {args.metric})...")
    best_runs = select_best_runs(
        metrics_list,
        key=args.metric,
        maximize=args.maximize,
    )
    
    print(f"  Selected {len(best_runs)} best runs:")
    for ic_type, metrics in best_runs.items():
        print(f"    {ic_type}: {metrics.name} ({args.metric}={getattr(metrics, args.metric):.4f})")
    print()
    
    # Load predictions
    pred_dir = args.eval_dir / "predictions"
    print(f"Loading predictions from {pred_dir}...")
    
    if not pred_dir.exists():
        print(f"Error: {pred_dir} not found")
        print("Run rom_mvar_eval_unseen.py with --save_predictions")
        return 1
    
    predictions_dict = load_predictions_from_npz(pred_dir)
    print(f"  Loaded {len(predictions_dict)} prediction sets")
    print()
    
    # Generate error plots
    print("Generating error plots for best runs...")
    make_best_run_error_plots(
        best_runs,
        predictions_dict,
        args.out_dir,
    )
    print()
    
    # Load simulation samples for order parameters
    print(f"Loading test simulations from {args.unseen_root}...")
    samples = load_unseen_simulations(
        args.unseen_root,
        require_density=True,
        require_traj=False,  # Try to load, but don't fail if missing
    )
    
    # Create lookup dict
    samples_dict = {(s.ic_type, s.name): s for s in samples}
    print(f"  Loaded {len(samples)} samples")
    print()
    
    # Generate order parameter plots
    print("Generating order parameter plots for best runs...")
    make_best_run_order_param_plots(
        best_runs,
        samples_dict,
        args.out_dir,
        predictions_dict=predictions_dict,
    )
    print()
    
    print("=" * 70)
    print("Visualization complete!")
    print("=" * 70)
    print(f"Plots saved to: {args.out_dir}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
