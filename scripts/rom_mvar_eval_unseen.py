#!/usr/bin/env python3
"""CLI script to evaluate ROM/MVAR on unseen IC simulations.

This script:
1. Loads a trained ROM/MVAR model
2. Loads test simulations organized by IC type
3. Runs forecasts and computes error metrics
4. Writes results to CSV and JSON files

Usage:
    python scripts/rom_mvar_eval_unseen.py \\
        --rom_dir rom_mvar/exp1/model \\
        --unseen_root simulations_unseen \\
        --out_dir results/eval_unseen \\
        --train_frac 0.8 \\
        --tol 0.1

Author: Maria
Date: November 2025
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rectsim.rom_eval_pipeline import (
    evaluate_unseen_rom,
    aggregate_metrics,
    print_aggregated_metrics,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ROM/MVAR on unseen IC simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--rom_dir",
        type=Path,
        required=True,
        help="Directory with ROM model files (pod_basis.npz, mvar_params.npz, train_summary.json)",
    )
    
    parser.add_argument(
        "--unseen_root",
        type=Path,
        required=True,
        help="Root directory with test simulations (organized by IC type subdirectories)",
    )
    
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Output directory for results (metrics CSV/JSON)",
    )
    
    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.8,
        help="Fraction of trajectory for initialization (default: 0.8)",
    )
    
    parser.add_argument(
        "--tol",
        type=float,
        default=0.1,
        help="Tolerance for tau computation (relative L2 error threshold, default: 0.1)",
    )
    
    parser.add_argument(
        "--ic_types",
        type=str,
        default=None,
        help="Comma-separated list of IC types to evaluate (default: auto-detect all)",
    )
    
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save predictions and error timeseries to disk (can be large)",
    )
    
    args = parser.parse_args()
    
    # Parse IC types if provided
    ic_types = None
    if args.ic_types:
        ic_types = [s.strip() for s in args.ic_types.split(",")]
    
    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("ROM/MVAR Evaluation on Unseen IC Simulations")
    print("=" * 70)
    print(f"ROM model:       {args.rom_dir}")
    print(f"Simulations:     {args.unseen_root}")
    print(f"Output dir:      {args.out_dir}")
    print(f"Train fraction:  {args.train_frac}")
    print(f"Tolerance (tau): {args.tol}")
    if ic_types:
        print(f"IC types:        {', '.join(ic_types)}")
    print("=" * 70)
    print()
    
    # Run evaluation
    metrics_list, predictions_dict = evaluate_unseen_rom(
        rom_dir=args.rom_dir,
        unseen_root=args.unseen_root,
        ic_types=ic_types,
        train_frac=args.train_frac,
        tol=args.tol,
        return_predictions=args.save_predictions,
    )
    
    if not metrics_list:
        print("No simulations evaluated successfully. Exiting.")
        return
    
    # Aggregate metrics
    print("Aggregating metrics...")
    aggregated = aggregate_metrics(metrics_list)
    print_aggregated_metrics(aggregated)
    
    # Write per-simulation metrics to CSV
    csv_path = args.out_dir / "metrics_per_sim.csv"
    print(f"Writing per-simulation metrics to {csv_path}...")
    
    with open(csv_path, "w", newline="") as f:
        if metrics_list:
            fieldnames = list(metrics_list[0].to_dict().keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in metrics_list:
                writer.writerow(m.to_dict())
    
    # Write per-simulation metrics to JSON
    json_path = args.out_dir / "metrics_per_sim.json"
    print(f"Writing per-simulation metrics to {json_path}...")
    
    with open(json_path, "w") as f:
        json.dump(
            [m.to_dict() for m in metrics_list],
            f,
            indent=2,
        )
    
    # Write aggregated metrics to JSON
    agg_path = args.out_dir / "metrics_aggregated.json"
    print(f"Writing aggregated metrics to {agg_path}...")
    
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    
    # Save predictions if requested
    if args.save_predictions and predictions_dict:
        import numpy as np
        
        pred_dir = args.out_dir / "predictions"
        pred_dir.mkdir(exist_ok=True)
        
        print(f"Saving predictions to {pred_dir}...")
        
        for (ic_type, name), preds in predictions_dict.items():
            sim_dir = pred_dir / ic_type
            sim_dir.mkdir(exist_ok=True)
            
            npz_path = sim_dir / f"{name}.npz"
            
            np.savez_compressed(
                npz_path,
                density_pred=preds["density_pred"],
                density_true=preds["density_true"],
                latent_pred=preds["latent_pred"],
                latent_true=preds["latent_true"],
                times=preds["times"],
                T0=preds["T0"],
                **preds["errors"],  # e1, e2, einf, rel_e2, mass_error
            )
        
        print(f"  Saved {len(predictions_dict)} prediction files")
    
    print()
    print("=" * 70)
    print("Evaluation complete!")
    print("=" * 70)
    print(f"Results written to: {args.out_dir}")
    print()


if __name__ == "__main__":
    main()
