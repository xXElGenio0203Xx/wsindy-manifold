#!/usr/bin/env python3
"""Complete ROM/MVAR evaluation pipeline (local execution).

This script runs the full evaluation workflow:
1. Load ROM/MVAR model
2. Load unseen test simulations
3. Run predictions and compute metrics
4. Aggregate metrics by IC type
5. Select best runs per IC type
6. Generate plots (error, order params)
7. Generate truth vs prediction videos

Designed for local execution (no SLURM dependencies).

Usage:
    python scripts/rom_mvar_full_eval_local.py \\
        --rom_dir rom_mvar/vicsek_morse_base/model \\
        --unseen_root simulations_unseen \\
        --out_root rom_mvar/vicsek_morse_base/unseen_eval \\
        --train_frac 0.8 \\
        --tol 0.1 \\
        --fps 20

Output Structure:
    out_root/
        metrics_per_sim.csv
        metrics_per_sim.json
        metrics_aggregated.json
        ic_type/
            best_error.png
            best_order_params.png
            best_truth_vs_pred.mp4

Author: Maria
Date: November 2025
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rectsim.rom_mvar_model import PODMVARModel
from rectsim.rom_eval_data import load_unseen_simulations
from rectsim.rom_eval_pipeline import (
    evaluate_unseen_rom,
    aggregate_metrics,
    print_aggregated_metrics,
)
from rectsim.rom_eval_viz import (
    select_best_runs,
    make_best_run_error_plots,
    make_best_run_order_param_plots,
)
from rectsim.rom_video_utils import make_best_run_videos


def main():
    parser = argparse.ArgumentParser(
        description="Complete ROM/MVAR evaluation pipeline (local)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full evaluation
  python scripts/rom_mvar_full_eval_local.py \\
      --rom_dir rom_mvar/vicsek_morse_base/model \\
      --unseen_root simulations_unseen \\
      --out_root rom_mvar/vicsek_morse_base/unseen_eval

  # Skip videos (faster)
  python scripts/rom_mvar_full_eval_local.py \\
      --rom_dir rom_mvar/exp1/model \\
      --unseen_root simulations_unseen \\
      --out_root rom_mvar/exp1/unseen_eval \\
      --no-videos

Output structure:
  out_root/
    metrics_per_sim.csv          # Per-simulation metrics
    metrics_per_sim.json
    metrics_aggregated.json      # Aggregated stats by IC type
    ring/
      best_error.png             # Error vs time
      best_order_params.png      # Order parameters
      best_truth_vs_pred.mp4     # Video comparison
    gaussian/
      ...
        """,
    )
    
    parser.add_argument(
        "--rom_dir",
        type=Path,
        required=True,
        help="Directory with ROM model (pod_basis.npz, mvar_params.npz, train_summary.json)",
    )
    
    parser.add_argument(
        "--unseen_root",
        type=Path,
        required=True,
        help="Root directory with test simulations (organized by IC type)",
    )
    
    parser.add_argument(
        "--out_root",
        type=Path,
        required=True,
        help="Output root directory",
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
        help="Tolerance for tau computation (default: 0.1)",
    )
    
    parser.add_argument(
        "--ic_types",
        type=str,
        default=None,
        help="Comma-separated IC types to evaluate (default: auto-detect)",
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Video frames per second (default: 20)",
    )
    
    parser.add_argument(
        "--metric",
        type=str,
        default="r2",
        help="Metric for selecting best runs (default: r2)",
    )
    
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Skip video generation (faster)",
    )
    
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )
    
    args = parser.parse_args()
    
    # Parse IC types
    ic_types = None
    if args.ic_types:
        ic_types = [s.strip() for s in args.ic_types.split(",")]
    
    # Create output directory
    args.out_root.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("ROM/MVAR Complete Evaluation Pipeline")
    print("=" * 70)
    print(f"ROM model:       {args.rom_dir}")
    print(f"Simulations:     {args.unseen_root}")
    print(f"Output root:     {args.out_root}")
    print(f"Train fraction:  {args.train_frac}")
    print(f"Tolerance:       {args.tol}")
    print(f"Best run metric: {args.metric}")
    if ic_types:
        print(f"IC types:        {', '.join(ic_types)}")
    print(f"Generate videos: {not args.no_videos}")
    print(f"Generate plots:  {not args.no_plots}")
    print("=" * 70)
    print()
    
    # Step 1: Run predictions and compute metrics
    print("STEP 1: Running predictions and computing metrics...")
    print("-" * 70)
    
    metrics_list, predictions_dict = evaluate_unseen_rom(
        rom_dir=args.rom_dir,
        unseen_root=args.unseen_root,
        ic_types=ic_types,
        train_frac=args.train_frac,
        tol=args.tol,
        return_predictions=True,  # Need predictions for videos
    )
    
    if not metrics_list:
        print("No simulations evaluated successfully. Exiting.")
        return 1
    
    print()
    
    # Step 2: Aggregate metrics
    print("STEP 2: Aggregating metrics by IC type...")
    print("-" * 70)
    
    aggregated = aggregate_metrics(metrics_list)
    print_aggregated_metrics(aggregated)
    
    # Step 3: Write metrics to disk
    print("STEP 3: Writing metrics to disk...")
    print("-" * 70)
    
    # CSV
    csv_path = args.out_root / "metrics_per_sim.csv"
    print(f"  {csv_path}")
    with open(csv_path, "w", newline="") as f:
        fieldnames = list(metrics_list[0].to_dict().keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics_list:
            writer.writerow(m.to_dict())
    
    # JSON per-sim
    json_path = args.out_root / "metrics_per_sim.json"
    print(f"  {json_path}")
    with open(json_path, "w") as f:
        json.dump([m.to_dict() for m in metrics_list], f, indent=2)
    
    # JSON aggregated
    agg_path = args.out_root / "metrics_aggregated.json"
    print(f"  {agg_path}")
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    
    print()
    
    # Step 4: Select best runs
    print("STEP 4: Selecting best runs per IC type...")
    print("-" * 70)
    
    best_runs = select_best_runs(metrics_list, key=args.metric, maximize=True)
    
    print(f"  Selected {len(best_runs)} best runs:")
    for ic_type, metrics in best_runs.items():
        print(f"    {ic_type}: {metrics.name} ({args.metric}={getattr(metrics, args.metric):.4f})")
    print()
    
    # Step 5: Generate plots
    if not args.no_plots:
        print("STEP 5: Generating plots for best runs...")
        print("-" * 70)
        
        # Error plots
        print("  Error plots:")
        make_best_run_error_plots(
            best_runs,
            predictions_dict,
            args.out_root,
        )
        
        # Load samples for order parameters
        print("  Loading samples for order parameters...")
        samples = load_unseen_simulations(
            args.unseen_root,
            ic_types=ic_types,
            require_density=True,
            require_traj=False,
        )
        samples_dict = {(s.ic_type, s.name): s for s in samples}
        
        print("  Order parameter plots:")
        make_best_run_order_param_plots(
            best_runs,
            samples_dict,
            args.out_root,
            predictions_dict=predictions_dict,
        )
        
        print()
    else:
        print("STEP 5: Skipping plots (--no-plots)")
        print()
    
    # Step 6: Generate videos
    if not args.no_videos:
        print("STEP 6: Generating videos for best runs...")
        print("-" * 70)
        
        # Load model
        print("  Loading ROM model...")
        model = PODMVARModel.load(args.rom_dir)
        
        # Load samples if not already loaded
        if args.no_plots:
            print("  Loading samples...")
            samples = load_unseen_simulations(
                args.unseen_root,
                ic_types=ic_types,
                require_density=True,
                require_traj=False,
            )
            samples_dict = {(s.ic_type, s.name): s for s in samples}
        
        make_best_run_videos(
            best_runs,
            model,
            samples_dict,
            args.out_root,
            train_frac=args.train_frac,
            fps=args.fps,
        )
        
        print()
    else:
        print("STEP 6: Skipping videos (--no-videos)")
        print()
    
    # Summary
    print("=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)
    print(f"Results saved to: {args.out_root}")
    print()
    print("Output structure:")
    print(f"  {args.out_root}/")
    print("    metrics_per_sim.csv")
    print("    metrics_per_sim.json")
    print("    metrics_aggregated.json")
    for ic_type in sorted(best_runs.keys()):
        print(f"    {ic_type}/")
        if not args.no_plots:
            print("      best_error.png")
            print("      best_order_params.png")
        if not args.no_videos:
            print("      best_truth_vs_pred.mp4")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
