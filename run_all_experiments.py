#!/usr/bin/env python3
"""
run_all_experiments.py — Master orchestrator for the DYN1-7 systematic suite
=============================================================================

Runs every DYN experiment through the full 3-method pipeline
(MVAR + LSTM + WSINDy) and generates:
  1. Per-experiment outputs in oscar_output/<EXP>/
  2. Per-experiment plots   in predictions/<EXP>/
  3. Cross-experiment suite comparison in predictions/SUITE_COMPARISON/

Usage:
  # Full suite (Oscar cluster)
  python run_all_experiments.py

  # Specific experiments only
  python run_all_experiments.py --experiments DYN1_gentle DYN4_blackhole

  # Skip pipeline, only generate visualizations from existing data
  python run_all_experiments.py --viz-only

  # Skip pipeline + per-experiment viz, only cross-experiment comparison
  python run_all_experiments.py --compare-only

  # Dry run: just print what would be done
  python run_all_experiments.py --dry-run
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# ── All DYN experiments in order ────────────────────────────────
ALL_EXPERIMENTS = [
    ("DYN1_gentle",         "configs/DYN1_gentle.yaml"),
    ("DYN2_hypervelocity",  "configs/DYN2_hypervelocity.yaml"),
    ("DYN3_hypernoisy",     "configs/DYN3_hypernoisy.yaml"),
    ("DYN4_blackhole",      "configs/DYN4_blackhole.yaml"),
    ("DYN5_supernova",      "configs/DYN5_supernova.yaml"),
    ("DYN6_varspeed",       "configs/DYN6_varspeed.yaml"),
    ("DYN7_pure_vicsek",    "configs/DYN7_pure_vicsek.yaml"),
]

ROOT = Path(__file__).resolve().parent


def run_cmd(cmd, label, dry_run=False):
    """Run a subprocess with timing and error handling."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    if dry_run:
        print("  [DRY RUN] Would execute above command")
        return True

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  *** FAILED ({elapsed/60:.1f}m): {label} ***\n")
        return False
    else:
        print(f"\n  Completed ({elapsed/60:.1f}m): {label}")
        return True


def run_pipeline(exp_name, config_path, dry_run=False):
    """Run ROM_WSINDY_pipeline.py for one experiment."""
    cmd = [
        sys.executable, "ROM_WSINDY_pipeline.py",
        "--config", config_path,
        "--experiment_name", exp_name,
    ]
    return run_cmd(cmd, f"PIPELINE: {exp_name}", dry_run=dry_run)


def run_visualizations(exp_name, dry_run=False):
    """Run run_visualizations.py for one experiment."""
    cmd = [
        sys.executable, "run_visualizations.py",
        "--experiment_name", exp_name,
    ]
    return run_cmd(cmd, f"VISUALIZE: {exp_name}", dry_run=dry_run)


def run_cross_comparison(experiments, dry_run=False):
    """Run cross-experiment comparison."""
    exp_names = [e[0] for e in experiments]
    cmd = [
        sys.executable, "scripts/compare_experiments.py",
        "--experiments", *exp_names,
    ]
    return run_cmd(cmd, "CROSS-EXPERIMENT COMPARISON", dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser(
        description="Master orchestrator for DYN1-7 systematic suite")
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="Specific experiment names to run (default: all)")
    parser.add_argument("--viz-only", action="store_true",
                        help="Skip pipeline, only generate visualizations")
    parser.add_argument("--compare-only", action="store_true",
                        help="Only generate cross-experiment comparison")
    parser.add_argument("--skip-viz", action="store_true",
                        help="Skip per-experiment visualization")
    parser.add_argument("--skip-compare", action="store_true",
                        help="Skip cross-experiment comparison")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    # Filter experiments if specified
    if args.experiments:
        experiments = [
            (name, cfg) for name, cfg in ALL_EXPERIMENTS
            if name in args.experiments
        ]
        if not experiments:
            print(f"ERROR: No matching experiments. Available: "
                  f"{[e[0] for e in ALL_EXPERIMENTS]}")
            sys.exit(1)
    else:
        experiments = ALL_EXPERIMENTS

    print(f"\n{'#'*70}")
    print(f"  DYN SYSTEMATIC SUITE — {len(experiments)} experiments")
    print(f"  Methods: MVAR + LSTM + WSINDy (3-field PDE)")
    print(f"  Pipeline: ROM_WSINDY_pipeline.py")
    print(f"{'#'*70}\n")
    for name, cfg in experiments:
        print(f"  {name:25s} → {cfg}")

    results = {}
    t_total = time.time()

    # ── Phase 1: Run pipelines ──────────────────────────────────
    if not args.viz_only and not args.compare_only:
        print(f"\n\n{'#'*70}")
        print("  PHASE 1: PIPELINE EXECUTION")
        print(f"{'#'*70}")

        for name, cfg in experiments:
            ok = run_pipeline(name, cfg, dry_run=args.dry_run)
            results[name] = {"pipeline": "ok" if ok else "FAILED"}
            if not ok:
                print(f"  WARNING: {name} failed, continuing with next...")

    # ── Phase 2: Per-experiment visualization ────────────────────
    if not args.compare_only and not args.skip_viz:
        print(f"\n\n{'#'*70}")
        print("  PHASE 2: PER-EXPERIMENT VISUALIZATION")
        print(f"{'#'*70}")

        for name, _ in experiments:
            ok = run_visualizations(name, dry_run=args.dry_run)
            results.setdefault(name, {})["viz"] = "ok" if ok else "FAILED"

    # ── Phase 3: Cross-experiment comparison ─────────────────────
    if not args.skip_compare:
        print(f"\n\n{'#'*70}")
        print("  PHASE 3: CROSS-EXPERIMENT COMPARISON")
        print(f"{'#'*70}")

        run_cross_comparison(experiments, dry_run=args.dry_run)

    # ── Summary ──────────────────────────────────────────────────
    elapsed = time.time() - t_total
    print(f"\n\n{'#'*70}")
    print(f"  SUITE COMPLETE — {elapsed/60:.1f} minutes total")
    print(f"{'#'*70}\n")

    for name, status in results.items():
        pipe = status.get("pipeline", "skipped")
        viz = status.get("viz", "skipped")
        print(f"  {name:25s}  pipeline={pipe:8s}  viz={viz:8s}")

    # Save suite status
    status_path = ROOT / "predictions" / "suite_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with open(status_path, "w") as f:
        json.dump({
            "experiments": {n: r for n, r in results.items()},
            "total_time_minutes": elapsed / 60,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    print(f"\n  Status saved: {status_path}")


if __name__ == "__main__":
    main()
