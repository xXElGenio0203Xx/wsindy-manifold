"""
Curate density time series for MVAR modeling.

This script:
1. Loads density time series from ensemble simulation
2. Applies POD to extract low-dimensional latent representation
3. Tests stationarity of latent coordinates using ADF
4. Applies recommended transforms (differencing/detrending) to non-stationary coordinates
5. Saves curated latent time series to curated_outputs/

Usage:
    python scripts/curate_latent_timeseries.py outputs/ensemble_stationarity_test

Output structure:
    curated_outputs/
        <run_name>/
            case_001/
                latent_raw.csv          # Original latent coordinates (d × time)
                latent_curated.csv      # Stationary latent coordinates
                curation_metadata.json  # Transform decisions per coordinate
            case_002/
                ...
            pod_projector.npz          # POD basis for reconstruction
            curation_summary.json      # Global summary of transforms applied
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

from rectsim.pod import PODProjector
from rectsim.nonstationarity import NonStationarityProcessor
from rectsim.stationarity_testing import load_density_timeseries_from_csv


def main():
    parser = argparse.ArgumentParser(
        description="Curate latent density time series for MVAR modeling"
    )
    parser.add_argument(
        "simulation_dir",
        type=Path,
        help="Path to simulation output directory (e.g., outputs/ensemble_stationarity_test)"
    )
    parser.add_argument(
        "--pod-energy",
        type=float,
        default=0.99,
        help="POD energy threshold for dimension selection (default: 0.99)"
    )
    parser.add_argument(
        "--adf-alpha",
        type=float,
        default=0.01,
        help="ADF test significance level (default: 0.01)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: curated_outputs/<run_name>)"
    )
    
    args = parser.parse_args()
    
    simulation_dir = args.simulation_dir.resolve()
    if not simulation_dir.exists():
        raise FileNotFoundError(f"Simulation directory not found: {simulation_dir}")
    
    # Find case directories
    case_dirs = sorted(simulation_dir.glob("case_*"))
    if not case_dirs:
        raise ValueError(f"No case_* subdirectories found in {simulation_dir}")
    
    n_cases = len(case_dirs)
    print(f"\n{'='*80}")
    print(f"LATENT DENSITY CURATION")
    print(f"{'='*80}")
    print(f"\nSimulation: {simulation_dir.name}")
    print(f"Cases found: {n_cases}")
    print(f"POD energy threshold: {args.pod_energy}")
    print(f"ADF significance level: α={args.adf_alpha}")
    
    # Load density time series from all cases
    print(f"\n{'─'*80}")
    print("Loading density time series...")
    print(f"{'─'*80}")
    
    density_csv_paths = []
    all_density_matrices = []
    
    for case_dir in case_dirs:
        density_csv = case_dir / "density.csv"
        if not density_csv.exists():
            print(f"  ⚠ Skipping {case_dir.name}: density.csv not found")
            continue
        
        density_matrix = load_density_timeseries_from_csv(density_csv)
        all_density_matrices.append(density_matrix)
        density_csv_paths.append(density_csv)
        print(f"  ✓ {case_dir.name}: {density_matrix.shape[0]} timesteps × {density_matrix.shape[1]} grid points")
    
    if not all_density_matrices:
        raise ValueError("No valid density.csv files found")
    
    n_cases = len(all_density_matrices)
    n_t = all_density_matrices[0].shape[0]
    n_grid = all_density_matrices[0].shape[1]
    
    # Apply POD to extract latent representation
    print(f"\n{'─'*80}")
    print("Applying POD to extract latent representation...")
    print(f"{'─'*80}")
    
    # Stack all snapshots: (C*time, grid)
    all_snapshots = np.vstack(all_density_matrices)
    
    # Infer grid shape (assume square or pad to square)
    grid_size = int(np.sqrt(n_grid))
    if grid_size * grid_size != n_grid:
        grid_size = int(np.ceil(np.sqrt(n_grid)))
        if grid_size * grid_size > n_grid:
            padding = grid_size * grid_size - n_grid
            all_snapshots = np.hstack([all_snapshots, np.zeros((all_snapshots.shape[0], padding))])
    
    n_x = n_y = grid_size
    n_snapshots_total = n_cases * n_t
    
    # Reshape to (n_snapshots, n_x, n_y)
    density_snapshots = all_snapshots.reshape(n_snapshots_total, n_x, n_y)
    
    print(f"  Snapshots: {n_snapshots_total} ({n_x}×{n_y} grid)")
    
    # Fit POD
    delta_x = delta_y = 1.0  # Placeholder (doesn't affect SVD)
    
    projector = PODProjector(
        energy_threshold=args.pod_energy,
        use_weighted_mass=False
    )
    projector.fit(density_snapshots, delta_x, delta_y)
    
    d = projector.d
    energy_achieved = projector.energy_curve[d - 1]
    
    print(f"  ✓ POD dimension: d = {d}")
    print(f"  ✓ Energy captured: {energy_achieved:.4f}")
    print(f"  ✓ Top 5 singular values: {projector.s[:5]}")
    
    # Project each case to latent coordinates
    print(f"\n{'─'*80}")
    print(f"Projecting cases to {d}-dimensional latent space...")
    print(f"{'─'*80}")
    
    latent_cases_raw = []
    for case_idx, density_matrix in enumerate(all_density_matrices):
        case_snapshots = density_matrix.reshape(n_t, n_x, n_y)
        Y_case = projector.transform(case_snapshots)  # (d, n_t)
        latent_cases_raw.append(Y_case)
        print(f"  Case {case_idx + 1}: Y.shape = {Y_case.shape}")
    
    # Test stationarity and apply transforms
    print(f"\n{'─'*80}")
    print(f"Testing stationarity and applying transforms...")
    print(f"{'─'*80}")
    
    processor = NonStationarityProcessor(
        adf_alpha=args.adf_alpha,
        verbose=True
    )
    
    # Fit processor to all latent time series
    processor.fit(latent_cases_raw)
    
    # Transform to stationary coordinates
    latent_cases_curated = processor.transform(latent_cases_raw)
    
    # Print summary
    print(f"\n{'─'*80}")
    print("Stationarity Analysis Summary:")
    print(f"{'─'*80}")
    
    for case_idx, case_meta in enumerate(processor.case_meta):
        print(f"\nCase {case_idx + 1}:")
        stationary_count = sum(1 for d in case_meta.decisions if d.mode == 'raw')
        print(f"  Stationary coordinates: {stationary_count}/{d}")
        
        # Show non-stationary coords and their transforms
        non_stat = [(i, d) for i, d in enumerate(case_meta.decisions) if d.mode != 'raw']
        if non_stat:
            print(f"  Transforms applied:")
            for coord_idx, decision in non_stat:
                print(f"    - Latent dim {coord_idx}: {decision.mode} (p={decision.adf_pvalue:.4f})")
        else:
            print(f"  ✓ All coordinates stationary!")
    
    # Save curated outputs
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = Path("curated_outputs") / simulation_dir.name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'─'*80}")
    print(f"Saving curated outputs to: {output_dir}")
    print(f"{'─'*80}")
    
    # Save POD projector
    pod_path = output_dir / "pod_projector.npz"
    projector.save(pod_path)
    print(f"  ✓ Saved {pod_path.name}")
    
    # Save each case
    for case_idx, (Y_raw, Y_curated, case_meta, orig_path) in enumerate(
        zip(latent_cases_raw, latent_cases_curated, processor.case_meta, density_csv_paths)
    ):
        case_num = case_idx + 1
        case_dir_out = output_dir / f"case_{case_num:03d}"
        case_dir_out.mkdir(exist_ok=True)
        
        # Save raw latent coordinates
        df_raw = pd.DataFrame(
            Y_raw.T,  # Transpose to (time, d)
            columns=[f"latent_{i}" for i in range(d)]
        )
        df_raw.insert(0, "time", np.arange(n_t))
        raw_path = case_dir_out / "latent_raw.csv"
        df_raw.to_csv(raw_path, index=False)
        
        # Save curated latent coordinates
        d_curated = Y_curated.shape[0]
        n_t_curated = Y_curated.shape[1]
        df_curated = pd.DataFrame(
            Y_curated.T,  # Transpose to (time, d)
            columns=[f"latent_{i}" for i in range(d_curated)]
        )
        df_curated.insert(0, "time", np.arange(n_t_curated))
        curated_path = case_dir_out / "latent_curated.csv"
        df_curated.to_csv(curated_path, index=False)
        
        # Save curation metadata
        metadata = {
            "case_num": case_num,
            "original_density_csv": str(orig_path),
            "n_timesteps_original": int(n_t),
            "n_timesteps_curated": int(n_t_curated),
            "pod_dimension": int(d),
            "pod_energy_threshold": float(args.pod_energy),
            "pod_energy_achieved": float(energy_achieved),
            "adf_alpha": float(args.adf_alpha),
            "transforms": []
        }
        
        for coord_idx, decision in enumerate(case_meta.decisions):
            metadata["transforms"].append({
                "latent_dim": int(coord_idx),
                "transform": decision.mode,
                "adf_pvalue": float(decision.adf_pvalue),
                "adf_lag": int(decision.adf_lag),
                "adf_variant": decision.adf_variant,
                "is_stationary": bool(decision.mode == 'raw'),
                "notes": decision.notes
            })
        
        metadata_path = case_dir_out / "curation_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Case {case_num}:")
        print(f"      {raw_path.name}")
        print(f"      {curated_path.name}")
        print(f"      {metadata_path.name}")
    
    # Save global summary
    summary = {
        "simulation_dir": str(simulation_dir),
        "num_cases": n_cases,
        "pod_dimension": int(d),
        "pod_energy_threshold": float(args.pod_energy),
        "pod_energy_achieved": float(energy_achieved),
        "adf_alpha": float(args.adf_alpha),
        "n_timesteps_original": int(n_t),
        "top_singular_values": [float(s) for s in projector.s[:10]],
        "cases_summary": []
    }
    
    for case_idx, case_meta in enumerate(processor.case_meta):
        stationary_count = sum(1 for d in case_meta.decisions if d.mode == 'raw')
        transforms_used = {}
        for decision in case_meta.decisions:
            transforms_used[decision.mode] = transforms_used.get(decision.mode, 0) + 1
        
        summary["cases_summary"].append({
            "case_num": case_idx + 1,
            "stationary_coords": stationary_count,
            "non_stationary_coords": d - stationary_count,
            "stationary_fraction": float(stationary_count / d),
            "transforms_used": transforms_used
        })
    
    summary_path = output_dir / "curation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ {summary_path.name}")
    
    print(f"\n{'='*80}")
    print("CURATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nCurated latent time series saved to:")
    print(f"  {output_dir}")
    print(f"\nEach case contains:")
    print(f"  • latent_raw.csv       - Original {d}-dimensional latent coordinates")
    print(f"  • latent_curated.csv   - Stationary transformed coordinates")
    print(f"  • curation_metadata.json - Transform decisions")
    print(f"\nGlobal files:")
    print(f"  • pod_projector.npz      - POD basis for reconstruction")
    print(f"  • curation_summary.json  - Summary of all transforms")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
