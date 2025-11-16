"""
Stationarity testing utilities for density time series.

Analyzes density fields from KDE snapshots to determine if they exhibit
non-stationary behavior that would require preprocessing before MVAR modeling.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

from .nonstationarity import NonStationarityProcessor


def load_density_timeseries_from_csv(density_csv_path: Path) -> np.ndarray:
    """
    Load density time series from CSV file.
    
    Parameters
    ----------
    density_csv_path : Path
        Path to density.csv file with columns: time, x, y, density
    
    Returns
    -------
    density_matrix : ndarray, shape (n_timesteps, n_gridpoints)
        Density values reshaped as (time × space)
    """
    df = pd.read_csv(density_csv_path)
    
    # Get unique times and grid points
    times = df['time'].unique()
    n_t = len(times)
    
    # Assume all times have same grid
    first_time_data = df[df['time'] == times[0]]
    n_grid = len(first_time_data)
    
    # Reshape to (n_timesteps, n_gridpoints)
    density_matrix = np.zeros((n_t, n_grid))
    for i, t in enumerate(times):
        time_data = df[df['time'] == t]
        density_matrix[i, :] = time_data['density'].values
    
    return density_matrix


def test_density_stationarity(
    density_csv_paths: List[Path],
    adf_alpha: float = 0.01,
    pod_energy_threshold: float = 0.99,
    output_json: Path = None
) -> Dict[str, Any]:
    """
    Test stationarity of POD latent coordinates from density time series.
    
    This is the paper-correct approach: we test stationarity of the LOW-DIMENSIONAL
    latent representation (POD coordinates), NOT the raw grid points.
    
    Workflow:
    1. Load all C density time series (each: time × gridpoints)
    2. Reshape to (C, time, gridpoints) → (time*C, gridpoints) for POD
    3. Apply POD/SVD to get latent space dimension d (typically d ≈ 13)
    4. Project each case to latent coordinates: Y_c ∈ R^{d × time}
    5. Run ADF tests on each of the d latent coordinates per case
    6. Report which latent coordinates are non-stationary
    
    Parameters
    ----------
    density_csv_paths : list of Path
        Paths to density.csv files (one per case)
    adf_alpha : float, default=0.01
        Significance level for ADF tests (paper uses 0.01)
    pod_energy_threshold : float, default=0.99
        Energy threshold for POD dimension selection (paper uses 0.99)
    output_json : Path, optional
        If provided, save results to JSON file
    
    Returns
    -------
    results : dict
        Dictionary with structure:
        {
            'adf_alpha': float,
            'num_cases': int,
            'pod_dimension': int,  # d = latent dimension
            'pod_energy_threshold': float,
            'pod_energy_achieved': float,
            'cases': [
                {
                    'case_num': int,
                    'file': str,
                    'n_timesteps': int,
                    'n_gridpoints': int,
                    'latent_coordinates': [  # d latent coords per case
                        {
                            'latent_dim': int,  # 0 to d-1
                            'adf_pvalue': float,
                            'is_stationary': bool,
                            'adf_lag': int,
                            'adf_variant': str,
                            'recommended_transform': str
                        },
                        ...
                    ]
                },
                ...
            ],
            'summary': {
                'total_latent_coords_tested': int,  # d * C
                'stationary_count': int,
                'non_stationary_count': int,
                'stationary_fraction': float,
                'requires_preprocessing': bool,
                'cases_fully_stationary': int,
                'cases_needing_preprocessing': int
            }
        }
    """
    from .pod import PODProjector
    
    if not density_csv_paths:
        raise ValueError("density_csv_paths cannot be empty")
    
    # Load all density time series
    print("Loading density time series from all cases...")
    all_density_matrices = []
    for csv_path in density_csv_paths:
        density_matrix = load_density_timeseries_from_csv(csv_path)
        all_density_matrices.append(density_matrix)
    
    n_cases = len(all_density_matrices)
    n_t = all_density_matrices[0].shape[0]
    n_grid = all_density_matrices[0].shape[1]
    
    print(f"  Cases: {n_cases}")
    print(f"  Time steps per case: {n_t}")
    print(f"  Grid points: {n_grid}")
    
    # Reshape for POD: (C, time, grid) → (time, C*grid) → transpose → (C*grid, time)
    # Actually we want (n_snapshots, n_x, n_y) but we have flattened grid
    # So: stack all cases' time series → (C*time, grid) then reshape
    
    # Stack all time steps from all cases
    all_snapshots = np.vstack(all_density_matrices)  # (C*time, grid)
    
    # Need to unflatten grid to (n_x, n_y) - assume square grid or get from CSV
    # For now, infer grid shape (assume it's from density CSV with regular grid)
    # We'll reshape assuming a reasonable grid (e.g., 40×40 = 1600)
    grid_size = int(np.sqrt(n_grid))
    if grid_size * grid_size != n_grid:
        # Not square, try to infer from density CSV metadata
        # For now, use closest square
        grid_size = int(np.ceil(np.sqrt(n_grid)))
        # Pad if needed
        if grid_size * grid_size > n_grid:
            padding = grid_size * grid_size - n_grid
            all_snapshots = np.hstack([all_snapshots, np.zeros((all_snapshots.shape[0], padding))])
    
    n_x = n_y = grid_size
    n_snapshots_total = n_cases * n_t
    
    # Reshape to (n_snapshots, n_x, n_y)
    density_snapshots = all_snapshots.reshape(n_snapshots_total, n_x, n_y)
    
    print(f"\nFitting POD to {n_snapshots_total} density snapshots ({n_x}×{n_y} grid)...")
    
    # Fit POD
    # Assume uniform grid spacing (doesn't affect SVD, just mass checks)
    delta_x = delta_y = 1.0  # Placeholder
    
    projector = PODProjector(
        energy_threshold=pod_energy_threshold,
        use_weighted_mass=False,
        verbose=False
    )
    projector.fit(density_snapshots, delta_x, delta_y)
    
    d = projector.d
    energy_achieved = projector.energy_curve[d - 1]
    
    print(f"  POD dimension selected: d = {d}")
    print(f"  Energy captured: {energy_achieved:.4f}")
    print(f"  Top 5 singular values: {projector.s[:5]}")
    
    # Project each case to latent coordinates
    print(f"\nProjecting cases to {d}-dimensional latent space...")
    latent_cases = []
    for case_idx, density_matrix in enumerate(all_density_matrices):
        # Reshape case to (n_t, n_x, n_y)
        case_snapshots = density_matrix.reshape(n_t, n_x, n_y)
        # Project: returns (d, n_t)
        Y_case = projector.transform(case_snapshots)
        latent_cases.append(Y_case)
        print(f"  Case {case_idx + 1}: Y.shape = {Y_case.shape}")
    
    # Test stationarity on latent coordinates
    print(f"\nTesting stationarity of {d} latent coordinates per case...")
    
    results = {
        'adf_alpha': adf_alpha,
        'num_cases': n_cases,
        'pod_dimension': d,
        'pod_energy_threshold': pod_energy_threshold,
        'pod_energy_achieved': float(energy_achieved),
        'n_timesteps': n_t,
        'n_gridpoints': n_grid,
        'cases': []
    }
    
    total_stationary = 0
    total_non_stationary = 0
    cases_fully_stationary = 0
    
    # Test each case's latent coordinates
    for case_idx, (Y_case, csv_path) in enumerate(zip(latent_cases, density_csv_paths)):
        case_num = case_idx + 1
        
        # Y_case is (d, n_t), need to test stationarity of each of d coordinates
        # Run stationarity analysis on latent coordinates
        processor = NonStationarityProcessor(
            adf_alpha=adf_alpha,
            verbose=False
        )
        processor.fit([Y_case])
        
        # Extract results for this case
        case_meta = processor.case_meta[0]
        latent_coord_results = []
        
        for latent_dim, decision in enumerate(case_meta.decisions):
            latent_coord_results.append({
                'latent_dim': int(latent_dim),
                'adf_pvalue': float(decision.adf_pvalue),
                'is_stationary': bool(decision.adf_pvalue < adf_alpha),
                'adf_lag': int(decision.adf_lag),
                'adf_variant': decision.adf_variant,
                'recommended_transform': decision.mode,
                'notes': decision.notes
            })
        
        stationary_count = sum(1 for c in latent_coord_results if c['is_stationary'])
        non_stationary_count = d - stationary_count
        
        total_stationary += stationary_count
        total_non_stationary += non_stationary_count
        
        if non_stationary_count == 0:
            cases_fully_stationary += 1
        
        case_result = {
            'case_num': case_num,
            'file': str(csv_path),
            'n_timesteps': int(n_t),
            'n_gridpoints': int(n_grid),
            'latent_dimension': d,
            'stationary_count': stationary_count,
            'non_stationary_count': non_stationary_count,
            'stationary_fraction': float(stationary_count / d) if d > 0 else 0.0,
            'latent_coordinates': latent_coord_results
        }
        
        results['cases'].append(case_result)
    
    # Summary across all cases
    total_latent_coords = d * n_cases
    results['summary'] = {
        'total_latent_coords_tested': total_latent_coords,
        'stationary_count': total_stationary,
        'non_stationary_count': total_non_stationary,
        'stationary_fraction': float(total_stationary / total_latent_coords) if total_latent_coords > 0 else 0.0,
        'requires_preprocessing': total_non_stationary > 0,
        'cases_fully_stationary': cases_fully_stationary,
        'cases_needing_preprocessing': n_cases - cases_fully_stationary
    }
    
    # Save to JSON if requested
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


def print_stationarity_summary(results: Dict[str, Any]):
    """
    Print human-readable summary of stationarity test results.
    
    Parameters
    ----------
    results : dict
        Results dictionary from test_density_stationarity()
    """
    print("\n" + "="*80)
    print("LATENT SPACE STATIONARITY ANALYSIS (POD Coordinates)")
    print("="*80)
    print(f"\nADF significance level: α={results['adf_alpha']}")
    print(f"Cases tested: {results['num_cases']}")
    print(f"POD dimension: d = {results['pod_dimension']}")
    print(f"POD energy threshold: {results['pod_energy_threshold']:.2f}")
    print(f"POD energy achieved: {results['pod_energy_achieved']:.4f}")
    print(f"\nTesting {results['pod_dimension']} latent coordinates per case")
    print(f"(This is the paper-correct approach: ADF on low-dimensional POD space)")
    
    print(f"\n{'─'*80}")
    print("Per-Case Results:")
    print(f"{'─'*80}")
    
    for case in results['cases']:
        d = case['latent_dimension']
        status_icon = "✓" if case['non_stationary_count'] == 0 else "⚠" if case['stationary_fraction'] > 0.5 else "✗"
        print(f"\n{status_icon} Case {case['case_num']}: {case['stationary_count']}/{d} latent coords stationary "
              f"({case['stationary_fraction']:.1%})")
        print(f"  File: {case['file']}")
        print(f"  Time steps: {case['n_timesteps']}, Grid points: {case['n_gridpoints']}")
        
        # Show non-stationary latent coordinates
        non_stat_coords = [c for c in case['latent_coordinates'] if not c['is_stationary']]
        if non_stat_coords:
            print(f"  Non-stationary latent dimensions:")
            for c in non_stat_coords:
                print(f"    - Latent dim {c['latent_dim']}: p={c['adf_pvalue']:.4f}, "
                      f"transform={c['recommended_transform']}")
        else:
            print(f"  ✓ All latent coordinates are stationary!")
    
    print(f"\n{'─'*80}")
    print("Overall Summary:")
    print(f"{'─'*80}")
    summary = results['summary']
    print(f"  Total latent coordinates tested: {summary['total_latent_coords_tested']} ({results['pod_dimension']} × {results['num_cases']} cases)")
    print(f"  Stationary: {summary['stationary_count']} ({summary['stationary_fraction']:.1%})")
    print(f"  Non-stationary: {summary['non_stationary_count']}")
    print(f"  Cases fully stationary: {summary['cases_fully_stationary']}/{results['num_cases']}")
    print(f"  Cases needing preprocessing: {summary['cases_needing_preprocessing']}/{results['num_cases']}")
    
    if summary['requires_preprocessing']:
        print(f"\n⚠️  Preprocessing required: {summary['non_stationary_count']} non-stationary latent coordinates detected")
        print(f"  Recommended actions:")
        print(f"    - Review ADF test results in stationarity_report.json")
        print(f"    - Apply recommended transforms (differencing/detrending) to latent coordinates")
        print(f"    - Use NonStationarityProcessor.transform() before MVAR fitting")
    else:
        print(f"\n✓ All latent coordinates are stationary (ready for MVAR modeling!)")
    
    print("="*80 + "\n")
