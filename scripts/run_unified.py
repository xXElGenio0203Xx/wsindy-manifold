#!/usr/bin/env python3
"""Run simulations with unified configuration format.

This script accepts the standardized configuration schema and automatically
dispatches to the appropriate backend (discrete or continuous) based on the
model type specified in the config.

All simulations produce standardized outputs:
- Order parameters CSV (polarization, angular momentum, mean speed, density variance)
- Trajectory and density animations (MP4)
- Summary plots
- Full trajectory and density data (CSV and NPZ)
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rectsim import simulate_backend
from rectsim.io_outputs import save_standardized_outputs
from rectsim.unified_config import apply_defaults, validate_config, convert_to_legacy_format
from rectsim.stationarity_testing import test_density_stationarity, print_stationarity_summary


def load_unified_config(config_path: Path) -> dict:
    """Load and validate unified configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply defaults
    config = apply_defaults(config)
    
    # Validate
    errors = validate_config(config)
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"  • {error}")
        sys.exit(1)
    
    return config


def _save_case_data(times, positions, velocities, domain_bounds, case_dir, boundary_condition, density_resolution):
    """Save trajectory and density data for a single case (minimal outputs)."""
    from rectsim.io_outputs import save_density_csv
    import pandas as pd
    
    # Save trajectory CSV
    traj_path = case_dir / 'trajectory.csv'
    T, N, _ = positions.shape
    
    # Build trajectory dataframe
    data = []
    for t_idx, t in enumerate(times):
        for i in range(N):
            data.append({
                'time': t,
                'particle_id': i,
                'x': positions[t_idx, i, 0],
                'y': positions[t_idx, i, 1],
                'vx': velocities[t_idx, i, 0],
                'vy': velocities[t_idx, i, 1],
            })
    
    df = pd.DataFrame(data)
    df.to_csv(traj_path, index=False)
    print(f"  ✓ Saved {traj_path}")
    
    # Save density CSV
    density_path = case_dir / 'density.csv'
    save_density_csv(
        times, positions, domain_bounds, density_resolution, density_path,
        bandwidth_mode="manual", manual_H=(3.0, 2.0),
        periodic_x=(boundary_condition == "periodic")
    )
    print(f"  ✓ Saved {density_path}")


def print_config_summary(config: dict):
    """Print human-readable configuration summary."""
    print("\n" + "="*80)
    print("SIMULATION CONFIGURATION")
    print("="*80)
    
    # Ensemble
    C = config['ensemble']['cases']
    O = config['ensemble']['outputs']
    if C > 1:
        print(f"\nEnsemble: {C} cases (visualizations for first {O})")
    
    # Model and domain
    model_type = config['model']['type']
    model_name = "Discrete Vicsek" if model_type == "discrete" else "Continuous D'Orsogna"
    print(f"\nModel: {model_name}")
    print(f"  • Domain: {config['domain']['Lx']} × {config['domain']['Ly']}")
    print(f"  • Boundary: {config['domain']['bc']}")
    print(f"  • Particles: {config['particles']['N']}")
    print(f"  • Initial distribution: {config['particles']['initial_distribution']}")
    print(f"  • Initial speed: {config['particles']['initial_speed']}")
    
    # Dynamics
    print(f"\nDynamics:")
    
    # Alignment
    if config['dynamics']['alignment']['enabled']:
        print(f"  • Alignment: ENABLED")
        print(f"    - Radius: {config['dynamics']['alignment']['radius']}")
        if model_type == "continuous":
            print(f"    - Rate (μ_r): {config['dynamics']['alignment']['rate']}")
    else:
        print(f"  • Alignment: DISABLED")
    
    # Forces
    if config['dynamics']['forces']['enabled']:
        print(f"  • Forces: ENABLED (Morse potential)")
        forces = config['dynamics']['forces']
        print(f"    - Repulsion: Cr={forces['Cr']}, lr={forces['lr']}")
        print(f"    - Attraction: Ca={forces['Ca']}, la={forces['la']}")
        print(f"    - Cutoff: {forces['rcut_factor']} × max(lr, la)")
        if model_type == "discrete":
            print(f"    - Coupling (μ_t): {forces['mu_t']}")
    else:
        print(f"  • Forces: DISABLED")
    
    # Noise
    noise = config['dynamics']['noise']
    print(f"  • Noise: {noise['type']}")
    if model_type == "discrete":
        print(f"    - Strength (η): {noise['eta']} rad")
    else:
        print(f"    - Diffusion (D_θ): {noise['Dtheta']}")
    
    # Self-propulsion (continuous only)
    if model_type == "continuous":
        sp = config['dynamics']['self_propulsion']
        print(f"  • Self-propulsion:")
        print(f"    - α: {sp['alpha']}")
        print(f"    - β: {sp['beta']}")
        print(f"    - Natural speed: {sp['alpha']/sp['beta']:.3f}")
    
    # Integration
    integ = config['integration']
    print(f"\nIntegration:")
    print(f"  • Time: 0 → {integ['T']}")
    print(f"  • Time step: {integ['dt']}")
    print(f"  • Save every: {integ['save_every']} steps")
    print(f"  • Neighbor rebuild: every {integ['neighbor_rebuild']} steps")
    if model_type == "continuous":
        print(f"  • Integrator: {integ['integrator']}")
    else:
        # Discrete model
        integrator = integ.get('integrator', 'euler')
        if integrator == 'euler_semiimplicit':
            print(f"  • Integrator: Semi-Implicit Euler")
        else:
            print(f"  • Integrator: Explicit Euler")
    print(f"  • Random seed: {integ['seed']}")
    
    # Outputs
    print(f"\nOutputs:")
    print(f"  • Directory: {config['outputs']['directory']}")
    print(f"  • Order parameters: {config['outputs']['order_parameters']}")
    print(f"  • Animations: {config['outputs']['animations']}")
    print(f"  • CSV data: {config['outputs']['save_csv']}")
    arrows = config['outputs']['arrows']
    if arrows['enabled']:
        print(f"  • Arrows: {arrows['scale']} (scale={arrows['scale_factor']})")
    else:
        print(f"  • Arrows: disabled (show dots)")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run collective motion simulation with unified configuration'
    )
    parser.add_argument(
        'config',
        type=Path,
        help='Path to unified YAML configuration file'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=Path,
        default=None,
        help='Override output directory'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Override random seed'
    )
    parser.add_argument(
        '--no-animations',
        action='store_true',
        help='Skip animation generation'
    )
    parser.add_argument(
        '-C', '--cases',
        type=int,
        default=None,
        help='Number of ensemble cases to run (override config)'
    )
    parser.add_argument(
        '-O', '--outputs',
        type=int,
        default=None,
        help='Number of cases to generate visualizations for (override config)'
    )
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration: {args.config}")
    config = load_unified_config(args.config)
    
    # Apply overrides
    if args.output_dir:
        config['outputs']['directory'] = str(args.output_dir)
    if args.seed is not None:
        config['integration']['seed'] = args.seed
    if args.no_animations:
        config['outputs']['animations'] = False
    if args.cases is not None:
        config['ensemble']['cases'] = args.cases
    if args.outputs is not None:
        config['ensemble']['outputs'] = args.outputs
    
    # Validate C and O
    C = config['ensemble']['cases']
    O = config['ensemble']['outputs']
    if O > C:
        print(f"⚠️  Warning: outputs ({O}) > cases ({C}), setting outputs = cases")
        config['ensemble']['outputs'] = C
        O = C
    
    # Print summary
    print_config_summary(config)
    
    # Convert to legacy format for backend
    model_type = config['model']['type']
    legacy_config = convert_to_legacy_format(config, model_type)
    
    # Create output directory
    output_dir = Path(config['outputs']['directory'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensemble parameters
    C = config['ensemble']['cases']
    O = config['ensemble']['outputs']
    seed = config['integration']['seed']
    
    # Domain bounds
    domain_bounds = (0, config['domain']['Lx'], 0, config['domain']['Ly'])
    
    # Run simulation(s)
    if C == 1:
        # Single case (original behavior)
        print("Running simulation...")
        rng = np.random.default_rng(seed)
        
        # Add initial_distribution to legacy config for backend
        legacy_config['initial_distribution'] = config['particles']['initial_distribution']
        
        result = simulate_backend(legacy_config, rng)
        
        print(f"✓ Simulation complete")
        print(f"  • Frames saved: {len(result['times'])}")
        print(f"  • Particles: {result['traj'].shape[1]}")
        
        # Extract data
        times = result['times']
        positions = result['traj']
        velocities = result['vel']
        
        # Save standardized outputs
        print("\n" + "-"*80)
        print("Generating outputs...")
        print("-"*80)
        
        metrics = save_standardized_outputs(
            times, positions, velocities, domain_bounds,
            output_dir, config['outputs'],
            boundary_condition=config['domain']['bc']
        )
    else:
        # Multi-case ensemble
        print(f"Running ensemble of {C} cases...")
        print(f"(Visualizations will be generated for first {O} case(s))\n")
        
        # Add initial_distribution to legacy config for backend
        legacy_config['initial_distribution'] = config['particles']['initial_distribution']
        
        # Storage for all cases
        all_results = []
        metrics = None  # Only computed for output cases
        
        for case_idx in range(C):
            case_num = case_idx + 1
            print(f"\n{'─'*80}")
            print(f"Case {case_num}/{C}")
            print(f"{'─'*80}")
            
            # Create case-specific RNG with different seed
            case_seed = seed + case_idx * 1000
            rng = np.random.default_rng(case_seed)
            
            # Update seed in legacy config
            legacy_config['seed'] = case_seed
            
            # Run simulation
            result = simulate_backend(legacy_config, rng)
            
            print(f"✓ Simulation complete")
            print(f"  • Frames: {len(result['times'])}")
            print(f"  • Particles: {result['traj'].shape[1]}")
            
            # Store result
            all_results.append({
                'times': result['times'],
                'positions': result['traj'],
                'velocities': result['vel'],
                'case_num': case_num,
                'case_seed': case_seed,
            })
            
            # Create case subfolder
            case_dir = output_dir / f"case_{case_num:03d}"
            case_dir.mkdir(parents=True, exist_ok=True)
            
            # Save trajectory and density CSVs (always for all cases)
            print(f"\nSaving case {case_num} data...")
            _save_case_data(
                result['times'], result['traj'], result['vel'],
                domain_bounds, case_dir, config['domain']['bc'],
                config['outputs']['density_resolution']
            )
            
            # Generate visualizations only for first O cases
            if case_idx < O:
                print(f"Generating visualizations for case {case_num}...")
                case_metrics = save_standardized_outputs(
                    result['times'], result['traj'], result['vel'],
                    domain_bounds, case_dir, config['outputs'],
                    boundary_condition=config['domain']['bc']
                )
                
                # Keep metrics from first case for summary
                if case_idx == 0:
                    metrics = case_metrics
            else:
                print(f"Skipping visualizations for case {case_num} (O={O})")
        
        # Use first case times for summary
        times = all_results[0]['times']
        positions = all_results[0]['positions']
        velocities = all_results[0]['velocities']
        
        # Test stationarity of density time series across all cases
        print(f"\n{'='*80}")
        print("Testing stationarity of density time series...")
        print(f"{'='*80}")
        
        density_csv_paths = []
        for case_idx in range(C):
            case_num = case_idx + 1
            case_dir = output_dir / f"case_{case_num:03d}"
            density_path = case_dir / "density.csv"
            if density_path.exists():
                density_csv_paths.append(density_path)
        
        if density_csv_paths:
            stationarity_json = output_dir / "stationarity_report.json"
            try:
                stationarity_results = test_density_stationarity(
                    density_csv_paths,
                    adf_alpha=0.01,
                    pod_energy_threshold=0.99,
                    output_json=stationarity_json
                )
                print(f"✓ Saved {stationarity_json}")
                print_stationarity_summary(stationarity_results)
            except Exception as e:
                print(f"⚠️  Stationarity testing failed: {e}")
        else:
            print("⚠️  No density CSV files found for stationarity testing")
    
    # Save metadata
    print("\nSaving metadata...")
    
    # Save unified config
    config_path = output_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"✓ Saved {config_path}")
    
    # Save complete run configuration as JSON (for compatibility with existing outputs)
    run_config = {
        'timestamp': datetime.now().isoformat(),
        'config_file': str(args.config),
        'model_type': model_type,
        'configuration': config,  # Complete unified config
        'legacy_config': legacy_config,  # Backend-specific format
    }
    
    run_json_path = output_dir / 'run.json'
    with open(run_json_path, 'w') as f:
        json.dump(run_config, f, indent=2)
    print(f"✓ Saved {run_json_path}")
    
    # Save metadata JSON
    metadata = {
        'model_type': model_type,
        'timestamp': datetime.now().isoformat(),
        'config_file': str(args.config),
        'seed': seed,
        'N': config['particles']['N'],
        'T': config['integration']['T'],
        'dt': config['integration']['dt'],
        'frames_saved': len(times),
        'initial_distribution': config['particles']['initial_distribution'],
        'forces_enabled': config['dynamics']['forces']['enabled'],
        'alignment_enabled': config['dynamics']['alignment']['enabled'],
        'ensemble_cases': C,
        'ensemble_outputs': O,
    }
    
    if C == 1 and 'result' in locals() and 'meta' in result and 'force_evals' in result['meta']:
        metadata['force_evaluations'] = result['meta']['force_evals']
        metadata['force_time'] = result['meta']['force_time']
    
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved {metadata_path}")
    
    # Save NPZ
    npz_path = output_dir / 'results.npz'
    np.savez(
        npz_path,
        times=times,
        positions=positions,
        velocities=velocities,
        config=config
    )
    print(f"✓ Saved {npz_path}")
    
    # Print final summary
    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    
    if C > 1:
        print(f"\nEnsemble structure:")
        print(f"  • Total cases: {C}")
        print(f"  • Visualizations generated: {O}")
        print(f"  • Case subfolders: case_001/ ... case_{C:03d}/")
        print(f"\nEach case folder contains:")
        print(f"  • trajectory.csv         - Full trajectory data")
        print(f"  • density.csv            - Density field data")
        
        if O > 0:
            print(f"\nFirst {O} case(s) also contain:")
            print(f"  • order_parameters.csv   - Time series of all 5 metrics")
            print(f"  • order_summary.png      - Summary plots")
            if config['outputs']['animations']:
                print(f"  • traj_animation.mp4     - Particle trajectory video")
                print(f"  • density_animation.mp4  - Density field video")
    else:
        print(f"\nGenerated files:")
        print(f"  • order_parameters.csv   - Time series of all 5 metrics")
        print(f"  • order_summary.png      - Summary plots")
        
        if config['outputs']['save_csv']:
            print(f"  • traj.csv               - Full trajectory data")
            print(f"  • density.csv            - Density field data")
        
        if config['outputs']['animations']:
            print(f"  • traj_animation.mp4     - Particle trajectory video")
            print(f"  • density_animation.mp4  - Density field video")
    
    print(f"\nGlobal metadata:")
    print(f"  • results.npz            - Binary data for analysis")
    print(f"  • config.yaml            - Unified configuration")
    print(f"  • run.json               - Complete running configuration")
    print(f"  • metadata.json          - Run metadata")
    
    if metrics:
        print(f"\nFinal order parameters (t={times[-1]:.1f}):")
        print(f"  • Polarization:       {metrics['polarization'][-1]:.4f}")
        print(f"  • Angular momentum:   {metrics['angular_momentum'][-1]:.4f}")
        print(f"  • Mean speed:         {metrics['mean_speed'][-1]:.4f}")
        print(f"  • Density variance:   {metrics['density_variance'][-1]:.4f}")
        if 'total_mass' in metrics:
            print(f"  • Total mass:         {metrics['total_mass'][-1]:.4f}")
    
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
