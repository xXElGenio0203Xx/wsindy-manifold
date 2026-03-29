#!/usr/bin/env python3
"""Run D'Orsogna continuous model with Morse forces and standardized outputs.

This script runs the continuous force-based model (dynamics.py) instead of the
discrete Vicsek model. It includes Morse potential forces and optional Vicsek
alignment, saving all standardized outputs (metrics, plots, animations, metadata).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rectsim.dynamics import simulate_backend
from rectsim.io_outputs import save_standardized_outputs


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config: dict) -> None:
    """Validate configuration has required sections."""
    required = ['model', 'sim', 'params', 'outputs']
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Config missing required sections: {missing}")
    
    # Check model type
    if config['model']['type'] != 'dorsogna':
        raise ValueError(f"This script is for 'dorsogna' model, got: {config['model']['type']}")
    
    # Validate sim parameters
    sim = config['sim']
    required_sim = ['N', 'T', 'dt', 'Lx', 'Ly', 'bc', 'save_every', 'neighbor_rebuild']
    missing_sim = [key for key in required_sim if key not in sim]
    if missing_sim:
        raise ValueError(f"sim section missing: {missing_sim}")
    
    # Validate params
    params = config['params']
    required_params = ['alpha', 'beta', 'Cr', 'Ca', 'lr', 'la', 'rcut_factor']
    missing_params = [key for key in required_params if key not in params]
    if missing_params:
        raise ValueError(f"params section missing: {missing_params}")


def main():
    parser = argparse.ArgumentParser(
        description='Run D\'Orsogna continuous model with Morse forces'
    )
    parser.add_argument(
        'config',
        type=Path,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=Path,
        default=None,
        help='Override output directory (default from config)'
    )
    args = parser.parse_args()
    
    # Load and validate config
    print(f"Loading config: {args.config}")
    config = load_config(args.config)
    validate_config(config)
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = Path(config['outputs']['save_dir'])
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Print configuration summary
    sim = config['sim']
    params = config['params']
    print("\n=== Simulation Configuration ===")
    print(f"Model: D'Orsogna (continuous, force-based)")
    print(f"Particles: {sim['N']}")
    print(f"Domain: {sim['Lx']} × {sim['Ly']} ({sim['bc']} BC)")
    print(f"Time: T={sim['T']}, dt={sim['dt']}")
    print(f"Integrator: {sim.get('integrator', 'euler_semiimplicit')}")
    print(f"\nForce parameters:")
    print(f"  α (speed): {params['alpha']}")
    print(f"  β (damping): {params['beta']}")
    print(f"  Cr (repulsion): {params['Cr']}")
    print(f"  Ca (attraction): {params['Ca']}")
    print(f"  lr (repulsive length): {params['lr']}")
    print(f"  la (attractive length): {params['la']}")
    
    # Check for alignment
    if 'alignment' in params and params['alignment'].get('enabled', False):
        align = params['alignment']
        print(f"\nVicsek alignment:")
        print(f"  radius: {align.get('radius', 1.5)}")
        print(f"  rate: {align.get('rate', 0.0)}")
        print(f"  Dθ: {align.get('Dtheta', 0.0)}")
    
    print("\n=== Running Simulation ===")
    
    # Run simulation using standardized backend
    seed = sim.get('seed', 42)
    rng = np.random.default_rng(seed)
    results = simulate_backend(config, rng)
    
    # Extract results from standardized format
    traj = results['traj']
    vel = results['vel']
    times = results['times']
    meta = results['meta']
    
    T_frames = traj.shape[0]
    N = traj.shape[1]
    
    print(f"\n✓ Simulation complete: {T_frames} frames, {N} particles")
    print(f"  Force evaluations: {meta['force_evals']}")
    print(f"  Force computation time: {meta['force_time']:.2f}s")
    
    # Save results.npz
    results_path = output_dir / 'results.npz'
    np.savez_compressed(
        results_path,
        traj=traj,
        vel=vel,
        times=times,
        Lx=sim['Lx'],
        Ly=sim['Ly'],
        bc=sim['bc']
    )
    print(f"\n✓ Saved trajectory: {results_path}")
    
    # Save metadata.json
    metadata = {
        'model': config['model'],
        'sim': config['sim'],
        'params': config['params'],
        'outputs': config['outputs'],
        'simulation_info': {
            'total_frames': int(T_frames),
            'total_particles': int(N),
            'simulation_time': float(sim['T']),
            'timestep': float(sim['dt']),
            'domain_size': [float(sim['Lx']), float(sim['Ly'])],
            'boundary_condition': sim['bc'],
            'integrator': sim.get('integrator', 'euler_semiimplicit'),
            'force_evaluations': int(results['force_evals']),
            'force_computation_time_s': float(results['force_time'])
        }
    }
    
    if 'seed' in config:
        metadata['seed'] = config['seed']
    
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_path}")
    
    # Compute and save standardized outputs
    print("\n=== Computing Standardized Outputs ===")
    
    outputs_config = config['outputs']
    domain_bounds = (0, sim['Lx'], 0, sim['Ly'])
    
    save_standardized_outputs(
        times=times,
        positions=traj,
        velocities=vel,
        domain_bounds=domain_bounds,
        output_dir=output_dir,
        config_outputs=outputs_config
    )
    
    print("\n=== Complete! ===")
    print(f"All outputs saved to: {output_dir}")
    
    # List output files
    print("\nGenerated files:")
    for file in sorted(output_dir.glob('*')):
        size = file.stat().st_size
        size_str = f"{size/1024:.1f}KB" if size < 1024*1024 else f"{size/(1024**2):.1f}MB"
        print(f"  {file.name:30s} ({size_str})")


if __name__ == '__main__':
    main()
