#!/usr/bin/env python3
"""Run discrete Vicsek model (with optional D'Orsogna forces) and standardized outputs.

This script runs the discrete-time Vicsek model (vicsek_discrete.py) with optional
Morse forces enabled. It saves all standardized outputs: metrics CSV, animations,
plots, and trajectory data.

Examples
--------
Pure Vicsek (no forces):
    python scripts/run_vicsek_discrete.py examples/configs/vicsek_pure.yaml

Hybrid Vicsek-D'Orsogna (with forces):
    python scripts/run_vicsek_discrete.py examples/configs/vicsek_dorsogna_discrete.yaml
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

from rectsim.vicsek_discrete import simulate_backend
from rectsim.io_outputs import save_standardized_outputs


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config: dict) -> None:
    """Validate configuration has required sections."""
    required = ['sim', 'model', 'params', 'noise', 'outputs']
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Config missing required sections: {missing}")
    
    # Validate sim parameters
    sim = config['sim']
    required_sim = ['N', 'T', 'dt', 'Lx', 'Ly', 'bc', 'save_every', 'neighbor_rebuild']
    missing_sim = [key for key in required_sim if key not in sim]
    if missing_sim:
        raise ValueError(f"sim section missing: {missing_sim}")
    
    # Validate model
    if 'speed' not in config['model']:
        raise ValueError("model section missing 'speed' (v0)")
    
    # Validate params
    if 'R' not in config['params']:
        raise ValueError("params section missing 'R' (alignment radius)")
    
    # Validate noise
    noise = config['noise']
    if 'kind' not in noise:
        raise ValueError("noise section missing 'kind' (gaussian or uniform)")
    
    if noise['kind'] == 'gaussian' and 'eta' not in noise:
        raise ValueError("Gaussian noise requires 'eta' parameter")
    elif noise['kind'] == 'uniform' and 'eta' not in noise:
        raise ValueError("Uniform noise requires 'eta' parameter")


def main():
    parser = argparse.ArgumentParser(
        description='Run discrete Vicsek model with optional D\'Orsogna forces'
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
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Override random seed'
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
        output_dir = Path(config['outputs'].get('directory', 'outputs/vicsek_discrete_run'))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Override seed if provided
    if args.seed is not None:
        config['sim']['seed'] = args.seed
    
    # Print configuration summary
    print("\n" + "="*70)
    print("DISCRETE VICSEK SIMULATION")
    print("="*70)
    print(f"\nModel: Discrete Vicsek")
    print(f"  • N particles: {config['sim']['N']}")
    print(f"  • Domain: {config['sim']['Lx']} × {config['sim']['Ly']}")
    print(f"  • Boundary: {config['sim']['bc']}")
    print(f"  • Speed v₀: {config['model']['speed']}")
    print(f"  • Alignment radius R: {config['params']['R']}")
    print(f"  • Noise: {config['noise']['kind']} (η={config['noise'].get('eta', 'N/A')})")
    
    # Check if forces enabled
    forces_enabled = config.get('forces', {}).get('enabled', False)
    if forces_enabled:
        print(f"\nD'Orsogna Forces: ENABLED")
        fparams = config['forces']['params']
        print(f"  • Repulsion: Cr={fparams['Cr']}, lr={fparams['lr']}")
        print(f"  • Attraction: Ca={fparams['Ca']}, la={fparams['la']}")
        print(f"  • Coupling: μₜ={fparams['mu_t']}")
        print(f"  • Cutoff: {fparams.get('rcut_factor', 5.0)} × max(lr, la)")
    else:
        print(f"\nD'Orsogna Forces: DISABLED (pure Vicsek)")
    
    print(f"\nIntegration:")
    print(f"  • Time: 0 → {config['sim']['T']}")
    print(f"  • dt: {config['sim']['dt']}")
    print(f"  • Save every: {config['sim']['save_every']} steps")
    print(f"  • Neighbor rebuild: every {config['sim']['neighbor_rebuild']} steps")
    
    print(f"\nOutput: {output_dir}")
    print("="*70 + "\n")
    
    # Run simulation
    print("Running simulation...")
    seed = config['sim'].get('seed', 42)
    rng = np.random.default_rng(seed)
    
    result = simulate_backend(config, rng)
    
    print(f"✓ Simulation complete")
    print(f"  • Frames saved: {len(result['times'])}")
    print(f"  • Particles: {result['traj'].shape[1]}")
    
    # Extract data
    times = result['times']
    positions = result['traj']
    velocities = result['vel']
    
    # Domain bounds
    Lx = config['sim']['Lx']
    Ly = config['sim']['Ly']
    domain_bounds = (0, Lx, 0, Ly)
    
    # Save standardized outputs
    print("\n" + "-"*70)
    print("Generating outputs...")
    print("-"*70)
    
    config_outputs = config['outputs']
    metrics = save_standardized_outputs(
        times, positions, velocities, domain_bounds,
        output_dir, config_outputs
    )
    
    # Save configuration and metadata
    print("\nSaving metadata...")
    
    # Save config
    config_path = output_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"✓ Saved {config_path}")
    
    # Save metadata JSON
    metadata = {
        'model': 'vicsek_discrete',
        'forces_enabled': forces_enabled,
        'timestamp': datetime.now().isoformat(),
        'config_file': str(args.config),
        'seed': seed,
        'N': config['sim']['N'],
        'T': config['sim']['T'],
        'dt': config['sim']['dt'],
        'frames_saved': len(times),
        'output_directory': str(output_dir)
    }
    
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved {metadata_path}")
    
    # Save NPZ for easy reload
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
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  • order_parameters.csv   - Time series of metrics")
    print(f"  • order_summary.png      - Summary plots")
    print(f"  • traj.csv               - Trajectory data")
    print(f"  • density.csv            - Density field data")
    
    if config_outputs.get('animations', True):
        print(f"  • traj_animation.mp4     - Particle trajectory video")
        print(f"  • density_animation.mp4  - Density field video")
    
    print(f"  • results.npz            - Binary data for analysis")
    print(f"  • config.yaml            - Configuration used")
    print(f"  • metadata.json          - Run metadata")
    
    if metrics:
        print(f"\nFinal order parameters (t={times[-1]:.1f}):")
        print(f"  • Polarization:       {metrics['polarization'][-1]:.4f}")
        print(f"  • Angular momentum:   {metrics['angular_momentum'][-1]:.4f}")
        print(f"  • Mean speed:         {metrics['mean_speed'][-1]:.4f}")
        print(f"  • Density variance:   {metrics['density_variance'][-1]:.4f}")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
