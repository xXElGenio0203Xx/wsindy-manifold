#!/usr/bin/env python
"""
Run simulation with standardized outputs.

This script demonstrates the new unified output system that works across
all model types (discrete, continuous, force-coupled).
"""

import numpy as np
import yaml
from pathlib import Path
from rectsim.vicsek_discrete import simulate_backend
from rectsim.io_outputs import save_standardized_outputs


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_simulation_with_outputs(config_path):
    """
    Run simulation and generate all standardized outputs.
    
    Parameters
    ----------
    config_path : str or Path
        Path to YAML configuration file
    """
    # Load configuration
    print(f"Loading configuration from {config_path}...")
    config = load_config(config_path)
    
    # Extract parameters
    run_name = config['outputs'].get('run_name', 'run')
    output_dir = Path('outputs') / run_name
    
    # Print simulation info
    print("\n" + "="*60)
    print("SIMULATION CONFIGURATION")
    print("="*60)
    print(f"Model type:       {config['model']['type']}")
    print(f"Particles:        {config['sim']['N']}")
    print(f"Domain:           {config['sim']['Lx']} × {config['sim']['Ly']}")
    print(f"Boundary:         {config['sim']['bc']}")
    print(f"Time:             0 → {config['sim']['T']} (dt = {config['sim']['dt']})")
    print(f"Speed:            v0 = {config['model']['speed']}")
    print(f"Radius:           R = {config['params']['R']}")
    print(f"Noise:            {config['noise']['kind']}, η = {config['noise']['eta']}")
    print(f"Output dir:       {output_dir}")
    print("="*60 + "\n")
    
    # Validate constraint
    v0 = config['model']['speed']
    dt = config['sim']['dt']
    R = config['params']['R']
    constraint_value = v0 * dt
    constraint_limit = 0.5 * R
    
    if constraint_value > constraint_limit:
        print(f"⚠️  WARNING: Constraint violated!")
        print(f"   v0*dt = {constraint_value:.3f} > 0.5*R = {constraint_limit:.3f}")
        print(f"   Consider reducing dt or increasing R.\n")
    else:
        print(f"✓ Constraint satisfied: v0*dt = {constraint_value:.3f} < 0.5*R = {constraint_limit:.3f}\n")
    
    # Run simulation
    print("Running simulation...")
    rng = np.random.default_rng(config['sim']['seed'])
    result = simulate_backend(config, rng)
    
    times = result['times']
    positions = result['traj']
    velocities = result['vel']
    
    T_frames, N, _ = positions.shape
    print(f"✓ Simulation complete: {T_frames} frames, {N} particles\n")
    
    # Define domain bounds
    domain_bounds = (0, config['sim']['Lx'], 0, config['sim']['Ly'])
    
    # Save basic results first
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / 'results.npz',
        times=times,
        traj=positions,
        vel=velocities,
        head=result['head']
    )
    print(f"✓ Saved results.npz")
    
    # Save metadata.json with complete configuration
    import json
    metadata = {
        'model': config['model'],
        'sim': config['sim'],
        'params': config['params'],
        'noise': config['noise'],
        'forces': config['forces'],
        'outputs': config['outputs'],
        'simulation_info': {
            'total_frames': int(T_frames),
            'total_particles': int(N),
            'simulation_time': float(config['sim']['T']),
            'timestep': float(config['sim']['dt']),
            'domain_size': [float(config['sim']['Lx']), float(config['sim']['Ly'])],
            'boundary_condition': config['sim']['bc']
        }
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata.json\n")
    
    # Generate standardized outputs
    print("="*60)
    print("GENERATING STANDARDIZED OUTPUTS")
    print("="*60)
    
    config_outputs = {
        'order_parameters': config['outputs'].get('order_parameters', True),
        'animations': config['outputs'].get('animations', True),
        'save_csv': config['outputs'].get('save_csv', True),
        'fps': config['outputs'].get('fps', 20),
        'density_resolution': config['outputs'].get('density_resolution', 100)
    }
    
    metrics = save_standardized_outputs(
        times, positions, velocities, domain_bounds,
        output_dir, config_outputs
    )
    
    # Print summary
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    
    if metrics is not None:
        print("\nOrder Parameters:")
        print(f"  Polarization Φ:")
        print(f"    Initial: {metrics['polarization'][0]:.4f}")
        print(f"    Final:   {metrics['polarization'][-1]:.4f}")
        print(f"    Mean:    {np.mean(metrics['polarization']):.4f}")
        print(f"    Std:     {np.std(metrics['polarization']):.4f}")
        
        print(f"\n  Angular Momentum L:")
        print(f"    Initial: {metrics['angular_momentum'][0]:.4f}")
        print(f"    Final:   {metrics['angular_momentum'][-1]:.4f}")
        print(f"    Mean:    {np.mean(metrics['angular_momentum']):.4f}")
        print(f"    Std:     {np.std(metrics['angular_momentum']):.4f}")
        
        print(f"\n  Mean Speed:")
        print(f"    Initial: {metrics['mean_speed'][0]:.4f}")
        print(f"    Final:   {metrics['mean_speed'][-1]:.4f}")
        print(f"    Mean:    {np.mean(metrics['mean_speed']):.4f}")
        print(f"    Std:     {np.std(metrics['mean_speed']):.4f}")
        
        print(f"\n  Density Variance:")
        print(f"    Initial: {metrics['density_variance'][0]:.6f}")
        print(f"    Final:   {metrics['density_variance'][-1]:.6f}")
        print(f"    Mean:    {np.mean(metrics['density_variance']):.6f}")
        print(f"    Std:     {np.std(metrics['density_variance']):.6f}")
    
    print("\n" + "="*60)
    print("OUTPUT FILES")
    print("="*60)
    
    # List all output files
    if output_dir.exists():
        files = sorted(output_dir.iterdir())
        total_size = 0
        for f in files:
            if f.is_file():
                size = f.stat().st_size
                total_size += size
                size_str = f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/1024/1024:.1f} MB"
                print(f"  {f.name:<30} {size_str:>10}")
        
        print(f"\n  Total: {total_size/1024/1024:.1f} MB")
    
    print("\n" + "="*60)
    print("✓ COMPLETE")
    print("="*60)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = 'examples/configs/standardized_demo.yaml'
    
    run_simulation_with_outputs(config_file)
