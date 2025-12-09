#!/usr/bin/env python
"""
Generate animations from existing simulation results.

This script creates trajectory and density animations from results.npz files.
Useful when animations failed during simulation or when ffmpeg wasn't available.
"""

import numpy as np
from pathlib import Path
import sys


def create_animations_from_results(results_path, config=None):
    """
    Create animations from existing results.npz file.
    
    Parameters
    ----------
    results_path : str or Path
        Path to results.npz file
    config : dict, optional
        Configuration with fps, density_resolution, etc.
        If None, uses defaults.
    """
    from rectsim.io_outputs import create_traj_animation, create_density_animation
    
    results_path = Path(results_path)
    output_dir = results_path.parent
    
    print(f"Loading results from {results_path}...")
    data = np.load(results_path)
    
    times = data['times']
    positions = data['traj']
    velocities = data['vel']
    
    T, N, _ = positions.shape
    print(f"  Loaded: {T} frames, {N} particles")
    
    # Extract configuration or use defaults
    if config is None:
        config = {
            'fps': 20,
            'density_resolution': 100
        }
    
    fps = config.get('fps', 20)
    resolution = config.get('density_resolution', 100)
    
    # Infer domain bounds from data
    x_min, x_max = positions[:, :, 0].min(), positions[:, :, 0].max()
    y_min, y_max = positions[:, :, 1].min(), positions[:, :, 1].max()
    
    # Add 10% padding
    x_pad = 0.1 * (x_max - x_min)
    y_pad = 0.1 * (y_max - y_min)
    domain_bounds = (x_min - x_pad, x_max + x_pad, 
                    y_min - y_pad, y_max + y_pad)
    
    print(f"\nDomain bounds: ({domain_bounds[0]:.1f}, {domain_bounds[1]:.1f}) × ({domain_bounds[2]:.1f}, {domain_bounds[3]:.1f})")
    print(f"Animation settings: {fps} fps, {resolution}×{resolution} grid\n")
    
    # Create trajectory animation
    traj_output = output_dir / 'traj_animation.mp4'
    print(f"Creating trajectory animation...")
    print(f"  Output: {traj_output}")
    try:
        create_traj_animation(
            times, positions, velocities, domain_bounds,
            traj_output, fps=fps, figsize=(8, 8), dpi=100
        )
        print(f"  ✓ Created {traj_output}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Create density animation
    density_output = output_dir / 'density_animation.mp4'
    print(f"\nCreating density animation...")
    print(f"  Output: {density_output}")
    try:
        create_density_animation(
            times, positions, domain_bounds, resolution,
            density_output, fps=fps, figsize=(8, 8), dpi=100
        )
        print(f"  ✓ Created {density_output}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    print("\n✓ Animation generation complete")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python create_animations.py <path/to/results.npz> [fps] [resolution]")
        print("\nExample:")
        print("  python create_animations.py outputs/my_run/results.npz")
        print("  python create_animations.py outputs/my_run/results.npz 30 150")
        sys.exit(1)
    
    results_path = sys.argv[1]
    
    config = {}
    if len(sys.argv) > 2:
        config['fps'] = int(sys.argv[2])
    if len(sys.argv) > 3:
        config['density_resolution'] = int(sys.argv[3])
    
    create_animations_from_results(results_path, config)
