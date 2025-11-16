#!/usr/bin/env python
"""
Run a complete Vicsek discrete simulation using the unified backend.

This script demonstrates how to:
1. Load a configuration file
2. Run a simulation with the unified backend
3. Analyze and save results
4. Generate basic visualizations
"""

import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from rectsim.vicsek_discrete import simulate_backend

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def compute_order_parameter(headings):
    """Compute polarization order parameter from headings.
    
    Parameters
    ----------
    headings : ndarray, shape (N, 2)
        Unit heading vectors
        
    Returns
    -------
    float
        Order parameter phi = ||<headings>|| in [0, 1]
    """
    mean_heading = np.mean(headings, axis=0)
    return float(np.linalg.norm(mean_heading))

def save_results(result, output_dir):
    """Save simulation results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save trajectories as NPZ
    np.savez_compressed(
        output_dir / 'results.npz',
        times=result['times'],
        traj=result['traj'],
        vel=result['vel'],
        head=result['head']
    )
    print(f"✓ Saved results.npz")
    
    # Compute and save metrics
    order_series = np.array([compute_order_parameter(result['head'][t]) 
                             for t in range(len(result['times']))])
    mean_speeds = np.array([np.mean(np.linalg.norm(result['vel'][t], axis=1))
                           for t in range(len(result['times']))])
    
    metrics = np.column_stack([result['times'], order_series, mean_speeds])
    np.savetxt(
        output_dir / 'metrics.csv',
        metrics,
        delimiter=',',
        header='time,order_parameter,mean_speed',
        comments=''
    )
    print(f"✓ Saved metrics.csv")
    
    # Save config metadata
    import json
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(result['meta'], f, indent=2)
    print(f"✓ Saved metadata.json")
    
    return order_series, mean_speeds

def plot_results(result, order_series, output_dir):
    """Generate basic plots."""
    output_dir = Path(output_dir)
    
    # Plot 1: Order parameter evolution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(result['times'], order_series, 'b-', linewidth=2)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Order Parameter φ', fontsize=12)
    ax.set_title('Collective Order Evolution', fontsize=14)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_dir / 'order_parameter.png', dpi=150)
    plt.close(fig)
    print(f"✓ Saved order_parameter.png")
    
    # Plot 2: Final state snapshot
    fig, ax = plt.subplots(figsize=(8, 8))
    pos = result['traj'][-1]
    vel = result['vel'][-1]
    
    # Scatter particles
    ax.scatter(pos[:, 0], pos[:, 1], c='steelblue', s=30, alpha=0.6, edgecolors='navy')
    
    # Quiver plot for velocities
    ax.quiver(pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1],
              angles='xy', scale_units='xy', scale=2.5,
              width=0.003, color='red', alpha=0.7)
    
    ax.set_xlim(0, result['meta']['sim']['Lx'])
    ax.set_ylim(0, result['meta']['sim']['Ly'])
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f"Final State (t = {result['times'][-1]:.1f})", fontsize=14)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(output_dir / 'final_state.png', dpi=150)
    plt.close(fig)
    print(f"✓ Saved final_state.png")

def print_summary(result, order_series):
    """Print simulation summary."""
    config = result['meta']
    print("\n" + "="*70)
    print("SIMULATION SUMMARY")
    print("="*70)
    print(f"Model:        {config['model']['type']}")
    print(f"Particles:    {config['sim']['N']}")
    print(f"Domain:       {config['sim']['Lx']} × {config['sim']['Ly']} ({config['sim']['bc']})")
    print(f"Time:         {config['sim']['T']} (dt = {config['sim']['dt']})")
    print(f"Speed:        v0 = {config['model']['speed']}")
    print(f"Radius:       R = {config['params']['R']}")
    print(f"Noise:        {config['noise']['kind']}, η = {config['noise']['eta']}")
    print(f"Constraint:   v0*dt = {config['model']['speed']*config['sim']['dt']:.3f} < 0.5*R = {0.5*config['params']['R']:.3f} ✓")
    print("-"*70)
    print(f"Frames saved: {len(result['times'])}")
    print(f"Initial φ:    {order_series[0]:.4f}")
    print(f"Final φ:      {order_series[-1]:.4f}")
    print(f"Mean φ:       {np.mean(order_series):.4f}")
    print(f"Std φ:        {np.std(order_series):.4f}")
    print("="*70 + "\n")

def main():
    """Main simulation workflow."""
    # Load configuration
    config_path = 'examples/configs/complete_demo.yaml'
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Setup output directory
    output_dir = Path('outputs') / config['outputs']['run_name']
    print(f"Output directory: {output_dir}\n")
    
    # Initialize RNG
    rng = np.random.default_rng(config['sim']['seed'])
    
    # Run simulation
    print("Running simulation...")
    print(f"  N = {config['sim']['N']} particles")
    print(f"  T = {config['sim']['T']} time units")
    print(f"  Saving every {config['sim']['save_every']} steps\n")
    
    result = simulate_backend(config, rng)
    
    print(f"✓ Simulation complete!\n")
    
    # Save results
    print("Saving results...")
    order_series, mean_speeds = save_results(result, output_dir)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_results(result, order_series, output_dir)
    
    # Print summary
    print_summary(result, order_series)
    
    print(f"All outputs saved to: {output_dir}/")
    print("\nFiles created:")
    print(f"  - results.npz           (trajectory data)")
    print(f"  - metrics.csv           (time series)")
    print(f"  - metadata.json         (configuration)")
    print(f"  - order_parameter.png   (evolution plot)")
    print(f"  - final_state.png       (snapshot)")

if __name__ == '__main__':
    main()
