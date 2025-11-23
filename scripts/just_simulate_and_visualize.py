"""
Simple script to simulate and visualize a single trajectory.
No POD, no MVAR - just observe the behavior!

Usage:
    python scripts/just_simulate_and_visualize.py --config configs/interesting_behavior.yaml
    python scripts/just_simulate_and_visualize.py --config configs/strong_clustering.yaml --output custom_name
"""

import argparse
import yaml
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rectsim.vicsek_discrete import simulate_backend
from rectsim.legacy_functions import (
    trajectory_video,
    side_by_side_video,
    kde_density_movie
)


def load_config(config_path: str) -> dict:
    """Load and parse YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def simulate_and_visualize(config_path: str, output_name: str = None):
    """Run simulation and create videos."""
    
    print("="*80)
    print("SIMPLE SIMULATION + VISUALIZATION")
    print("="*80)
    print(f"\n📄 Config: {config_path}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Set output name
    if output_name is None:
        output_name = config.get('sim_name', 'simulation')
    
    output_dir = Path('simulations') / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output: {output_dir}")
    
    # Extract simulation parameters
    sim = config['sim']
    N = sim['N']
    T = sim['T']
    dt = sim['dt']
    Lx = sim['Lx']
    Ly = sim['Ly']
    
    steps = int(T / dt)
    print(f"\n⚙️  Parameters:")
    print(f"   N = {N} particles")
    print(f"   Domain = {Lx}×{Ly}")
    print(f"   T = {T}s, dt = {dt} → {steps} timesteps")
    print(f"   Save every {sim['save_every']} steps")
    
    # Show force parameters if enabled
    if config.get('forces', {}).get('enabled', False):
        forces = config['forces']['params']
        print(f"\n🔧 Morse Forces:")
        print(f"   Ca = {forces['Ca']}, Cr = {forces['Cr']}")
        print(f"   la = {forces['la']}, lr = {forces['lr']}")
        print(f"   mu_t = {forces['mu_t']}")
    
    # Show model parameters
    model = config.get('model', {})
    print(f"\n🎯 Model:")
    print(f"   Speed = {model.get('speed', 0.5)}")
    print(f"   Speed mode = {model.get('speed_mode', 'constant')}")
    
    print(f"\n{'='*80}")
    print("Running Simulation...")
    print(f"{'='*80}\n")
    
    # Run simulation
    rng = np.random.default_rng(sim.get('seed', 42))
    result = simulate_backend(config, rng)
    
    traj = result['traj']      # (T, N, 2)
    times = result['times']    # (T,)
    
    print(f"\n✓ Simulation complete!")
    print(f"   Generated {len(times)} frames")
    print(f"   Shape: {traj.shape}")
    
    # Verify speeds are constant in constant or constant_with_forces modes
    speed_mode = config.get('model', {}).get('speed_mode', 'constant')
    if speed_mode in ['constant', 'constant_with_forces']:
        # Compute speeds from trajectory (with periodic boundary handling)
        speeds_per_frame = []
        for t in range(len(times) - 1):
            dt_actual = times[t+1] - times[t]
            displacements = traj[t+1] - traj[t]
            
            # Handle periodic boundaries
            if config['sim']['bc'] == 'periodic':
                displacements[:, 0] = np.where(
                    displacements[:, 0] > Lx/2, 
                    displacements[:, 0] - Lx, 
                    displacements[:, 0]
                )
                displacements[:, 0] = np.where(
                    displacements[:, 0] < -Lx/2, 
                    displacements[:, 0] + Lx, 
                    displacements[:, 0]
                )
                displacements[:, 1] = np.where(
                    displacements[:, 1] > Ly/2, 
                    displacements[:, 1] - Ly, 
                    displacements[:, 1]
                )
                displacements[:, 1] = np.where(
                    displacements[:, 1] < -Ly/2, 
                    displacements[:, 1] + Ly, 
                    displacements[:, 1]
                )
            
            speeds_t = np.linalg.norm(displacements, axis=1) / dt_actual
            speeds_per_frame.append(speeds_t.mean())
        
        speeds_array = np.array(speeds_per_frame)
        print(f"   Average speed (periodic-corrected): {speeds_array.mean():.3f} ± {speeds_array.std():.3f}")
        print(f"   Expected: {config['model']['speed']:.3f}")
        print(f"   Speed range: [{speeds_array.min():.3f}, {speeds_array.max():.3f}]")
    
    # Save trajectory data
    traj_file = output_dir / "trajectory.npz"
    np.savez(
        traj_file,
        traj=traj,
        times=times,
        vel=result['vel'],
        head=result['head'],
        config=config
    )
    print(f"   Saved: {traj_file}")
    
    print(f"\n{'='*80}")
    print("Computing Density Fields...")
    print(f"{'='*80}\n")
    
    # Compute density (use config value or high-quality default)
    density_res = config.get('outputs', {}).get('density_resolution', 128)
    nx = density_res
    ny = density_res
    
    # Bandwidth in grid units (pixels) - need enough smoothing for clustering visualization
    # For a 128×128 grid, bandwidth of 3-5 pixels gives smooth clusters
    # Scale with resolution: higher resolution needs proportionally larger bandwidth
    bandwidth = density_res / 25.0  # ~5 pixels for 128×128
    
    print(f"   Grid: {nx}×{ny}")
    print(f"   Bandwidth: {bandwidth:.2f} pixels")
    print(f"   Physical smoothing scale: {bandwidth * min(Lx, Ly) / density_res:.2f} units")
    
    density, density_meta = kde_density_movie(
        traj=traj,
        Lx=Lx,
        Ly=Ly,
        nx=nx,
        ny=ny,
        bandwidth=bandwidth,
        bc='periodic'
    )
    
    print(f"   Density shape: {density.shape}")
    
    print(f"\n{'='*80}")
    print("Creating Videos...")
    print(f"{'='*80}\n")
    
    # Create trajectory video (standalone)
    traj_video = output_dir / "trajectory.mp4"
    print(f"🎬 Creating trajectory video...")
    trajectory_video(
        path=output_dir,
        traj=traj,
        times=times,
        Lx=Lx,
        Ly=Ly,
        name="trajectory",
        fps=20,
        marker_size=50,
        title=f"{output_name} - Particle Trajectories"
    )
    print(f"   ✓ Saved: {traj_video}")
    
    # Create density video (standalone) - reuse density as both left and right
    density_video = output_dir / "density.mp4"
    print(f"🎬 Creating density video...")
    # Use side_by_side_video but show the same density on both sides as a workaround
    # Or we can create a simple density-only video
    side_by_side_video(
        path=output_dir,
        left_frames=density,
        right_frames=density,  # Same frames for now
        name="density",
        fps=20,
        cmap='hot',
        titles=('Density Field', 'Density Field')
    )
    print(f"   ✓ Saved: {density_video}")
    
    # Create overlay comparison video
    combined_video = output_dir / "trajectory_with_density.mp4"
    print(f"🎬 Creating trajectory+density overlay video...")
    
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def update_overlay(frame):
        ax.clear()
        # Show density as background
        extent = [0, Lx, 0, Ly]
        ax.imshow(density[frame], extent=extent, origin='lower', cmap='hot', alpha=0.6, vmin=0, vmax=density.max())
        # Overlay particle positions
        ax.scatter(traj[frame, :, 0], traj[frame, :, 1], c='cyan', s=30, alpha=0.8, edgecolors='white', linewidths=0.5)
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{output_name} - Trajectory + Density (t={times[frame]:.2f}s)')
        ax.set_aspect('equal')
    
    anim = FuncAnimation(fig, update_overlay, frames=len(traj), interval=50)
    anim.save(combined_video, writer='ffmpeg', fps=20, dpi=100)
    plt.close()
    print(f"   ✓ Saved: {combined_video}")
    
    print(f"\n{'='*80}")
    print("COMPLETE! 🎉")
    print(f"{'='*80}")
    print(f"\n📂 All outputs in: {output_dir}")
    print(f"   • {traj_video.name}")
    print(f"   • {density_video.name}")
    print(f"   • {combined_video.name}")
    print(f"   • {traj_file.name}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Simulate and visualize a single trajectory"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory name (default: use sim_name from config)'
    )
    
    args = parser.parse_args()
    
    simulate_and_visualize(args.config, args.output)


if __name__ == '__main__':
    main()
