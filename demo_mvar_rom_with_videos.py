#!/usr/bin/env python3
"""
Comprehensive MVAR-ROM demonstration with video generation.
Shows the complete pipeline: simulation → POD → MVAR → forecast → videos
"""
import numpy as np
from pathlib import Path
from wsindy_manifold.mvar_rom import run_mvar_rom_evaluation, MVARROMConfig

def generate_synthetic_swarm(T=600, nx=50, ny=50, n_particles=3):
    """Generate synthetic density evolution with moving particle-like blobs."""
    print(f"Generating synthetic swarm: T={T}, grid={nx}×{ny}, n_particles={n_particles}")
    
    np.random.seed(42)
    densities = np.zeros((T, nx, ny))
    
    # Generate particle trajectories
    particles = []
    for i in range(n_particles):
        # Random initial position
        x0 = np.random.uniform(10, nx-10)
        y0 = np.random.uniform(10, ny-10)
        
        # Random motion parameters
        vx = np.random.uniform(-0.3, 0.3)
        vy = np.random.uniform(-0.3, 0.3)
        omega = np.random.uniform(-0.05, 0.05)  # Rotation
        
        particles.append({
            'x0': x0, 'y0': y0,
            'vx': vx, 'vy': vy,
            'omega': omega,
            'sigma': np.random.uniform(2, 4)  # Width
        })
    
    # Generate density fields
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    
    for t in range(T):
        for p in particles:
            # Position with periodic boundary conditions
            x = (p['x0'] + p['vx']*t + 5*np.cos(p['omega']*t)) % nx
            y = (p['y0'] + p['vy']*t + 5*np.sin(p['omega']*t)) % ny
            
            # Gaussian blob
            densities[t] += np.exp(-((xx-x)**2 + (yy-y)**2) / (2*p['sigma']**2))
        
        # Add small diffusion noise
        densities[t] += 0.01 * np.random.randn(nx, ny)
        densities[t] = np.clip(densities[t], 0, None)
    
    print(f"✓ Generated density range: [{densities.min():.3f}, {densities.max():.3f}]")
    return densities

def main():
    print("="*80)
    print("MVAR-ROM Demo with Video Generation")
    print("="*80)
    
    # Generate synthetic data
    T = 600
    nx, ny = 50, 50
    densities = generate_synthetic_swarm(T=T, nx=nx, ny=ny, n_particles=3)
    
    # Configure evaluation
    config = MVARROMConfig(
        pod_energy=0.99,
        mvar_order=4,
        ridge=1e-6,
        train_frac=0.8,
        tolerance_threshold=0.10,
        save_snapshots=True,
        save_videos=True,  # Enable video generation
        fps=30,  # High framerate for smooth visualization
        output_dir=Path("outputs/demo_with_videos")
    )
    
    print("\nRunning MVAR-ROM evaluation pipeline...")
    print("-" * 80)
    
    # Run evaluation
    results = run_mvar_rom_evaluation(densities, nx, ny, config)
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    summary = results['summary']
    print(f"\n📊 Forecast Quality:")
    print(f"   R² Score:           {summary['r2']:.4f}")
    print(f"   Median L² Error:    {summary['median_e2']:.4f}")
    print(f"   Tolerance Horizon:  {summary['tau_tol']} frames")
    
    print(f"\n📐 Dimensionality:")
    print(f"   Original space:     {nx*ny} cells")
    print(f"   POD latent space:   {results['Ud'].shape[1]} modes")
    print(f"   Compression ratio:  {(nx*ny)/results['Ud'].shape[1]:.1f}×")
    
    print(f"\n🎬 Video Outputs:")
    videos_dir = Path(config.output_dir) / f"mvar_w{config.mvar_order}_lam{config.ridge:.0e}" / "videos"
    for video in videos_dir.glob("*.mp4"):
        size_mb = video.stat().st_size / (1024**2)
        print(f"   ✓ {video.name:<35} ({size_mb:.2f} MB)")
    
    print(f"\n📁 Full results saved to:")
    print(f"   {config.output_dir}/")
    print("\n" + "="*80)
    print("✓ Demo complete!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
