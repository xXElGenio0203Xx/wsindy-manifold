#!/usr/bin/env python3
"""
Quick test of legacy functions to ensure they work correctly.
"""
import numpy as np
from pathlib import Path
from rectsim.legacy_functions import (
    kde_density_movie,
    save_video,
    side_by_side_video,
    compute_order_params
)

print("Testing legacy functions...")

# Test 1: KDE density movie
print("\n1. Testing kde_density_movie...")
traj = np.random.rand(20, 30, 2) * 15.0  # 20 frames, 30 particles, domain [0,15]
rho, meta = kde_density_movie(traj, Lx=15.0, Ly=15.0, nx=64, ny=64, bandwidth=2.0, bc='periodic')
print(f"  ✓ Density shape: {rho.shape}")
print(f"  ✓ Metadata keys: {list(meta.keys())}")
print(f"  ✓ Grid extent: {meta['extent']}")
assert rho.shape == (20, 64, 64), f"Wrong shape: {rho.shape}"
assert 'extent' in meta, "Missing extent in metadata"
assert 'bandwidth' in meta, "Missing bandwidth in metadata"

# Test 2: Order parameters
print("\n2. Testing order parameters...")
vel = np.random.randn(30, 2)  # 30 particles, 2D velocities
params = compute_order_params(vel, include_nematic=True)
print(f"  ✓ Order params: {params}")
assert 'phi' in params, "Missing polarization"
assert 'mean_speed' in params, "Missing mean_speed"
assert 'nematic' in params, "Missing nematic"

# Test 3: Save single video
print("\n3. Testing save_video...")
output_dir = Path("outputs/legacy_test")
output_dir.mkdir(parents=True, exist_ok=True)

# Create dummy frames
frames_single = np.random.rand(15, 32, 32)
save_video(
    path=output_dir,
    frames=frames_single,
    fps=10,
    name="test_single",
    cmap='hot',
    title="Test Single Video"
)
assert (output_dir / "test_single.mp4").exists(), "Video not created"
print(f"  ✓ Created: {output_dir / 'test_single.mp4'}")

# Test 4: Side-by-side comparison video
print("\n4. Testing side_by_side_video...")
left_frames = np.random.rand(15, 32, 32)
right_frames = np.random.rand(15, 32, 32) * 0.9  # Slightly different
error_timeseries = np.random.rand(15) * 0.1  # Error values

side_by_side_video(
    path=output_dir,
    left_frames=left_frames,
    right_frames=right_frames,
    lower_strip_timeseries=error_timeseries,
    name="test_comparison",
    fps=10,
    cmap='viridis',
    titles=('Left Panel', 'Right Panel')
)
assert (output_dir / "test_comparison.mp4").exists(), "Comparison video not created"
print(f"  ✓ Created: {output_dir / 'test_comparison.mp4'}")

print("\n" + "="*60)
print("ALL TESTS PASSED! ✅")
print("="*60)
print(f"\nTest videos saved to: {output_dir.absolute()}")
print("\nLegacy functions are working correctly and ready to use in production pipeline.")
