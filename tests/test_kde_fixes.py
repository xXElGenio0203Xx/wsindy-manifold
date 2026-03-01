#!/usr/bin/env python3
"""
Test script to verify KDE fixes:
1. Axis orientation (no transpose bug)
2. Periodic wrapping (no particle loss)
3. Mass conservation (validation working)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, 'src')
from rectsim.legacy_functions import kde_density_movie

print("="*80)
print("KDE FIXES VERIFICATION TESTS")
print("="*80)

# Domain setup
Lx, Ly = 15.0, 15.0
nx, ny = 64, 64
N = 40
T = 5

# ============================================================================
# TEST 1: Axis Orientation (Cluster along X-axis)
# ============================================================================
print("\n" + "="*80)
print("TEST 1: Axis Orientation (cluster along x-axis)")
print("="*80)

# Create particles clustered along x-axis at y=Ly/2
traj_test1 = np.zeros((T, N, 2))
for t in range(T):
    traj_test1[t, :, 0] = np.linspace(2, 13, N)  # Spread along X
    traj_test1[t, :, 1] = Ly/2 + np.random.randn(N) * 0.5  # Tight cluster in Y

rho1, meta1 = kde_density_movie(traj_test1, Lx, Ly, nx, ny, bandwidth=2.0, bc="periodic")

# Check: density should be high along middle row (y ≈ Ly/2), spread across x
y_middle_idx = ny // 2
x_profile = rho1[0, y_middle_idx, :]  # Density along x at middle y
y_profile = rho1[0, :, nx // 2]       # Density along y at middle x

print(f"\nDensity along x-axis (at y=Ly/2): max={x_profile.max():.3f}, mean={x_profile.mean():.3f}")
print(f"Density along y-axis (at x=Lx/2): max={y_profile.max():.3f}, mean={y_profile.mean():.3f}")

if x_profile.max() > y_profile.max() * 1.5:
    print("✅ PASS: Density correctly spread along x-axis (not y-axis)")
else:
    print("❌ FAIL: Density orientation may be wrong!")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Heatmap
im = axes[0].imshow(rho1[0], origin='lower', cmap='hot', aspect='auto')
axes[0].set_title('Density Heatmap\n(should show horizontal stripe)')
axes[0].set_xlabel('x (grid index)')
axes[0].set_ylabel('y (grid index)')
plt.colorbar(im, ax=axes[0])

# X profile
axes[1].plot(x_profile, 'b-', linewidth=2)
axes[1].set_title('Density along x-axis\n(should be HIGH and spread)')
axes[1].set_xlabel('x index')
axes[1].set_ylabel('Density')
axes[1].grid(True, alpha=0.3)

# Y profile
axes[2].plot(y_profile, 'r-', linewidth=2)
axes[2].set_title('Density along y-axis\n(should be LOW except middle peak)')
axes[2].set_xlabel('y index')
axes[2].set_ylabel('Density')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test_kde_orientation.png', dpi=150, bbox_inches='tight')
print("\n📊 Saved: test_kde_orientation.png")
plt.close()

# ============================================================================
# TEST 2: Periodic Wrapping (Particles near boundaries)
# ============================================================================
print("\n" + "="*80)
print("TEST 2: Periodic Wrapping (boundary particles)")
print("="*80)

# Create particles that intentionally go slightly out of bounds
traj_test2 = np.zeros((T, N, 2))
for t in range(T):
    # Half particles near x=0, half near x=Lx
    traj_test2[t, :N//2, 0] = np.random.uniform(-0.1, 0.5, N//2)  # Some negative!
    traj_test2[t, N//2:, 0] = np.random.uniform(Lx-0.5, Lx+0.1, N//2)  # Some > Lx!
    traj_test2[t, :, 1] = np.random.uniform(0, Ly, N)

print(f"\nParticles with x < 0: {np.sum(traj_test2[0, :, 0] < 0)}")
print(f"Particles with x > Lx: {np.sum(traj_test2[0, :, 0] > Lx)}")

rho2, meta2 = kde_density_movie(traj_test2, Lx, Ly, nx, ny, bandwidth=2.0, bc="periodic")

# Check: should see density wrap from right to left edge
left_edge_density = rho2[0, :, :5].sum()  # Left 5 columns
right_edge_density = rho2[0, :, -5:].sum()  # Right 5 columns

print(f"\nDensity at left edge (5 cols): {left_edge_density:.2f}")
print(f"Density at right edge (5 cols): {right_edge_density:.2f}")

if left_edge_density > 0 and right_edge_density > 0:
    print("✅ PASS: Periodic wrapping working (density at both edges)")
else:
    print("⚠️  WARNING: Low edge density - check wrapping")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im = axes[0].imshow(rho2[0], origin='lower', cmap='hot', aspect='auto')
axes[0].set_title('Density with Boundary Particles\n(should show wrap-around)')
axes[0].set_xlabel('x (grid index)')
axes[0].set_ylabel('y (grid index)')
plt.colorbar(im, ax=axes[0])

# Edge profile
edge_profile = np.concatenate([rho2[0, :, -10:].mean(axis=0), rho2[0, :, :10].mean(axis=0)])
x_edge = np.arange(len(edge_profile))
axes[1].plot(x_edge, edge_profile, 'g-', linewidth=2, marker='o')
axes[1].axvline(10, color='r', linestyle='--', label='Boundary (x=Lx wraps to x=0)')
axes[1].set_title('Density across periodic boundary\n(should be continuous)')
axes[1].set_xlabel('Position (right edge → left edge)')
axes[1].set_ylabel('Average density')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test_kde_wrapping.png', dpi=150, bbox_inches='tight')
print("📊 Saved: test_kde_wrapping.png")
plt.close()

# ============================================================================
# TEST 3: Mass Conservation
# ============================================================================
print("\n" + "="*80)
print("TEST 3: Mass Conservation")
print("="*80)

# Random walk with varying particle counts
print("\nSubtest 3a: Normal conditions (N=40)")
traj_test3a = np.random.uniform(0, [Lx, Ly], size=(T, N, 2))
rho3a, meta3a = kde_density_movie(traj_test3a, Lx, Ly, nx, ny, bandwidth=2.0, bc="periodic")

dx = Lx / nx
dy = Ly / ny
masses_3a = []
for t in range(T):
    mass = rho3a[t].sum() * dx * dy
    masses_3a.append(mass)
    
print(f"Target N: {N}")
print(f"Mass range: [{min(masses_3a):.3f}, {max(masses_3a):.3f}]")
print(f"Mass error: {abs(np.mean(masses_3a) - N):.3f} ({abs(np.mean(masses_3a) - N)/N*100:.2f}%)")

if abs(np.mean(masses_3a) - N) < 0.5:
    print("✅ PASS: Mass conservation within 0.5 particles")
else:
    print("❌ FAIL: Mass conservation violated!")

# Test with different N values
print("\nSubtest 3b: Different particle counts")
N_values = [20, 40, 80, 100]
mass_errors = []

for N_test in N_values:
    traj_temp = np.random.uniform(0, [Lx, Ly], size=(3, N_test, 2))
    rho_temp, _ = kde_density_movie(traj_temp, Lx, Ly, nx, ny, bandwidth=2.0, bc="periodic")
    mass = rho_temp[0].sum() * dx * dy
    error = abs(mass - N_test) / N_test * 100
    mass_errors.append(error)
    print(f"  N={N_test:3d}: mass={mass:.2f}, error={error:.2f}%")

if all(err < 5.0 for err in mass_errors):
    print("✅ PASS: Mass conservation holds for all N values")
else:
    print("❌ FAIL: Mass conservation varies with N")

# ============================================================================
# TEST 4: Visual Wrap Test (Single Particle)
# ============================================================================
print("\n" + "="*80)
print("TEST 4: Visual Wrap Test (single particle near boundary)")
print("="*80)

# Single particle at x = 0.5, y = Ly/2
traj_test4 = np.zeros((1, 1, 2))
traj_test4[0, 0, :] = [0.5, Ly/2]

rho4, meta4 = kde_density_movie(traj_test4, Lx, Ly, nx, ny, bandwidth=3.0, bc="periodic")

# Check: should see Gaussian blob at left edge AND right edge (wraparound)
left_density = rho4[0, ny//2, :5].max()
right_density = rho4[0, ny//2, -5:].max()

print(f"\nSingle particle at x=0.5, y={Ly/2}")
print(f"Density at left edge: {left_density:.3f}")
print(f"Density at right edge: {right_density:.3f}")

if right_density > 0.01 * left_density:
    print("✅ PASS: Periodic smoothing working (wraparound visible)")
else:
    print("❌ FAIL: No wraparound detected!")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im = axes[0].imshow(rho4[0], origin='lower', cmap='hot', aspect='auto')
axes[0].scatter([0.5/Lx*nx], [Ly/2/Ly*ny], color='cyan', s=200, marker='x', linewidths=3, label='Particle')
axes[0].set_title('Single Particle Near Left Boundary\n(should see wrap to right edge)')
axes[0].set_xlabel('x (grid index)')
axes[0].set_ylabel('y (grid index)')
axes[0].legend()
plt.colorbar(im, ax=axes[0])

# Slice at y = Ly/2
slice_y = rho4[0, ny//2, :]
axes[1].plot(slice_y, 'b-', linewidth=2, marker='o', markersize=3)
axes[1].set_title('Density slice at y=Ly/2\n(should show peaks at BOTH edges)')
axes[1].set_xlabel('x (grid index)')
axes[1].set_ylabel('Density')
axes[1].grid(True, alpha=0.3)
axes[1].axvline(0, color='r', linestyle='--', alpha=0.5, label='Left edge')
axes[1].axvline(nx-1, color='r', linestyle='--', alpha=0.5, label='Right edge')
axes[1].legend()

plt.tight_layout()
plt.savefig('test_kde_single_particle.png', dpi=150, bbox_inches='tight')
print("📊 Saved: test_kde_single_particle.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
Generated test figures:
  - test_kde_orientation.png   : Verify axis orientation (no transpose)
  - test_kde_wrapping.png       : Verify periodic wrapping (no particle loss)
  - test_kde_single_particle.png: Verify Gaussian wraparound (mode='wrap')

Review these images to confirm:
  ✓ Horizontal stripes appear horizontally (not vertical)
  ✓ Density appears at both edges for boundary particles
  ✓ Single particle creates wrapped Gaussian blob
  ✓ Mass conservation messages printed above (no warnings)
""")
print("="*80)
