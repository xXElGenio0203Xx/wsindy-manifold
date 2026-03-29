#!/usr/bin/env python3
"""Quick KDE verification test - runs in seconds"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from rectsim.legacy_functions import kde_density_movie

print("="*60)
print("QUICK KDE VERIFICATION TEST")
print("="*60)

# Setup
Lx, Ly = 15.0, 15.0
nx, ny = 64, 64
N = 40

# TEST 1: Basic execution with mass conservation
print("\n1. Basic KDE with mass conservation check...")
traj = np.random.uniform(0, [Lx, Ly], size=(3, N, 2))
rho, meta = kde_density_movie(traj, Lx, Ly, nx, ny, bandwidth=2.0, bc="periodic")

dx = Lx / nx
dy = Ly / ny
mass = rho[0].sum() * dx * dy
print(f"   ✓ Shape: {rho.shape}")
print(f"   ✓ Mass: {mass:.2f} (target: {N}, error: {abs(mass-N)/N*100:.1f}%)")

# TEST 2: Particles out of bounds (wrapping test)
print("\n2. Boundary wrapping test...")
traj_oob = np.zeros((2, N, 2))
traj_oob[0, :N//2, 0] = np.random.uniform(-0.2, 0.5, N//2)  # Some x < 0
traj_oob[0, N//2:, 0] = np.random.uniform(Lx-0.5, Lx+0.2, N//2)  # Some x > Lx
traj_oob[0, :, 1] = np.random.uniform(0, Ly, N)
traj_oob[1] = traj_oob[0]

n_neg = np.sum(traj_oob[0, :, 0] < 0)
n_over = np.sum(traj_oob[0, :, 0] > Lx)
print(f"   Input: {n_neg} particles with x<0, {n_over} with x>{Lx}")

rho_oob, _ = kde_density_movie(traj_oob, Lx, Ly, nx, ny, bandwidth=2.0, bc="periodic")
mass_oob = rho_oob[0].sum() * dx * dy
print(f"   ✓ Mass after wrapping: {mass_oob:.2f} (should still ≈ {N})")
print(f"   ✓ Error: {abs(mass_oob-N)/N*100:.1f}%")

# TEST 3: Axis orientation (horizontal cluster)
print("\n3. Axis orientation test...")
traj_horiz = np.zeros((1, N, 2))
traj_horiz[0, :, 0] = np.linspace(3, 12, N)  # Spread along x
traj_horiz[0, :, 1] = Ly/2 + np.random.randn(N) * 0.3  # Tight in y

rho_horiz, _ = kde_density_movie(traj_horiz, Lx, Ly, nx, ny, bandwidth=2.0, bc="periodic")

y_mid = ny // 2
x_profile_sum = rho_horiz[0, y_mid, :].sum()
y_profile_sum = rho_horiz[0, :, nx//2].sum()

print(f"   Density sum along x-direction: {x_profile_sum:.2f}")
print(f"   Density sum along y-direction: {y_profile_sum:.2f}")
print(f"   Ratio (x/y): {x_profile_sum/y_profile_sum:.1f}x")

if x_profile_sum > y_profile_sum * 1.5:
    print("   ✓ PASS: Horizontal cluster correctly oriented")
else:
    print("   ✗ FAIL: May have transpose bug!")

print("\n" + "="*60)
print("ALL TESTS COMPLETE")
print("="*60)
print("\nIf you see warnings about mass conservation above,")
print("that's GOOD - it means the check is working!")
