import sys
sys.path.insert(0, '/Users/maria_1/Desktop/wsindy-manifold/src')

import numpy as np
import matplotlib.pyplot as plt
from wsindy_manifold.density import kde_density_movie

# Create test trajectory
np.random.seed(42)
T, N = 20, 40
Lx, Ly = 15.0, 15.0

# Simulate particles moving
traj = np.zeros((T, N, 2))
pos = np.random.uniform(0, [Lx, Ly], size=(N, 2))
vel = np.random.randn(N, 2) * 0.5

for t in range(T):
    pos = pos + vel
    pos = pos % [Lx, Ly]  # periodic wrap
    traj[t] = pos

print("="*70)
print("COMPARING OLD vs CURRENT DENSITY SETTINGS")
print("="*70)

# OLD STYLE: High resolution, larger bandwidth
rho_old, meta_old = kde_density_movie(
    traj, Lx=Lx, Ly=Ly, 
    nx=64, ny=64, 
    bandwidth=2.0, 
    bc="periodic"
)

# CURRENT STYLE: Low resolution, small bandwidth
from rectsim.density import density_movie_kde
rho_current = density_movie_kde(
    traj, Lx=Lx, Ly=Ly,
    nx=16, ny=16,
    bandwidth=0.5,
    bc="periodic"
)

# Plot comparison
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, frame_idx in enumerate([0, T//2, T-1]):
    # Old style
    ax = axes[0, i]
    im = ax.imshow(rho_old[frame_idx], origin='lower', cmap='hot', interpolation='bilinear')
    ax.set_title(f'OLD Style (64×64, BW=2.0)\nFrame {frame_idx}', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Current style
    ax = axes[1, i]
    im = ax.imshow(rho_current[frame_idx], origin='lower', cmap='hot', interpolation='bilinear')
    ax.set_title(f'CURRENT Style (16×16, BW=0.5)\nFrame {frame_idx}', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig('density_old_vs_current.png', dpi=150, bbox_inches='tight')

print(f"\n📊 OLD PARAMETERS (what you used to have):")
print(f"   Resolution: {meta_old['nx']}×{meta_old['ny']} = {meta_old['nx']*meta_old['ny']:,} cells")
print(f"   Bandwidth: {meta_old['bandwidth']}")
print(f"   Particles/cell: {N/(meta_old['nx']*meta_old['ny']):.3f}")
print(f"   Result: SMOOTH CONTINUOUS HEATMAP ✅")

print(f"\n⚠️  CURRENT PARAMETERS (what you have now):")
print(f"   Resolution: 16×16 = 256 cells")
print(f"   Bandwidth: 0.5")
print(f"   Particles/cell: {N/256:.3f}")
print(f"   Result: BLOCKY SPARSE FIELD ❌")

print(f"\n💡 SOLUTION:")
print(f"   Change resolution from 16×16 to 64×64")
print(f"   Change bandwidth from 0.5 to 2.0-2.5")
print(f"   This will give you smooth continuous heatmaps like before!")

print(f"\n✅ Saved comparison: density_old_vs_current.png")
print("="*70)
