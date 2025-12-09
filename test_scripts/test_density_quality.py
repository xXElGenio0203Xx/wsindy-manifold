"""Test density field quality with different parameters."""

import sys
sys.path.insert(0, '/Users/maria_1/Desktop/wsindy-manifold/src')

import numpy as np
import matplotlib.pyplot as plt
from rectsim.density import compute_density_grid
from rectsim import simulate_backend

# Simulate one short run
config = {
    "sim": {"N": 40, "T": 2.0, "dt": 0.1, "save_every": 1, "Lx": 15.0, "Ly": 15.0, "bc": "periodic"},
    "model": {"speed": 1.0},
    "params": {"R": 2.0},
    "noise": {"kind": "gaussian", "eta": 0.3}
}

result = simulate_backend(config, rng=np.random.default_rng(42))
pos_final = result["traj"][-1]  # Take final snapshot

# Test different resolutions and bandwidths
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

test_configs = [
    (16, 0.5, "Current: 16×16, BW=0.5"),
    (16, 2.0, "Better BW: 16×16, BW=2.0"),
    (32, 1.0, "Better Resolution: 32×32, BW=1.0"),
    (64, 2.0, "High Resolution: 64×64, BW=2.0"),
    (32, 2.5, "Smooth: 32×32, BW=2.5"),
    (64, 3.0, "Very Smooth: 64×64, BW=3.0"),
]

for idx, (res, bw, title) in enumerate(test_configs):
    ax = axes[idx // 3, idx % 3]
    
    rho, _, _ = compute_density_grid(
        pos_final, 
        nx=res, ny=res, 
        Lx=config["sim"]["Lx"], 
        Ly=config["sim"]["Ly"],
        bandwidth=bw,
        bc=config["sim"]["bc"]
    )
    
    im = ax.imshow(rho, origin='lower', cmap='hot', interpolation='bilinear')
    ax.set_title(f"{title}\nRange: [{rho.min():.2f}, {rho.max():.2f}]", fontsize=10)
    ax.set_xlabel(f"Nonzero: {(rho > 0.01).sum()}/{rho.size}")
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('density_comparison.png', dpi=150, bbox_inches='tight')
print("\n" + "="*70)
print("DENSITY QUALITY COMPARISON")
print("="*70)
print(f"\nGenerated comparison image: density_comparison.png")
print(f"\nWith N={config['sim']['N']} particles:")
print(f"  • 16×16 grid = {config['sim']['N']/256:.2f} particles per cell")
print(f"  • 32×32 grid = {config['sim']['N']/1024:.2f} particles per cell")
print(f"  • 64×64 grid = {config['sim']['N']/4096:.2f} particles per cell")
print(f"\n⚠️  LOW RESOLUTION ISSUE:")
print(f"  Current 16×16 is too coarse - creates blocky, sparse fields")
print(f"\n✅ RECOMMENDED:")
print(f"  Use at least 32×32 resolution with bandwidth=2.0-2.5")
print(f"  Or 64×64 with bandwidth=2.0-3.0 for smooth continuous heatmaps")
print("="*70)
