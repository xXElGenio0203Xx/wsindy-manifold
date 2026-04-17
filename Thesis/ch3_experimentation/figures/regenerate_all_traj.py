#!/usr/bin/env python3
"""Regenerate ALL trajectory panels for Figure 2.1 with larger heading arrows.

Produces traj_DYN{1,4,6,7}_t{0,15,38}.png  (12 files total).
Arrow length increased ~3x vs original to remain visible at 0.30\\textwidth print size.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rectsim.config import load_config
from rectsim.vicsek_discrete import simulate_backend

# Config mapping: (DYN label, config path)
PANELS = [
    ('DYN1', 'configs/DYN1_gentle.yaml'),
    ('DYN4', 'configs/DYN4_blackhole.yaml'),
    ('DYN7', 'configs/DYN7_pure_vicsek.yaml'),
    ('DYN6', 'configs/DYN6_varspeed.yaml'),
]

TARGET_TIMES = [1.2, 6.0, 15.0]
FRAME_LABELS = ['t0', 't15', 't38']
OUT_DIR = os.path.dirname(__file__)

for dyn_label, cfg_path in PANELS:
    print(f"\n=== {dyn_label} ({cfg_path}) ===")
    cfg = load_config(cfg_path)
    rng = np.random.default_rng(42)
    result = simulate_backend(cfg, rng)
    times = result['times']
    traj = result['traj']   # (T, N, 2)
    vel = result['vel']     # (T, N, 2)
    N = traj.shape[1]
    Lx = cfg['sim']['Lx']
    Ly = cfg['sim']['Ly']

    for target_t, label in zip(TARGET_TIMES, FRAME_LABELS):
        frame = np.argmin(np.abs(times - target_t))
        t_val = times[frame]
        pos = traj[frame]
        v = vel[frame]
        angles = np.arctan2(v[:, 1], v[:, 0])
        speeds = np.linalg.norm(v, axis=1)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Ground Truth Trajectory - Uniform')

        norm = matplotlib.colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cmap = plt.cm.hsv
        colors = cmap(norm(angles))

        # Smaller dots so arrows are not occluded
        ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=12, zorder=3,
                   edgecolors='none')

        # Arrows ~3x longer than original (scale=0.4 vs 1.2)
        speed_safe = np.maximum(speeds, 1e-10)
        vn = v / speed_safe[:, None]
        ax.quiver(pos[:, 0], pos[:, 1], vn[:, 0], vn[:, 1],
                  angles, cmap='hsv', clim=[-np.pi, np.pi],
                  scale=0.4, scale_units='xy', width=0.005,
                  headwidth=4, headlength=5, headaxislength=4,
                  alpha=0.85, zorder=2)

        sm = plt.cm.ScalarMappable(cmap='hsv', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Heading Angle (rad)')
        cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cbar.set_ticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

        total_frames = len(times)
        ax.text(0.02, 0.98, f't = {t_val:.2f}s\nFrame {frame}/{total_frames}',
                transform=ax.transAxes, verticalalignment='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.text(0.02, 0.06, f'N = {N} particles',
                transform=ax.transAxes, verticalalignment='bottom', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        out = os.path.join(OUT_DIR, f'traj_{dyn_label}_{label}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {out}  (t={t_val:.2f}s, frame {frame}/{total_frames})")

print("\nDone — all 12 trajectory PNGs regenerated with larger arrows.")
