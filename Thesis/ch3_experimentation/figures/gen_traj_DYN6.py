"""Generate DYN6 (blackhole VS / variable speed) trajectory snapshots for Fig 2.1(d)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rectsim.config import load_config
from rectsim.vicsek_discrete import simulate_backend

# Load DYN6 config
cfg = load_config("configs/DYN6_varspeed.yaml")
rng = np.random.default_rng(42)

# Run simulation
result = simulate_backend(cfg, rng)
times = result["times"]
traj = result["traj"]   # (T, N, 2)
vel = result["vel"]      # (T, N, 2)

N = traj.shape[1]
Lx = cfg["sim"]["Lx"]
Ly = cfg["sim"]["Ly"]

# Pick 3 representative timesteps matching footer labels: t≈0s, t≈6s, t≈15s
target_times = [1.2, 6.0, 15.0]
labels = ["t0", "t15", "t38"]  # match existing naming convention

for target_t, label in zip(target_times, labels):
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

    # Heading-angle-colored scatter + quiver
    norm = matplotlib.colors.Normalize(vmin=-np.pi, vmax=np.pi)
    cmap = plt.cm.hsv
    colors = cmap(norm(angles))

    ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=18, zorder=3, edgecolors='none')

    # Unit-length arrows (direction only) for consistent display
    speed_safe = np.maximum(speeds, 1e-10)
    vn = v / speed_safe[:, None]
    q = ax.quiver(pos[:, 0], pos[:, 1], vn[:, 0], vn[:, 1],
                  angles, cmap='hsv', clim=[-np.pi, np.pi],
                  scale=1.2, scale_units='xy', width=0.006,
                  alpha=0.85, zorder=2)

    sm = plt.cm.ScalarMappable(cmap='hsv', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Heading Angle (rad)')
    cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.set_ticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

    # Info boxes
    total_frames = len(times)
    ax.text(0.02, 0.98, f't = {t_val:.2f}s\nFrame {frame}/{total_frames}',
            transform=ax.transAxes, verticalalignment='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(0.02, 0.06, f'N = {N} particles',
            transform=ax.transAxes, verticalalignment='bottom', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    out = os.path.join(os.path.dirname(__file__), f'traj_DYN6_{label}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}  (t={t_val:.2f}s, frame {frame}/{total_frames})")

print("Done — 3 DYN6 trajectory PNGs generated.")
