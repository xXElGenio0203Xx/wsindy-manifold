import numpy as np
import matplotlib.pyplot as plt

traj = np.load('simulations/vicsek_discrete_eta0p30__latest/arrays/traj.npy')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
times = [0, 500, -1]
titles = ['Initial', 'Middle', 'Final']

for ax, t, title in zip(axes, times, titles):
    ax.scatter(traj[t, :, 0], traj[t, :, 1], s=10, alpha=0.6)
    ax.set_aspect('equal')
    ax.set_title(f'{title} (t={t if t>=0 else len(traj)-1})')
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/cluster_check.png', dpi=100)
print('Saved to /tmp/cluster_check.png')

pos_final = traj[-1]
com = pos_final.mean(axis=0)
dists = np.linalg.norm(pos_final - com, axis=1)

print(f'\nFinal state analysis:')
print(f'  Center of mass: ({com[0]:.2f}, {com[1]:.2f})')
print(f'  Mean distance from COM: {dists.mean():.2f}')
print(f'  Max distance from COM: {dists.max():.2f}')
print(f'  Std distance from COM: {dists.std():.2f}')

# Check for multiple clusters using distance distribution
print(f'\nDistance percentiles:')
print(f'  25%: {np.percentile(dists, 25):.2f}')
print(f'  50%: {np.percentile(dists, 50):.2f}')
print(f'  75%: {np.percentile(dists, 75):.2f}')
print(f'  90%: {np.percentile(dists, 90):.2f}')
