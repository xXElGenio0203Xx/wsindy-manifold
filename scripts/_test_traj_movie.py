from pathlib import Path
import numpy as np
from rectsim.density import traj_movie

run_dir = Path('outputs/single')
print('Loading', run_dir / 'traj.npz')
adata = np.load(run_dir / 'traj.npz')
traj = adata['x'] if 'x' in adata.files else adata['traj'] if 'traj' in adata.files else adata['x']
vel = adata['v'] if 'v' in adata.files else None
print('traj shape', traj.shape)
print('vel shape', None if vel is None else vel.shape)

try:
    out = traj_movie(traj, vel, adata['times'], 20.0, 20.0, run_dir, fps=12, marker_size=6, draw_vectors=False)
    print('traj_movie returned:', out)
except Exception as e:
    import traceback
    traceback.print_exc()
    print('Exception:', e)
