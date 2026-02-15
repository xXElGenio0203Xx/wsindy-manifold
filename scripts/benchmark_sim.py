"""Benchmark simulation times for different N and T."""
import time
import numpy as np
from rectsim.vicsek_discrete import VicsekMorseSimulation

config = {
    'sim': {'N': 100, 'T': 6.0, 'dt': 0.04, 'Lx': 15.0, 'Ly': 15.0, 'bc': 'periodic'},
    'model': {'type': 'discrete', 'speed': 1.5, 'speed_mode': 'constant_with_forces'},
    'params': {'R': 2.5},
    'noise': {'kind': 'gaussian', 'eta': 0.2, 'match_variance': True},
    'forces': {'enabled': True, 'type': 'morse', 'params': {
        'Ca': 0.8, 'Cr': 0.3, 'la': 1.5, 'lr': 0.5, 'mu_t': 0.3, 'rcut_factor': 5.0}},
    'alignment': {'enabled': True}
}

np.random.seed(42)

# Test 1: N=100, T=6 (current)
t0 = time.time()
sim = VicsekMorseSimulation(config)
pos = np.random.uniform(0, 15, (100, 2))
theta = np.random.uniform(0, 2*np.pi, 100)
sim.run(pos, theta)
dt1 = time.time() - t0
n_steps_1 = sim.trajectory.shape[0]
print(f'N=100, T=6.0:  {dt1:.2f}s, {n_steps_1} steps')

# Test 2: N=100, T=30
config2 = dict(config)
config2['sim'] = dict(config['sim'], T=30.0)
t0 = time.time()
sim2 = VicsekMorseSimulation(config2)
pos2 = np.random.uniform(0, 15, (100, 2))
theta2 = np.random.uniform(0, 2*np.pi, 100)
sim2.run(pos2, theta2)
dt2 = time.time() - t0
n_steps_2 = sim2.trajectory.shape[0]
print(f'N=100, T=30.0: {dt2:.2f}s, {n_steps_2} steps')

# Test 3: N=200, T=30
config3 = dict(config)
config3['sim'] = dict(config['sim'], T=30.0, N=200)
t0 = time.time()
sim3 = VicsekMorseSimulation(config3)
pos3 = np.random.uniform(0, 15, (200, 2))
theta3 = np.random.uniform(0, 2*np.pi, 200)
sim3.run(pos3, theta3)
dt3 = time.time() - t0
n_steps_3 = sim3.trajectory.shape[0]
print(f'N=200, T=30.0: {dt3:.2f}s, {n_steps_3} steps')

# Extrapolate
print(f'\n--- Extrapolations ---')
cost_per_step_N100 = dt2 / n_steps_2
cost_per_step_N200 = dt3 / n_steps_3
print(f'Cost per step: N=100: {cost_per_step_N100*1000:.2f}ms, N=200: {cost_per_step_N200*1000:.2f}ms')

# With subsample=3, T=30 gives 30/0.04/3 = 250 frames per IC
for T in [30, 60, 120]:
    for N_ic in [100, 200, 500]:
        n_steps = int(T / 0.04)
        n_frames = n_steps // 3  # subsample=3
        sim_time = N_ic * cost_per_step_N100 * n_steps
        # MVAR training samples per IC: n_frames - lag
        n_samples = N_ic * (n_frames - 5)
        print(f'  T={T:3d}s, {N_ic:3d} ICs: sim_time={sim_time/60:.1f}min, '
              f'{n_frames} frames/IC, {n_samples} total samples')
