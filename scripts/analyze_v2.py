#!/usr/bin/env python3
"""Quick V2 analysis script."""
import numpy as np
import json

# Load V2 MVAR model
mvar = np.load('oscar_output/synthesis_v2/MVAR/mvar_model.npz')
print('=== MVAR Model ===')
for k in mvar.files:
    v = mvar[k]
    if v.ndim == 0:
        print(f'  {k}: {v}')
    else:
        print(f'  {k}: shape={v.shape}, range=[{v.min():.6f}, {v.max():.6f}]')

# Load POD basis
pod = np.load('oscar_output/synthesis_v2/rom_common/pod_basis.npz')
print()
print('=== POD Basis ===')
for k in pod.files:
    v = pod[k]
    if v.ndim == 0:
        print(f'  {k}: {v}')
    elif v.size < 20:
        print(f'  {k}: {v}')
    else:
        print(f'  {k}: shape={v.shape}')

print()
svs = pod['singular_values']
print(f'Singular values (d=8): {svs}')
total_e = float(pod['total_energy'])
energies = svs**2 / total_e
cum_e = np.cumsum(energies)
print(f'Energy per mode: {energies}')
print(f'Cumulative: {cum_e}')

# How many modes for various energy thresholds?
all_svs = pod['all_singular_values']
all_cum = np.cumsum(all_svs**2) / total_e
d99 = np.searchsorted(all_cum, 0.99) + 1
d95 = np.searchsorted(all_cum, 0.95) + 1
d90 = np.searchsorted(all_cum, 0.90) + 1
d80 = np.searchsorted(all_cum, 0.80) + 1
print(f'\nModes for 80% energy: {d80}')
print(f'Modes for 90% energy: {d90}')
print(f'Modes for 95% energy: {d95}')
print(f'Modes for 99% energy: {d99}')
print(f'Top 30 cumulative energy: {all_cum[:30]}')

# Reconstruct companion matrix and analyze
A_comp = mvar['A_companion']
p = int(mvar['p'])
r = int(mvar['r'])
print(f'\n=== Companion Analysis ===')
print(f'p={p}, r={r}')
print(f'A_companion shape: {A_comp.shape}')

companion_dim = p * r
C = np.zeros((companion_dim, companion_dim))
C[:r, :] = A_comp
for k in range(p - 1):
    C[(k+1)*r:(k+2)*r, k*r:(k+1)*r] = np.eye(r)

evals = np.linalg.eigvals(C)
moduli = np.abs(evals)
print(f'Current spectral radius: {np.max(moduli):.6f}')
print(f'Eigenvalue moduli: {np.sort(moduli)[::-1]}')
