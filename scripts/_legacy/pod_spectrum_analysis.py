"""Analyze full POD spectrum to determine d needed for 98% energy."""
import numpy as np

# Load the FULL singular value spectrum from V2.2
pod = np.load('oscar_output/synthesis_v2_2/rom_common/pod_basis.npz')
print('Keys:', list(pod.keys()))

sv_all = pod['all_singular_values']
print(f'Total singular values: {len(sv_all)}')

energy = sv_all**2 / np.sum(sv_all**2)
cum = np.cumsum(energy)

print(f'\nPOD Energy Spectrum:')
for d in [5, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100]:
    if d <= len(cum):
        print(f'  d={d:3d}: {cum[d-1]*100:.2f}% energy  (mode {d} contributes {energy[d-1]*100:.3f}%)')

# Find exact d for key thresholds
print(f'\nModes needed for energy thresholds:')
for target in [0.90, 0.95, 0.98, 0.99, 0.995, 0.999]:
    d_needed = np.searchsorted(cum, target) + 1
    print(f'  {target*100:.1f}% energy needs d={d_needed}')

print(f'\nTop 30 singular values:')
print([f'{s:.2f}' for s in sv_all[:30]])

print(f'\nCondition number sigma_1/sigma_d for various d:')
for d in [10, 15, 20, 25, 30, 35, 40]:
    if d <= len(sv_all):
        print(f'  d={d}: sigma_1/sigma_d = {sv_all[0]/sv_all[d-1]:.2f}')

# Also check: how many training samples do we have?
lat = np.load('oscar_output/synthesis_v2_2/rom_common/latent_dataset.npz')
print(f'\nTraining data shape: X={lat["X_all"].shape}, Y={lat["Y_all"].shape}')
n_samples = lat['X_all'].shape[0]
print(f'Total training samples: {n_samples}')

# MVAR parameter count for various d and p
print(f'\nMVAR parameter counts (d*p*d + d) and sample ratios:')
for d in [15, 20, 25, 30, 35]:
    for p in [3, 5, 7, 10]:
        n_params = d * p * d + d
        ratio = n_samples / n_params if n_params > 0 else 0
        print(f'  d={d:2d}, p={p:2d}: {n_params:6d} params, ratio={ratio:.1f}x')
