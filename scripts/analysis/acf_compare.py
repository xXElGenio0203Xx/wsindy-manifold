"""Compute autocorrelation for CS01 vs DR01 latent modes, plus a few more experiments for contrast."""
import numpy as np
import sys

experiments = [
    'DO_CS01_swarm_C01_l05',
    'DO_DR01_dring_C01_l01',
    'DO_SM01_mill_C05_l01',
    'DO_EC01_esccol_C2_l3',
    'DO_DM01_dmill_C09_l05',
]

for name in experiments:
    path = f'oscar_output/{name}/rom_common/latent_dataset.npz'
    try:
        data = np.load(path)
    except FileNotFoundError:
        print(f"\n=== {name}: NOT FOUND ===")
        continue

    print(f"\n=== {name} ===")
    print(f"Keys: {list(data.keys())}")
    for k in data.keys():
        print(f"  {k}: shape={data[k].shape}")

    # Find training latent data
    Y = None
    for key_try in ['Y_train', 'Y_train_latent', 'y_train']:
        if key_try in data:
            Y = data[key_try]
            break
    if Y is None:
        for k in data.keys():
            if 'train' in k.lower() and 'test' not in k.lower():
                Y = data[k]
                print(f"  Using key: {k}")
                break

    if Y is None:
        print("  No training data found!")
        continue

    print(f"  Training data shape: {Y.shape}")

    # Shape: (n_trajs, T, d) or (T, d) ?
    if Y.ndim == 3:
        n_trajs, T, d = Y.shape
    elif Y.ndim == 2:
        T, d = Y.shape
        Y = Y[np.newaxis, :, :]  # (1, T, d)
        n_trajs = 1
    else:
        print(f"  Unexpected ndim={Y.ndim}")
        continue

    print(f"  n_trajs={n_trajs}, T={T}, d={d}")

    # Compute autocorrelation averaged over first 10 trajectories and all modes
    max_lag = min(T - 1, 60)
    n_use = min(n_trajs, 10)

    # Per-mode ACF (averaged over trajectories)
    acf_avg = np.zeros((d, max_lag + 1))
    for m in range(d):
        for i in range(n_use):
            y = Y[i, :, m]
            y = y - y.mean()
            var = y.var()
            if var < 1e-15:
                continue
            n = len(y)
            full_acf = np.correlate(y, y, mode='full')[n-1:]
            full_acf = full_acf / (n * var)
            acf_avg[m, :min(len(full_acf), max_lag + 1)] += full_acf[:max_lag + 1]
        acf_avg[m] /= n_use

    # Summary: mean ACF across all modes
    mean_acf = acf_avg.mean(axis=0)

    # Find decorrelation lag: where mean ACF drops below 0.5 and below 0.1
    lag_half = np.where(mean_acf < 0.5)[0]
    lag_tenth = np.where(mean_acf < 0.1)[0]

    print(f"\n  Mean ACF across all {d} modes:")
    for lag in [1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50]:
        if lag <= max_lag:
            print(f"    ACF(lag={lag:2d}): {mean_acf[lag]:.4f}")

    if len(lag_half) > 0:
        print(f"  ** Decorrelation (ACF < 0.5) at lag = {lag_half[0]}")
    else:
        print(f"  ** ACF never drops below 0.5!")

    if len(lag_tenth) > 0:
        print(f"  ** Near-zero (ACF < 0.1) at lag = {lag_tenth[0]}")
    else:
        print(f"  ** ACF never drops below 0.1!")

    # Also report the per-mode spread
    zero_crosses = []
    for m in range(d):
        zc = np.where(acf_avg[m] < 0)[0]
        if len(zc) > 0:
            zero_crosses.append(zc[0])
        else:
            zero_crosses.append(max_lag)

    print(f"  Zero-crossing lags per mode: min={min(zero_crosses)}, "
          f"median={np.median(zero_crosses):.0f}, max={max(zero_crosses)}")

print("\n\nDone.")
