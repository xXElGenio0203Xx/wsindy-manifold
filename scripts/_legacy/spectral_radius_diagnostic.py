#!/usr/bin/env python3
"""Compute spectral radius of MVAR companion matrix for all CONT_FIX experiments."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import shared components from suite script
from rectsim.vicsek_discrete import simulate_backend
from rectsim.legacy_functions import kde_density_movie

# Re-use functions from suite_CONT_v1
import importlib.util
spec = importlib.util.spec_from_file_location("suite", str(Path(__file__).parent / "suite_CONT_v1.py"))
suite = importlib.util.module_from_spec(spec)

# We need to prevent main() from running
import types
original_argv = sys.argv
sys.argv = ['suite_CONT_v1.py', '--experiment', 'CF1_raw_singleIC_d7_p3_alpha1_H37']
try:
    spec.loader.exec_module(suite)
except SystemExit:
    pass
sys.argv = original_argv

# Pull functions we need
run_single_sim = suite.run_single_sim
build_pod_from_data = suite.build_pod_from_data
train_mvar = suite.train_mvar
make_ic_params = suite.make_ic_params
ROM_SUBSAMPLE = suite.ROM_SUBSAMPLE
ROM_DT = suite.ROM_DT


def spectral_radius(model, lag, d):
    """Compute spectral radius of MVAR(p) companion matrix."""
    W = model.coef_  # (d, lag*d)

    # Our feature vector is [z_{t-lag}, z_{t-lag+1}, ..., z_{t-1}] (oldest first)
    # W[:, i*d:(i+1)*d] multiplies z_{t-lag+i}
    # So W[:, 0:d] = A_lag, W[:, (lag-1)*d:lag*d] = A_1
    A_mats_reversed = [W[:, i*d:(i+1)*d] for i in range(lag)]
    A_mats = A_mats_reversed[::-1]  # Now A_1, A_2, ..., A_p

    pd = lag * d
    C = np.zeros((pd, pd))
    for i, A in enumerate(A_mats):
        C[:d, i*d:(i+1)*d] = A
    if lag > 1:
        C[d:, :(lag-1)*d] = np.eye((lag-1)*d)

    eigs = np.linalg.eigvals(C)
    rho = np.max(np.abs(eigs))
    return rho, eigs


def run_single_ic_diagnostic(name, rom_d, lag, alpha, T_train, transform, seed):
    """Train MVAR on one IC and return spectral radius."""
    ic_params = make_ic_params(seed)
    T_total = T_train + 200 * ROM_DT + 1.0
    rho_full, _ = run_single_sim(T_total, seed, 'gaussian', ic_params)
    rho_sub = rho_full[::ROM_SUBSAMPLE]
    T_train_frames = min(int(round(T_train / ROM_DT)), rho_sub.shape[0])
    X_train = rho_sub[:T_train_frames].reshape(T_train_frames, -1)

    pod_data = build_pod_from_data(X_train, rom_d=rom_d, transform=transform)

    if transform == 'sqrt':
        X_work = np.sqrt(X_train + pod_data['density_transform_eps'])
    else:
        X_work = X_train.copy()
    z_train = (X_work - pod_data['X_mean']) @ pod_data['U_r']

    model, _, r2_train, diag = train_mvar([z_train], lag=lag, alpha=alpha)
    if diag['ABORT']:
        return None

    rho, eigs = spectral_radius(model, lag, rom_d)
    mags = np.sort(np.abs(eigs))[::-1]

    return {
        'name': name, 'rho': rho, 'mags': mags,
        'r2_train': r2_train, 'diag': diag,
    }


def run_multi_ic_diagnostic(name, transform):
    """Train MVAR on 30 ICs and return spectral radius."""
    T_total = 5.0 + 162 * ROM_DT + 1.0
    T_train_frames = int(round(5.0 / ROM_DT))

    all_X = []
    for i in range(30):
        s = 8000 + i * 71
        ic = make_ic_params(s)
        rho_full, _ = run_single_sim(T_total, s, 'gaussian', ic)
        rho_sub = rho_full[::ROM_SUBSAMPLE]
        T_avail = min(T_train_frames, rho_sub.shape[0])
        all_X.append(rho_sub[:T_avail].reshape(T_avail, -1))

    X_concat = np.vstack(all_X)
    pod_data = build_pod_from_data(X_concat, rom_d=19, transform=transform)

    y_trajs = []
    for X_raw in all_X:
        if transform == 'sqrt':
            X_work = np.sqrt(X_raw + pod_data['density_transform_eps'])
        else:
            X_work = X_raw.copy()
        z = (X_work - pod_data['X_mean']) @ pod_data['U_r']
        y_trajs.append(z)

    model, _, r2_train, diag = train_mvar(y_trajs, lag=5, alpha=1e-4)
    rho, eigs = spectral_radius(model, 5, 19)
    mags = np.sort(np.abs(eigs))[::-1]

    return {
        'name': name, 'rho': rho, 'mags': mags,
        'r2_train': r2_train, 'diag': diag,
    }


def print_result(r):
    stable = "STABLE" if r['rho'] < 1.0 else "UNSTABLE"
    print(f"  {r['name']}")
    print(f"    n_samples={r['diag']['n_samples']}  n_features={r['diag']['n_features']}  "
          f"r2_train={r['r2_train']:.6f}")
    print(f"    SPECTRAL RADIUS = {r['rho']:.6f}  ({stable})")
    print(f"    Top 5 |eig|: {', '.join(f'{m:.4f}' for m in r['mags'][:5])}")
    print(f"    # |eig| > 1.0:  {int(np.sum(r['mags'] > 1.0))}")
    print(f"    # |eig| > 0.99: {int(np.sum(r['mags'] > 0.99))}")
    print()


if __name__ == '__main__':
    print("=" * 80)
    print("  SPECTRAL RADIUS DIAGNOSTIC — MVAR Companion Matrix")
    print("=" * 80)
    print()

    # --- Single-IC experiments (use 3 representative seeds for statistics) ---
    seeds = [7000, 7411, 7822]  # trial 0, 3, 6

    single_ic_configs = [
        ('CF1/CF4 (d=7,p=3,a=1,5s,raw)', 7, 3, 1.0, 5.0, 'raw'),
        ('CF1/CF4 (d=7,p=3,a=1,5s,sqrt)', 7, 3, 1.0, 5.0, 'sqrt'),
        ('CF7/CF8 (d=19,p=5,a=0.01,72s,raw)', 19, 5, 1e-2, 72.0, 'raw'),
        ('CF7/CF8 (d=19,p=5,a=0.01,72s,sqrt)', 19, 5, 1e-2, 72.0, 'sqrt'),
    ]

    print("--- Single-IC experiments (3 seeds for statistics) ---\n")
    for name, rom_d, lag, alpha, T_train, transform in single_ic_configs:
        rhos = []
        for seed in seeds:
            r = run_single_ic_diagnostic(name, rom_d, lag, alpha, T_train, transform, seed)
            if r:
                rhos.append(r['rho'])
        if rhos:
            # Print last result as representative
            print_result(r)
            if len(rhos) > 1:
                print(f"    Across {len(rhos)} seeds: "
                      f"rho = {np.mean(rhos):.4f} +/- {np.std(rhos):.4f}  "
                      f"[{np.min(rhos):.4f}, {np.max(rhos):.4f}]")
                print()

    # --- Multi-IC experiments ---
    print("\n--- Multi-IC in-sample (30 ICs, d=19, p=5, a=1e-4) ---\n")
    for transform, label in [('raw', 'CONT3 (raw, d=19, p=5, a=1e-4)'),
                             ('sqrt', 'CONT4 (sqrt, d=19, p=5, a=1e-4)')]:
        r = run_multi_ic_diagnostic(label, transform)
        print_result(r)
