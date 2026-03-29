#!/usr/bin/env python3
"""Quick check of AL and CF experiment results."""
import numpy as np
import os

ROOT = '/Users/maria_1/Desktop/wsindy-manifold/oscar_output'

def get_r2_stats(exp_dir):
    test_dir = os.path.join(exp_dir, 'test')
    if not os.path.isdir(test_dir):
        return None, None, None, 0

    test_runs = sorted([
        d for d in os.listdir(test_dir)
        if d.startswith('test_') and os.path.isdir(os.path.join(test_dir, d))
    ])

    r2_list = []
    mass_err_list = []

    for run in test_runs:
        rdir = os.path.join(test_dir, run)
        pred_file = os.path.join(rdir, 'density_pred_mvar.npz')
        if not os.path.exists(pred_file):
            pred_file = os.path.join(rdir, 'density_pred.npz')
        if not os.path.exists(pred_file):
            continue

        true_file = os.path.join(rdir, 'density_true.npz')
        if not os.path.exists(true_file):
            continue

        d_pred = np.load(pred_file)
        d_true = np.load(true_file)
        rho_pred = d_pred['rho']
        rho_true = d_true['rho']
        fsi = int(d_pred['forecast_start_idx']) if 'forecast_start_idx' in d_pred else 0

        M0 = float(rho_true[max(fsi - 1, 0)].sum())

        n_f = rho_pred.shape[0] - fsi
        tf = rho_true[fsi:fsi + n_f]
        pf = rho_pred[fsi:fsi + n_f]
        L = min(tf.shape[0], pf.shape[0])
        tf, pf = tf[:L], pf[:L]
        if L == 0:
            continue

        ss_tot = np.sum((tf - tf.mean()) ** 2)
        if ss_tot < 1e-15:
            continue
        r2 = 1 - np.sum((tf - pf) ** 2) / ss_tot
        mass_err = abs(pf[-1].sum() - M0) / max(M0, 1e-8) * 100

        r2_list.append(r2)
        mass_err_list.append(mass_err)

    if not r2_list:
        return None, None, None, 0
    return float(np.mean(r2_list)), float(np.std(r2_list)), float(np.mean(mass_err_list)), len(r2_list)


print("=" * 95)
print(f"  {'Experiment':<58s} {'R²':>7s} {'±std':>6s} {'Mass%':>7s} {'n':>3s}")
print("=" * 95)

for prefix, label in [('AL', 'AL EXPERIMENTS'), ('CF', 'CF EXPERIMENTS')]:
    dirs = sorted([
        d for d in os.listdir(ROOT)
        if d.startswith(prefix) and os.path.isdir(os.path.join(ROOT, d))
    ])
    if dirs:
        print(f"\n  --- {label} ---")
    for d in dirs:
        exp_path = os.path.join(ROOT, d)
        r2_mean, r2_std, merr, n = get_r2_stats(exp_path)
        if r2_mean is not None:
            print(f"  {d:<58s} {r2_mean:>+7.4f} {r2_std:>5.3f} {merr:>6.1f}% {n:>3d}")
        else:
            print(f"  {d:<58s} {'N/A':>7s}")

print()
