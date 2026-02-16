#!/usr/bin/env python3
"""
SUITE_CONT_FIX (V1) — Continuation vs Generalization Diagnostic
================================================================

Goal: make single-IC continuation identifiable; separate "sample" vs "intrinsic" causes.

A) Low-complexity single-IC continuation (d=7, p=3, α=1.0)
   CF1–CF6: raw and sqrt+simplex at H37/H100/H162

B) Same best model, long prefix to get enough samples (d=19, p=5, α=1e-2)
   CF7–CF10: raw and sqrt+simplex at H37/H162
   K_train = 600 frames ~ 72s  (enough to make regression well-posed)

C) Multi-IC in-sample continuation (unchanged from original CONT-3/4)
   CONT3/4: d=19, p=5, 30 training ICs, in-sample test on 10 ICs

INSURANCE: every experiment reports n_samples, n_features (=p*d),
condition number, r2_train. Auto-aborts if n_samples < n_features.

Usage:
  PYTHONPATH=src python scripts/suite_CONT_v1.py [--experiment CF1_raw_singleIC_d7_p3_alpha1_H37]
  PYTHONPATH=src python scripts/suite_CONT_v1.py --group A   # CF1-CF6 only
  PYTHONPATH=src python scripts/suite_CONT_v1.py --group B   # CF7-CF10 only
  PYTHONPATH=src python scripts/suite_CONT_v1.py --group C   # CONT3-CONT4 only
  PYTHONPATH=src python scripts/suite_CONT_v1.py             # all 12 experiments
"""

import numpy as np
import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

try:
    from sklearn.linear_model import Ridge
except ImportError:
    class Ridge:
        """Fallback Ridge regression (no sklearn dependency)."""
        def __init__(self, alpha=1.0, fit_intercept=True):
            self.alpha = alpha
            self.fit_intercept = fit_intercept
            self.coef_ = None
            self.intercept_ = None

        def fit(self, X, Y):
            if self.fit_intercept:
                X_mean = X.mean(axis=0)
                Y_mean = Y.mean(axis=0)
                Xc = X - X_mean
                Yc = Y - Y_mean
            else:
                Xc, Yc = X, Y
                Y_mean = np.zeros(Y.shape[1])
                X_mean = np.zeros(X.shape[1])
            A = Xc.T @ Xc + self.alpha * np.eye(Xc.shape[1])
            self.coef_ = np.linalg.solve(A, Xc.T @ Yc).T
            self.intercept_ = Y_mean - self.coef_ @ X_mean
            return self

        def predict(self, X):
            return X @ self.coef_.T + self.intercept_

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from rectsim.vicsek_discrete import simulate_backend
from rectsim.legacy_functions import kde_density_movie


# ─────────────────────────────────────────────────────────────
# SIMPLEX PROJECTION (Duchi et al.)
# ─────────────────────────────────────────────────────────────

def project_simplex(rho_flat: np.ndarray, M0: float) -> np.ndarray:
    """Euclidean projection onto {rho >= 0, sum(rho) = M0}."""
    n = len(rho_flat)
    mu = np.sort(rho_flat)[::-1]
    cumsum = np.cumsum(mu)
    arange = np.arange(1, n + 1, dtype=np.float64)
    test = mu - (cumsum - M0) / arange
    rho_max = np.max(np.where(test > 0)[0]) if np.any(test > 0) else 0
    theta = (cumsum[rho_max] - M0) / (rho_max + 1)
    return np.maximum(rho_flat - theta, 0.0)


# ─────────────────────────────────────────────────────────────
# SIMULATION
# ─────────────────────────────────────────────────────────────

BASE_SIM_CONFIG = {
    'sim': {
        'N': 100, 'dt': 0.04,
        'Lx': 15.0, 'Ly': 15.0, 'bc': 'periodic',
    },
    'model': {
        'type': 'discrete', 'speed': 1.5,
        'speed_mode': 'constant_with_forces',
    },
    'params': {'R': 2.5},
    'noise': {'kind': 'gaussian', 'eta': 0.2, 'match_variance': True},
    'forces': {
        'enabled': True, 'type': 'morse',
        'params': {
            'Ca': 0.8, 'Cr': 0.3, 'la': 1.5, 'lr': 0.5,
            'mu_t': 0.3, 'rcut_factor': 5.0,
        },
    },
    'alignment': {'enabled': True},
}

DENSITY_NX = 48
DENSITY_NY = 48
DENSITY_BW = 4.0
ROM_SUBSAMPLE = 3
ROM_DT = 0.04 * ROM_SUBSAMPLE  # 0.12s


def run_single_sim(T, seed, ic_type='gaussian', ic_params=None):
    """Run one simulation and return density movie."""
    config = {**BASE_SIM_CONFIG}
    config['sim'] = {**config['sim'], 'T': T}
    config['seed'] = seed
    config['initial_distribution'] = ic_type
    if ic_params is None:
        ic_params = {'center': (7.5, 7.5), 'sigma': 2.0}
    config['ic_params'] = ic_params

    rng = np.random.default_rng(seed)
    result = simulate_backend(config, rng)

    rho, _ = kde_density_movie(
        result['traj'],
        Lx=config['sim']['Lx'], Ly=config['sim']['Ly'],
        nx=DENSITY_NX, ny=DENSITY_NY,
        bandwidth=DENSITY_BW,
        bc=config['sim'].get('bc', 'periodic'),
    )
    return rho, result['times']


# ─────────────────────────────────────────────────────────────
# ROM COMPONENTS
# ─────────────────────────────────────────────────────────────

def build_pod_from_data(X_all, rom_d=None, energy_thresh=0.90,
                        transform='raw', eps=1e-8):
    """Build POD basis from (T, n_spatial) data matrix."""
    if transform == 'sqrt':
        X_work = np.sqrt(X_all + eps)
    elif transform == 'log':
        X_work = np.log(X_all + eps)
    else:
        X_work = X_all.copy()

    X_mean = X_work.mean(axis=0)
    X_centered = X_work - X_mean
    U, S, Vt = np.linalg.svd(X_centered.T, full_matrices=False)

    total_energy = np.sum(S**2)
    cumulative_energy = np.cumsum(S**2) / total_energy

    if rom_d is not None:
        R = min(rom_d, len(S))
    else:
        R = np.searchsorted(cumulative_energy, energy_thresh) + 1

    U_r = U[:, :R]
    X_latent = X_centered @ U_r

    return {
        'U_r': U_r, 'S': S, 'X_mean': X_mean, 'X_latent': X_latent,
        'R_POD': R,
        'energy_captured': float(cumulative_energy[R - 1]),
        'density_transform': transform, 'density_transform_eps': eps,
    }


def build_mvar_matrices(y_trajs, lag):
    """Build (X, Y) regression matrices from list of latent trajectories."""
    X_rows, Y_rows = [], []
    for y in y_trajs:
        T, d = y.shape
        for t in range(lag, T):
            X_rows.append(y[t - lag:t, :].ravel())
            Y_rows.append(y[t, :])
    return np.array(X_rows), np.array(Y_rows)


def train_mvar(y_trajs, lag=5, alpha=1e-4):
    """
    Train MVAR model.
    Returns (model, lag, r2_train, diagnostics_dict).
    diagnostics_dict always has: n_samples, n_features, sample_feature_ratio,
    condition_number, ABORT (bool), and optionally 'reason'.
    """
    X, Y = build_mvar_matrices(y_trajs, lag)
    n_samples, n_features = X.shape
    d = Y.shape[1]

    # INSURANCE: condition number & identifiability check
    cond = float(np.linalg.cond(X)) if n_samples >= n_features else float('inf')
    ratio = n_samples / n_features if n_features > 0 else float('inf')

    diag = {
        'n_samples': int(n_samples),
        'n_features': int(n_features),
        'd': int(d),
        'sample_feature_ratio': round(ratio, 2),
        'condition_number': cond,
    }

    if n_samples < n_features:
        diag['ABORT'] = True
        diag['reason'] = (f"UNDER-DETERMINED: n_samples={n_samples} < "
                          f"n_features={n_features} (p={lag}*d={d})")
        print(f"    !! AUTO-ABORT: {diag['reason']}")
        return None, lag, None, diag

    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, Y)

    Y_hat = model.predict(X)
    ss_res = np.sum((Y - Y_hat) ** 2)
    ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

    diag['r2_train'] = round(float(r2), 6)
    diag['ABORT'] = False
    return model, lag, r2, diag


def mvar_forecast(model, z_init, lag, n_steps):
    """Closed-loop MVAR forecast. z_init: (lag, d). Returns (n_steps, d)."""
    buffer = list(z_init)
    forecast = []
    for _ in range(n_steps):
        x = np.concatenate(buffer[-lag:]).reshape(1, -1)
        y_next = model.predict(x).ravel()
        buffer.append(y_next)
        forecast.append(y_next)
    return np.array(forecast)


def inverse_transform(X_latent, pod_data):
    """Lift from latent to physical space."""
    X_recon = X_latent @ pod_data['U_r'].T + pod_data['X_mean']
    transform = pod_data['density_transform']
    if transform == 'sqrt':
        X_physical = np.maximum(X_recon, 0.0) ** 2
    elif transform == 'log':
        X_physical = np.exp(X_recon) - pod_data['density_transform_eps']
    else:
        X_physical = X_recon
    return np.maximum(X_physical, 0.0)


# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────

def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0


def compute_frame_r2(true_f, pred_f):
    T = true_f.shape[0]
    return np.array([compute_r2(true_f[t].ravel(), pred_f[t].ravel())
                     for t in range(T)])


# ─────────────────────────────────────────────────────────────
# EXPERIMENT DEFINITIONS
# ─────────────────────────────────────────────────────────────

N_REPEAT_SINGLE = 10  # trials per single-IC experiment

EXPERIMENTS = {
    # ═══════════════════════════════════════════════════════════
    # A) LOW-COMPLEXITY SINGLE-IC  (d=7, p=3, alpha=1.0)
    #    T_train=5s -> 42 ROM frames -> 39 samples, 21 features
    #    ratio = 39/21 ~ 1.86  (safely > 1)
    # ═══════════════════════════════════════════════════════════
    'CF1_raw_singleIC_d7_p3_alpha1_H37': {
        'mode': 'single_ic', 'transform': 'raw', 'simplex': False,
        'H': 37, 'T_train': 5.0,
        'rom_d': 7, 'lag': 3, 'alpha': 1.0,
        'n_repeat': N_REPEAT_SINGLE,
    },
    'CF2_raw_singleIC_d7_p3_alpha1_H100': {
        'mode': 'single_ic', 'transform': 'raw', 'simplex': False,
        'H': 100, 'T_train': 5.0,
        'rom_d': 7, 'lag': 3, 'alpha': 1.0,
        'n_repeat': N_REPEAT_SINGLE,
    },
    'CF3_raw_singleIC_d7_p3_alpha1_H162': {
        'mode': 'single_ic', 'transform': 'raw', 'simplex': False,
        'H': 162, 'T_train': 5.0,
        'rom_d': 7, 'lag': 3, 'alpha': 1.0,
        'n_repeat': N_REPEAT_SINGLE,
    },
    'CF4_sqrtSimplex_singleIC_d7_p3_alpha1_H37': {
        'mode': 'single_ic', 'transform': 'sqrt', 'simplex': True,
        'H': 37, 'T_train': 5.0,
        'rom_d': 7, 'lag': 3, 'alpha': 1.0,
        'n_repeat': N_REPEAT_SINGLE,
    },
    'CF5_sqrtSimplex_singleIC_d7_p3_alpha1_H100': {
        'mode': 'single_ic', 'transform': 'sqrt', 'simplex': True,
        'H': 100, 'T_train': 5.0,
        'rom_d': 7, 'lag': 3, 'alpha': 1.0,
        'n_repeat': N_REPEAT_SINGLE,
    },
    'CF6_sqrtSimplex_singleIC_d7_p3_alpha1_H162': {
        'mode': 'single_ic', 'transform': 'sqrt', 'simplex': True,
        'H': 162, 'T_train': 5.0,
        'rom_d': 7, 'lag': 3, 'alpha': 1.0,
        'n_repeat': N_REPEAT_SINGLE,
    },

    # ═══════════════════════════════════════════════════════════
    # B) LONG-PREFIX SINGLE-IC  (d=19, p=5, alpha=1e-2)
    #    K_train=600 frames ~ 72s -> 595 samples, 95 features
    #    ratio = 595/95 ~ 6.26  (well-posed)
    # ═══════════════════════════════════════════════════════════
    'CF7_raw_singleIC_longPrefix_d19_p5_H37': {
        'mode': 'single_ic', 'transform': 'raw', 'simplex': False,
        'H': 37, 'T_train': 72.0,  # 600 ROM frames
        'rom_d': 19, 'lag': 5, 'alpha': 1e-2,
        'n_repeat': N_REPEAT_SINGLE,
    },
    'CF8_sqrtSimplex_singleIC_longPrefix_d19_p5_H37': {
        'mode': 'single_ic', 'transform': 'sqrt', 'simplex': True,
        'H': 37, 'T_train': 72.0,
        'rom_d': 19, 'lag': 5, 'alpha': 1e-2,
        'n_repeat': N_REPEAT_SINGLE,
    },
    'CF9_raw_singleIC_longPrefix_d19_p5_H162': {
        'mode': 'single_ic', 'transform': 'raw', 'simplex': False,
        'H': 162, 'T_train': 72.0,
        'rom_d': 19, 'lag': 5, 'alpha': 1e-2,
        'n_repeat': N_REPEAT_SINGLE,
    },
    'CF10_sqrtSimplex_singleIC_longPrefix_d19_p5_H162': {
        'mode': 'single_ic', 'transform': 'sqrt', 'simplex': True,
        'H': 162, 'T_train': 72.0,
        'rom_d': 19, 'lag': 5, 'alpha': 1e-2,
        'n_repeat': N_REPEAT_SINGLE,
    },

    # ═══════════════════════════════════════════════════════════
    # C) MULTI-IC IN-SAMPLE CONTINUATION (unchanged, well-posed)
    #    d=19, p=5, 30 ICs x 37 frames/IC -> ~960 samples, 95 features
    # ═══════════════════════════════════════════════════════════
    'CONT3_raw_multiIC_insample_H162': {
        'mode': 'multi_ic_insample', 'transform': 'raw', 'simplex': False,
        'H': 162, 'T_train': 5.0,
        'n_train_runs': 30, 'n_test_insample': 10,
        'rom_d': 19, 'lag': 5, 'alpha': 1e-4,
    },
    'CONT4_sqrtSimplex_multiIC_insample_H162': {
        'mode': 'multi_ic_insample', 'transform': 'sqrt', 'simplex': True,
        'H': 162, 'T_train': 5.0,
        'n_train_runs': 30, 'n_test_insample': 10,
        'rom_d': 19, 'lag': 5, 'alpha': 1e-4,
    },
}


# ─────────────────────────────────────────────────────────────
# IC GENERATORS
# ─────────────────────────────────────────────────────────────

def make_ic_params(seed):
    """Generate a reproducible IC from seed."""
    rng = np.random.default_rng(seed)
    return {
        'center': (float(rng.uniform(2.0, 13.0)),
                   float(rng.uniform(2.0, 13.0))),
        'sigma': float(rng.uniform(1.0, 3.0)),
    }


# ─────────────────────────────────────────────────────────────
# SINGLE-IC CONTINUATION
# ─────────────────────────────────────────────────────────────

def run_single_ic_experiment(name, cfg, root):
    """Train on prefix of ONE trajectory, forecast its own suffix."""
    H = cfg['H']
    T_train = cfg['T_train']
    T_total = T_train + H * ROM_DT + 1.0
    n_repeat = cfg.get('n_repeat', 10)
    rom_d = cfg['rom_d']
    lag = cfg['lag']
    alpha = cfg['alpha']
    transform = cfg['transform']
    do_simplex = cfg['simplex']

    expected_frames = int(round(T_train / ROM_DT))
    expected_samples = expected_frames - lag
    expected_features = lag * rom_d

    print(f"\n{'='*70}")
    print(f"  {name}  (single-IC continuation)")
    print(f"  transform={transform}  simplex={do_simplex}  H={H}")
    print(f"  d={rom_d}  p(lag)={lag}  alpha={alpha}  T_train={T_train:.1f}s")
    print(f"  Expected: {expected_frames} train frames -> "
          f"{expected_samples} samples, {expected_features} features  "
          f"(ratio={expected_samples/expected_features:.2f})")
    print(f"{'='*70}")

    out_dir = Path(root) / 'oscar_output' / name
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    all_diags = []

    for trial in range(n_repeat):
        seed = 7000 + trial * 137
        ic_params = make_ic_params(seed)

        print(f"\n  Trial {trial}/{n_repeat}: seed={seed}, "
              f"IC=({ic_params['center'][0]:.1f}, "
              f"{ic_params['center'][1]:.1f}, "
              f"sigma={ic_params['sigma']:.1f})")

        # 1. Simulate
        t0 = time.time()
        rho_full, _ = run_single_sim(T_total, seed, 'gaussian', ic_params)
        sim_time = time.time() - t0

        # 2. Subsample
        rho_sub = rho_full[::ROM_SUBSAMPLE]
        T_train_frames = min(int(round(T_train / ROM_DT)), rho_sub.shape[0])
        T_forecast = min(H, rho_sub.shape[0] - T_train_frames)
        if T_forecast < 1:
            print(f"    !! Not enough frames for forecast, skipping")
            continue

        rho_train = rho_sub[:T_train_frames]
        rho_test = rho_sub[T_train_frames:T_train_frames + T_forecast]
        M0 = float(rho_train[-1].sum())
        X_train = rho_train.reshape(T_train_frames, -1)

        # 3. POD
        pod_data = build_pod_from_data(X_train, rom_d=rom_d, transform=transform)
        R = pod_data['R_POD']

        # 4. Latent trajectory
        if transform == 'sqrt':
            X_work = np.sqrt(X_train + pod_data['density_transform_eps'])
        else:
            X_work = X_train.copy()
        z_train = (X_work - pod_data['X_mean']) @ pod_data['U_r']

        # 5. Train MVAR (with insurance)
        model, _, r2_train, diag = train_mvar([z_train], lag=lag, alpha=alpha)
        all_diags.append(diag)

        print(f"    MVAR: n_samples={diag['n_samples']}, "
              f"n_features={diag['n_features']}, "
              f"ratio={diag['sample_feature_ratio']:.2f}, "
              f"cond={diag['condition_number']:.1e}")

        if diag['ABORT']:
            continue

        print(f"    r2_train={r2_train:.4f}")

        # 6. Forecast
        z_init = z_train[-lag:]
        z_forecast = mvar_forecast(model, z_init, lag, T_forecast)

        # 7. Lift
        rho_pred_flat = inverse_transform(z_forecast, pod_data)
        rho_pred = rho_pred_flat.reshape(T_forecast, DENSITY_NY, DENSITY_NX)

        # 8. Simplex
        if do_simplex:
            for t in range(T_forecast):
                rho_pred[t] = project_simplex(
                    rho_pred[t].ravel(), M0
                ).reshape(DENSITY_NY, DENSITY_NX)

        # 9. Metrics
        ss_tot = np.sum((rho_test - rho_test.mean()) ** 2)
        r2_overall = (1.0 - np.sum((rho_test - rho_pred) ** 2) / ss_tot
                      if ss_tot > 1e-12 else 0.0)
        mass_err = (abs(rho_pred[-1].sum() - M0) / M0 * 100
                    if M0 > 1e-12 else 0.0)
        r2_frames = compute_frame_r2(rho_test, rho_pred)

        result = {
            'trial': trial, 'seed': seed,
            'R_POD': R, 'r2_train_mvar': float(r2_train),
            'r2_overall': float(r2_overall),
            'r2_first_frame': float(r2_frames[0]) if len(r2_frames) > 0 else None,
            'r2_last_frame': float(r2_frames[-1]) if len(r2_frames) > 0 else None,
            'mass_err_final_pct': float(mass_err),
            'T_train_frames': T_train_frames, 'T_forecast': T_forecast,
            'n_mvar_samples': diag['n_samples'],
            'n_mvar_features': diag['n_features'],
            'condition_number': diag['condition_number'],
            'sim_time_s': float(sim_time),
            'ic_params': ic_params,
        }
        results.append(result)
        print(f"    R2={r2_overall:+.4f}  mass_err={mass_err:.1f}%")

    # Aggregate
    if not results:
        print("  !! No valid trials (all aborted)")
        summary = {
            'experiment': name, 'ABORTED': True,
            'reason': 'All trials auto-aborted (under-determined)',
            'diagnostics': all_diags,
            'timestamp': datetime.now().isoformat(),
        }
        with open(out_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        return summary

    r2s = [r['r2_overall'] for r in results]
    summary = {
        'experiment': name, 'ABORTED': False,
        'mode': cfg['mode'], 'transform': transform, 'simplex': do_simplex,
        'H': H, 'rom_d': rom_d, 'lag': lag, 'alpha': alpha,
        'T_train': T_train,
        'n_trials': len(results),
        'r2_mean': float(np.mean(r2s)),
        'r2_std': float(np.std(r2s)),
        'r2_median': float(np.median(r2s)),
        'r2_min': float(np.min(r2s)),
        'r2_max': float(np.max(r2s)),
        'mass_err_mean': float(np.mean([r['mass_err_final_pct'] for r in results])),
        'r2_train_mean': float(np.mean([r['r2_train_mvar'] for r in results])),
        'n_samples': results[0]['n_mvar_samples'],
        'n_features': results[0]['n_mvar_features'],
        'condition_number_mean': float(np.mean([r['condition_number'] for r in results])),
        'timestamp': datetime.now().isoformat(),
    }

    print(f"\n  SUMMARY: R2={summary['r2_mean']:+.4f} +/- {summary['r2_std']:.4f}  "
          f"[{summary['r2_min']:+.4f}, {summary['r2_max']:+.4f}]")
    print(f"  r2_train={summary['r2_train_mean']:.4f}  "
          f"samples={summary['n_samples']}  features={summary['n_features']}  "
          f"cond={summary['condition_number_mean']:.1e}")

    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / 'per_trial_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return summary


# ─────────────────────────────────────────────────────────────
# MULTI-IC IN-SAMPLE CONTINUATION
# ─────────────────────────────────────────────────────────────

def run_multi_ic_insample_experiment(name, cfg, root):
    """Train MVAR on N trajectory prefixes, forecast continuation of same ICs."""
    H = cfg['H']
    T_train = cfg['T_train']
    T_total = T_train + H * ROM_DT + 1.0
    n_train = cfg['n_train_runs']
    n_test = cfg['n_test_insample']
    rom_d = cfg['rom_d']
    lag = cfg['lag']
    alpha = cfg['alpha']
    transform = cfg['transform']
    do_simplex = cfg['simplex']

    T_train_frames = int(round(T_train / ROM_DT))
    expected_samples = n_train * (T_train_frames - lag)
    expected_features = lag * rom_d

    print(f"\n{'='*70}")
    print(f"  {name}  (multi-IC in-sample)")
    print(f"  transform={transform}  simplex={do_simplex}  H={H}")
    print(f"  d={rom_d}  p(lag)={lag}  alpha={alpha}  "
          f"n_train={n_train}  n_test={n_test}")
    print(f"  Expected: {n_train} x {T_train_frames} frames -> "
          f"~{expected_samples} samples, {expected_features} features  "
          f"(ratio={expected_samples/expected_features:.2f})")
    print(f"{'='*70}")

    out_dir = Path(root) / 'oscar_output' / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run all simulations
    print(f"\n  Running {n_train} training simulations (T={T_total:.1f}s)...")
    all_rho, all_M0 = [], []
    for i in range(n_train):
        seed = 8000 + i * 71
        ic = make_ic_params(seed)
        rho_full, _ = run_single_sim(T_total, seed, 'gaussian', ic)
        rho_sub = rho_full[::ROM_SUBSAMPLE]
        all_rho.append(rho_sub)
        M0 = float(rho_sub[min(T_train_frames - 1, rho_sub.shape[0] - 1)].sum())
        all_M0.append(M0)
        if i < 3 or i == n_train - 1:
            print(f"    sim {i}: T_sub={rho_sub.shape[0]}, M0={M0:.1f}")

    # 2. POD from all training prefixes
    print(f"\n  Building POD from {n_train} training prefixes...")
    X_train_all = []
    for rho_sub in all_rho:
        T_avail = min(T_train_frames, rho_sub.shape[0])
        X_train_all.append(rho_sub[:T_avail].reshape(T_avail, -1))
    X_concat = np.vstack(X_train_all)
    pod_data = build_pod_from_data(X_concat, rom_d=rom_d, transform=transform)
    R = pod_data['R_POD']
    print(f"    POD d={R}, energy={pod_data['energy_captured']:.4f}")

    # 3. Latent trajectories
    y_trajs = []
    for rho_sub in all_rho:
        T_avail = min(T_train_frames, rho_sub.shape[0])
        X_raw = rho_sub[:T_avail].reshape(T_avail, -1)
        if transform == 'sqrt':
            X_work = np.sqrt(X_raw + pod_data['density_transform_eps'])
        else:
            X_work = X_raw.copy()
        z = (X_work - pod_data['X_mean']) @ pod_data['U_r']
        y_trajs.append(z)

    # 4. Train MVAR (with insurance)
    model, _, r2_train, diag = train_mvar(y_trajs, lag=lag, alpha=alpha)

    print(f"    MVAR: n_samples={diag['n_samples']}, "
          f"n_features={diag['n_features']}, "
          f"ratio={diag['sample_feature_ratio']:.2f}, "
          f"cond={diag['condition_number']:.1e}")

    if diag['ABORT']:
        summary = {
            'experiment': name, 'ABORTED': True,
            'reason': diag['reason'], 'diagnostics': diag,
            'timestamp': datetime.now().isoformat(),
        }
        with open(out_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        return summary

    print(f"    r2_train={r2_train:.4f}")

    # 5. In-sample continuation
    results = []
    for idx in range(min(n_test, n_train)):
        rho_sub = all_rho[idx]
        M0 = all_M0[idx]
        z = y_trajs[idx]

        T_avail_forecast = min(H, rho_sub.shape[0] - T_train_frames)
        if T_avail_forecast < 1:
            continue

        z_init = z[-lag:]
        z_forecast = mvar_forecast(model, z_init, lag, T_avail_forecast)
        rho_pred_flat = inverse_transform(z_forecast, pod_data)
        rho_pred = rho_pred_flat.reshape(T_avail_forecast, DENSITY_NY, DENSITY_NX)
        rho_test = rho_sub[T_train_frames:T_train_frames + T_avail_forecast]

        if do_simplex:
            for t in range(T_avail_forecast):
                rho_pred[t] = project_simplex(
                    rho_pred[t].ravel(), M0
                ).reshape(DENSITY_NY, DENSITY_NX)

        ss_tot = np.sum((rho_test - rho_test.mean()) ** 2)
        r2 = (1.0 - np.sum((rho_test - rho_pred) ** 2) / ss_tot
              if ss_tot > 1e-12 else 0.0)
        mass_err = (abs(rho_pred[-1].sum() - M0) / M0 * 100
                    if M0 > 1e-12 else 0.0)

        results.append({
            'test_idx': idx, 'r2_overall': float(r2),
            'mass_err_final_pct': float(mass_err), 'T_forecast': T_avail_forecast,
        })
        print(f"    test_{idx}: R2={r2:+.4f}  mass_err={mass_err:.1f}%")

    if not results:
        print("  !! No valid test runs")
        return None

    r2s = [r['r2_overall'] for r in results]
    summary = {
        'experiment': name, 'ABORTED': False,
        'mode': cfg['mode'], 'transform': transform, 'simplex': do_simplex,
        'H': H, 'rom_d': rom_d, 'lag': lag, 'alpha': alpha,
        'n_train': n_train, 'n_test': len(results),
        'r2_mean': float(np.mean(r2s)),
        'r2_std': float(np.std(r2s)),
        'r2_median': float(np.median(r2s)),
        'r2_min': float(np.min(r2s)),
        'r2_max': float(np.max(r2s)),
        'mass_err_mean': float(np.mean([r['mass_err_final_pct'] for r in results])),
        'mvar_r2_train': float(r2_train),
        'n_mvar_samples': diag['n_samples'],
        'n_mvar_features': diag['n_features'],
        'condition_number': diag['condition_number'],
        'pod_d': R,
        'timestamp': datetime.now().isoformat(),
    }

    print(f"\n  SUMMARY: R2={summary['r2_mean']:+.4f} +/- {summary['r2_std']:.4f}")
    print(f"  r2_train={r2_train:.4f}  samples={diag['n_samples']}  "
          f"features={diag['n_features']}  cond={diag['condition_number']:.1e}")

    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / 'per_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return summary


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Suite CONT_FIX V1: continuation vs generalization')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Run a single experiment by name')
    parser.add_argument('--group', type=str, default=None,
                        choices=['A', 'B', 'C', 'all'],
                        help='Run experiment group: A (CF1-6), B (CF7-10), '
                             'C (CONT3-4), or all')
    args = parser.parse_args()

    root = str(Path(__file__).parent.parent)

    # Select experiments
    if args.experiment:
        if args.experiment not in EXPERIMENTS:
            print(f"Unknown experiment: {args.experiment}")
            print(f"Available: {', '.join(EXPERIMENTS.keys())}")
            sys.exit(1)
        exp_list = {args.experiment: EXPERIMENTS[args.experiment]}
    elif args.group:
        if args.group == 'A':
            exp_list = {k: v for k, v in EXPERIMENTS.items()
                        if k.startswith('CF') and 'longPrefix' not in k}
        elif args.group == 'B':
            exp_list = {k: v for k, v in EXPERIMENTS.items()
                        if 'longPrefix' in k}
        elif args.group == 'C':
            exp_list = {k: v for k, v in EXPERIMENTS.items()
                        if k.startswith('CONT')}
        else:  # all
            exp_list = EXPERIMENTS
    else:
        exp_list = EXPERIMENTS

    print("=" * 80)
    print("  SUITE CONT_FIX (V1) -- Continuation vs Generalization")
    print(f"  Running {len(exp_list)} experiments")
    print("=" * 80)

    # Print experiment table
    print(f"\n  {'Name':<50s} {'d':>3s} {'p':>3s} {'alpha':>8s} {'H':>4s}")
    print("  " + "-" * 72)
    for exp_name, cfg in exp_list.items():
        print(f"  {exp_name:<50s} {cfg['rom_d']:>3d} {cfg['lag']:>3d} "
              f"{cfg['alpha']:>8.0e} {cfg['H']:>4d}")
    print()

    summaries = []
    t_start = time.time()

    for exp_name, cfg in exp_list.items():
        if cfg['mode'] == 'single_ic':
            s = run_single_ic_experiment(exp_name, cfg, root)
        elif cfg['mode'] == 'multi_ic_insample':
            s = run_multi_ic_insample_experiment(exp_name, cfg, root)
        else:
            print(f"Unknown mode: {cfg['mode']}")
            continue
        if s:
            summaries.append(s)

    elapsed = time.time() - t_start

    # Combined table
    print("\n\n" + "=" * 105)
    print("  SUITE CONT_FIX V1 -- COMBINED RESULTS")
    print("=" * 105)
    hdr = (f"  {'Experiment':<50s} {'R2':>8s} {'+/-':>6s} "
           f"{'MassErr':>8s} {'r2_tr':>6s} {'n_s':>5s} {'n_f':>5s} {'cond':>9s}")
    print(hdr)
    print("  " + "-" * 103)

    for s in summaries:
        if s.get('ABORTED', False):
            print(f"  {s['experiment']:<50s}  ** ABORTED: {s.get('reason', '?')}")
            continue
        n_s = s.get('n_samples', s.get('n_mvar_samples', '?'))
        n_f = s.get('n_features', s.get('n_mvar_features', '?'))
        cond = s.get('condition_number_mean', s.get('condition_number', '?'))
        r2_tr = s.get('r2_train_mean', s.get('mvar_r2_train', '?'))

        cond_str = f"{cond:.1e}" if isinstance(cond, float) else str(cond)
        r2_tr_str = f"{r2_tr:.4f}" if isinstance(r2_tr, float) else str(r2_tr)

        print(f"  {s['experiment']:<50s} {s['r2_mean']:>+7.4f} "
              f"{s['r2_std']:>5.4f} {s['mass_err_mean']:>7.1f}% "
              f"{r2_tr_str:>6s} {str(n_s):>5s} {str(n_f):>5s} {cond_str:>9s}")

    print(f"\n  Total wall time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Save combined
    combined_path = Path(root) / 'oscar_output' / 'CONT_fix_v1_combined.json'
    with open(combined_path, 'w') as f:
        json.dump(summaries, f, indent=2)
    print(f"  -> Combined results: {combined_path}")


if __name__ == '__main__':
    main()
