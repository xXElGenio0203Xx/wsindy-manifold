#!/usr/bin/env python3
"""
Unified thesis data extraction: extract ALL lightweight artifacts from a
single experiment in ONE pass (loading heavy density files only once per
test run).

Outputs (written to the experiment directory):
  1. kde_snapshots.npz     — density frames + particle positions at target times
  2. mass_timeseries.npz   — total mass over full forecast for all models
  3. spatial_order.npz     — spatial order (std ρ) time-series per model
  4. lp_errors.npz         — relative L1/L2/L∞ error time-series per model per test

Usage (single experiment):
    python scripts/extract_thesis_data.py oscar_output/DO_DM01_dmill_C09_l05

Usage (from SLURM array via manifest):
    # The SLURM wrapper reads the config name from manifest.txt,
    # derives the experiment name, and calls this script.
"""
import argparse
import sys
import numpy as np
from pathlib import Path


# Target times for KDE snapshots (seconds)
TARGET_TIMES = [0.6, 3.0, 25.0, 75.0, 200.0]

# Models to extract predictions for
MODELS = ['mvar', 'lstm', 'wsindy']


def _compute_spatial_order(rho):
    """Compute spatial order parameter: std(ρ) per frame."""
    return np.array([np.std(rho[t]) for t in range(rho.shape[0])])


def _compute_errors(rho_true, rho_pred):
    """Compute relative L1, L2, L∞ errors per frame."""
    T = rho_true.shape[0]
    e1, e2, einf = [], [], []
    for t in range(T):
        diff = rho_true[t] - rho_pred[t]
        norm_true_l2 = np.sqrt(np.sum(rho_true[t] ** 2))
        norm_true_l1 = np.sum(np.abs(rho_true[t]))
        norm_true_linf = np.max(np.abs(rho_true[t]))
        e2.append(np.sqrt(np.sum(diff ** 2)) / (norm_true_l2 + 1e-12))
        e1.append(np.sum(np.abs(diff)) / (norm_true_l1 + 1e-12))
        einf.append(np.max(np.abs(diff)) / (norm_true_linf + 1e-12))
    return np.array(e1), np.array(e2), np.array(einf)


def _load_density_pred(test_run, model):
    """Find and load the density prediction file for a model."""
    # Model-specific file first
    pred_path = test_run / f'density_pred_{model}.npz'
    if not pred_path.exists() and model == 'wsindy':
        pred_path = test_run / 'density_pred_wsindy.npz'
    if not pred_path.exists():
        # Generic fallback (only if no model-specific files exist)
        generic = test_run / 'density_pred.npz'
        if generic.exists():
            pred_path = generic
        else:
            return None, None
    dp = np.load(pred_path)
    return dp['rho'], dp['times']


def extract_experiment(exp_dir, target_times=None, force=False):
    """Extract all thesis data from one experiment."""
    exp_dir = Path(exp_dir)
    if target_times is None:
        target_times = TARGET_TIMES

    exp_name = exp_dir.name
    test_base = exp_dir / 'test'
    if not test_base.exists():
        print(f"  SKIP {exp_name}: no test/ directory")
        return False

    # Discover all test runs
    test_runs = sorted([d for d in test_base.iterdir()
                        if d.is_dir() and d.name.startswith('test_')])
    if not test_runs:
        print(f"  SKIP {exp_name}: no test_NNN/ directories")
        return False

    print(f"\n  Processing {exp_name} ({len(test_runs)} test runs) ...")

    # -------------------------------------------------------------------
    # Storage accumulators
    # -------------------------------------------------------------------
    kde_dict = {}
    mass_dict = {}
    so_dict = {}
    lp_dict = {}

    # Track if we've extracted KDE (only from test_000)
    kde_extracted = False

    # Track spatial order: accumulate per-model across test_000
    # (spatial order is expensive; just use test_000)
    so_models_done = set()

    for test_run in test_runs:
        test_idx = int(test_run.name.split('_')[1])
        test_tag = f'test{test_idx:03d}'

        # Load true density (needed for everything)
        true_path = test_run / 'density_true.npz'
        if not true_path.exists():
            print(f"    SKIP {test_run.name}: no density_true.npz")
            continue

        dens_true = np.load(true_path)
        rho_true = dens_true['rho']        # (T, ny, nx)
        times_true = dens_true['times']

        # ---------------------------------------------------------------
        # 1. KDE Snapshots (from test_000 only)
        # ---------------------------------------------------------------
        if test_idx == 0 and not kde_extracted:
            traj_path = test_run / 'trajectory.npz'
            if traj_path.exists():
                traj_data = np.load(traj_path)
                x_traj = traj_data['traj']          # (T_traj, N, 2)
                times_traj = traj_data['times']

                actual_times = []
                for target_t in target_times:
                    traj_idx = int(np.argmin(np.abs(times_traj - target_t)))
                    dens_idx = int(np.argmin(np.abs(times_true - target_t)))
                    actual_t = float(f'{times_true[dens_idx]:.1f}')
                    actual_times.append(actual_t)

                    kde_dict[f'particles_t{actual_t}'] = x_traj[traj_idx]
                    kde_dict[f'rho_true_t{actual_t}'] = rho_true[dens_idx]

                kde_dict['times_actual'] = np.array(actual_times)
                kde_extracted = True
            else:
                print(f"    WARN: no trajectory.npz in {test_run.name}")

        # ---------------------------------------------------------------
        # Process each model's predictions
        # ---------------------------------------------------------------
        for model in MODELS:
            rho_pred, times_pred = _load_density_pred(test_run, model)
            if rho_pred is None:
                continue

            # Align lengths (prediction may be shorter than truth)
            T_pred = rho_pred.shape[0]

            # For mass + spatial order, we need aligned true/pred
            # Find the forecast start in true density
            # Use time alignment: find where times_pred[0] falls in times_true
            start_idx = int(np.argmin(np.abs(times_true - times_pred[0])))
            rho_true_aligned = rho_true[start_idx:start_idx + T_pred]
            if len(rho_true_aligned) < T_pred:
                T_pred = len(rho_true_aligned)
                rho_pred = rho_pred[:T_pred]
                times_pred = times_pred[:T_pred]
                rho_true_aligned = rho_true_aligned[:T_pred]

            # -----------------------------------------------------------
            # 1b. KDE model predictions (test_000 only)
            # -----------------------------------------------------------
            if test_idx == 0 and kde_extracted:
                for actual_t in kde_dict.get('times_actual', []):
                    pidx = int(np.argmin(np.abs(times_pred - actual_t)))
                    kde_dict[f'rho_pred_{model}_t{actual_t}'] = rho_pred[pidx]

            # -----------------------------------------------------------
            # 2. Mass time-series (test_000 only for simplicity)
            # -----------------------------------------------------------
            if test_idx == 0:
                mass_true = np.array([float(np.sum(rho_true_aligned[t]))
                                      for t in range(T_pred)])
                mass_pred = np.array([float(np.sum(rho_pred[t]))
                                      for t in range(T_pred)])
                mass_dict[f'mass_pred_{model}'] = mass_pred
                if 'times' not in mass_dict:
                    mass_dict['times'] = times_pred
                    mass_dict['mass_true'] = mass_true

            # -----------------------------------------------------------
            # 3. Spatial order (test_000 only)
            # -----------------------------------------------------------
            if test_idx == 0 and model not in so_models_done:
                so_pred = _compute_spatial_order(rho_pred)
                so_dict[f'so_pred_{model}'] = so_pred
                if 'times' not in so_dict:
                    so_dict['times'] = times_pred       # forecast time axis
                    # Compute true spatial order from FULL rho_true (t=0 onwards)
                    so_dict['so_true'] = _compute_spatial_order(rho_true)
                    so_dict['times_true'] = times_true  # full time axis
                so_models_done.add(model)

            # -----------------------------------------------------------
            # 4. L^p errors (all test runs)
            # -----------------------------------------------------------
            e1, e2, einf = _compute_errors(rho_true_aligned, rho_pred)
            lp_dict[f'rel_e1_{model}_{test_tag}'] = e1
            lp_dict[f'rel_e2_{model}_{test_tag}'] = e2
            lp_dict[f'rel_einf_{model}_{test_tag}'] = einf
            lp_dict[f'times_{model}_{test_tag}'] = times_pred

    # -------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------

    # 1. KDE snapshots
    kde_out = exp_dir / 'kde_snapshots.npz'
    if kde_dict and (force or not kde_out.exists()):
        np.savez_compressed(kde_out, **kde_dict)
        size_kb = kde_out.stat().st_size / 1024
        print(f"    -> kde_snapshots.npz ({size_kb:.0f} KB)")
    elif kde_out.exists() and not force:
        print(f"    SKIP kde_snapshots.npz (exists)")

    # 2. Mass time-series
    mass_out = exp_dir / 'mass_timeseries.npz'
    if mass_dict and (force or not mass_out.exists()):
        np.savez_compressed(mass_out, **mass_dict)
        size_kb = mass_out.stat().st_size / 1024
        print(f"    -> mass_timeseries.npz ({size_kb:.0f} KB)")
    elif mass_out.exists() and not force:
        print(f"    SKIP mass_timeseries.npz (exists)")

    # 3. Spatial order
    so_out = exp_dir / 'spatial_order.npz'
    if so_dict and (force or not so_out.exists()):
        np.savez_compressed(so_out, **so_dict)
        size_kb = so_out.stat().st_size / 1024
        print(f"    -> spatial_order.npz ({size_kb:.0f} KB)")
    elif so_out.exists() and not force:
        print(f"    SKIP spatial_order.npz (exists)")

    # 4. L^p errors
    lp_out = exp_dir / 'lp_errors.npz'
    if lp_dict and (force or not lp_out.exists()):
        np.savez_compressed(lp_out, **lp_dict)
        size_kb = lp_out.stat().st_size / 1024
        print(f"    -> lp_errors.npz ({size_kb:.0f} KB)")
    elif lp_out.exists() and not force:
        print(f"    SKIP lp_errors.npz (exists)")

    # 5. Mass-conservation figure (render on Oscar, export PNG)
    mass_fig_out = exp_dir / 'mass_conservation_plot.png'
    if mass_out.exists() and (force or not mass_fig_out.exists()):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            mdata = np.load(mass_out)
            if 'times' in mdata and 'mass_true' in mdata:
                times = mdata['times']
                m_true = mdata['mass_true']
                M0 = m_true[0] if m_true[0] != 0 else 1.0

                colors = {'mvar': '#1f77b4', 'lstm': '#d62728', 'wsindy': '#2ca02c'}
                labels = {'mvar': 'MVAR', 'lstm': 'LSTM', 'wsindy': 'WSINDy'}

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.axhline(1.0, color='black', ls='--', lw=1.5, alpha=0.7,
                            label='True mass')
                for model in MODELS:
                    key = f'mass_pred_{model}'
                    if key in mdata:
                        m_pred = mdata[key]
                        ax.plot(times[:len(m_pred)], m_pred / M0, '-',
                                color=colors[model], lw=1.5, alpha=0.8,
                                label=labels[model])
                ax.set_xlabel('Time (s)')
                ax.set_ylabel(r'$M(t)/M_0$')
                ax.set_title(f'{exp_name} — Mass Conservation', fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(mass_fig_out, dpi=200, bbox_inches='tight')
                plt.close(fig)
                print(f"    -> mass_conservation_plot.png")
        except Exception as e:
            print(f"    WARN: mass plot failed: {e}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Extract ALL thesis data from a single experiment')
    parser.add_argument('exp_dir', type=str,
                        help='Path to experiment directory '
                             '(e.g. oscar_output/DO_DM01_dmill_C09_l05)')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing output files')
    parser.add_argument('--times', nargs='*', type=float, default=None,
                        help=f'Target snapshot times (default: {TARGET_TIMES})')
    args = parser.parse_args()

    extract_experiment(args.exp_dir, target_times=args.times, force=args.force)


if __name__ == '__main__':
    main()
