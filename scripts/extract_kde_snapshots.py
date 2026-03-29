#!/usr/bin/env python3
"""
Extract lightweight KDE snapshots + mass time-series from a single experiment.
Designed to run on Oscar as an array job, producing small .npz files
that can be rsynced back for local thesis-figure generation.

Outputs (written to the experiment directory):
  1. kde_snapshots.npz   — density frames + particle positions at target times
  2. mass_timeseries.npz — total mass over full forecast for MVAR & LSTM

Usage (single experiment):
    python scripts/extract_kde_snapshots.py oscar_output/DO_DM01_dmill_C09_l05

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
MODELS = ['mvar', 'lstm']


def extract_experiment(exp_dir, test_idx=0, target_times=None, force=False):
    """Extract snapshots and mass time-series from one experiment."""
    exp_dir = Path(exp_dir)
    if target_times is None:
        target_times = TARGET_TIMES

    exp_name = exp_dir.name
    test_run = exp_dir / 'test' / f'test_{test_idx:03d}'
    if not test_run.exists():
        print(f"  SKIP {exp_name}: no test_{test_idx:03d}/")
        return False

    # -------------------------------------------------------------------
    # 1. KDE Snapshots
    # -------------------------------------------------------------------
    kde_out = exp_dir / 'kde_snapshots.npz'
    if kde_out.exists() and not force:
        print(f"  SKIP kde_snapshots.npz (exists) for {exp_name}")
    else:
        traj_path = test_run / 'trajectory.npz'
        true_path = test_run / 'density_true.npz'
        if not traj_path.exists() or not true_path.exists():
            print(f"  SKIP {exp_name}: missing trajectory.npz or density_true.npz")
        else:
            print(f"  Extracting KDE snapshots for {exp_name} ...")
            traj_data = np.load(traj_path)
            dens_true = np.load(true_path)

            x_traj = traj_data['traj']            # (T_traj, N, 2)
            times_traj = traj_data['times']
            rho_true = dens_true['rho']            # (T_dens, ny, nx)
            times_dens = dens_true['times']

            save_dict = {}
            actual_times = []

            for target_t in target_times:
                traj_idx = int(np.argmin(np.abs(times_traj - target_t)))
                dens_idx = int(np.argmin(np.abs(times_dens - target_t)))
                actual_t = float(f'{times_dens[dens_idx]:.1f}')
                actual_times.append(actual_t)

                save_dict[f'particles_t{actual_t}'] = x_traj[traj_idx]
                save_dict[f'rho_true_t{actual_t}'] = rho_true[dens_idx]

            # Model predictions at the same target times
            for model in MODELS:
                pred_path = test_run / f'density_pred_{model}.npz'
                if not pred_path.exists():
                    pred_path = test_run / 'density_pred.npz'
                if not pred_path.exists():
                    continue
                dp = np.load(pred_path)
                rho_pred = dp['rho']
                times_pred = dp['times']
                for actual_t in actual_times:
                    pidx = int(np.argmin(np.abs(times_pred - actual_t)))
                    save_dict[f'rho_pred_{model}_t{actual_t}'] = rho_pred[pidx]

            save_dict['times_actual'] = np.array(actual_times)
            np.savez_compressed(kde_out, **save_dict)
            size_kb = kde_out.stat().st_size / 1024
            print(f"    -> {kde_out.name} ({size_kb:.0f} KB, "
                  f"{len(actual_times)} snapshots)")

    # -------------------------------------------------------------------
    # 2. Mass Time-Series
    # -------------------------------------------------------------------
    mass_out = exp_dir / 'mass_timeseries.npz'
    if mass_out.exists() and not force:
        print(f"  SKIP mass_timeseries.npz (exists) for {exp_name}")
    else:
        true_path = test_run / 'density_true.npz'
        if not true_path.exists():
            print(f"  SKIP mass: no density_true.npz for {exp_name}")
        else:
            print(f"  Extracting mass time-series for {exp_name} ...")
            dens_true = np.load(true_path)
            rho_true = dens_true['rho']
            times_true = dens_true['times']

            mass_dict = {}
            # True mass (before alignment — use raw frames)
            mass_true = np.array([float(np.sum(rho_true[t]))
                                  for t in range(rho_true.shape[0])])

            for model in MODELS:
                pred_path = test_run / f'density_pred_{model}.npz'
                if not pred_path.exists():
                    pred_path = test_run / 'density_pred.npz'
                if not pred_path.exists():
                    continue
                dp = np.load(pred_path)
                rho_pred = dp['rho']
                times_pred = dp['times']
                start_idx = int(dp.get('forecast_start_idx', 0))

                mass_pred = np.array([float(np.sum(rho_pred[t]))
                                      for t in range(rho_pred.shape[0])])

                # Align true mass to the prediction time window
                mass_true_aligned = mass_true[start_idx:start_idx + len(mass_pred)]
                if len(mass_true_aligned) < len(mass_pred):
                    mass_pred = mass_pred[:len(mass_true_aligned)]
                    times_pred = times_pred[:len(mass_true_aligned)]

                mass_dict[f'mass_pred_{model}'] = mass_pred
                if 'times' not in mass_dict:
                    mass_dict['times'] = times_pred
                    mass_dict['mass_true'] = mass_true_aligned

            if mass_dict:
                np.savez_compressed(mass_out, **mass_dict)
                size_kb = mass_out.stat().st_size / 1024
                print(f"    -> {mass_out.name} ({size_kb:.0f} KB)")
            else:
                print(f"    No model predictions found for mass.")

    # -------------------------------------------------------------------
    # 3. Mass-conservation figure (render on Oscar, export PNG)
    # -------------------------------------------------------------------
    mass_fig_out = exp_dir / 'mass_conservation_plot.png'
    if mass_fig_out.exists() and not force:
        print(f"  SKIP mass_conservation_plot.png (exists) for {exp_name}")
    elif mass_out.exists():
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        mdata = np.load(mass_out)
        if 'times' in mdata and 'mass_true' in mdata:
            times = mdata['times']
            m_true = mdata['mass_true']
            M0 = m_true[0] if m_true[0] != 0 else 1.0

            colors = {'mvar': '#1f77b4', 'lstm': '#d62728'}
            labels = {'mvar': 'MVAR', 'lstm': 'LSTM'}

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

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Extract KDE snapshots + mass time-series from experiment')
    parser.add_argument('exp_dir', type=str,
                        help='Path to experiment directory '
                             '(e.g. oscar_output/DO_DM01_dmill_C09_l05)')
    parser.add_argument('--test_idx', type=int, default=0,
                        help='Test run index (default: 0)')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing output files')
    parser.add_argument('--times', nargs='*', type=float, default=None,
                        help=f'Target snapshot times (default: {TARGET_TIMES})')
    args = parser.parse_args()

    extract_experiment(args.exp_dir, test_idx=args.test_idx,
                       target_times=args.times, force=args.force)


if __name__ == '__main__':
    main()
