#!/usr/bin/env python3
"""
Verify KDE density is correctly aligned with particle trajectories.

Creates a side-by-side figure:
  1) Particle scatter at time t
  2) Ground-truth density at time t with particle overlay
  3) Predicted density at time t with particle overlay

If KDE axes are transposed relative to trajectories, the red dots
will NOT overlap the density peaks.
"""
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def verify_alignment(exp='DYN1_gentle_v2', test_id=0, target_times=[2.0, 5.0, 10.0, 20.0]):
    base = Path('oscar_output') / exp / 'test' / f'test_{test_id:03d}'
    
    traj_data = np.load(base / 'trajectory.npz')
    dens_true = np.load(base / 'density_true.npz')
    dens_pred = np.load(base / 'density_pred.npz')
    
    x = traj_data['traj']         # (T_traj, N, 2)
    times_traj = traj_data['times']
    rho_true = dens_true['rho']   # (T_dens, ny, nx)
    times_dens = dens_true['times']
    rho_pred = dens_pred['rho']
    times_pred = dens_pred['times']
    forecast_start = int(dens_pred.get('forecast_start_idx', 0))
    
    print(f"Experiment: {exp}, test_{test_id:03d}")
    print(f"  Trajectory:   shape={x.shape}, t=[{times_traj[0]:.2f}, {times_traj[-1]:.2f}], dt={times_traj[1]-times_traj[0]:.4f}")
    print(f"  Density true: shape={rho_true.shape}, t=[{times_dens[0]:.2f}, {times_dens[-1]:.2f}], dt={times_dens[1]-times_dens[0]:.4f}")
    print(f"  Density pred: shape={rho_pred.shape}, t=[{times_pred[0]:.2f}, {times_pred[-1]:.2f}], dt={times_pred[1]-times_pred[0]:.4f}")
    print(f"  Forecast starts at idx {forecast_start} (t={times_pred[forecast_start]:.2f})")
    
    # Load domain bounds
    cfg_path = Path('oscar_output') / exp / 'config_used.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    Lx = cfg['sim']['Lx']
    Ly = cfg['sim']['Ly']
    print(f"  Domain: Lx={Lx}, Ly={Ly}")
    
    n_times = len(target_times)
    fig, axes = plt.subplots(n_times, 3, figsize=(18, 5*n_times))
    if n_times == 1:
        axes = axes[np.newaxis, :]
    
    for row, target_t in enumerate(target_times):
        traj_idx = np.argmin(np.abs(times_traj - target_t))
        dens_idx = np.argmin(np.abs(times_dens - target_t))
        pred_idx = np.argmin(np.abs(times_pred - target_t))
        
        pos = x[traj_idx]  # (N, 2)
        
        # Col 0: Particles only
        ax = axes[row, 0]
        ax.scatter(pos[:, 0], pos[:, 1], s=3, alpha=0.6, c='blue')
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)
        ax.set_aspect('equal')
        ax.set_xlabel('x = traj[:, 0]')
        ax.set_ylabel('y = traj[:, 1]')
        ax.set_title(f'Particles (t={times_traj[traj_idx]:.2f}s)')
        
        # Col 1: Ground truth density + particles
        ax = axes[row, 1]
        vmax = max(rho_true[dens_idx].max(), 1e-6)
        im = ax.imshow(rho_true[dens_idx], origin='lower', extent=[0, Lx, 0, Ly],
                       cmap='viridis', vmin=0, vmax=vmax)
        ax.scatter(pos[:, 0], pos[:, 1], s=1, c='red', alpha=0.4)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'True density + particles (t={times_dens[dens_idx]:.2f}s)')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Col 2: Predicted density + particles
        ax = axes[row, 2]
        is_forecast = pred_idx >= forecast_start
        vmax_p = max(rho_pred[pred_idx].max(), 1e-6)
        im = ax.imshow(rho_pred[pred_idx], origin='lower', extent=[0, Lx, 0, Ly],
                       cmap='viridis', vmin=0, vmax=vmax_p)
        ax.scatter(pos[:, 0], pos[:, 1], s=1, c='red', alpha=0.4)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        label = "FORECAST" if is_forecast else "conditioning"
        ax.set_title(f'Pred density [{label}] + particles (t={times_pred[pred_idx]:.2f}s)')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    fig.suptitle(f'{exp} test_{test_id:03d} — KDE ↔ Trajectory Alignment Check\n'
                 f'Red dots = particle positions. They should overlap density peaks.',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    out = Path('artifacts/thesis_figures') / f'kde_alignment_check_{exp}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: {out}")


def recompute_and_compare(exp='DYN1_gentle_v2', test_id=0, target_t=5.0):
    """Recompute KDE from trajectory and compare with stored density."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.rectsim.legacy_functions import kde_density_movie
    
    base = Path('oscar_output') / exp / 'test' / f'test_{test_id:03d}'
    traj_data = np.load(base / 'trajectory.npz')
    dens_true = np.load(base / 'density_true.npz')
    
    x = traj_data['traj']
    times_traj = traj_data['times']
    rho_stored = dens_true['rho']
    times_dens = dens_true['times']
    
    cfg_path = Path('oscar_output') / exp / 'config_used.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    Lx = cfg['sim']['Lx']
    Ly = cfg['sim']['Ly']
    
    # Recompute KDE from trajectory using same parameters as config
    bw = cfg.get('density', {}).get('bandwidth', cfg.get('outputs', {}).get('density_bandwidth', 2.0))
    nx = cfg.get('density', {}).get('nx', cfg.get('outputs', {}).get('density_resolution', 64))
    ny = cfg.get('density', {}).get('ny', nx)
    print(f"\nRecomputing KDE from trajectory (nx={nx}, ny={ny}, bw={bw})...")
    rho_recomp, meta = kde_density_movie(
        x, Lx=Lx, Ly=Ly, nx=nx, ny=ny, bandwidth=bw, bc='periodic'
    )
    print(f"  Recomputed shape: {rho_recomp.shape}")
    print(f"  Stored shape:     {rho_stored.shape}")
    
    # Find matching frame
    dens_idx = np.argmin(np.abs(times_dens - target_t))
    traj_idx = np.argmin(np.abs(times_traj - target_t))
    
    # The stored density may be at a different time resolution
    # Find the trajectory frame closest to the stored density time
    dens_time = times_dens[dens_idx]
    traj_idx_for_dens = np.argmin(np.abs(times_traj - dens_time))
    
    print(f"\n  Comparing at t={dens_time:.3f}:")
    print(f"    Stored density frame {dens_idx}")
    print(f"    Recomputed from traj frame {traj_idx_for_dens}")
    
    diff = rho_stored[dens_idx] - rho_recomp[traj_idx_for_dens]
    print(f"    Max abs diff: {np.abs(diff).max():.6f}")
    print(f"    Stored  max:  {rho_stored[dens_idx].max():.4f}")
    print(f"    Recomp  max:  {rho_recomp[traj_idx_for_dens].max():.4f}")
    
    # Check if transposed version matches better
    diff_T = rho_stored[dens_idx] - rho_recomp[traj_idx_for_dens].T
    print(f"    Max abs diff (transposed): {np.abs(diff_T).max():.6f}")
    
    if np.abs(diff).max() < 0.01:
        print("  ✓ MATCH: Stored density matches recomputed KDE")
    elif np.abs(diff_T).max() < 0.01 and np.abs(diff).max() > 0.01:
        print("  ✗ TRANSPOSED: Stored density matches TRANSPOSED recomputed KDE!")
    else:
        print(f"  ⚠ Neither version is a close match — may be different bandwidth or subsampling")

    # Also check: is the stored density the same as recomputed at same trajectory index?
    # (accounting for possible subsample factor)
    subsample_ratio = len(times_traj) / len(times_dens)
    print(f"\n  Time subsampling ratio: {subsample_ratio:.1f}x")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    im = ax.imshow(rho_stored[dens_idx], origin='lower', extent=[0, Lx, 0, Ly], cmap='viridis')
    ax.set_title('Stored density_true')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = axes[1]
    im = ax.imshow(rho_recomp[traj_idx_for_dens], origin='lower', extent=[0, Lx, 0, Ly], cmap='viridis')
    ax.set_title('Recomputed KDE')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = axes[2]
    dmax = max(abs(diff.min()), abs(diff.max()), 1e-6)
    im = ax.imshow(diff, origin='lower', extent=[0, Lx, 0, Ly], cmap='RdBu_r', vmin=-dmax, vmax=dmax)
    ax.set_title('Stored - Recomputed')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    fig.suptitle(f'{exp} — Stored vs Recomputed KDE (t={dens_time:.2f}s)', fontsize=13)
    fig.tight_layout()
    out = Path('artifacts/thesis_figures') / f'kde_recompute_check_{exp}.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


if __name__ == '__main__':
    import sys
    exp = sys.argv[1] if len(sys.argv) > 1 else 'DYN1_gentle_v2'
    
    # Test 1: Visual alignment check
    verify_alignment(exp, test_id=0, target_times=[2.0, 5.0, 15.0, 30.0])
    
    # Test 2: Recompute and numerical comparison
    recompute_and_compare(exp, test_id=0, target_t=5.0)
