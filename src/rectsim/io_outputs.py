"""
Standardized I/O for all Vicsek-type simulations.

This module provides unified output functions for:
- Order parameter time series (CSV)
- Trajectory data (CSV and NPZ)
- Density fields (CSV)
- Summary plots (PNG)
- Animations (MP4)

All output functions work across discrete, continuous, and force-coupled models.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.stats import gaussian_kde
from pathlib import Path
import csv


def save_order_parameters_csv(times, metrics, output_path):
    """
    Save order parameter time series to CSV.
    
    Creates: order_parameters.csv with columns:
    - time
    - polarization
    - angular_momentum
    - mean_speed
    - density_variance
    
    Parameters
    ----------
    times : ndarray, shape (T,)
        Time points
    metrics : dict
        Dictionary with keys 'polarization', 'angular_momentum',
        'mean_speed', 'density_variance', each array of shape (T,)
    output_path : str or Path
        Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'polarization', 'angular_momentum', 
                        'mean_speed', 'density_variance'])
        
        for i, t in enumerate(times):
            writer.writerow([
                t,
                metrics['polarization'][i],
                metrics['angular_momentum'][i],
                metrics['mean_speed'][i],
                metrics['density_variance'][i]
            ])
    
    print(f"✓ Saved {output_path}")


def save_trajectory_csv(times, positions, velocities, output_path):
    """
    Save trajectory data to CSV.
    
    Creates: traj.csv with columns:
    - time, particle_id, x, y, vx, vy
    
    Parameters
    ----------
    times : ndarray, shape (T,)
        Time points
    positions : ndarray, shape (T, N, 2)
        Positions over time
    velocities : ndarray, shape (T, N, 2)
        Velocities over time
    output_path : str or Path
        Output file path
        
    Notes
    -----
    For large simulations, this can produce large files.
    Consider saving only every nth frame.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    T, N, _ = positions.shape
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'particle_id', 'x', 'y', 'vx', 'vy'])
        
        for t_idx, t in enumerate(times):
            for i in range(N):
                writer.writerow([
                    t,
                    i,
                    positions[t_idx, i, 0],
                    positions[t_idx, i, 1],
                    velocities[t_idx, i, 0],
                    velocities[t_idx, i, 1]
                ])
    
    print(f"✓ Saved {output_path}")


def save_density_csv(times, positions, domain_bounds, resolution, output_path,
                    bandwidth_mode="manual", manual_H=(3.0, 2.0), periodic_x=True):
    """
    Save KDE density fields to CSV using paper-accurate KDE (Alvarez et al., 2025).
    
    Creates: density.csv with columns:
    - time, x, y, density
    
    Parameters
    ----------
    times : ndarray, shape (T,)
        Time points
    positions : ndarray, shape (T, N, 2)
        Positions over time
    domain_bounds : tuple of (xmin, xmax, ymin, ymax)
        Domain boundaries
    resolution : int
        Grid resolution
    output_path : str or Path
        Output file path
    bandwidth_mode : str, optional
        "silverman" or "manual" (default: "manual")
    manual_H : tuple, optional
        Manual bandwidth (h_x, h_y) when bandwidth_mode="manual"
    periodic_x : bool, optional
        Apply periodic boundary handling in x-direction
    """
    from .kde_density import kde_density_snapshot
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    xmin, xmax, ymin, ymax = domain_bounds
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'x', 'y', 'density'])
        
        for t_idx, t in enumerate(times):
            try:
                # Use paper-accurate KDE
                rho, H, S, meta = kde_density_snapshot(
                    positions=positions[t_idx],
                    domain=(xmin, xmax, ymin, ymax),
                    nx=resolution,
                    ny=resolution,
                    bandwidth_mode=bandwidth_mode,
                    manual_H=manual_H,
                    periodic_x=periodic_x,
                    periodic_extension_n=5,
                    obstacle_rect=None
                )
                
                # Get grid coordinates
                x_grid = np.linspace(xmin, xmax, resolution)
                y_grid = np.linspace(ymin, ymax, resolution)
                X, Y = np.meshgrid(x_grid, y_grid)
                
                # Write density values
                for i in range(resolution):
                    for j in range(resolution):
                        writer.writerow([t, X[j, i], Y[j, i], rho[j, i]])
                        
            except (np.linalg.LinAlgError, ValueError):
                # KDE failed, write zeros
                x_grid = np.linspace(xmin, xmax, resolution)
                y_grid = np.linspace(ymin, ymax, resolution)
                X, Y = np.meshgrid(x_grid, y_grid)
                for i in range(resolution):
                    for j in range(resolution):
                        writer.writerow([t, X[j, i], Y[j, i], 0.0])
    
    print(f"✓ Saved {output_path}")


def plot_order_summary(times, metrics, output_path, figsize=(10, 8)):
    """
    Create summary plot with 4 subplots for all order parameters.
    
    Creates: order_summary.png with:
    - subplot(1): Polarization Φ(t)
    - subplot(2): Angular momentum L(t)
    - subplot(3): Mean speed ⟨|v|⟩(t)
    - subplot(4): Density variance Var(ρ)(t)
    
    All subplots share x-axis (time).
    
    Parameters
    ----------
    times : ndarray, shape (T,)
        Time points
    metrics : dict
        Dictionary with metric arrays
    output_path : str or Path
        Output file path
    figsize : tuple, optional
        Figure size (width, height)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    
    # Polarization
    axes[0].plot(times, metrics['polarization'], 'b-', linewidth=1.5)
    axes[0].set_ylabel('Polarization Φ', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.05, 1.05)
    
    # Angular momentum
    axes[1].plot(times, metrics['angular_momentum'], 'g-', linewidth=1.5)
    axes[1].set_ylabel('Angular Momentum L', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # Mean speed
    axes[2].plot(times, metrics['mean_speed'], 'r-', linewidth=1.5)
    axes[2].set_ylabel('Mean Speed ⟨|v|⟩', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    
    # Density variance
    axes[3].plot(times, metrics['density_variance'], 'm-', linewidth=1.5)
    axes[3].set_ylabel('Density Variance', fontsize=11)
    axes[3].set_xlabel('Time', fontsize=11)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved {output_path}")


def create_traj_animation(times, positions, velocities, domain_bounds,
                         output_path, fps=20, arrow_scale=1.0,
                         arrow_mode='speed', show_arrows=True,
                         figsize=(8, 8), dpi=100):
    """
    Create trajectory animation with optional velocity arrows.
    
    Creates: traj_animation.mp4
    - Particles shown as arrows colored by heading angle
    - Arrow length scaled by speed (proportional) or uniform
    
    Parameters
    ----------
    times : ndarray, shape (T,)
        Time points
    positions : ndarray, shape (T, N, 2)
        Positions over time
    velocities : ndarray, shape (T, N, 2)
        Velocities over time
    domain_bounds : tuple of (xmin, xmax, ymin, ymax)
        Domain boundaries
    output_path : str or Path
        Output file path
    fps : int, optional
        Frames per second (default: 20)
    arrow_scale : float, optional
        Scale factor for arrow sizes (default: 1.0)
    arrow_mode : str, optional
        'speed': arrow length proportional to speed (default)
        'uniform': fixed arrow length
    show_arrows : bool, optional
        If False, show particles as dots instead of arrows (default: True)
    figsize : tuple, optional
        Figure size
    dpi : int, optional
        Resolution
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    xmin, xmax, ymin, ymax = domain_bounds
    T, N, _ = positions.shape
    
    # Compute speeds and angles
    speeds = np.linalg.norm(velocities, axis=2)
    angles = np.arctan2(velocities[:, :, 1], velocities[:, :, 0])
    
    # Setup figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       verticalalignment='top', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if show_arrows:
        # Normalize arrows if uniform mode
        if arrow_mode == 'uniform':
            # Fixed arrow length, direction only
            vel_norm = velocities.copy()
            for t in range(T):
                norms = speeds[t, :, np.newaxis]
                norms[norms < 1e-10] = 1.0  # Avoid division by zero
                vel_norm[t] = velocities[t] / norms
        else:
            # Speed-proportional arrows (default)
            vel_norm = velocities
        
        # Initialize quiver
        quiver = ax.quiver(positions[0, :, 0], positions[0, :, 1],
                          vel_norm[0, :, 0], vel_norm[0, :, 1],
                          angles[0], cmap='hsv', clim=[-np.pi, np.pi],
                          scale=arrow_scale, scale_units='xy')
        
        # Add colorbar
        cbar = plt.colorbar(quiver, ax=ax, label='Heading angle')
        cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cbar.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
        
        def update(frame):
            quiver.set_offsets(positions[frame])
            quiver.set_UVC(vel_norm[frame, :, 0], vel_norm[frame, :, 1], angles[frame])
            time_text.set_text(f't = {times[frame]:.2f}')
            return quiver, time_text
    else:
        # Show as colored dots (no arrows)
        scatter = ax.scatter(positions[0, :, 0], positions[0, :, 1],
                           c=angles[0], cmap='hsv', vmin=-np.pi, vmax=np.pi,
                           s=50, alpha=0.8)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, label='Heading angle')
        cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cbar.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
        
        def update(frame):
            scatter.set_offsets(positions[frame])
            scatter.set_array(angles[frame])
            time_text.set_text(f't = {times[frame]:.2f}')
            return scatter, time_text
    
    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=True)
    
    # Save animation
    writer = FFMpegWriter(fps=fps, bitrate=1800)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close()
    
    print(f"✓ Saved {output_path}")


def create_density_animation(times, positions, domain_bounds, resolution,
                            output_path, fps=20, figsize=(8, 8), dpi=100,
                            vmin=None, vmax=None, bandwidth_mode="manual",
                            manual_H=(3.0, 2.0), periodic_x=True):
    """
    Create density field animation using paper-accurate KDE (Alvarez et al., 2025).
    
    Creates: density_animation.mp4
    - Heatmap of particle density
    - Fixed color scale across all frames
    
    Parameters
    ----------
    times : ndarray, shape (T,)
        Time points
    positions : ndarray, shape (T, N, 2)
        Positions over time
    domain_bounds : tuple of (xmin, xmax, ymin, ymax)
        Domain boundaries
    resolution : int
        Grid resolution
    output_path : str or Path
        Output file path
    fps : int, optional
        Frames per second
    figsize : tuple, optional
        Figure size
    dpi : int, optional
        Resolution
    vmin, vmax : float, optional
        Color scale limits. If None, computed from data.
    bandwidth_mode : str, optional
        "silverman" or "manual" (default: "manual")
    manual_H : tuple, optional
        Manual bandwidth (h_x, h_y) when bandwidth_mode="manual"
    periodic_x : bool, optional
        Apply periodic boundary handling in x-direction
    """
    from .kde_density import kde_density_snapshot
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    xmin, xmax, ymin, ymax = domain_bounds
    T = len(times)
    
    # Compute all density fields using paper-accurate KDE
    densities = []
    for t in range(T):
        try:
            rho, H, S, meta = kde_density_snapshot(
                positions=positions[t],
                domain=(xmin, xmax, ymin, ymax),
                nx=resolution,
                ny=resolution,
                bandwidth_mode=bandwidth_mode,
                manual_H=manual_H,
                periodic_x=periodic_x,
                periodic_extension_n=5,
                obstacle_rect=None
            )
            densities.append(rho)
        except (np.linalg.LinAlgError, ValueError):
            densities.append(np.zeros((resolution, resolution)))
    
    if vmin is None:
        vmin = min(d.min() for d in densities)
    if vmax is None:
        vmax = max(d.max() for d in densities)
    
    # Setup figure
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(densities[0], extent=[xmin, xmax, ymin, ymax],
                   origin='lower', cmap='hot', vmin=vmin, vmax=vmax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Density Field')
    
    plt.colorbar(im, ax=ax, label='Density')
    
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       verticalalignment='top', fontsize=12, color='white',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    def update(frame):
        im.set_data(densities[frame])
        time_text.set_text(f't = {times[frame]:.2f}')
        return im, time_text
    
    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=True)
    
    # Save animation
    writer = FFMpegWriter(fps=fps, bitrate=1800)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close()
    
    print(f"✓ Saved {output_path}")


def save_standardized_outputs(times, positions, velocities, domain_bounds,
                              output_dir, config_outputs, boundary_condition="periodic"):
    """
    Generate all standardized outputs based on config.
    
    This is the main entry point for output generation.
    
    Parameters
    ----------
    times : ndarray, shape (T,)
        Time points
    positions : ndarray, shape (T, N, 2)
        Positions over time
    velocities : ndarray, shape (T, N, 2)
        Velocities over time
    domain_bounds : tuple of (xmin, xmax, ymin, ymax)
        Domain boundaries
    output_dir : str or Path
        Output directory
    config_outputs : dict
        Output configuration with keys:
        - order_parameters: bool
        - animations: bool
        - save_csv: bool
        - fps: int
        - density_resolution: int
    boundary_condition : str, optional
        "periodic" or "reflecting" (default: "periodic")
        Used for paper-accurate KDE with periodic boundary handling
        
    Returns
    -------
    dict
        Dictionary of computed metrics (if order_parameters=True)
    """
    from .standard_metrics import compute_metrics_series
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract config
    compute_metrics = config_outputs.get('order_parameters', True)
    make_animations = config_outputs.get('animations', True)
    save_csv = config_outputs.get('save_csv', True)
    fps = config_outputs.get('fps', 20)
    resolution = config_outputs.get('density_resolution', 100)
    
    # KDE configuration (paper defaults)
    bandwidth_mode = config_outputs.get('kde_bandwidth_mode', 'manual')
    manual_H = tuple(config_outputs.get('kde_manual_H', [3.0, 2.0]))
    periodic_x = (boundary_condition == "periodic")
    
    # Arrow configuration
    arrows_config = config_outputs.get('arrows', {})
    show_arrows = arrows_config.get('enabled', True)
    arrow_mode = arrows_config.get('scale', 'speed')  # 'speed' or 'uniform'
    arrow_scale_factor = arrows_config.get('scale_factor', 1.0)
    
    metrics = None
    
    # Compute metrics
    if compute_metrics:
        print("\nComputing order parameters...")
        metrics = compute_metrics_series(positions, velocities, domain_bounds,
                                        resolution=resolution, verbose=True,
                                        boundary_condition=boundary_condition,
                                        bandwidth_mode=bandwidth_mode,
                                        manual_H=manual_H)
        
        # Save order parameters CSV
        save_order_parameters_csv(times, metrics, 
                                 output_dir / 'order_parameters.csv')
        
        # Plot summary
        plot_order_summary(times, metrics,
                          output_dir / 'order_summary.png')
    
    # Save trajectory CSV
    if save_csv:
        print("\nSaving trajectory data...")
        save_trajectory_csv(times, positions, velocities,
                           output_dir / 'traj.csv')
        
        print("\nSaving density data...")
        save_density_csv(times, positions, domain_bounds, resolution,
                        output_dir / 'density.csv',
                        bandwidth_mode=bandwidth_mode,
                        manual_H=manual_H,
                        periodic_x=periodic_x)
    
    # Create animations
    if make_animations:
        # Check if ffmpeg is available
        import shutil
        ffmpeg_available = shutil.which('ffmpeg') is not None
        
        if not ffmpeg_available:
            print("\n⚠️  Warning: ffmpeg not found!")
            print("   Animations require ffmpeg to be installed.")
            print("   Install with: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)")
            print("   You can create animations later using: python scripts/create_animations.py")
            print(f"   {output_dir / 'results.npz'}\n")
        else:
            print("\nCreating trajectory animation...")
            try:
                create_traj_animation(times, positions, velocities, domain_bounds,
                                    output_dir / 'traj_animation.mp4', 
                                    fps=fps, arrow_scale=arrow_scale_factor,
                                    arrow_mode=arrow_mode, show_arrows=show_arrows)
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                print(f"  You can retry later with: python scripts/create_animations.py {output_dir / 'results.npz'}")
            
            print("\nCreating density animation...")
            try:
                create_density_animation(times, positions, domain_bounds, resolution,
                                       output_dir / 'density_animation.mp4',
                                       fps=fps,
                                       bandwidth_mode=bandwidth_mode,
                                       manual_H=manual_H,
                                       periodic_x=periodic_x)
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                print(f"  You can retry later with: python scripts/create_animations.py {output_dir / 'results.npz'}")
    
    return metrics
