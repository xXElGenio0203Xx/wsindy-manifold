"""
Movie generation for CrowdROM pipeline.

Generates three types of movies:
1. Trajectory movie: Agent positions/velocities over time
2. Density movie: KDE density field heatmap
3. Latent movie: POD latent coordinates time series
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Tuple, Optional, List


def create_trajectory_movie(
    times: np.ndarray,
    positions: np.ndarray,
    velocities: Optional[np.ndarray],
    domain: dict,
    output_path: Path,
    fps: int = 20,
    max_frames: int = 500,
    obstacles: Optional[List[dict]] = None
) -> None:
    """
    Create trajectory movie showing agent positions over time.
    
    Parameters
    ----------
    times : ndarray, shape (n_frames,)
        Time values
    positions : ndarray, shape (n_frames, n_agents, 2)
        Agent positions
    velocities : ndarray or None, shape (n_frames, n_agents, 2)
        Agent velocities (for quiver plot). If None, uses scatter plot
    domain : dict
        Domain bounds: {"xmin": 0, "xmax": 20, "ymin": 0, "ymax": 20}
    output_path : Path
        Output MP4 file path
    fps : int, default=20
        Frames per second
    max_frames : int, default=500
        Maximum frames (subsample if needed)
    obstacles : list of dict, optional
        Obstacles to overlay: [{"type": "rect", "xmin": ..., ...}]
    """
    n_frames = len(times)
    n_agents = positions.shape[1]
    
    # Subsample if needed
    if n_frames > max_frames:
        indices = np.linspace(0, n_frames - 1, max_frames, dtype=int)
        times_sub = times[indices]
        positions_sub = positions[indices]
        velocities_sub = velocities[indices] if velocities is not None else None
    else:
        times_sub = times
        positions_sub = positions
        velocities_sub = velocities
    
    n_frames_sub = len(times_sub)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    xmin = domain.get("xmin", 0)
    xmax = domain.get("xmax", 20)
    ymin = domain.get("ymin", 0)
    ymax = domain.get("ymax", 20)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    
    # Draw obstacles
    if obstacles:
        for obs in obstacles:
            if obs["type"] == "rect":
                rect = Rectangle(
                    (obs["xmin"], obs["ymin"]),
                    obs["xmax"] - obs["xmin"],
                    obs["ymax"] - obs["ymin"],
                    facecolor="gray",
                    alpha=0.3,
                    edgecolor="black"
                )
                ax.add_patch(rect)
    
    # Initialize plot elements
    if velocities_sub is not None:
        # Quiver plot
        quiver = ax.quiver(
            positions_sub[0, :, 0],
            positions_sub[0, :, 1],
            velocities_sub[0, :, 0],
            velocities_sub[0, :, 1],
            scale=20,
            color="blue",
            alpha=0.7
        )
        title = ax.set_title(f"t = {times_sub[0]:.2f}")
    else:
        # Scatter plot
        scatter = ax.scatter(
            positions_sub[0, :, 0],
            positions_sub[0, :, 1],
            s=20,
            c="blue",
            alpha=0.7
        )
        title = ax.set_title(f"t = {times_sub[0]:.2f}")
    
    def update(frame):
        """Update function for animation."""
        if velocities_sub is not None:
            # Update quiver
            quiver.set_offsets(positions_sub[frame, :, :])
            quiver.set_UVC(velocities_sub[frame, :, 0], velocities_sub[frame, :, 1])
        else:
            # Update scatter
            scatter.set_offsets(positions_sub[frame, :, :])
        
        title.set_text(f"t = {times_sub[frame]:.2f}")
        return (quiver if velocities_sub is not None else scatter, title)
    
    # Create animation
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames_sub,
        interval=1000 / fps,
        blit=False
    )
    
    # Save
    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    anim.save(str(output_path), writer=writer)
    plt.close(fig)


def create_density_movie(
    times: np.ndarray,
    densities: np.ndarray,
    domain: dict,
    output_path: Path,
    fps: int = 20,
    max_frames: int = 500,
    show_mass: bool = True
) -> None:
    """
    Create density movie showing KDE density field heatmap.
    
    Parameters
    ----------
    times : ndarray, shape (n_frames,)
        Time values
    densities : ndarray, shape (n_frames, nx, ny)
        Density snapshots
    domain : dict
        Domain bounds
    output_path : Path
        Output MP4 file path
    fps : int, default=20
        Frames per second
    max_frames : int, default=500
        Maximum frames
    show_mass : bool, default=True
        Show mass conservation text overlay
    """
    n_frames, nx, ny = densities.shape
    
    # Subsample if needed
    if n_frames > max_frames:
        indices = np.linspace(0, n_frames - 1, max_frames, dtype=int)
        times_sub = times[indices]
        densities_sub = densities[indices]
    else:
        times_sub = times
        densities_sub = densities
    
    n_frames_sub = len(times_sub)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    xmin = domain.get("xmin", 0)
    xmax = domain.get("xmax", 20)
    ymin = domain.get("ymin", 0)
    ymax = domain.get("ymax", 20)
    
    # Initial density plot
    im = ax.imshow(
        densities_sub[0].T,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        aspect="equal",
        cmap="viridis",
        vmin=densities_sub.min(),
        vmax=densities_sub.max()
    )
    
    cbar = plt.colorbar(im, ax=ax, label="density")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    title = ax.set_title(f"t = {times_sub[0]:.2f}")
    
    if show_mass:
        dx = (xmax - xmin) / nx
        dy = (ymax - ymin) / ny
        mass = densities_sub[0].sum() * dx * dy
        mass_text = ax.text(
            0.02, 0.98,
            f"Mass: {mass:.6f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )
    else:
        mass_text = None
    
    def update(frame):
        """Update function for animation."""
        im.set_data(densities_sub[frame].T)
        title.set_text(f"t = {times_sub[frame]:.2f}")
        
        if mass_text is not None:
            dx = (xmax - xmin) / nx
            dy = (ymax - ymin) / ny
            mass = densities_sub[frame].sum() * dx * dy
            mass_text.set_text(f"Mass: {mass:.6f}")
            return im, title, mass_text
        
        return im, title
    
    # Create animation
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames_sub,
        interval=1000 / fps,
        blit=False
    )
    
    # Save
    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    anim.save(str(output_path), writer=writer)
    plt.close(fig)


def create_latent_movie(
    times: np.ndarray,
    latents: np.ndarray,
    output_path: Path,
    fps: int = 20,
    max_frames: int = 500,
    mode: str = "timeseries"
) -> None:
    """
    Create latent coordinate movie.
    
    Parameters
    ----------
    times : ndarray, shape (n_frames,)
        Time values
    latents : ndarray, shape (d, n_frames)
        Latent coordinates
    output_path : Path
        Output MP4 file path
    fps : int, default=20
        Frames per second
    max_frames : int, default=500
        Maximum frames
    mode : str, default="timeseries"
        "timeseries": rolling time series plot of all d coordinates
        "embedding": 2D phase space (y1 vs y2) with time-synchronized marker
    """
    d, n_frames = latents.shape
    
    # Subsample if needed
    if n_frames > max_frames:
        indices = np.linspace(0, n_frames - 1, max_frames, dtype=int)
        times_sub = times[indices]
        latents_sub = latents[:, indices]
    else:
        times_sub = times
        latents_sub = latents
    
    n_frames_sub = len(times_sub)
    
    if mode == "timeseries":
        # Rolling time series plot
        fig, axes = plt.subplots(min(d, 5), 1, figsize=(12, 2 * min(d, 5)), sharex=True)
        if d == 1:
            axes = [axes]
        
        # Show first 5 coordinates
        n_show = min(d, 5)
        lines = []
        
        for i in range(n_show):
            ax = axes[i]
            line, = ax.plot([], [], 'b-', linewidth=1)
            ax.set_ylabel(f"y{i+1}")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(latents_sub[i].min() * 1.1, latents_sub[i].max() * 1.1)
            lines.append(line)
        
        axes[-1].set_xlabel("time")
        axes[0].set_title(f"Latent Coordinates (d={d})")
        
        # Window size for rolling display
        window = min(100, n_frames_sub)
        
        def update(frame):
            """Update function for animation."""
            start = max(0, frame - window)
            end = frame + 1
            
            for i in range(n_show):
                lines[i].set_data(times_sub[start:end], latents_sub[i, start:end])
                axes[i].set_xlim(times_sub[start], times_sub[min(start + window, n_frames_sub - 1)])
            
            return lines
        
    elif mode == "embedding":
        # 2D phase space plot
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot full trajectory
        ax.plot(latents_sub[0], latents_sub[1], 'b-', alpha=0.3, linewidth=0.5)
        
        # Current position marker
        marker, = ax.plot([], [], 'ro', markersize=10)
        trail, = ax.plot([], [], 'r-', linewidth=2, alpha=0.7)
        
        ax.set_xlabel("y1")
        ax.set_ylabel("y2")
        ax.set_title(f"Latent Space Embedding (d={d})")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
        
        # Set limits
        margin = 0.1
        y1_range = latents_sub[0].max() - latents_sub[0].min()
        y2_range = latents_sub[1].max() - latents_sub[1].min()
        ax.set_xlim(latents_sub[0].min() - margin * y1_range,
                   latents_sub[0].max() + margin * y1_range)
        ax.set_ylim(latents_sub[1].min() - margin * y2_range,
                   latents_sub[1].max() + margin * y2_range)
        
        time_text = ax.text(
            0.02, 0.98,
            "",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )
        
        trail_length = 20
        
        def update(frame):
            """Update function for animation."""
            marker.set_data([latents_sub[0, frame]], [latents_sub[1, frame]])
            
            # Trail
            start = max(0, frame - trail_length)
            trail.set_data(latents_sub[0, start:frame+1], latents_sub[1, start:frame+1])
            
            time_text.set_text(f"t = {times_sub[frame]:.2f}")
            
            return marker, trail, time_text
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Create animation
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames_sub,
        interval=1000 / fps,
        blit=False
    )
    
    # Save
    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    anim.save(str(output_path), writer=writer)
    plt.close(fig)
