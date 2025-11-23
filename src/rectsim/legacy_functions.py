"""
Legacy functions from the old working pipeline (commit 67655d3).

These are the exact functions that worked perfectly before:
1. kde_density_movie - KDE with proper metadata
2. side_by_side_video - Comparison video generation
3. save_video - Single density video generation
4. Order parameters - polarization, nematic_order, etc.

Copied from wsindy-manifold-OLD to restore working functionality.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple, Dict, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

Array = np.ndarray


# ============================================================================
# KDE Density Computation (from wsindy_manifold/density.py)
# ============================================================================

def kde_density_movie(
    traj: Array,
    Lx: float,
    Ly: float,
    nx: int,
    ny: int,
    bandwidth: float,
    bc: str = "periodic"
) -> Tuple[Array, Dict]:
    """
    Compute Gaussian-smoothed KDE density movie from particle trajectories.
    
    Args:
        traj: Particle positions (T, N, 2) with (x, y) coordinates
        Lx: Domain width
        Ly: Domain height
        nx: Number of grid points in x
        ny: Number of grid points in y
        bandwidth: Gaussian smoothing bandwidth (in grid units)
        bc: Boundary conditions ('periodic' or 'reflecting')
        
    Returns:
        rho: Density movie (T, ny, nx)  # Note: ny first for image convention
        meta: Metadata dict with 'bandwidth', 'nx', 'ny', 'extent', 'Lx', 'Ly', 'bc'
        
    Example:
        >>> traj = np.random.rand(100, 50, 2) * 30  # 100 frames, 50 particles
        >>> rho, meta = kde_density_movie(traj, Lx=30, Ly=30, nx=50, ny=50, bandwidth=1.5)
        >>> print(rho.shape)  # (100, 50, 50)
        >>> print(meta['extent'])  # [0, 30, 0, 30]
    """
    T, N, d = traj.shape
    
    if d != 2:
        raise ValueError(f"traj must have shape (T, N, 2), got {traj.shape}")
    
    # Create grid edges
    x_edges = np.linspace(0.0, Lx, nx + 1)
    y_edges = np.linspace(0.0, Ly, ny + 1)
    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]
    
    # Mode for Gaussian filter
    mode = "wrap" if bc == "periodic" else "nearest"
    
    # Compute density for each frame
    rho = np.zeros((T, ny, nx))
    
    for t in range(T):
        # 2D histogram
        # histogram2d returns hist[i,j] where i is x-bins, j is y-bins
        # But for images, we need [row, col] = [y, x] indexing
        hist, _, _ = np.histogram2d(
            traj[t, :, 1],  # y coordinates (rows)
            traj[t, :, 0],  # x coordinates (columns)
            bins=[y_edges, x_edges],
            range=[[0.0, Ly], [0.0, Lx]]
        )
        
        # Normalize to density (particles per unit area)
        density = hist / (dx * dy)
        
        # Apply Gaussian smoothing
        if bandwidth > 0:
            density = gaussian_filter(density, sigma=bandwidth, mode=mode)
        
        # Store directly (already in [y, x] = [row, col] format for images)
        rho[t] = density
    
    # Metadata
    meta = {
        'bandwidth': bandwidth,
        'nx': nx,
        'ny': ny,
        'Lx': Lx,
        'Ly': Ly,
        'extent': [0, Lx, 0, Ly],  # [xmin, xmax, ymin, ymax]
        'bc': bc,
        'N_particles': N,
        'T_frames': T
    }
    
    return rho, meta


def estimate_bandwidth(Lx: float, Ly: float, N: int, nx: int, ny: int) -> float:
    """
    Estimate reasonable KDE bandwidth based on problem size.
    
    Rule of thumb: bandwidth ~ (L / sqrt(N)) * (grid_resolution_factor)
    
    Args:
        Lx: Domain width
        Ly: Domain height
        N: Number of particles
        nx: Grid points in x
        ny: Grid points in y
        
    Returns:
        bandwidth: Suggested bandwidth in grid units
    """
    # Average domain size
    L_avg = (Lx + Ly) / 2
    
    # Average grid spacing
    dx = Lx / nx
    dy = Ly / ny
    dx_avg = (dx + dy) / 2
    
    # Scott's rule: h ~ N^(-1/(d+4)) where d=2
    scott_factor = N ** (-1/6)
    
    # Bandwidth in physical units
    h_physical = L_avg * scott_factor * 0.5
    
    # Convert to grid units
    bandwidth = h_physical / dx_avg
    
    # Clamp to reasonable range
    bandwidth = np.clip(bandwidth, 0.5, 5.0)
    
    return bandwidth


# ============================================================================
# Order Parameters (from wsindy_manifold/standard_metrics.py)
# ============================================================================

def polarization(vel: Array, eps: float = 1e-10) -> float:
    """
    Compute polarization order parameter Φ(t).
    
    Φ = (1/N) || Σᵢ vᵢ/||vᵢ|| ||
    
    Args:
        vel: Velocities (N, 2) or (N, d)
        eps: Small constant to avoid division by zero
        
    Returns:
        phi: Polarization in [0, 1]
    """
    if vel.ndim != 2:
        raise ValueError(f"vel must have shape (N, d), got {vel.shape}")
    
    N = vel.shape[0]
    if N == 0:
        return 0.0
    
    # Normalize velocities
    speeds = np.linalg.norm(vel, axis=1, keepdims=True)
    normalized = vel / (speeds + eps)
    
    # Sum and compute magnitude
    mean_direction = np.mean(normalized, axis=0)
    phi = np.linalg.norm(mean_direction)
    
    return float(phi)


def mean_speed(vel: Array) -> float:
    """
    Compute mean speed.
    
    Args:
        vel: Velocities (N, 2) or (N, d)
        
    Returns:
        v_mean: Mean speed
    """
    speeds = np.linalg.norm(vel, axis=1)
    return float(np.mean(speeds))


def speed_std(vel: Array) -> float:
    """
    Compute standard deviation of speeds.
    
    Args:
        vel: Velocities (N, 2) or (N, d)
        
    Returns:
        v_std: Speed standard deviation
    """
    speeds = np.linalg.norm(vel, axis=1)
    return float(np.std(speeds))


def nematic_order(vel: Array, eps: float = 1e-10) -> float:
    """
    Compute nematic order parameter (2nd moment of headings).
    
    Q = max eigenvalue of (1/N) Σᵢ (nᵢ ⊗ nᵢ - I/d)
    where nᵢ = vᵢ/||vᵢ||
    
    Args:
        vel: Velocities (N, 2)
        eps: Small constant
        
    Returns:
        q: Nematic order in [0, 1]
    """
    if vel.ndim != 2 or vel.shape[1] != 2:
        raise ValueError(f"nematic_order requires (N, 2) velocities, got {vel.shape}")
    
    N, d = vel.shape
    if N == 0:
        return 0.0
    
    # Normalize
    speeds = np.linalg.norm(vel, axis=1, keepdims=True)
    n = vel / (speeds + eps)
    
    # Compute Q tensor
    Q = np.zeros((d, d))
    for i in range(N):
        Q += np.outer(n[i], n[i])
    Q = Q / N - np.eye(d) / d
    
    # Max eigenvalue
    eigvals = np.linalg.eigvalsh(Q)
    q = float(np.max(eigvals))
    
    return q


def compute_order_params(
    vel: Array,
    include_nematic: bool = False
) -> Dict[str, float]:
    """
    Compute all order parameters for a velocity snapshot.
    
    Args:
        vel: Velocities (N, 2) or (N, d)
        include_nematic: Whether to compute nematic order
        
    Returns:
        params: Dictionary with 'phi', 'mean_speed', 'speed_std', 'nematic' (optional)
    """
    params = {
        'phi': polarization(vel),
        'mean_speed': mean_speed(vel),
        'speed_std': speed_std(vel)
    }
    
    if include_nematic and vel.shape[1] == 2:
        params['nematic'] = nematic_order(vel)
    
    return params


# ============================================================================
# Error Metrics (from wsindy_manifold/mvar_rom.py)
# ============================================================================

def compute_frame_metrics(
    X_true: Array,
    X_pred: Array
) -> Dict[str, Array]:
    """
    Compute frame-wise error metrics.
    
    Args:
        X_true: True density snapshots (T, n_c) - flattened density fields
        X_pred: Predicted density snapshots (T, n_c) - flattened density fields
        
    Returns:
        metrics: Dict with arrays for each metric over time
            - e1: Relative L¹ error
            - e2: Relative L² error
            - einf: Relative L∞ error
            - rmse: Root mean squared error
            - mass_error: Relative mass conservation error
    """
    T = X_true.shape[0]
    
    e1 = np.zeros(T)
    e2 = np.zeros(T)
    einf = np.zeros(T)
    rmse = np.zeros(T)
    mass_error = np.zeros(T)
    
    for t in range(T):
        diff = X_pred[t] - X_true[t]
        
        # Norms
        norm_true_1 = np.linalg.norm(X_true[t], ord=1)
        norm_true_2 = np.linalg.norm(X_true[t], ord=2)
        norm_true_inf = np.linalg.norm(X_true[t], ord=np.inf)
        
        # Relative errors
        e1[t] = np.linalg.norm(diff, ord=1) / (norm_true_1 + 1e-16)
        e2[t] = np.linalg.norm(diff, ord=2) / (norm_true_2 + 1e-16)
        einf[t] = np.linalg.norm(diff, ord=np.inf) / (norm_true_inf + 1e-16)
        
        # RMSE
        rmse[t] = np.sqrt(np.mean(diff ** 2))
        
        # Mass error
        mass_true = np.sum(X_true[t])
        mass_pred = np.sum(X_pred[t])
        mass_error[t] = np.abs(mass_pred - mass_true) / (np.abs(mass_true) + 1e-16)
    
    return {
        "e1": e1,
        "e2": e2,
        "einf": einf,
        "rmse": rmse,
        "mass_error": mass_error,
    }


def compute_summary_metrics(
    X_true: Array,
    X_pred: Array,
    X_train_mean: Array,
    frame_metrics: Dict[str, Array],
    tolerance_threshold: float = 0.10
) -> Dict:
    """
    Compute aggregate summary metrics.
    
    Args:
        X_true: True density (T, n_c) - flattened density fields
        X_pred: Predicted density (T, n_c) - flattened density fields
        X_train_mean: Mean of training data (n_c,)
        frame_metrics: Frame-wise metrics from compute_frame_metrics
        tolerance_threshold: Threshold for tolerance horizon (default 0.10)
        
    Returns:
        summary: Dict with scalar metrics
            - r2: R² score
            - median_e2: Median relative L² error
            - p10_e2: 10th percentile L² error
            - p90_e2: 90th percentile L² error
            - tau_tol: Tolerance horizon (frames until error exceeds threshold)
            - mean_mass_error: Mean mass conservation error
            - max_mass_error: Max mass conservation error
    """
    e2 = frame_metrics["e2"]
    
    # R² score
    ss_res = np.sum((X_true - X_pred) ** 2)
    ss_tot = np.sum((X_true - X_train_mean) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-16)
    
    # Percentiles
    median_e2 = np.median(e2)
    p10_e2 = np.percentile(e2, 10)
    p90_e2 = np.percentile(e2, 90)
    
    # Tolerance horizon: first time rolling mean exceeds threshold
    window = min(10, len(e2))
    rolling_e2 = np.convolve(e2, np.ones(window)/window, mode='valid')
    tau_tol_idx = np.where(rolling_e2 >= tolerance_threshold)[0]
    tau_tol = tau_tol_idx[0] if len(tau_tol_idx) > 0 else len(e2)
    
    return {
        "r2": float(r2),
        "median_e2": float(median_e2),
        "p10_e2": float(p10_e2),
        "p90_e2": float(p90_e2),
        "tau_tol": int(tau_tol),
        "mean_mass_error": float(np.mean(frame_metrics["mass_error"])),
        "max_mass_error": float(np.max(frame_metrics["mass_error"])),
    }


def plot_errors_timeseries(
    frame_metrics: Dict[str, Array],
    summary: Dict,
    T0: int = 0,
    save_path: Optional[Path] = None,
    title: str = 'MVAR-ROM: Error Metrics Over Time'
) -> plt.Figure:
    """
    Plot error metrics over time.
    
    Args:
        frame_metrics: Frame-wise metrics from compute_frame_metrics
        summary: Summary metrics with R², median_e2, tau_tol
        T0: Starting time frame (default 0)
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    T = len(frame_metrics["e1"])
    t = np.arange(T0, T0 + T)
    
    # Plot relative errors
    axes[0].plot(t, frame_metrics["e1"], 'b-', alpha=0.7, linewidth=2, label='Relative L¹')
    if T0 > 0:
        axes[0].axvline(T0, color='k', linestyle='--', alpha=0.5, label='Train/Test Split')
    axes[0].set_ylabel('Relative L¹ Error', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(t, frame_metrics["e2"], 'g-', alpha=0.7, linewidth=2, label='Relative L²')
    if T0 > 0:
        axes[1].axvline(T0, color='k', linestyle='--', alpha=0.5)
    axes[1].axhline(0.10, color='r', linestyle=':', alpha=0.5, label='10% Threshold')
    if summary["tau_tol"] < T and summary["tau_tol"] > 0:
        axes[1].axvline(T0 + summary["tau_tol"], color='r', linestyle='--', 
                       alpha=0.7, linewidth=2, label=f'τ_tol = {summary["tau_tol"]}')
    axes[1].set_ylabel('Relative L² Error', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    axes[2].plot(t, frame_metrics["einf"], 'r-', alpha=0.7, linewidth=2, label='Relative L∞')
    if T0 > 0:
        axes[2].axvline(T0, color='k', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Time Frame', fontsize=11)
    axes[2].set_ylabel('Relative L∞ Error', fontsize=11, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # Add summary text
    text_str = f"R² = {summary['r2']:.4f}\n"
    text_str += f"Median L² = {summary['median_e2']:.4f}\n"
    text_str += f"τ_tol = {summary['tau_tol']} frames"
    axes[1].text(0.02, 0.98, text_str, transform=axes[1].transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.8), fontsize=11)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved error timeseries: {save_path}")
    
    return fig


# ============================================================================
# Video Generation (from wsindy_manifold/io.py)
# ============================================================================

def save_video(
    path: Path,
    frames: Array,
    fps: int,
    name: str,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None
) -> None:
    """
    Save 2D heatmap frames as MP4 video.
    
    Args:
        path: Directory to save video
        frames: Array of shape (T, ny, nx)
        fps: Frames per second
        name: Filename (without .mp4 extension)
        cmap: Colormap name
        vmin: Min value for colormap (auto if None)
        vmax: Max value for colormap (auto if None)
        title: Video title
    """
    path.mkdir(parents=True, exist_ok=True)
    video_path = path / f"{name}.mp4"
    
    if vmin is None:
        vmin = frames.min()
    if vmax is None:
        vmax = frames.max()
    
    T, ny, nx = frames.shape
    
    # Subsample if too many frames
    max_frames = 500
    if T > max_frames:
        indices = np.linspace(0, T - 1, max_frames, dtype=int)
        frames = frames[indices]
        T = max_frames
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Initial frame
    im = ax.imshow(frames[0].T, origin='lower', cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect='auto')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Density', fontsize=11)
    
    # Time text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       color='white', fontsize=11, va='top',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    
    # Write video
    writer = FFMpegWriter(fps=fps, bitrate=2000)
    
    with writer.saving(fig, video_path, dpi=100):
        for t in range(T):
            im.set_data(frames[t].T)
            time_text.set_text(f'Frame {t+1}/{T}')
            writer.grab_frame()
    
    plt.close(fig)
    print(f"Saved video: {video_path}")


def side_by_side_video(
    path: Path,
    left_frames: Array,
    right_frames: Array,
    lower_strip_timeseries: Optional[Array] = None,
    name: str = "comparison",
    fps: int = 20,
    cmap: str = "viridis",
    titles: tuple = ("Ground Truth", "Prediction")
) -> None:
    """
    Create side-by-side comparison video with optional error timeseries below.
    
    Args:
        path: Directory to save video
        left_frames: Left panel frames (T, ny, nx)
        right_frames: Right panel frames (T, ny, nx)
        lower_strip_timeseries: Optional timeseries (T,) to plot below
        name: Filename (without .mp4 extension)
        fps: Frames per second
        cmap: Colormap name
        titles: (left_title, right_title)
    """
    path.mkdir(parents=True, exist_ok=True)
    video_path = path / f"{name}.mp4"
    
    T = len(left_frames)
    
    # Subsample if needed
    max_frames = 500
    if T > max_frames:
        indices = np.linspace(0, T - 1, max_frames, dtype=int)
        left_frames = left_frames[indices]
        right_frames = right_frames[indices]
        if lower_strip_timeseries is not None:
            lower_strip_timeseries = lower_strip_timeseries[indices]
        T = max_frames
    
    # Shared colormap limits
    vmin = min(left_frames.min(), right_frames.min())
    vmax = max(left_frames.max(), right_frames.max())
    
    # Create figure
    if lower_strip_timeseries is not None:
        fig = plt.figure(figsize=(14, 7))
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.3, wspace=0.3)
        ax_left = fig.add_subplot(gs[0, 0])
        ax_right = fig.add_subplot(gs[0, 1])
        ax_ts = fig.add_subplot(gs[1, :])
    else:
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))
        ax_ts = None
    
    # Left panel
    im_left = ax_left.imshow(left_frames[0].T, origin='lower', cmap=cmap,
                             vmin=vmin, vmax=vmax, aspect='auto')
    ax_left.set_title(titles[0], fontsize=13, fontweight='bold')
    ax_left.set_xlabel('x', fontsize=11)
    ax_left.set_ylabel('y', fontsize=11)
    plt.colorbar(im_left, ax=ax_left, fraction=0.046, pad=0.04)
    
    # Right panel
    im_right = ax_right.imshow(right_frames[0].T, origin='lower', cmap=cmap,
                               vmin=vmin, vmax=vmax, aspect='auto')
    ax_right.set_title(titles[1], fontsize=13, fontweight='bold')
    ax_right.set_xlabel('x', fontsize=11)
    ax_right.set_ylabel('y', fontsize=11)
    plt.colorbar(im_right, ax=ax_right, fraction=0.046, pad=0.04)
    
    # Timeseries panel
    if ax_ts is not None and lower_strip_timeseries is not None:
        time_steps = np.arange(T)
        line, = ax_ts.plot(time_steps, lower_strip_timeseries, 'b-', linewidth=2)
        marker, = ax_ts.plot([0], [lower_strip_timeseries[0]], 'ro', markersize=8)
        ax_ts.set_xlabel('Time Step', fontsize=11)
        ax_ts.set_ylabel('Error', fontsize=11)
        ax_ts.set_title('Relative L² Error Over Time', fontsize=12)
        ax_ts.grid(True, alpha=0.3)
        ax_ts.set_xlim([0, T-1])
        ax_ts.set_ylim([0, lower_strip_timeseries.max() * 1.1])
    
    plt.tight_layout()
    
    # Write video
    writer = FFMpegWriter(fps=fps, bitrate=3000)
    
    with writer.saving(fig, video_path, dpi=100):
        for t in range(T):
            im_left.set_data(left_frames[t].T)
            im_right.set_data(right_frames[t].T)
            
            if ax_ts is not None and lower_strip_timeseries is not None:
                marker.set_data([t], [lower_strip_timeseries[t]])
            
            writer.grab_frame()
    
    plt.close(fig)
    print(f"Saved comparison video: {video_path}")


def trajectory_video(
    path: Path,
    traj: Array,
    times: Array,
    Lx: float,
    Ly: float,
    name: str = "trajectory",
    fps: int = 20,
    marker_size: float = 30,
    marker_color: str = 'C0',
    title: Optional[str] = None,
    show_velocities: bool = True,
    quiver_scale: float = 3.0,
    quiver_width: float = 0.004,
    quiver_alpha: float = 0.8
) -> None:
    """
    Create trajectory video showing particle positions and velocities over time.
    Particles are colored by heading angle (-π to π) with a colorbar.
    Arrow lengths are proportional to particle speeds.
    
    Args:
        path: Directory to save video
        traj: Particle trajectories (T, N, 2)
        times: Time points (T,)
        Lx: Domain width
        Ly: Domain height
        name: Filename (without .mp4 extension)
        fps: Frames per second
        marker_size: Particle marker size
        marker_color: Ignored (particles colored by heading angle)
        title: Video title
        show_velocities: If True, show velocity arrows (quiver)
        quiver_scale: Scale factor for velocity arrows (higher = shorter arrows)
        quiver_width: Width of velocity arrows
        quiver_alpha: Transparency of velocity arrows
    """
    path.mkdir(parents=True, exist_ok=True)
    video_path = path / f"{name}.mp4"
    
    T, N, _ = traj.shape
    
    # Subsample if too many frames
    max_frames = 500
    if T > max_frames:
        indices = np.linspace(0, T - 1, max_frames, dtype=int)
        traj = traj[indices]
        times = times[indices]
        T = max_frames
    
    # Compute velocities from positions (finite differences)
    # Handle periodic boundaries to avoid spurious large velocities
    vel = np.zeros_like(traj)
    if T > 1:
        dt = times[1] - times[0] if len(times) > 1 else 1.0
        for t in range(T - 1):
            disp = traj[t + 1] - traj[t]
            
            # Wrap displacements for periodic boundaries (minimum image convention)
            disp[:, 0] = np.where(disp[:, 0] > Lx/2, disp[:, 0] - Lx, disp[:, 0])
            disp[:, 0] = np.where(disp[:, 0] < -Lx/2, disp[:, 0] + Lx, disp[:, 0])
            disp[:, 1] = np.where(disp[:, 1] > Ly/2, disp[:, 1] - Ly, disp[:, 1])
            disp[:, 1] = np.where(disp[:, 1] < -Ly/2, disp[:, 1] + Ly, disp[:, 1])
            
            vel[t] = disp / dt
        vel[-1] = vel[-2]  # Copy last velocity
    
    # Create figure with colorbar
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    # Compute heading angles for first frame
    angles = np.arctan2(vel[0, :, 1], vel[0, :, 0])  # Range: -π to π
    
    # Initial scatter plot colored by heading angle
    scatter = ax.scatter(
        traj[0, :, 0], 
        traj[0, :, 1], 
        c=angles,
        s=marker_size, 
        alpha=0.8,
        edgecolors='none',
        cmap='hsv',  # Circular colormap for angles
        vmin=-np.pi,
        vmax=np.pi
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Heading Angle (rad)', fraction=0.046, pad=0.04)
    cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    
    # Initial quiver plot (velocity arrows) - arrows proportional to speed
    quiver = None
    if show_velocities:
        # Don't use scale_units='xy' to make arrows speed-proportional
        quiver = ax.quiver(
            traj[0, :, 0],
            traj[0, :, 1],
            vel[0, :, 0],
            vel[0, :, 1],
            angles='xy',
            scale_units='xy',
            scale=quiver_scale,  # Controls arrow length
            width=quiver_width,
            alpha=quiver_alpha,
            color='orange',  # Orange arrows on colored particles
            edgecolors='white',
            linewidth=0.5
        )
    
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Time text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       color='black', fontsize=11, va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Particle count text
    ax.text(0.02, 0.02, f'N = {N} particles', transform=ax.transAxes,
           color='black', fontsize=10, va='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Write video using FFMpegWriter (best quality)
    writer = FFMpegWriter(fps=fps, bitrate=2000)
    
    with writer.saving(fig, video_path, dpi=100):
        for t in range(T):
            # Compute heading angles for this frame
            angles = np.arctan2(vel[t, :, 1], vel[t, :, 0])
            
            # Update particle positions and colors
            scatter.set_offsets(traj[t, :, :2])
            scatter.set_array(angles)
            
            # Update velocity arrows (proportional to speed)
            if show_velocities and quiver is not None:
                quiver.set_offsets(traj[t, :, :2])
                quiver.set_UVC(vel[t, :, 0], vel[t, :, 1])
            
            time_text.set_text(f't = {times[t]:.2f}s\nFrame {t+1}/{T}')
            writer.grab_frame()
    
    plt.close(fig)
    print(f"Saved trajectory video: {video_path}")
