"""
MVAR-ROM: Multivariate Autoregressive Reduced-Order Modeling for Density Forecasting.

Complete evaluation pipeline following EF-ROM best practices:
- KDE → POD/SVD → MVAR latent dynamics → Lift to density
- Closed-loop multi-step forecasting
- Comprehensive metrics (relative L1/L2/L∞, R², mass error, tolerance horizon)
- Full visualization suite (error plots, snapshots, videos, latent diagnostics)

Reference: Alvarez et al. (2025), "Equation-Free Reduced-Order Modeling"
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FFMpegWriter

Array = np.ndarray


@dataclass
class MVARROMConfig:
    """Configuration for MVAR-ROM evaluation."""
    pod_energy: float = 0.99           # POD energy threshold
    mvar_order: int = 4                # MVAR order (w)
    ridge: float = 1e-6                # Ridge regularization parameter
    train_frac: float = 0.8            # Training fraction
    tolerance_threshold: float = 0.10   # Tolerance for τ_tol metric
    save_snapshots: bool = True        # Whether to save snapshots
    save_videos: bool = True           # Whether to save videos
    fps: int = 20                      # Video frames per second
    output_dir: Path = Path("outputs/mvar_rom_evaluation")


# ============================================================================
# POD (Proper Orthogonal Decomposition)
# ============================================================================

def fit_pod(
    X: Array, 
    energy: float = 0.99
) -> Tuple[Array, Array, int, Array]:
    """
    Fit POD basis from density snapshots.
    
    Args:
        X: Density snapshots (T, n_c) where n_c = nx * ny
        energy: Cumulative energy threshold (0 < energy <= 1)
        
    Returns:
        Ud: POD basis (n_c, d)
        xbar: Mean snapshot (n_c,)
        d: Number of modes retained
        energy_curve: Cumulative energy per mode
    """
    if X.ndim != 2:
        raise ValueError(f"X must have shape (T, n_c), got {X.shape}")
    
    T, n_c = X.shape
    
    # Compute mean
    xbar = np.mean(X, axis=0)
    
    # Center data
    X_centered = X - xbar
    
    # SVD: X_centered = U @ S @ Vt
    # For T << n_c, use economy SVD on covariance
    if T < n_c:
        # Compute temporal covariance C = (1/T) * X_centered @ X_centered.T
        C = (X_centered @ X_centered.T) / T
        eigvals, eigvecs = np.linalg.eigh(C)
        
        # Sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Compute spatial modes
        Ud = X_centered.T @ eigvecs  # (n_c, T)
        
        # Normalize
        norms = np.linalg.norm(Ud, axis=0)
        Ud = Ud / (norms + 1e-16)
        
        sigma = np.sqrt(eigvals * T)
    else:
        # Standard SVD
        U, sigma, Vt = np.linalg.svd(X_centered, full_matrices=False)
        Ud = U  # (n_c, min(T, n_c))
    
    # Compute energy
    energy_vals = sigma ** 2
    total_energy = np.sum(energy_vals)
    cumulative_energy = np.cumsum(energy_vals) / (total_energy + 1e-16)
    
    # Find d that captures desired energy
    d = np.searchsorted(cumulative_energy, energy) + 1
    d = min(d, len(cumulative_energy))
    
    print(f"POD: Retained {d}/{len(cumulative_energy)} modes ({cumulative_energy[d-1]*100:.2f}% energy)")
    
    return Ud[:, :d], xbar, d, cumulative_energy


def restrict(X: Array, Ud: Array, xbar: Array) -> Array:
    """
    Restrict density snapshots to latent space.
    
    Args:
        X: Density snapshots (T, n_c)
        Ud: POD basis (n_c, d)
        xbar: Mean snapshot (n_c,)
        
    Returns:
        Y: Latent trajectory (T, d)
    """
    X_centered = X - xbar
    Y = X_centered @ Ud
    return Y


def lift(Y: Array, Ud: Array, xbar: Array) -> Array:
    """
    Lift latent trajectory to density space.
    
    Args:
        Y: Latent trajectory (T, d)
        Ud: POD basis (n_c, d)
        xbar: Mean snapshot (n_c,)
        
    Returns:
        X: Density snapshots (T, n_c)
    """
    X = Y @ Ud.T + xbar
    return X


# ============================================================================
# MVAR (Multivariate Autoregressive) Model
# ============================================================================

def fit_mvar(
    Y: Array,
    w: int = 4,
    ridge: float = 1e-6
) -> Tuple[Array, Array]:
    """
    Fit MVAR model to latent trajectory.
    
    Model: y(t) = A0 + sum_{j=1}^w A_j * y(t-j) + epsilon(t)
    
    Args:
        Y: Latent trajectory (T, d)
        w: Lag order (default 4)
        ridge: Ridge regularization λ (default 1e-6)
        
    Returns:
        A0: Bias vector (d,)
        A: Lag matrices (w, d, d)
    """
    from wsindy_manifold.latent.mvar import fit_mvar as _fit_mvar
    
    model = _fit_mvar(Y, w=w, ridge_lambda=ridge)
    A0 = model["A0"]
    A = model["A"]
    
    return A0, A


def forecast_closed_loop(
    Y_seed: Array,
    A0: Array,
    A: Array,
    steps: int,
    add_noise: bool = False,
    noise_cov: Optional[Array] = None
) -> Array:
    """
    Closed-loop multi-step MVAR forecast.
    
    Args:
        Y_seed: Initial conditions (w, d) - last w latent states
        A0: Bias vector (d,)
        A: Lag matrices (w, d, d)
        steps: Number of steps to forecast
        add_noise: Whether to add stochastic noise
        noise_cov: Noise covariance (d, d) if add_noise=True
        
    Returns:
        Y_forecast: Forecasted latent trajectory (steps, d)
    """
    from wsindy_manifold.latent.mvar import rollout
    
    w, d = Y_seed.shape
    model = {"A0": A0, "A": A, "w": w, "ridge_lambda": 0.0}
    
    # Rollout (no noise in basic implementation)
    Y_forecast = rollout(Y_seed, steps=steps, model=model)
    
    # Optionally add noise
    if add_noise and noise_cov is not None:
        for t in range(steps):
            Y_forecast[t] += np.random.multivariate_normal(np.zeros(d), noise_cov)
    
    return Y_forecast


def forecast_one_step(
    Y: Array,
    A0: Array,
    A: Array
) -> Array:
    """
    One-step ahead forecast with teacher forcing (diagnostic).
    
    Args:
        Y: True latent trajectory (T, d)
        A0: Bias vector (d,)
        A: Lag matrices (w, d, d)
        
    Returns:
        Y_pred: One-step predictions (T-w, d)
    """
    from wsindy_manifold.latent.mvar import forecast_step
    
    w, d, _ = A.shape
    T = Y.shape[0]
    model = {"A0": A0, "A": A, "w": w, "ridge_lambda": 0.0}
    
    Y_pred = np.zeros((T - w, d))
    for t in range(w, T):
        Y_pred[t - w] = forecast_step(Y[t-w:t], model)
    
    return Y_pred


# ============================================================================
# Evaluation Metrics
# ============================================================================

def compute_frame_metrics(
    X_true: Array,
    X_pred: Array
) -> Dict[str, Array]:
    """
    Compute frame-wise error metrics.
    
    Args:
        X_true: True density snapshots (T, n_c)
        X_pred: Predicted density snapshots (T, n_c)
        
    Returns:
        metrics: Dict with arrays for each metric over time
            - e1: Relative L1 error
            - e2: Relative L2 error
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
        X_true: True density (T, n_c)
        X_pred: Predicted density (T, n_c)
        X_train_mean: Mean of training data (n_c,)
        frame_metrics: Frame-wise metrics from compute_frame_metrics
        tolerance_threshold: Threshold for tolerance horizon (default 0.10)
        
    Returns:
        summary: Dict with scalar metrics
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


# ============================================================================
# Visualization
# ============================================================================

def plot_errors_timeseries(
    frame_metrics: Dict[str, Array],
    summary: Dict,
    T0: int,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot error metrics over time.
    
    Args:
        frame_metrics: Frame-wise metrics
        summary: Summary metrics with R², median_e2, tau_tol
        T0: Training window end
        save_path: Path to save figure
        
    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    T = len(frame_metrics["e1"])
    t = np.arange(T0, T0 + T)
    
    # Plot relative errors
    axes[0].plot(t, frame_metrics["e1"], 'b-', alpha=0.7, label='Relative L¹')
    axes[0].axvline(T0, color='k', linestyle='--', alpha=0.5, label='Train/Test Split')
    axes[0].set_ylabel('Relative L¹ Error', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(t, frame_metrics["e2"], 'g-', alpha=0.7, label='Relative L²')
    axes[1].axvline(T0, color='k', linestyle='--', alpha=0.5)
    axes[1].axhline(0.10, color='r', linestyle=':', alpha=0.5, label='10% Threshold')
    if summary["tau_tol"] < T:
        axes[1].axvline(T0 + summary["tau_tol"], color='r', linestyle='--', 
                       alpha=0.7, label=f'τ_tol = {summary["tau_tol"]}')
    axes[1].set_ylabel('Relative L² Error', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    axes[2].plot(t, frame_metrics["einf"], 'r-', alpha=0.7, label='Relative L∞')
    axes[2].axvline(T0, color='k', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Time Frame', fontsize=11)
    axes[2].set_ylabel('Relative L∞ Error', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # Add summary text
    text_str = f"R² = {summary['r2']:.4f}\n"
    text_str += f"Median L² = {summary['median_e2']:.4f}\n"
    text_str += f"τ_tol = {summary['tau_tol']} frames"
    axes[1].text(0.02, 0.98, text_str, transform=axes[1].transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5), fontsize=10)
    
    plt.suptitle('MVAR-ROM: Error Metrics Over Time', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved error timeseries: {save_path}")
    
    return fig


def plot_snapshots(
    X_true: Array,
    X_pred: Array,
    nx: int,
    ny: int,
    T0: int,
    n_snapshots: int = 6,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot truth, prediction, and difference snapshots.
    
    Args:
        X_true: True densities (T, n_c)
        X_pred: Predicted densities (T, n_c)
        nx, ny: Grid dimensions
        T0: Training window end
        n_snapshots: Number of snapshots (default 6)
        save_path: Path to save figure
        
    Returns:
        fig: Matplotlib figure
    """
    T = X_true.shape[0]
    times = np.linspace(0, T-1, n_snapshots, dtype=int)
    
    fig = plt.figure(figsize=(15, 3*n_snapshots))
    gs = GridSpec(n_snapshots, 3, figure=fig, hspace=0.3, wspace=0.2)
    
    vmin = min(X_true.min(), X_pred.min())
    vmax = max(X_true.max(), X_pred.max())
    
    for i, t in enumerate(times):
        # Reshape to 2D
        true_2d = X_true[t].reshape(nx, ny)
        pred_2d = X_pred[t].reshape(nx, ny)
        diff_2d = np.abs(pred_2d - true_2d)
        
        # Truth
        ax1 = fig.add_subplot(gs[i, 0])
        im1 = ax1.imshow(true_2d.T, origin='lower', cmap='viridis', 
                        vmin=vmin, vmax=vmax)
        ax1.set_title(f't = {T0 + t}: Truth', fontsize=11)
        ax1.set_xticks([])
        ax1.set_yticks([])
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        # Prediction
        ax2 = fig.add_subplot(gs[i, 1])
        im2 = ax2.imshow(pred_2d.T, origin='lower', cmap='viridis',
                        vmin=vmin, vmax=vmax)
        ax2.set_title(f't = {T0 + t}: Prediction', fontsize=11)
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        # Difference
        ax3 = fig.add_subplot(gs[i, 2])
        im3 = ax3.imshow(diff_2d.T, origin='lower', cmap='hot')
        ax3.set_title(f't = {T0 + t}: |Difference|', fontsize=11)
        ax3.set_xticks([])
        ax3.set_yticks([])
        plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    plt.suptitle('MVAR-ROM: Truth vs Prediction Snapshots',
                fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved snapshots: {save_path}")
    
    return fig


def plot_pod_energy(
    energy_curve: Array,
    d: int,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot POD cumulative energy.
    
    Args:
        energy_curve: Cumulative energy per mode
        d: Number of modes retained
        save_path: Path to save figure
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    modes = np.arange(1, len(energy_curve) + 1)
    ax.plot(modes, energy_curve * 100, 'b-', linewidth=2)
    ax.axvline(d, color='r', linestyle='--', linewidth=2,
              label=f'd = {d} ({energy_curve[d-1]*100:.2f}% energy)')
    ax.axhline(99, color='k', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Number of Modes', fontsize=12)
    ax.set_ylabel('Cumulative Energy (%)', fontsize=12)
    ax.set_title('POD Energy Spectrum', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xlim([1, len(energy_curve)])
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved POD energy: {save_path}")
    
    return fig


def plot_latent_scatter(
    Y_true: Array,
    Y_pred: Array,
    n_modes: int = 3,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot scatter of true vs predicted latent modes.
    
    Args:
        Y_true: True latent trajectory (T, d)
        Y_pred: Predicted latent trajectory (T, d)
        n_modes: Number of leading modes to plot
        save_path: Path to save figure
        
    Returns:
        fig: Matplotlib figure
    """
    d = min(n_modes, Y_true.shape[1])
    
    fig, axes = plt.subplots(1, d, figsize=(5*d, 4))
    if d == 1:
        axes = [axes]
    
    for i in range(d):
        ax = axes[i]
        ax.scatter(Y_true[:, i], Y_pred[:, i], alpha=0.5, s=10)
        
        # Add diagonal
        lims = [min(Y_true[:, i].min(), Y_pred[:, i].min()),
                max(Y_true[:, i].max(), Y_pred[:, i].max())]
        ax.plot(lims, lims, 'r--', alpha=0.5, linewidth=2)
        
        # Compute correlation
        corr = np.corrcoef(Y_true[:, i], Y_pred[:, i])[0, 1]
        
        ax.set_xlabel(f'True y_{i+1}', fontsize=11)
        ax.set_ylabel(f'Predicted y_{i+1}', fontsize=11)
        ax.set_title(f'Mode {i+1} (ρ = {corr:.3f})', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.suptitle('MVAR-ROM: Latent Space Prediction Quality',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved latent scatter: {save_path}")
    
    return fig


# ============================================================================
# Video Generation
# ============================================================================

def create_density_movie(
    densities: Array,
    nx: int,
    ny: int,
    save_path: Path,
    title: str = "Density Evolution",
    fps: int = 20,
    max_frames: Optional[int] = None
):
    """
    Create density heatmap movie.
    
    Args:
        densities: Density snapshots (T, nx, ny)
        nx, ny: Grid dimensions
        save_path: Path to save MP4
        title: Movie title
        fps: Frames per second
        max_frames: Maximum frames (subsamples if T > max_frames)
    """
    T = densities.shape[0]
    
    # Subsample if needed
    if max_frames and T > max_frames:
        indices = np.linspace(0, T-1, max_frames, dtype=int)
        densities = densities[indices]
        T = max_frames
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(8, 7))
    
    vmin, vmax = densities.min(), densities.max()
    
    # Initial frame
    im = ax.imshow(densities[0].T, origin='lower', cmap='viridis',
                   vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    title_text = ax.set_title(f'{title}\nFrame 0/{T}', fontsize=13, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Density', fontsize=11)
    
    # Animation update function
    def update(frame):
        im.set_array(densities[frame].T)
        title_text.set_text(f'{title}\nFrame {frame}/{T}')
        return [im, title_text]
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=T, 
                                   interval=1000/fps, blit=True)
    
    # Save
    writer = FFMpegWriter(fps=fps, bitrate=2000, codec='h264')
    anim.save(save_path, writer=writer)
    plt.close(fig)
    
    print(f"Saved density movie: {save_path}")


def create_comparison_movie(
    densities_true: Array,
    densities_pred: Array,
    frame_metrics: Dict[str, Array],
    nx: int,
    ny: int,
    T0: int,
    save_path: Path,
    fps: int = 20,
    max_frames: Optional[int] = None
):
    """
    Create side-by-side comparison movie: Truth | Prediction | Difference.
    
    Args:
        densities_true: True density snapshots (T, nx, ny)
        densities_pred: Predicted density snapshots (T, nx, ny)
        frame_metrics: Frame-wise metrics dict
        nx, ny: Grid dimensions
        T0: Training window end
        save_path: Path to save MP4
        fps: Frames per second
        max_frames: Maximum frames
    """
    T = densities_true.shape[0]
    
    # Subsample if needed
    if max_frames and T > max_frames:
        indices = np.linspace(0, T-1, max_frames, dtype=int)
        densities_true = densities_true[indices]
        densities_pred = densities_pred[indices]
        for key in frame_metrics:
            frame_metrics[key] = frame_metrics[key][indices]
        T = max_frames
    
    # Setup figure with 3 columns + error plot
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[3, 1], hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax_err = fig.add_subplot(gs[1, :])
    
    vmin = min(densities_true.min(), densities_pred.min())
    vmax = max(densities_true.max(), densities_pred.max())
    
    # Initial frames
    im1 = ax1.imshow(densities_true[0].T, origin='lower', cmap='viridis',
                     vmin=vmin, vmax=vmax, aspect='auto')
    ax1.set_title('Ground Truth', fontsize=12, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    im2 = ax2.imshow(densities_pred[0].T, origin='lower', cmap='viridis',
                     vmin=vmin, vmax=vmax, aspect='auto')
    ax2.set_title('MVAR-ROM Prediction', fontsize=12, fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    diff = np.abs(densities_pred[0] - densities_true[0])
    im3 = ax3.imshow(diff.T, origin='lower', cmap='hot', aspect='auto')
    ax3.set_title('|Difference|', fontsize=12, fontweight='bold')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    # Error plot
    t_vals = np.arange(T) + T0
    line_e2, = ax_err.plot([], [], 'g-', linewidth=2, label='Relative L² Error')
    ax_err.axhline(0.10, color='r', linestyle=':', alpha=0.5, label='10% Threshold')
    ax_err.set_xlim([T0, T0 + T])
    ax_err.set_ylim([0, max(1.0, frame_metrics['e2'].max() * 1.1)])
    ax_err.set_xlabel('Time Frame', fontsize=11)
    ax_err.set_ylabel('Relative L² Error', fontsize=11)
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(loc='upper right')
    
    # Current frame marker
    vline = ax_err.axvline(T0, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    
    fig.suptitle(f'MVAR-ROM Forecast Evaluation (Frame {T0}/{T0+T})',
                fontsize=14, fontweight='bold')
    
    # Animation update function
    def update(frame):
        # Update density images
        im1.set_array(densities_true[frame].T)
        im2.set_array(densities_pred[frame].T)
        diff = np.abs(densities_pred[frame] - densities_true[frame])
        im3.set_array(diff.T)
        im3.set_clim(vmin=0, vmax=diff.max())
        
        # Update error plot
        line_e2.set_data(t_vals[:frame+1], frame_metrics['e2'][:frame+1])
        vline.set_xdata([T0 + frame, T0 + frame])
        
        # Update title
        fig.suptitle(f'MVAR-ROM Forecast Evaluation (Frame {T0 + frame}/{T0+T})',
                    fontsize=14, fontweight='bold')
        
        return [im1, im2, im3, line_e2, vline]
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=T,
                                   interval=1000/fps, blit=True)
    
    # Save
    writer = FFMpegWriter(fps=fps, bitrate=3000, codec='h264')
    anim.save(save_path, writer=writer)
    plt.close(fig)
    
    print(f"Saved comparison movie: {save_path}")


# ============================================================================
# Complete Evaluation Pipeline
# ============================================================================

def evaluate(
    X_true: Array,
    X_pred: Array,
    X_train_mean: Array,
    T0: int,
    tolerance_threshold: float = 0.10
) -> Tuple[Dict[str, Array], Dict]:
    """
    Complete evaluation: frame metrics + summary.
    
    Args:
        X_true: True density snapshots (T, n_c)
        X_pred: Predicted density snapshots (T, n_c)
        X_train_mean: Mean of training data (n_c,)
        T0: Training window end
        tolerance_threshold: Threshold for tolerance horizon
        
    Returns:
        frame_metrics: Frame-wise metrics dict
        summary: Summary metrics dict
    """
    frame_metrics = compute_frame_metrics(X_true, X_pred)
    summary = compute_summary_metrics(X_true, X_pred, X_train_mean, 
                                     frame_metrics, tolerance_threshold)
    
    return frame_metrics, summary


def run_mvar_rom_evaluation(
    densities: Array,
    nx: int,
    ny: int,
    config: MVARROMConfig,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Complete MVAR-ROM evaluation pipeline.
    
    Args:
        densities: Density snapshots (T, nx, ny)
        nx, ny: Grid dimensions
        config: Configuration object
        output_dir: Output directory (overrides config if provided)
        
    Returns:
        results: Dictionary with all results and paths
    """
    if output_dir is None:
        output_dir = Path(config.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("MVAR-ROM Evaluation Pipeline")
    print("="*80)
    
    # Flatten spatial dimensions
    T = densities.shape[0]
    n_c = nx * ny
    X = densities.reshape(T, n_c)
    
    # Split data
    T0 = int(config.train_frac * T)
    T1 = T - T0
    X_train = X[:T0]
    X_test = X[T0:]
    
    print(f"\nData split: T0 = {T0}, T1 = {T1} ({config.train_frac*100:.0f}% train)")
    
    # ========================================================================
    # 1. POD
    # ========================================================================
    print("\n[1/5] Fitting POD...")
    t_start = time.time()
    Ud, xbar, d, energy_curve = fit_pod(X_train, energy=config.pod_energy)
    t_pod = time.time() - t_start
    print(f"POD complete: d = {d}, time = {t_pod:.2f}s")
    
    # Save POD
    pod_dir = output_dir / "pod"
    pod_dir.mkdir(exist_ok=True)
    np.save(pod_dir / "Ud.npy", Ud)
    np.save(pod_dir / "xbar.npy", xbar)
    np.save(pod_dir / "energy_curve.npy", energy_curve)
    
    plot_pod_energy(energy_curve, d, save_path=pod_dir / "energy.png")
    
    # ========================================================================
    # 2. Restrict to latent space
    # ========================================================================
    print("\n[2/5] Restricting to latent space...")
    Y_train = restrict(X_train, Ud, xbar)
    Y_test = restrict(X_test, Ud, xbar)
    
    # ========================================================================
    # 3. Fit MVAR
    # ========================================================================
    print("\n[3/5] Fitting MVAR...")
    t_start = time.time()
    A0, A = fit_mvar(Y_train, w=config.mvar_order, ridge=config.ridge)
    t_train = time.time() - t_start
    print(f"MVAR complete: w = {config.mvar_order}, λ = {config.ridge}, time = {t_train:.2f}s")
    
    # Save MVAR
    mvar_dir = output_dir / f"mvar_w{config.mvar_order}_lam{config.ridge:.0e}"
    mvar_dir.mkdir(exist_ok=True)
    np.save(mvar_dir / "A0.npy", A0)
    np.save(mvar_dir / "Astack.npy", A)
    
    # ========================================================================
    # 4. Forecast (closed-loop)
    # ========================================================================
    print("\n[4/5] Forecasting (closed-loop)...")
    Y_seed = Y_train[-config.mvar_order:]
    
    t_start = time.time()
    Y_forecast = forecast_closed_loop(Y_seed, A0, A, steps=T1)
    t_forecast = time.time() - t_start
    forecast_fps = T1 / (t_forecast + 1e-16)
    print(f"Forecast complete: {T1} steps in {t_forecast:.2f}s ({forecast_fps:.1f} FPS)")
    
    # Lift to density space
    X_forecast = lift(Y_forecast, Ud, xbar)
    
    # ========================================================================
    # 5. Evaluate
    # ========================================================================
    print("\n[5/5] Evaluating...")
    frame_metrics, summary = evaluate(
        X_test, X_forecast, xbar, T0, 
        tolerance_threshold=config.tolerance_threshold
    )
    
    # Add timing and config to summary
    summary.update({
        "d": int(d),
        "w": int(config.mvar_order),
        "lambda": float(config.ridge),
        "train_frac": float(config.train_frac),
        "T0": int(T0),
        "T1": int(T1),
        "train_time_s": float(t_train + t_pod),
        "forecast_time_s": float(t_forecast),
        "forecast_fps": float(forecast_fps),
    })
    
    # Save metrics
    np.savetxt(
        mvar_dir / "metrics_over_time.csv",
        np.column_stack([
            np.arange(T1),
            frame_metrics["e1"],
            frame_metrics["e2"],
            frame_metrics["einf"],
            frame_metrics["rmse"],
            frame_metrics["mass_error"],
        ]),
        header="t,e1,e2,einf,rmse,mass_error",
        delimiter=",",
        comments=""
    )
    
    with open(mvar_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "-"*80)
    print("Summary Metrics:")
    print(f"  R² = {summary['r2']:.4f}")
    print(f"  Median L² = {summary['median_e2']:.4f}")
    print(f"  P10 L² = {summary['p10_e2']:.4f}, P90 L² = {summary['p90_e2']:.4f}")
    print(f"  Tolerance horizon (10%) = {summary['tau_tol']} frames")
    print(f"  Mean mass error = {summary['mean_mass_error']:.6f}")
    print(f"  Max mass error = {summary['max_mass_error']:.6f}")
    print("-"*80)
    
    # ========================================================================
    # Plots
    # ========================================================================
    print("\nGenerating plots...")
    
    plot_errors_timeseries(frame_metrics, summary, T0, 
                          save_path=mvar_dir / "errors_timeseries.png")
    
    if config.save_snapshots:
        plot_snapshots(X_test, X_forecast, nx, ny, T0, n_snapshots=6,
                      save_path=mvar_dir / "snapshots.png")
    
    plot_latent_scatter(Y_test, Y_forecast, n_modes=min(3, d),
                       save_path=mvar_dir / "latent_scatter.png")
    
    # ========================================================================
    # Videos
    # ========================================================================
    if config.save_videos:
        print("\nGenerating videos...")
        videos_dir = mvar_dir / "videos"
        videos_dir.mkdir(exist_ok=True)
        
        # Reshape back to 2D for videos
        densities_true_2d = X_test.reshape(T1, nx, ny)
        densities_pred_2d = X_forecast.reshape(T1, nx, ny)
        
        # Create individual density movies
        create_density_movie(
            densities_true_2d, nx, ny,
            save_path=videos_dir / "true_density.mp4",
            title="Ground Truth Density",
            fps=config.fps,
            max_frames=500
        )
        
        create_density_movie(
            densities_pred_2d, nx, ny,
            save_path=videos_dir / "pred_density.mp4",
            title="MVAR-ROM Predicted Density",
            fps=config.fps,
            max_frames=500
        )
        
        # Create comparison movie
        create_comparison_movie(
            densities_true_2d, densities_pred_2d,
            frame_metrics, nx, ny, T0,
            save_path=videos_dir / "true_vs_pred_comparison.mp4",
            fps=config.fps,
            max_frames=500
        )
    
    print("\n" + "="*80)
    print(f"✓ Evaluation complete! Results saved to: {mvar_dir}")
    print("="*80 + "\n")
    
    return {
        "summary": summary,
        "frame_metrics": frame_metrics,
        "output_dir": mvar_dir,
        "Ud": Ud,
        "xbar": xbar,
        "A0": A0,
        "A": A,
        "Y_test": Y_test,
        "Y_forecast": Y_forecast,
        "X_test": X_test,
        "X_forecast": X_forecast,
    }
