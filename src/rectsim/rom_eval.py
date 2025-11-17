"""ROM evaluation utilities for POD + MVAR pipeline.

This module provides standardized evaluation, metrics, and visualization tools
for Reduced-Order Models (ROM) based on global POD and MVAR forecasting.

Key features:
- Train/test splitting across runs and within runs
- Mass conservation checks
- Comprehensive error metrics (L1, L2, L∞, RMSE, R²)
- Order parameter tracking
- Standardized output directory structure
- Dashboard plots and comparison videos

Directory structure:
    rom/
    └── <experiment_name>/
        ├── pod/
        │   ├── basis.npz
        │   ├── pod_energy.png
        │   └── pod_info.json
        ├── latent/
        │   └── run_<id>_latent.npz
        └── mvar/
            ├── mvar_model.npz
            ├── train_info.json
            └── forecast/
                ├── forecast_run_<id>.npz
                ├── metrics_run_<id>.json
                ├── order_params_run_<id>.csv
                ├── errors_time_run_<id>.png
                ├── order_params_run_<id>.png
                ├── snapshot_grid_run_<id>.png
                ├── density_true_run_<id>.mp4
                ├── density_pred_run_<id>.mp4
                └── density_comparison_run_<id>.mp4

Author: Maria
Date: November 2025
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation


# ============================================================================
# Configuration and directory structure
# ============================================================================


@dataclass
class ROMConfig:
    """Configuration for ROM experiment.

    Attributes
    ----------
    experiment_name : str
        Unique identifier for this experiment.
    train_runs : list of int
        Indices of runs used for training POD and MVAR.
    test_runs : list of int
        Indices of runs used for testing.
    train_frac : float
        Fraction of each run's time steps used for MVAR training.
    mvar_order : int
        MVAR model order (number of lags).
    ridge : float
        Ridge regularization parameter.
    energy_threshold : float
        POD energy threshold for mode selection.
    latent_dim : int or None
        Number of POD modes (None if energy-based).
    sim_root : Path
        Root directory containing simulation runs.
    rom_root : Path
        Root directory for ROM outputs.
    """

    experiment_name: str
    train_runs: List[int]
    test_runs: List[int]
    train_frac: float = 0.8
    mvar_order: int = 4
    ridge: float = 1e-6
    energy_threshold: float = 0.995
    latent_dim: int | None = None
    sim_root: Path = field(default_factory=lambda: Path("simulations"))
    rom_root: Path = field(default_factory=lambda: Path("rom"))

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            "experiment_name": self.experiment_name,
            "train_runs": self.train_runs,
            "test_runs": self.test_runs,
            "train_frac": self.train_frac,
            "mvar_order": self.mvar_order,
            "ridge": self.ridge,
            "energy_threshold": self.energy_threshold,
            "latent_dim": self.latent_dim,
            "sim_root": str(self.sim_root),
            "rom_root": str(self.rom_root),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ROMConfig":
        """Load from dict."""
        d = d.copy()
        d["sim_root"] = Path(d["sim_root"])
        d["rom_root"] = Path(d["rom_root"])
        return cls(**d)


def setup_rom_directories(config: ROMConfig) -> Dict[str, Path]:
    """Create standardized ROM directory structure.

    Parameters
    ----------
    config : ROMConfig
        ROM configuration.

    Returns
    -------
    paths : dict
        Dictionary of created paths:
        - 'base': rom/<experiment_name>
        - 'pod': rom/<experiment_name>/pod
        - 'latent': rom/<experiment_name>/latent
        - 'mvar': rom/<experiment_name>/mvar
        - 'forecast': rom/<experiment_name>/mvar/forecast
    """
    base = config.rom_root / config.experiment_name
    paths = {
        "base": base,
        "pod": base / "pod",
        "latent": base / "latent",
        "mvar": base / "mvar",
        "forecast": base / "mvar" / "forecast",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


# ============================================================================
# Train/test splitting
# ============================================================================


def split_runs_train_test(
    run_dirs: List[Path],
    train_indices: List[int],
    test_indices: List[int],
) -> Tuple[List[Path], List[Path]]:
    """Split run directories into training and test sets.

    Parameters
    ----------
    run_dirs : list of Path
        All run directories.
    train_indices : list of int
        Indices for training runs.
    test_indices : list of int
        Indices for test runs.

    Returns
    -------
    train_dirs : list of Path
        Training run directories.
    test_dirs : list of Path
        Test run directories.

    Examples
    --------
    >>> run_dirs = [Path(f"run_{i:04d}") for i in range(10)]
    >>> train, test = split_runs_train_test(run_dirs, [0,1,2,3,4], [5,6])
    >>> len(train), len(test)
    (5, 2)
    """
    train_dirs = [run_dirs[i] for i in train_indices]
    test_dirs = [run_dirs[i] for i in test_indices]
    return train_dirs, test_dirs


def get_forecast_split_indices(T: int, train_frac: float) -> Tuple[int, int]:
    """Get time indices for train/forecast split within a run.

    Parameters
    ----------
    T : int
        Total number of time steps.
    train_frac : float
        Fraction of time steps for training (0.0 to 1.0).

    Returns
    -------
    T_train : int
        Index where training ends (exclusive).
    T_forecast : int
        Number of forecast steps (T - T_train).

    Examples
    --------
    >>> get_forecast_split_indices(1000, 0.8)
    (800, 200)
    """
    T_train = int(np.ceil(train_frac * T))
    T_train = max(T_train, 1)  # At least 1 training point
    T_train = min(T_train, T - 1)  # At least 1 forecast point
    T_forecast = T - T_train
    return T_train, T_forecast


# ============================================================================
# Mass conservation checks
# ============================================================================


def compute_mass(rho: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    """Compute total mass for density field(s).

    Parameters
    ----------
    rho : np.ndarray, shape (T, ny, nx) or (ny, nx)
        Density field(s).
    dx, dy : float, optional
        Grid spacing. Default is 1.0.

    Returns
    -------
    mass : np.ndarray, shape (T,) or scalar
        Total mass at each time step.

    Examples
    --------
    >>> rho = np.ones((10, 32, 32))
    >>> mass = compute_mass(rho, dx=0.1, dy=0.1)
    >>> mass.shape
    (10,)
    """
    dA = dx * dy

    if rho.ndim == 2:
        return float(np.sum(rho) * dA)
    elif rho.ndim == 3:
        return np.sum(rho, axis=(1, 2)) * dA
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {rho.shape}")


def check_mass_conservation(
    mass_true: np.ndarray,
    mass_pred: np.ndarray,
    tolerance: float = 1e-6,
) -> Dict:
    """Check if mass is conserved in forecast.

    Parameters
    ----------
    mass_true : np.ndarray, shape (T,)
        True mass over time.
    mass_pred : np.ndarray, shape (T,)
        Predicted mass over time.
    tolerance : float, optional
        Absolute tolerance for mass drift. Default is 1e-6.

    Returns
    -------
    result : dict
        Contains:
        - 'mass_error': np.ndarray, shape (T,) - mass_pred - mass_true
        - 'mass_drift_max': float - max absolute error
        - 'mass_conservation_ok': bool - True if max drift < tolerance
    """
    mass_error = mass_pred - mass_true
    mass_drift_max = float(np.max(np.abs(mass_error)))
    mass_conservation_ok = mass_drift_max < tolerance

    return {
        "mass_error": mass_error,
        "mass_drift_max": mass_drift_max,
        "mass_conservation_ok": mass_conservation_ok,
    }


# ============================================================================
# Error metrics
# ============================================================================


def compute_pointwise_errors(
    rho_true: np.ndarray,
    rho_pred: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute pointwise error metrics over time.

    Parameters
    ----------
    rho_true : np.ndarray, shape (T, ny, nx)
        True density fields.
    rho_pred : np.ndarray, shape (T, ny, nx)
        Predicted density fields.

    Returns
    -------
    errors : dict
        Contains arrays of shape (T,):
        - 'e1': L1 norm per time step
        - 'e2': L2 norm per time step
        - 'e_inf': L∞ norm per time step
        - 'rmse': Root mean square error per time step
    """
    T, ny, nx = rho_true.shape
    n_grid = ny * nx

    # Compute differences
    diff = rho_pred - rho_true

    # Flatten spatial dimensions
    diff_flat = diff.reshape(T, -1)
    true_flat = rho_true.reshape(T, -1)

    # Compute norms
    e1 = np.sum(np.abs(diff_flat), axis=1)
    e2 = np.linalg.norm(diff_flat, axis=1)
    e_inf = np.max(np.abs(diff_flat), axis=1)
    rmse = e2 / np.sqrt(n_grid)

    # Alternative: normalized RMSE
    true_norm = np.linalg.norm(true_flat, axis=1)
    rmse_normalized = e2 / (true_norm + 1e-12)

    return {
        "e1": e1,
        "e2": e2,
        "e_inf": e_inf,
        "rmse": rmse,
        "rmse_normalized": rmse_normalized,
    }


def compute_summary_metrics(errors: Dict[str, np.ndarray]) -> Dict:
    """Compute summary statistics over forecast window.

    Parameters
    ----------
    errors : dict
        Output from compute_pointwise_errors.

    Returns
    -------
    summary : dict
        Contains scalar metrics:
        - 'median_e1', 'median_e2', 'median_einf'
        - 'p10_e2', 'p90_e2'
        - 'mean_rmse', 'median_rmse'
        - 'mean_rmse_normalized'
    """
    return {
        "median_e1": float(np.median(errors["e1"])),
        "median_e2": float(np.median(errors["e2"])),
        "median_einf": float(np.median(errors["e_inf"])),
        "p10_e2": float(np.percentile(errors["e2"], 10)),
        "p90_e2": float(np.percentile(errors["e2"], 90)),
        "mean_rmse": float(np.mean(errors["rmse"])),
        "median_rmse": float(np.median(errors["rmse"])),
        "mean_rmse_normalized": float(np.mean(errors["rmse_normalized"])),
    }


def compute_r2_score(
    rho_true: np.ndarray,
    rho_pred: np.ndarray,
) -> float:
    """Compute R² coefficient of determination.

    Treats each (t, i, j) as one sample.

    Parameters
    ----------
    rho_true : np.ndarray, shape (T, ny, nx)
        True density fields.
    rho_pred : np.ndarray, shape (T, ny, nx)
        Predicted density fields.

    Returns
    -------
    r2 : float
        Coefficient of determination.
    """
    true_flat = rho_true.flatten()
    pred_flat = rho_pred.flatten()

    ss_res = np.sum((true_flat - pred_flat) ** 2)
    ss_tot = np.sum((true_flat - true_flat.mean()) ** 2)

    if ss_tot < 1e-12:
        return 0.0

    r2 = 1.0 - ss_res / ss_tot
    return float(r2)


def count_nans(arrays: Dict[str, np.ndarray]) -> int:
    """Count total NaNs in a collection of arrays.

    Parameters
    ----------
    arrays : dict
        Dictionary of arrays to check.

    Returns
    -------
    nan_count : int
        Total number of NaN values.
    """
    nan_count = 0
    for arr in arrays.values():
        nan_count += int(np.sum(np.isnan(arr)))
    return nan_count


# ============================================================================
# Order parameter tracking
# ============================================================================


def compute_density_order_parameter(rho: np.ndarray) -> np.ndarray:
    """Compute simple order parameter from density field.

    This is a placeholder - replace with actual order parameter
    computation based on your system (e.g., polarization, clustering).

    Parameters
    ----------
    rho : np.ndarray, shape (T, ny, nx)
        Density fields.

    Returns
    -------
    order : np.ndarray, shape (T,)
        Order parameter at each time step.

    Notes
    -----
    Current implementation: std(rho) as a proxy for spatial variation.
    Replace with proper order parameter for your system.
    """
    # Compute spatial standard deviation as proxy
    T = rho.shape[0]
    order = np.zeros(T)

    for t in range(T):
        order[t] = np.std(rho[t])

    return order


def compare_order_parameters(
    rho_true: np.ndarray,
    rho_pred: np.ndarray,
) -> pd.DataFrame:
    """Compute order parameters for true and predicted densities.

    Parameters
    ----------
    rho_true : np.ndarray, shape (T, ny, nx)
        True density fields.
    rho_pred : np.ndarray, shape (T, ny, nx)
        Predicted density fields.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns:
        - 'time': time step index
        - 'order_true': order parameter from true density
        - 'order_pred': order parameter from predicted density
        - 'order_error': absolute difference
    """
    T = rho_true.shape[0]

    order_true = compute_density_order_parameter(rho_true)
    order_pred = compute_density_order_parameter(rho_pred)
    order_error = np.abs(order_pred - order_true)

    df = pd.DataFrame({
        "time": np.arange(T),
        "order_true": order_true,
        "order_pred": order_pred,
        "order_error": order_error,
    })

    return df


# ============================================================================
# Visualization utilities
# ============================================================================


def plot_errors_dashboard(
    times: np.ndarray,
    errors: Dict[str, np.ndarray],
    mass_error: np.ndarray,
    out_path: Path,
) -> None:
    """Create dashboard plot with error metrics over time.

    Parameters
    ----------
    times : np.ndarray, shape (T,)
        Time values.
    errors : dict
        Output from compute_pointwise_errors.
    mass_error : np.ndarray, shape (T,)
        Mass conservation error.
    out_path : Path
        Output file path.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    # Top panel: L2 error and RMSE
    ax = axes[0]
    ax.plot(times, errors["e2"], "b-", linewidth=1.5, label="L2 error")
    ax.set_ylabel("L2 error", color="b")
    ax.tick_params(axis="y", labelcolor="b")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(times, errors["rmse"], "r-", linewidth=1.5, label="RMSE")
    ax2.set_ylabel("RMSE", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    ax.set_title("Forecast Errors Over Time")

    # Middle panel: RMSE normalized
    ax = axes[1]
    ax.plot(times, errors["rmse_normalized"], "g-", linewidth=1.5)
    ax.set_ylabel("Normalized RMSE")
    ax.set_title("Normalized RMSE (relative to ||ρ_true||)")
    ax.grid(True, alpha=0.3)

    # Bottom panel: Mass error
    ax = axes[2]
    ax.plot(times, mass_error, "m-", linewidth=1.5)
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mass error")
    ax.set_title("Mass Conservation")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_order_parameters(
    df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot order parameter comparison.

    Parameters
    ----------
    df : pd.DataFrame
        Output from compare_order_parameters.
    out_path : Path
        Output file path.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Top panel: True vs predicted
    ax = axes[0]
    ax.plot(df["time"], df["order_true"], "b-", linewidth=1.5, label="True")
    ax.plot(df["time"], df["order_pred"], "r--", linewidth=1.5, label="Predicted")
    ax.set_ylabel("Order parameter")
    ax.set_title("Order Parameter: True vs Predicted")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom panel: Error
    ax = axes[1]
    ax.plot(df["time"], df["order_error"], "m-", linewidth=1.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Absolute error")
    ax.set_title("Order Parameter Error")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_snapshot_grid(
    rho_true: np.ndarray,
    rho_pred: np.ndarray,
    times: np.ndarray,
    snapshot_indices: List[int],
    out_path: Path,
) -> None:
    """Create grid of snapshot comparisons at selected times.

    Parameters
    ----------
    rho_true : np.ndarray, shape (T, ny, nx)
        True density fields.
    rho_pred : np.ndarray, shape (T, ny, nx)
        Predicted density fields.
    times : np.ndarray, shape (T,)
        Time values.
    snapshot_indices : list of int
        Time indices for snapshots.
    out_path : Path
        Output file path.
    """
    n_snapshots = len(snapshot_indices)
    fig, axes = plt.subplots(n_snapshots, 3, figsize=(12, 4 * n_snapshots))

    if n_snapshots == 1:
        axes = axes.reshape(1, -1)

    # Determine common colorbar range
    vmin = min(rho_true.min(), rho_pred.min())
    vmax = max(rho_true.max(), rho_pred.max())

    for i, t_idx in enumerate(snapshot_indices):
        # True
        ax = axes[i, 0]
        im = ax.imshow(rho_true[t_idx], origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(f"True (t={times[t_idx]:.2f})")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Predicted
        ax = axes[i, 1]
        im = ax.imshow(rho_pred[t_idx], origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(f"Predicted (t={times[t_idx]:.2f})")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Difference
        ax = axes[i, 2]
        diff = rho_pred[t_idx] - rho_true[t_idx]
        diff_max = max(abs(diff.min()), abs(diff.max()))
        im = ax.imshow(diff, origin="lower", vmin=-diff_max, vmax=diff_max, cmap="RdBu_r")
        ax.set_title(f"Difference (t={times[t_idx]:.2f})")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_density_video(
    rho: np.ndarray,
    times: np.ndarray,
    out_path: Path,
    title: str = "Density",
    fps: int = 20,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Create video of density evolution.

    Parameters
    ----------
    rho : np.ndarray, shape (T, ny, nx)
        Density fields.
    times : np.ndarray, shape (T,)
        Time values.
    out_path : Path
        Output file path (.mp4 or .gif).
    title : str, optional
        Video title.
    fps : int, optional
        Frames per second. Default is 20.
    vmin, vmax : float or None, optional
        Colorbar limits. If None, use data min/max.
    """
    T, ny, nx = rho.shape

    if vmin is None:
        vmin = rho.min()
    if vmax is None:
        vmax = rho.max()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(rho[0], origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
    plt.colorbar(im, ax=ax, label="Density")
    ax.set_title(f"{title} (t={times[0]:.2f})")
    ax.axis("off")

    def update(frame):
        im.set_data(rho[frame])
        ax.set_title(f"{title} (t={times[frame]:.2f})")
        return [im]

    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=True)
    anim.save(out_path, fps=fps, dpi=100)
    plt.close(fig)


def create_comparison_video(
    rho_true: np.ndarray,
    rho_pred: np.ndarray,
    times: np.ndarray,
    out_path: Path,
    fps: int = 20,
) -> None:
    """Create side-by-side comparison video.

    Parameters
    ----------
    rho_true : np.ndarray, shape (T, ny, nx)
        True density fields.
    rho_pred : np.ndarray, shape (T, ny, nx)
        Predicted density fields.
    times : np.ndarray, shape (T,)
        Time values.
    out_path : Path
        Output file path.
    fps : int, optional
        Frames per second. Default is 20.
    """
    T = rho_true.shape[0]

    # Common colorbar range
    vmin = min(rho_true.min(), rho_pred.min())
    vmax = max(rho_true.max(), rho_pred.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im1 = axes[0].imshow(rho_true[0], origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0].set_title("True")
    axes[0].axis("off")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(rho_pred[0], origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
    axes[1].set_title("Predicted")
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(f"t={times[0]:.2f}")

    def update(frame):
        im1.set_data(rho_true[frame])
        im2.set_data(rho_pred[frame])
        fig.suptitle(f"t={times[frame]:.2f}")
        return [im1, im2]

    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=True)
    anim.save(out_path, fps=fps, dpi=100)
    plt.close(fig)
