"""Multivariate AutoRegressive (MVAR) modeling in POD latent space.

This module implements the global POD + MVAR pipeline for collective motion
density field forecasting. The approach follows the EF-ROM methodology:

1. Load KDE density movies from multiple simulations (varied ICs, same model)
2. Compute global POD basis across all runs and time steps
3. Project density fields to low-dimensional latent space
4. Train linear MVAR model on latent time series with ridge regularization
5. Evaluate multi-step forecasts on held-out test segments

The pipeline is designed for Oscar HPC and local execution with reproducible
outputs saved to structured directories.

Key references:
- Alvarez et al. "Autoregressive ROM for collective motion forecasting"
- Bhaskar & Ziegelmeier "D'Orsogna swarm KDE-based density representation"
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_density_movies(run_dirs: List[Path]) -> Dict[str, Dict[str, np.ndarray]]:
    """Load density movies from multiple simulation runs.

    Parameters
    ----------
    run_dirs : list of Path
        List of run directories, each containing density.npz.

    Returns
    -------
    density_dict : dict
        Keyed by run name (directory name), values are dicts with:
        - 'rho': np.ndarray, shape (T, ny, nx) - density movie
        - 'times': np.ndarray, shape (T,) - time stamps
        - 'meta': dict - optional metadata from run.json if present

    Examples
    --------
    >>> from pathlib import Path
    >>> run_dirs = [Path("outputs/simulations/run_001")]
    >>> density_dict = load_density_movies(run_dirs)
    >>> density_dict['run_001']['rho'].shape
    (1001, 128, 128)
    """
    density_dict = {}

    for run_dir in run_dirs:
        run_name = run_dir.name

        # Load density.npz
        density_path = run_dir / "density.npz"
        if not density_path.exists():
            print(f"Warning: {density_path} not found, skipping {run_name}")
            continue

        data = np.load(density_path)
        rho = data["rho"]  # (T, ny, nx) or (T, ny, nx, 1)
        times = data.get("times", np.arange(rho.shape[0]))

        # Squeeze singleton channel dimension if present
        if rho.ndim == 4 and rho.shape[-1] == 1:
            rho = rho.squeeze(-1)

        # Load optional metadata
        meta = {}
        run_json_path = run_dir / "run.json"
        if run_json_path.exists():
            try:
                with open(run_json_path, "r") as f:
                    meta = json.load(f)
            except Exception:
                pass

        density_dict[run_name] = {
            "rho": rho,
            "times": times,
            "meta": meta,
        }

    return density_dict


def build_global_snapshot_matrix(
    density_dict: Dict[str, Dict[str, np.ndarray]],
    subtract_mean: bool = True,
) -> Tuple[np.ndarray, Dict[str, slice], np.ndarray]:
    """Build global snapshot matrix from multiple density runs.

    Concatenates flattened density fields from all runs into a single matrix,
    optionally subtracting the global spatial mean.

    Parameters
    ----------
    density_dict : dict
        Output from load_density_movies. Keys are run names, values contain 'rho'.
    subtract_mean : bool, optional
        If True, subtract global temporal-ensemble mean. Default is True.

    Returns
    -------
    X : np.ndarray, shape (T_total, d)
        Snapshot matrix where d = ny * nx and T_total = sum of all T_r.
    run_slices : dict
        Maps run_name to slice object indicating time indices in X.
    global_mean_flat : np.ndarray, shape (d,)
        Global spatial mean. Zero array if subtract_mean=False.

    Examples
    --------
    >>> density_dict = {'run1': {'rho': np.random.rand(100, 32, 32)},
    ...                 'run2': {'rho': np.random.rand(80, 32, 32)}}
    >>> X, slices, mean = build_global_snapshot_matrix(density_dict)
    >>> X.shape
    (180, 1024)
    >>> slices['run1']
    slice(0, 100, None)
    """
    if not density_dict:
        raise ValueError("density_dict is empty")

    # Infer grid shape from first run
    first_rho = next(iter(density_dict.values()))["rho"]
    T_first, ny, nx = first_rho.shape
    d = ny * nx

    # Flatten each run and track slices
    flattened_runs = []
    run_slices = {}
    current_t = 0

    for run_name, run_data in density_dict.items():
        rho = run_data["rho"]
        T_r = rho.shape[0]

        # Validate grid shape consistency
        if rho.shape[1:] != (ny, nx):
            raise ValueError(
                f"Run {run_name} has grid shape {rho.shape[1:]}, "
                f"expected ({ny}, {nx})"
            )

        # Flatten: (T_r, ny, nx) -> (T_r, d)
        rho_flat = rho.reshape(T_r, -1)
        flattened_runs.append(rho_flat)

        # Track slice
        run_slices[run_name] = slice(current_t, current_t + T_r)
        current_t += T_r

    # Concatenate: (T_total, d)
    X = np.vstack(flattened_runs)

    # Compute and subtract global mean if requested
    if subtract_mean:
        global_mean_flat = X.mean(axis=0)
        X = X - global_mean_flat[np.newaxis, :]
    else:
        global_mean_flat = np.zeros(d)

    return X, run_slices, global_mean_flat


def compute_pod(
    X: np.ndarray,
    r: int | None = None,
    energy_threshold: float = 0.995,
) -> Dict[str, np.ndarray]:
    """Compute POD basis via SVD on snapshot matrix.

    Performs singular value decomposition and selects leading modes based on
    cumulative energy threshold.

    Parameters
    ----------
    X : np.ndarray, shape (T_total, d)
        Snapshot matrix (already mean-subtracted if desired).
    r : int or None, optional
        Number of POD modes to keep. If None, choose based on energy_threshold.
    energy_threshold : float, optional
        Cumulative energy fraction for mode selection. Default is 0.995 (99.5%).

    Returns
    -------
    pod_basis : dict
        Dictionary containing:
        - 'Phi': np.ndarray, shape (d, r) - POD spatial modes (columns)
        - 'S': np.ndarray, shape (min(T_total, d),) - all singular values
        - 'U': np.ndarray, shape (T_total, r) - temporal coefficients (truncated)
        - 'r': int - number of modes kept
        - 'energy': np.ndarray - cumulative energy fractions
        - 'ny': int - grid height
        - 'nx': int - grid width

    Examples
    --------
    >>> X = np.random.randn(1000, 1024)
    >>> pod = compute_pod(X, r=None, energy_threshold=0.99)
    >>> pod['Phi'].shape
    (1024, 42)
    >>> pod['energy'][-1] >= 0.99
    True

    Notes
    -----
    Energy is defined as sum(S[:k]**2) / sum(S**2).
    The POD modes Phi are the right singular vectors (Vt.T).
    """
    T_total, d = X.shape

    # Perform SVD: X = U @ diag(S) @ Vt
    # U: (T_total, k), S: (k,), Vt: (k, d) where k = min(T_total, d)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Compute cumulative energy
    energy_vals = S ** 2
    total_energy = energy_vals.sum()
    cum_energy = np.cumsum(energy_vals) / total_energy

    # Determine number of modes
    if r is None:
        r = int(np.searchsorted(cum_energy, energy_threshold) + 1)
        r = min(r, len(S))  # Cap at max available modes

    # Extract truncated components
    Phi = Vt[:r, :].T  # (d, r) - spatial modes as columns
    U_trunc = U[:, :r]  # (T_total, r)
    S_trunc = S[:r]
    energy_trunc = cum_energy[:r]

    return {
        "Phi": Phi,
        "S": S,  # Keep all singular values for diagnostics
        "U": U_trunc,
        "r": r,
        "energy": cum_energy,
        "energy_trunc": energy_trunc,
        "S_trunc": S_trunc,
    }


def plot_pod_energy(
    S: np.ndarray,
    out_path: Path,
    r_mark: int | None = None,
    energy_threshold: float = 0.995,
) -> None:
    """Plot cumulative energy vs POD mode index.

    Parameters
    ----------
    S : np.ndarray
        Singular values from POD.
    out_path : Path
        Output file path for PNG.
    r_mark : int or None, optional
        Mark the chosen number of modes with a vertical line.
    energy_threshold : float, optional
        Draw horizontal line at this energy threshold.
    """
    energy_vals = S ** 2
    cum_energy = np.cumsum(energy_vals) / energy_vals.sum()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(cum_energy) + 1), cum_energy, "b-", linewidth=2)
    ax.axhline(energy_threshold, color="r", linestyle="--", label=f"{energy_threshold:.1%} threshold")

    if r_mark is not None:
        ax.axvline(r_mark, color="g", linestyle="--", label=f"r = {r_mark}")

    ax.set_xlabel("Number of POD modes")
    ax.set_ylabel("Cumulative energy fraction")
    ax.set_title("POD Mode Energy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(left=1)
    ax.set_ylim([0, 1.05])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def project_to_pod(
    density_dict: Dict[str, Dict[str, np.ndarray]],
    Phi: np.ndarray,
    global_mean_flat: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Project density movies to POD latent space.

    Parameters
    ----------
    density_dict : dict
        Output from load_density_movies.
    Phi : np.ndarray, shape (d, r)
        POD spatial modes.
    global_mean_flat : np.ndarray, shape (d,)
        Global spatial mean to subtract.

    Returns
    -------
    latent_dict : dict
        Keyed by run name, values are dicts with:
        - 'Y': np.ndarray, shape (T_r, r) - latent coefficients
        - 'times': np.ndarray, shape (T_r,) - time stamps

    Examples
    --------
    >>> density_dict = {'run1': {'rho': np.random.rand(100, 32, 32),
    ...                          'times': np.arange(100)}}
    >>> Phi = np.random.randn(1024, 50)
    >>> mean = np.zeros(1024)
    >>> latent = project_to_pod(density_dict, Phi, mean)
    >>> latent['run1']['Y'].shape
    (100, 50)
    """
    latent_dict = {}
    d, r = Phi.shape

    for run_name, run_data in density_dict.items():
        rho = run_data["rho"]
        times = run_data["times"]
        T_r = rho.shape[0]

        # Flatten: (T_r, ny, nx) -> (T_r, d)
        rho_flat = rho.reshape(T_r, -1)

        # Subtract mean
        rho_centered = rho_flat - global_mean_flat[np.newaxis, :]

        # Project: Y = X @ Phi, shape (T_r, r)
        Y = rho_centered @ Phi

        latent_dict[run_name] = {
            "Y": Y,
            "times": times,
        }

    return latent_dict


def reconstruct_from_pod(
    Y: np.ndarray,
    Phi: np.ndarray,
    global_mean_flat: np.ndarray,
    ny: int,
    nx: int,
) -> np.ndarray:
    """Reconstruct density fields from POD latent coefficients.

    Parameters
    ----------
    Y : np.ndarray, shape (T, r)
        Latent POD coefficients.
    Phi : np.ndarray, shape (d, r)
        POD spatial modes.
    global_mean_flat : np.ndarray, shape (d,)
        Global spatial mean.
    ny, nx : int
        Grid dimensions.

    Returns
    -------
    rho_rec : np.ndarray, shape (T, ny, nx)
        Reconstructed density movie.

    Examples
    --------
    >>> Y = np.random.rand(100, 50)
    >>> Phi = np.random.randn(1024, 50)
    >>> mean = np.zeros(1024)
    >>> rho = reconstruct_from_pod(Y, Phi, mean, 32, 32)
    >>> rho.shape
    (100, 32, 32)
    """
    T, r = Y.shape

    # Reconstruct: X = Y @ Phi.T, shape (T, d)
    rho_flat = Y @ Phi.T

    # Add mean
    rho_flat = rho_flat + global_mean_flat[np.newaxis, :]

    # Reshape to (T, ny, nx)
    rho_rec = rho_flat.reshape(T, ny, nx)

    return rho_rec


@dataclass
class MVARModel:
    """Multivariate AutoRegressive model in POD latent space.

    Attributes
    ----------
    order : int
        AR order (number of lags).
    A0 : np.ndarray, shape (r,)
        Constant term.
    A : np.ndarray, shape (order, r, r)
        AR coefficient matrices. A[j-1] is the matrix for lag j.
    ridge : float
        Ridge regularization parameter used in training.
    latent_dim : int
        Latent space dimension (r).

    Notes
    -----
    The MVAR model predicts:
    y(t) = A0 + A[0] @ y(t-1) + A[1] @ y(t-2) + ... + A[order-1] @ y(t-order)
    """

    order: int
    A0: np.ndarray
    A: np.ndarray
    ridge: float
    latent_dim: int

    def save(self, path: Path) -> None:
        """Save model to npz file."""
        np.savez(
            path,
            order=self.order,
            A0=self.A0,
            A=self.A,
            ridge=self.ridge,
            latent_dim=self.latent_dim,
        )

    @classmethod
    def load(cls, path: Path) -> "MVARModel":
        """Load model from npz file."""
        data = np.load(path)
        return cls(
            order=int(data["order"]),
            A0=data["A0"],
            A=data["A"],
            ridge=float(data["ridge"]),
            latent_dim=int(data["latent_dim"]),
        )


def fit_mvar_from_runs(
    latent_dict: Dict[str, Dict[str, np.ndarray]],
    order: int = 4,
    ridge: float = 1e-6,
    train_frac: float = 0.8,
) -> Tuple[MVARModel, Dict]:
    """Fit MVAR model on latent time series from multiple runs.

    Uses ridge regression to learn AR coefficients from training segments
    of each run.

    Parameters
    ----------
    latent_dict : dict
        Output from project_to_pod. Keys are run names.
    order : int, optional
        AR model order (number of lags). Default is 4.
    ridge : float, optional
        Ridge regularization parameter. Default is 1e-6.
    train_frac : float, optional
        Fraction of each run to use for training. Default is 0.8.

    Returns
    -------
    model : MVARModel
        Fitted MVAR model.
    info : dict
        Training metadata including number of samples, runs, etc.

    Examples
    --------
    >>> latent_dict = {'run1': {'Y': np.random.rand(100, 50)}}
    >>> model, info = fit_mvar_from_runs(latent_dict, order=4, ridge=1e-6)
    >>> model.A.shape
    (4, 50, 50)
    """
    # Infer latent dimension from first run
    first_Y = next(iter(latent_dict.values()))["Y"]
    r = first_Y.shape[1]

    # Collect training data from all runs
    X_list = []  # Features
    Y_list = []  # Targets

    train_splits = {}

    for run_name, run_data in latent_dict.items():
        Y_run = run_data["Y"]  # (T_r, r)
        T_r = Y_run.shape[0]

        # Determine training split
        T_train = int(train_frac * T_r)
        T_train = max(order + 1, T_train)  # Need at least order+1 points
        train_splits[run_name] = T_train

        # Build regression dataset for this run
        for t in range(order, T_train):
            # Target: y(t)
            y_t = Y_run[t]  # (r,)

            # Features: [1, y(t-1), y(t-2), ..., y(t-order)]
            x_t = [1.0]  # Constant term
            for lag in range(1, order + 1):
                x_t.extend(Y_run[t - lag])  # Append y(t-lag)

            X_list.append(x_t)
            Y_list.append(y_t)

    # Convert to arrays
    X = np.array(X_list)  # (N_samples, 1 + order*r)
    Y = np.array(Y_list)  # (N_samples, r)
    N_samples = X.shape[0]

    # Ridge regression: W = (X^T X + ridge * I)^{-1} X^T Y
    XtX = X.T @ X
    XtY = X.T @ Y
    I = np.eye(XtX.shape[0])
    W = np.linalg.solve(XtX + ridge * I, XtY)  # (1 + order*r, r)

    # Extract A0 and A matrices
    A0 = W[0, :]  # (r,)
    A = np.zeros((order, r, r))
    for j in range(order):
        block_start = 1 + j * r
        block_end = 1 + (j + 1) * r
        A[j] = W[block_start:block_end, :].T  # Transpose to get (r, r)

    model = MVARModel(
        order=order,
        A0=A0,
        A=A,
        ridge=ridge,
        latent_dim=r,
    )

    info = {
        "order": order,
        "ridge": ridge,
        "train_frac": train_frac,
        "num_runs": len(latent_dict),
        "total_samples": N_samples,
        "latent_dim": r,
        "train_splits": train_splits,
    }

    return model, info


def mvar_forecast(
    model: MVARModel,
    Y_init: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Generate multi-step forecast with MVAR model.

    Parameters
    ----------
    model : MVARModel
        Fitted MVAR model.
    Y_init : np.ndarray, shape (order, r)
        Initial conditions: last `order` latent states before forecast.
    steps : int
        Number of steps to forecast.

    Returns
    -------
    Y_forecast : np.ndarray, shape (steps, r)
        Forecasted latent trajectories.

    Examples
    --------
    >>> model = MVARModel(...)  # fitted model
    >>> Y_init = np.random.rand(4, 50)  # last 4 states
    >>> Y_pred = mvar_forecast(model, Y_init, steps=100)
    >>> Y_pred.shape
    (100, 50)
    """
    r = model.latent_dim
    order = model.order

    # Initialize history with Y_init
    Y_hist = list(Y_init)  # List of (r,) arrays
    Y_forecast = []

    for _ in range(steps):
        # Predict next state
        y_next = model.A0.copy()

        for j in range(order):
            y_next += model.A[j] @ Y_hist[-(j + 1)]

        Y_forecast.append(y_next)
        Y_hist.append(y_next)

    return np.array(Y_forecast)


def evaluate_mvar_on_runs(
    model: MVARModel,
    latent_dict: Dict[str, Dict[str, np.ndarray]],
    density_dict: Dict[str, Dict[str, np.ndarray]],
    pod_basis: Dict[str, np.ndarray],
    global_mean_flat: np.ndarray,
    ny: int,
    nx: int,
    train_frac: float = 0.8,
) -> Dict:
    """Evaluate MVAR model on held-out test segments of each run.

    Computes per-run and aggregate forecast metrics at both latent and
    density field levels.

    Parameters
    ----------
    model : MVARModel
        Fitted MVAR model.
    latent_dict : dict
        Latent trajectories from project_to_pod.
    density_dict : dict
        Original density movies from load_density_movies.
    pod_basis : dict
        POD basis from compute_pod.
    global_mean_flat : np.ndarray
        Global spatial mean.
    ny, nx : int
        Grid dimensions.
    train_frac : float, optional
        Training fraction used in fit_mvar_from_runs.

    Returns
    -------
    results : dict
        Evaluation results with keys 'per_run' and 'aggregate'.

    Examples
    --------
    >>> results = evaluate_mvar_on_runs(model, latent_dict, density_dict,
    ...                                  pod_basis, mean, 32, 32)
    >>> results['aggregate']['mean_R2']
    0.8234
    """
    Phi = pod_basis["Phi"]
    order = model.order

    per_run_results = {}
    all_R2 = []
    all_rmse_series = []

    for run_name, run_data in latent_dict.items():
        Y_run = run_data["Y"]  # (T_r, r)
        rho_run = density_dict[run_name]["rho"]  # (T_r, ny, nx)
        T_r = Y_run.shape[0]

        # Determine train/test split
        T_train = int(train_frac * T_r)
        T_test_start = T_train

        if T_test_start + order >= T_r:
            print(f"Warning: Not enough test data for {run_name}, skipping")
            continue

        # Forecast horizon
        H = T_r - T_test_start - order

        if H <= 0:
            print(f"Warning: No forecast horizon for {run_name}, skipping")
            continue

        # Initial conditions: last `order` states from training
        Y_init = Y_run[T_test_start - order : T_test_start]  # (order, r)

        # Generate forecast
        Y_pred = mvar_forecast(model, Y_init, steps=H)  # (H, r)

        # Ground truth latent and density
        Y_true = Y_run[T_test_start + order : T_test_start + order + H]  # (H, r)
        rho_true = rho_run[T_test_start + order : T_test_start + order + H]  # (H, ny, nx)

        # Reconstruct predicted density
        rho_pred = reconstruct_from_pod(Y_pred, Phi, global_mean_flat, ny, nx)

        # Compute latent-level metrics
        latent_rmse_series = np.sqrt(np.mean((Y_pred - Y_true) ** 2, axis=1))

        # Compute density-level metrics
        rmse_series = np.sqrt(np.mean((rho_pred - rho_true) ** 2, axis=(1, 2)))

        # Compute R² at density level
        rho_mean = rho_true.mean(axis=0, keepdims=True)  # (1, ny, nx)
        ss_tot = np.sum((rho_true - rho_mean) ** 2)
        ss_res = np.sum((rho_true - rho_pred) ** 2)
        R2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        per_run_results[run_name] = {
            "R2": float(R2),
            "rmse_time_series": rmse_series.tolist(),
            "latent_rmse_time_series": latent_rmse_series.tolist(),
            "T_train": int(T_train),
            "T_test": int(H),
            "mean_rmse": float(rmse_series.mean()),
            "mean_latent_rmse": float(latent_rmse_series.mean()),
        }

        all_R2.append(R2)
        all_rmse_series.append(rmse_series)

    # Aggregate statistics
    if all_R2:
        all_R2_array = np.array(all_R2)
        aggregate = {
            "mean_R2": float(all_R2_array.mean()),
            "median_R2": float(np.median(all_R2_array)),
            "std_R2": float(all_R2_array.std()),
            "min_R2": float(all_R2_array.min()),
            "max_R2": float(all_R2_array.max()),
        }

        # Aggregate RMSE statistics across time
        if all_rmse_series:
            # Stack into (num_runs, H) - pad to max length
            max_H = max(len(s) for s in all_rmse_series)
            rmse_matrix = np.full((len(all_rmse_series), max_H), np.nan)
            for i, series in enumerate(all_rmse_series):
                rmse_matrix[i, :len(series)] = series

            aggregate["mean_rmse"] = float(np.nanmean(rmse_matrix))
            aggregate["median_rmse"] = float(np.nanmedian(rmse_matrix))
            aggregate["p10_rmse"] = float(np.nanpercentile(rmse_matrix.flatten(), 10))
            aggregate["p90_rmse"] = float(np.nanpercentile(rmse_matrix.flatten(), 90))
    else:
        aggregate = {}

    results = {
        "per_run": per_run_results,
        "aggregate": aggregate,
    }

    return results
