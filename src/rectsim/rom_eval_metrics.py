"""Metrics computation for ROM/MVAR evaluation on unseen IC simulations.

This module provides error metrics that are consistent with the existing
ROM/MVAR training pipeline (R², RMSE, L1/L2/Linf norms, mass conservation).

Author: Maria
Date: November 2025
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import numpy as np


@dataclass
class SimulationMetrics:
    """Forecast metrics for a single test simulation.
    
    Attributes
    ----------
    ic_type : str
        Initial condition type (e.g., "ring", "gaussian", "uniform").
    name : str
        Simulation identifier.
    r2 : float
        R² score over forecast horizon (1 - MSE/var).
    rmse_mean : float
        Mean RMSE over time steps.
    e1_median : float
        Median L¹ error (mean absolute error per grid cell).
    e2_median : float
        Median L² error (RMSE per time step).
    einf_median : float
        Median L∞ error (max absolute error per grid cell).
    mass_error_mean : float
        Mean relative mass error over time.
    mass_error_max : float
        Maximum relative mass error over time.
    tau : Optional[float]
        Time when error first exceeds tolerance (None if never).
    n_forecast : int
        Number of forecast steps.
    train_frac : float
        Fraction of trajectory used for initialization.
    """
    
    ic_type: str
    name: str
    r2: float
    rmse_mean: float
    e1_median: float
    e2_median: float
    einf_median: float
    mass_error_mean: float
    mass_error_max: float
    tau: Optional[float]
    n_forecast: int
    train_frac: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def compute_forecast_metrics(
    density_true: np.ndarray,
    density_pred: np.ndarray,
    times: Optional[np.ndarray] = None,
    tol: float = 0.1,
    train_frac: float = 0.8,
) -> Dict[str, Any]:
    """Compute forecast error metrics.
    
    Metrics are consistent with existing ROM/MVAR pipeline:
    - R² (coefficient of determination)
    - RMSE (root mean squared error)
    - L¹, L², L∞ norms per time step
    - Mass conservation error
    - τ (first time error exceeds tolerance)
    
    Parameters
    ----------
    density_true : np.ndarray
        Ground truth density, shape (T, Ny, Nx).
    density_pred : np.ndarray
        Predicted density, shape (T, Ny, Nx).
    times : Optional[np.ndarray]
        Time points, shape (T,). If None, uses indices.
    tol : float, default=0.1
        Tolerance for computing τ (relative L² error threshold).
    train_frac : float, default=0.8
        Fraction of trajectory used for initialization (for metadata).
        
    Returns
    -------
    metrics : dict
        Dictionary with keys:
        - r2, rmse_mean, e1_median, e2_median, einf_median
        - mass_error_mean, mass_error_max
        - tau (or None)
        - n_forecast, train_frac
    """
    T, Ny, Nx = density_true.shape
    assert density_pred.shape == density_true.shape, \
        f"Shape mismatch: {density_pred.shape} vs {density_true.shape}"
    
    if times is None:
        times = np.arange(T)
    
    # Flatten spatial dimensions for easier computation
    rho_true_flat = density_true.reshape(T, -1)  # (T, Ny*Nx)
    rho_pred_flat = density_pred.reshape(T, -1)
    
    # --- R² (coefficient of determination) ---
    # R² = 1 - SS_res / SS_tot
    # SS_res = sum of squared residuals
    # SS_tot = total sum of squares (variance)
    ss_res = np.sum((rho_true_flat - rho_pred_flat) ** 2)
    ss_tot = np.sum((rho_true_flat - rho_true_flat.mean()) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # --- Per-time-step errors ---
    # L² (RMSE per time)
    e2_t = np.sqrt(np.mean((density_true - density_pred) ** 2, axis=(1, 2)))
    
    # L¹ (MAE per time)
    e1_t = np.mean(np.abs(density_true - density_pred), axis=(1, 2))
    
    # L∞ (max error per time)
    einf_t = np.max(np.abs(density_true - density_pred), axis=(1, 2))
    
    # Summary statistics
    rmse_mean = e2_t.mean()
    e1_median = np.median(e1_t)
    e2_median = np.median(e2_t)
    einf_median = np.median(einf_t)
    
    # --- Mass conservation ---
    # Mass = sum over grid (assuming uniform cell area)
    mass_true = density_true.sum(axis=(1, 2))
    mass_pred = density_pred.sum(axis=(1, 2))
    
    # Relative mass error per time
    mass_error_t = np.abs(mass_pred - mass_true) / (np.abs(mass_true) + 1e-12)
    mass_error_mean = mass_error_t.mean()
    mass_error_max = mass_error_t.max()
    
    # --- τ (time when error exceeds tolerance) ---
    # Use relative L² error: ||ρ_pred - ρ_true||_2 / ||ρ_true||_2
    # Compute per time step
    norm_true_t = np.sqrt(np.sum(density_true ** 2, axis=(1, 2)))
    norm_error_t = np.sqrt(np.sum((density_pred - density_true) ** 2, axis=(1, 2)))
    rel_error_t = norm_error_t / (norm_true_t + 1e-12)
    
    # Find first time exceeding tolerance
    exceed_mask = rel_error_t > tol
    if np.any(exceed_mask):
        idx = np.where(exceed_mask)[0][0]
        tau = times[idx].item()
    else:
        tau = None
    
    return {
        "r2": float(r2),
        "rmse_mean": float(rmse_mean),
        "e1_median": float(e1_median),
        "e2_median": float(e2_median),
        "einf_median": float(einf_median),
        "mass_error_mean": float(mass_error_mean),
        "mass_error_max": float(mass_error_max),
        "tau": tau,
        "n_forecast": int(T),
        "train_frac": float(train_frac),
    }


def compute_relative_errors_timeseries(
    density_true: np.ndarray,
    density_pred: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute per-time-step error arrays for plotting.
    
    Parameters
    ----------
    density_true : np.ndarray
        Ground truth, shape (T, Ny, Nx).
    density_pred : np.ndarray
        Prediction, shape (T, Ny, Nx).
        
    Returns
    -------
    errors : dict
        Dictionary with keys:
        - "e1": L¹ error per time, shape (T,)
        - "e2": L² error per time, shape (T,)
        - "einf": L∞ error per time, shape (T,)
        - "rel_e2": Relative L² error per time, shape (T,)
        - "mass_error": Relative mass error per time, shape (T,)
    """
    T = density_true.shape[0]
    
    # Per-time errors
    e1_t = np.mean(np.abs(density_true - density_pred), axis=(1, 2))
    e2_t = np.sqrt(np.mean((density_true - density_pred) ** 2, axis=(1, 2)))
    einf_t = np.max(np.abs(density_true - density_pred), axis=(1, 2))
    
    # Relative L² error
    norm_true_t = np.sqrt(np.sum(density_true ** 2, axis=(1, 2)))
    norm_error_t = np.sqrt(np.sum((density_pred - density_true) ** 2, axis=(1, 2)))
    rel_e2_t = norm_error_t / (norm_true_t + 1e-12)
    
    # Mass error
    mass_true = density_true.sum(axis=(1, 2))
    mass_pred = density_pred.sum(axis=(1, 2))
    mass_error_t = np.abs(mass_pred - mass_true) / (np.abs(mass_true) + 1e-12)
    
    return {
        "e1": e1_t,
        "e2": e2_t,
        "einf": einf_t,
        "rel_e2": rel_e2_t,
        "mass_error": mass_error_t,
    }
