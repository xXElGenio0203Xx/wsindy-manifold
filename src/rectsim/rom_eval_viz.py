"""Visualization utilities for ROM/MVAR evaluation results.

This module provides:
1. Best run selection per IC type
2. Error vs time plotting
3. Order parameter plotting for contextualization

Author: Maria
Date: November 2025
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from rectsim.rom_eval_metrics import SimulationMetrics
from rectsim.rom_eval_data import SimulationSample

# Use non-interactive backend if needed
mpl.use('Agg')


def select_best_runs(
    metrics_list: List[SimulationMetrics],
    key: str = "r2",
    maximize: bool = True,
) -> Dict[str, SimulationMetrics]:
    """Select best simulation per IC type based on a metric.
    
    Parameters
    ----------
    metrics_list : List[SimulationMetrics]
        All simulation metrics.
    key : str, default="r2"
        Metric to optimize. Options: "r2", "rmse_mean", "e2_median", etc.
    maximize : bool, default=True
        If True, select max value (for R²). If False, select min (for RMSE).
        
    Returns
    -------
    best_runs : Dict[str, SimulationMetrics]
        Dictionary mapping IC type to best simulation metrics.
        
    Examples
    --------
    >>> best = select_best_runs(metrics, key="r2", maximize=True)
    >>> best = select_best_runs(metrics, key="rmse_mean", maximize=False)
    """
    if not metrics_list:
        return {}
    
    # Group by IC type
    by_ic_type: Dict[str, List[SimulationMetrics]] = {}
    for m in metrics_list:
        if m.ic_type not in by_ic_type:
            by_ic_type[m.ic_type] = []
        by_ic_type[m.ic_type].append(m)
    
    # Select best for each IC type
    best_runs = {}
    for ic_type, sims in by_ic_type.items():
        # Extract metric values
        values = [getattr(m, key) for m in sims]
        
        # Find best index
        if maximize:
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        
        best_runs[ic_type] = sims[best_idx]
    
    return best_runs


def plot_error_time_series(
    times: np.ndarray,
    errors: Dict[str, np.ndarray],
    out_path: Path,
    title: str = "Forecast Error vs Time",
    ic_type: Optional[str] = None,
) -> None:
    """Plot error metrics vs time.
    
    Parameters
    ----------
    times : np.ndarray
        Time points, shape (T,).
    errors : Dict[str, np.ndarray]
        Dictionary with error arrays:
        - "e1": L¹ error per time
        - "e2": L² error per time (RMSE)
        - "einf": L∞ error per time
        - "rel_e2": Relative L² error
        - "mass_error": Mass conservation error
    out_path : Path
        Output PNG file path.
    title : str
        Plot title.
    ic_type : Optional[str]
        IC type for subtitle.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    if ic_type:
        fig.text(0.5, 0.96, f"IC Type: {ic_type}", ha='center', fontsize=11)
    
    # L² error (RMSE)
    ax = axes[0, 0]
    ax.plot(times, errors["e2"], 'b-', linewidth=2)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("L² Error (RMSE)", fontsize=10)
    ax.set_title("L² Error", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Relative L² error
    ax = axes[0, 1]
    ax.plot(times, errors["rel_e2"], 'r-', linewidth=2)
    ax.axhline(0.1, color='k', linestyle='--', alpha=0.5, label='tol=0.1')
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Relative L² Error", fontsize=10)
    ax.set_title("Relative L² Error", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # L¹ and L∞ errors
    ax = axes[1, 0]
    ax.plot(times, errors["e1"], 'g-', linewidth=2, label='L¹ (MAE)')
    ax.plot(times, errors["einf"], 'm-', linewidth=2, label='L∞ (max)')
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Error", fontsize=10)
    ax.set_title("L¹ and L∞ Errors", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Mass conservation error
    ax = axes[1, 1]
    ax.plot(times, errors["mass_error"], 'orange', linewidth=2)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Relative Mass Error", fontsize=10)
    ax.set_title("Mass Conservation Error", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_order_params_from_sample(
    sample: SimulationSample,
    T0: Optional[int] = None,
) -> pd.DataFrame:
    """Compute order parameters from simulation trajectories.
    
    Parameters
    ----------
    sample : SimulationSample
        Simulation with trajectory data.
    T0 : Optional[int]
        If provided, only compute for times >= T0 (forecast horizon).
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns:
        - "time": time points
        - "polarization": global velocity alignment
        - "speed_mean": mean speed
        - "speed_std": speed standard deviation
        
    Notes
    -----
    Requires sample.traj_true with "x", "v", "times" arrays.
    """
    if sample.traj_true is None:
        raise ValueError(f"No trajectory data for {sample.ic_type}/{sample.name}")
    
    if "v" not in sample.traj_true:
        raise ValueError(f"No velocity data in {sample.ic_type}/{sample.name}")
    
    v = sample.traj_true["v"]  # (T, N, 2)
    times = sample.traj_true.get("times", np.arange(v.shape[0]))
    
    if T0 is not None:
        v = v[T0:]
        times = times[T0:]
    
    T, N, _ = v.shape
    
    # Compute order parameters
    polarization = np.zeros(T)
    speed_mean = np.zeros(T)
    speed_std = np.zeros(T)
    
    for t in range(T):
        vt = v[t]  # (N, 2)
        
        # Speed per agent
        speeds = np.linalg.norm(vt, axis=1)  # (N,)
        speed_mean[t] = speeds.mean()
        speed_std[t] = speeds.std()
        
        # Polarization: ||<v>|| / <||v||>
        v_mean = vt.mean(axis=0)  # (2,)
        polarization[t] = np.linalg.norm(v_mean) / (speeds.mean() + 1e-12)
    
    df = pd.DataFrame({
        "time": times,
        "polarization": polarization,
        "speed_mean": speed_mean,
        "speed_std": speed_std,
    })
    
    return df


def plot_order_params(
    df: pd.DataFrame,
    out_path: Path,
    title: str = "Order Parameters vs Time",
    ic_type: Optional[str] = None,
    T0: Optional[int] = None,
) -> None:
    """Plot order parameters vs time.
    
    Parameters
    ----------
    df : pd.DataFrame
        Order parameter dataframe from compute_order_params_from_sample.
    out_path : Path
        Output PNG file path.
    title : str
        Plot title.
    ic_type : Optional[str]
        IC type for subtitle.
    T0 : Optional[int]
        Forecast start time (draws vertical line).
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    if ic_type:
        fig.text(0.5, 0.96, f"IC Type: {ic_type}", ha='center', fontsize=11)
    
    # Polarization
    ax = axes[0]
    ax.plot(df["time"], df["polarization"], 'b-', linewidth=2)
    if T0 is not None and len(df) > 0:
        t0_val = df["time"].iloc[0]
        ax.axvline(t0_val, color='r', linestyle='--', alpha=0.7, label=f'Forecast start (T0)')
        ax.legend(fontsize=9)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Polarization", fontsize=10)
    ax.set_title("Global Velocity Alignment", fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    
    # Speed statistics
    ax = axes[1]
    ax.plot(df["time"], df["speed_mean"], 'g-', linewidth=2, label='Mean')
    ax.fill_between(
        df["time"],
        df["speed_mean"] - df["speed_std"],
        df["speed_mean"] + df["speed_std"],
        alpha=0.3,
        color='g',
        label='±1 std'
    )
    if T0 is not None and len(df) > 0:
        t0_val = df["time"].iloc[0]
        ax.axvline(t0_val, color='r', linestyle='--', alpha=0.7, label=f'Forecast start (T0)')
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Speed", fontsize=10)
    ax.set_title("Agent Speed Statistics", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def make_best_run_error_plots(
    best_runs: Dict[str, SimulationMetrics],
    predictions_dict: Dict[tuple, Dict[str, Any]],
    out_root: Path,
) -> None:
    """Generate error plots for best runs per IC type.
    
    Parameters
    ----------
    best_runs : Dict[str, SimulationMetrics]
        Best simulation metrics per IC type.
    predictions_dict : Dict[tuple, Dict]
        Predictions keyed by (ic_type, name), containing "errors" and "times".
    out_root : Path
        Output root directory.
    """
    for ic_type, metrics in best_runs.items():
        key = (ic_type, metrics.name)
        
        if key not in predictions_dict:
            print(f"  Warning: No predictions found for {ic_type}/{metrics.name}")
            continue
        
        preds = predictions_dict[key]
        times = preds["times"]
        errors = preds["errors"]
        
        out_path = out_root / ic_type / "best_error.png"
        
        plot_error_time_series(
            times,
            errors,
            out_path,
            title=f"Best Run Error vs Time (R²={metrics.r2:.4f})",
            ic_type=ic_type,
        )
        
        print(f"  ✓ {out_path}")


def make_best_run_order_param_plots(
    best_runs: Dict[str, SimulationMetrics],
    samples_dict: Dict[tuple, SimulationSample],
    out_root: Path,
    predictions_dict: Optional[Dict[tuple, Dict[str, Any]]] = None,
) -> None:
    """Generate order parameter plots for best runs per IC type.
    
    Parameters
    ----------
    best_runs : Dict[str, SimulationMetrics]
        Best simulation metrics per IC type.
    samples_dict : Dict[tuple, SimulationSample]
        Samples keyed by (ic_type, name).
    out_root : Path
        Output root directory.
    predictions_dict : Optional[Dict]
        If provided, extract T0 to mark forecast start.
    """
    for ic_type, metrics in best_runs.items():
        key = (ic_type, metrics.name)
        
        if key not in samples_dict:
            print(f"  Warning: No sample found for {ic_type}/{metrics.name}")
            continue
        
        sample = samples_dict[key]
        
        if sample.traj_true is None or "v" not in sample.traj_true:
            print(f"  Warning: No trajectory data for {ic_type}/{metrics.name}")
            continue
        
        # Get T0 if available
        T0 = None
        if predictions_dict and key in predictions_dict:
            T0 = predictions_dict[key].get("T0")
        
        try:
            # Compute order params for forecast horizon only
            df = compute_order_params_from_sample(sample, T0=T0)
            
            out_path = out_root / ic_type / "best_order_params.png"
            
            plot_order_params(
                df,
                out_path,
                title=f"Best Run Order Parameters (R²={metrics.r2:.4f})",
                ic_type=ic_type,
                T0=T0,
            )
            
            print(f"  ✓ {out_path}")
            
        except Exception as e:
            print(f"  Warning: Failed to plot order params for {ic_type}/{metrics.name}: {e}")
