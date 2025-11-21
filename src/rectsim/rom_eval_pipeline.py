"""ROM/MVAR evaluation pipeline for unseen IC simulations.

This module orchestrates prediction and evaluation:
1. Load trained ROM/MVAR model
2. Load unseen test simulations
3. Run forecasts and compute metrics
4. Aggregate results by IC type

Author: Maria
Date: November 2025
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np

from rectsim.rom_mvar_model import PODMVARModel
from rectsim.rom_eval_data import SimulationSample, load_unseen_simulations
from rectsim.rom_eval_metrics import (
    SimulationMetrics,
    compute_forecast_metrics,
    compute_relative_errors_timeseries,
)


def predict_single_simulation(
    model: PODMVARModel,
    sample: SimulationSample,
    train_frac: float = 0.8,
    tol: float = 0.1,
    return_predictions: bool = False,
) -> Tuple[SimulationMetrics, Optional[Dict[str, Any]]]:
    """Run ROM/MVAR forecast on a single simulation and compute metrics.
    
    Protocol:
    1. Encode entire trajectory to latent space
    2. Split at T0 = int(train_frac * T)
    3. Use last p latent states as initial history
    4. Forecast forward n_steps = T - T0
    5. Decode predictions back to density
    6. Compute error metrics
    
    Parameters
    ----------
    model : PODMVARModel
        Trained ROM/MVAR model.
    sample : SimulationSample
        Test simulation with ground truth density.
    train_frac : float, default=0.8
        Fraction of trajectory to use for initialization.
    tol : float, default=0.1
        Tolerance for τ computation (relative L² error threshold).
    return_predictions : bool, default=False
        If True, return predictions and error timeseries.
        
    Returns
    -------
    metrics : SimulationMetrics
        Computed forecast metrics.
    predictions : Optional[Dict]
        If return_predictions=True, dict with keys:
        - "density_pred": predicted density, shape (n_forecast, Ny, Nx)
        - "density_true": ground truth tail, shape (n_forecast, Ny, Nx)
        - "latent_pred": predicted latent, shape (n_forecast, d)
        - "latent_true": ground truth latent tail, shape (n_forecast, d)
        - "errors": per-time error arrays from compute_relative_errors_timeseries
        - "times": time points for forecast horizon
        
    Raises
    ------
    ValueError
        If grid shapes don't match or trajectory too short.
    """
    density_true = sample.density_true
    T, Ny, Nx = density_true.shape
    
    # Check grid compatibility
    if model.grid_shape != (Ny, Nx):
        raise ValueError(
            f"Grid shape mismatch: model={model.grid_shape}, "
            f"data={(Ny, Nx)} for {sample.ic_type}/{sample.name}"
        )
    
    # Check trajectory length
    min_T = model.mvar_order + 10  # Need at least p + some forecast steps
    if T < min_T:
        raise ValueError(
            f"Trajectory too short: T={T} < {min_T} for {sample.ic_type}/{sample.name}"
        )
    
    # Step 1: Encode to latent
    latent_true = model.encode(density_true)  # (T, d)
    
    # Step 2: Split at T0
    T0 = int(train_frac * T)
    if T0 < model.mvar_order:
        T0 = model.mvar_order
    
    n_forecast = T - T0
    
    # Step 3: Extract initial history (last p states before T0)
    latent_hist = latent_true[T0 - model.mvar_order : T0]  # (p, d)
    
    # Step 4: Forecast forward
    latent_pred = model.forecast(latent_hist, n_steps=n_forecast)  # (n_forecast, d)
    
    # Step 5: Decode predictions
    density_pred = model.decode(latent_pred)  # (n_forecast, Ny, Nx)
    
    # Ground truth tail for comparison
    density_true_tail = density_true[T0:]  # (n_forecast, Ny, Nx)
    latent_true_tail = latent_true[T0:]  # (n_forecast, d)
    
    # Get time points for forecast horizon
    times_forecast = sample.times[T0:] if len(sample.times) >= T else np.arange(n_forecast)
    
    # Step 6: Compute metrics
    metrics_dict = compute_forecast_metrics(
        density_true_tail,
        density_pred,
        times=times_forecast,
        tol=tol,
        train_frac=train_frac,
    )
    
    metrics = SimulationMetrics(
        ic_type=sample.ic_type,
        name=sample.name,
        **metrics_dict,
    )
    
    # Optionally return predictions
    predictions = None
    if return_predictions:
        errors = compute_relative_errors_timeseries(density_true_tail, density_pred)
        predictions = {
            "density_pred": density_pred,
            "density_true": density_true_tail,
            "latent_pred": latent_pred,
            "latent_true": latent_true_tail,
            "errors": errors,
            "times": times_forecast,
            "T0": T0,
        }
    
    return metrics, predictions


def evaluate_unseen_rom(
    rom_dir: Path,
    unseen_root: Path,
    ic_types: Optional[List[str]] = None,
    train_frac: float = 0.8,
    tol: float = 0.1,
    return_predictions: bool = False,
) -> Tuple[List[SimulationMetrics], Optional[Dict[str, Dict[str, Any]]]]:
    """Evaluate ROM/MVAR on unseen IC simulations.
    
    Parameters
    ----------
    rom_dir : Path
        Directory with ROM model files (pod_basis.npz, mvar_params.npz).
    unseen_root : Path
        Root directory with test simulations organized by IC type.
    ic_types : Optional[List[str]]
        Specific IC types to evaluate. If None, auto-detect all.
    train_frac : float, default=0.8
        Fraction of trajectory for initialization.
    tol : float, default=0.1
        Tolerance for τ computation.
    return_predictions : bool, default=False
        If True, also return predictions for all simulations.
        
    Returns
    -------
    metrics_list : List[SimulationMetrics]
        Metrics for each successfully evaluated simulation.
    predictions_dict : Optional[Dict]
        If return_predictions=True, dict mapping (ic_type, name) to predictions.
        
    Raises
    ------
    FileNotFoundError
        If ROM directory or simulations root not found.
    """
    rom_dir = Path(rom_dir)
    unseen_root = Path(unseen_root)
    
    # Load ROM model
    print(f"Loading ROM model from {rom_dir}...")
    model = PODMVARModel.load(rom_dir)
    print(f"  Latent dim: {model.latent_dim}, MVAR order: {model.mvar_order}")
    print()
    
    # Load simulations
    print(f"Loading test simulations from {unseen_root}...")
    samples = load_unseen_simulations(
        unseen_root,
        ic_types=ic_types,
        require_density=True,
        require_traj=False,
    )
    
    if not samples:
        raise ValueError(f"No simulations found in {unseen_root}")
    
    print(f"  Loaded {len(samples)} simulations")
    print()
    
    # Evaluate each simulation
    metrics_list = []
    predictions_dict = {} if return_predictions else None
    
    n_success = 0
    n_failed = 0
    
    for i, sample in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] {sample.ic_type}/{sample.name}...", end=" ")
        
        try:
            metrics, preds = predict_single_simulation(
                model,
                sample,
                train_frac=train_frac,
                tol=tol,
                return_predictions=return_predictions,
            )
            
            metrics_list.append(metrics)
            if return_predictions:
                predictions_dict[(sample.ic_type, sample.name)] = preds
            
            n_success += 1
            print(f"✓ R²={metrics.r2:.4f}, RMSE={metrics.rmse_mean:.6f}")
            
        except Exception as e:
            n_failed += 1
            warnings.warn(f"Failed: {e}", stacklevel=2)
            print(f"✗ {e}")
    
    print()
    print(f"Completed: {n_success} success, {n_failed} failed")
    print()
    
    return metrics_list, predictions_dict


def aggregate_metrics(
    metrics_list: List[SimulationMetrics],
) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics by IC type and overall.
    
    Parameters
    ----------
    metrics_list : List[SimulationMetrics]
        Metrics from evaluate_unseen_rom.
        
    Returns
    -------
    aggregated : Dict[str, Dict[str, float]]
        Dictionary with structure:
        {
            "overall": {
                "r2_mean": ..., "r2_median": ...,
                "rmse_mean": ..., "rmse_median": ...,
                ...
            },
            "by_ic_type": {
                "ring": {"r2_mean": ..., "rmse_mean": ..., "n": ...},
                "gaussian": {...},
                ...
            }
        }
    """
    if not metrics_list:
        return {"overall": {}, "by_ic_type": {}}
    
    # Extract arrays for overall stats
    r2_all = np.array([m.r2 for m in metrics_list])
    rmse_all = np.array([m.rmse_mean for m in metrics_list])
    e1_all = np.array([m.e1_median for m in metrics_list])
    e2_all = np.array([m.e2_median for m in metrics_list])
    einf_all = np.array([m.einf_median for m in metrics_list])
    mass_err_mean_all = np.array([m.mass_error_mean for m in metrics_list])
    mass_err_max_all = np.array([m.mass_error_max for m in metrics_list])
    
    # Overall aggregates
    overall = {
        "r2_mean": float(r2_all.mean()),
        "r2_median": float(np.median(r2_all)),
        "r2_std": float(r2_all.std()),
        "rmse_mean": float(rmse_all.mean()),
        "rmse_median": float(np.median(rmse_all)),
        "rmse_std": float(rmse_all.std()),
        "e1_mean": float(e1_all.mean()),
        "e1_median": float(np.median(e1_all)),
        "e2_mean": float(e2_all.mean()),
        "e2_median": float(np.median(e2_all)),
        "einf_mean": float(einf_all.mean()),
        "einf_median": float(np.median(einf_all)),
        "mass_error_mean": float(mass_err_mean_all.mean()),
        "mass_error_max_mean": float(mass_err_max_all.mean()),
        "n": len(metrics_list),
    }
    
    # Group by IC type
    by_ic_type = {}
    ic_types = sorted(set(m.ic_type for m in metrics_list))
    
    for ic_type in ic_types:
        subset = [m for m in metrics_list if m.ic_type == ic_type]
        
        r2 = np.array([m.r2 for m in subset])
        rmse = np.array([m.rmse_mean for m in subset])
        e1 = np.array([m.e1_median for m in subset])
        e2 = np.array([m.e2_median for m in subset])
        einf = np.array([m.einf_median for m in subset])
        mass_err_mean = np.array([m.mass_error_mean for m in subset])
        mass_err_max = np.array([m.mass_error_max for m in subset])
        
        by_ic_type[ic_type] = {
            "r2_mean": float(r2.mean()),
            "r2_median": float(np.median(r2)),
            "r2_std": float(r2.std()),
            "rmse_mean": float(rmse.mean()),
            "rmse_median": float(np.median(rmse)),
            "rmse_std": float(rmse.std()),
            "e1_mean": float(e1.mean()),
            "e1_median": float(np.median(e1)),
            "e2_mean": float(e2.mean()),
            "e2_median": float(np.median(e2)),
            "einf_mean": float(einf.mean()),
            "einf_median": float(np.median(einf)),
            "mass_error_mean": float(mass_err_mean.mean()),
            "mass_error_max_mean": float(mass_err_max.mean()),
            "n": len(subset),
        }
    
    return {
        "overall": overall,
        "by_ic_type": by_ic_type,
    }


def print_aggregated_metrics(aggregated: Dict[str, Dict[str, float]]) -> None:
    """Pretty print aggregated metrics.
    
    Parameters
    ----------
    aggregated : Dict
        Output from aggregate_metrics.
    """
    print("=" * 70)
    print("AGGREGATED METRICS")
    print("=" * 70)
    print()
    
    # Overall
    print("Overall (all IC types):")
    overall = aggregated["overall"]
    print(f"  n = {overall['n']}")
    print(f"  R²:        {overall['r2_mean']:.4f} ± {overall['r2_std']:.4f} (median={overall['r2_median']:.4f})")
    print(f"  RMSE:      {overall['rmse_mean']:.6f} ± {overall['rmse_std']:.6f} (median={overall['rmse_median']:.6f})")
    print(f"  L¹ error:  {overall['e1_mean']:.6f} (median={overall['e1_median']:.6f})")
    print(f"  L² error:  {overall['e2_mean']:.6f} (median={overall['e2_median']:.6f})")
    print(f"  L∞ error:  {overall['einf_mean']:.6f} (median={overall['einf_median']:.6f})")
    print(f"  Mass err:  {overall['mass_error_mean']:.6f} (max={overall['mass_error_max_mean']:.6f})")
    print()
    
    # By IC type
    print("By IC type:")
    for ic_type, stats in sorted(aggregated["by_ic_type"].items()):
        print(f"  {ic_type} (n={stats['n']}):")
        print(f"    R²:   {stats['r2_mean']:.4f} ± {stats['r2_std']:.4f}")
        print(f"    RMSE: {stats['rmse_mean']:.6f} ± {stats['rmse_std']:.6f}")
        print(f"    Mass: {stats['mass_error_mean']:.6f}")
    print()
