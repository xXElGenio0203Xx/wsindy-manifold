"""Quick unit test for ROM/MVAR evaluation pipeline components.

Tests basic functionality without requiring actual model files or simulations.

Author: Maria
Date: November 2025
"""

import numpy as np
import tempfile
from pathlib import Path

from rectsim.rom_eval_metrics import (
    compute_forecast_metrics,
    compute_relative_errors_timeseries,
)


def test_compute_forecast_metrics():
    """Test metrics computation with synthetic data."""
    print("Testing compute_forecast_metrics...")
    
    # Create synthetic data
    T, Ny, Nx = 100, 32, 32
    np.random.seed(42)
    
    density_true = np.random.rand(T, Ny, Nx) + 1.0
    density_pred = density_true + 0.1 * np.random.randn(T, Ny, Nx)
    
    times = np.linspace(0, 10, T)
    
    # Compute metrics
    metrics = compute_forecast_metrics(
        density_true,
        density_pred,
        times=times,
        tol=0.1,
        train_frac=0.8,
    )
    
    # Check all expected keys present
    expected_keys = {
        "r2", "rmse_mean", "e1_median", "e2_median", "einf_median",
        "mass_error_mean", "mass_error_max", "tau", "n_forecast", "train_frac"
    }
    assert set(metrics.keys()) == expected_keys, f"Missing keys: {expected_keys - set(metrics.keys())}"
    
    # Check reasonable values
    assert 0.0 <= metrics["r2"] <= 1.0, f"R² out of range: {metrics['r2']}"
    assert metrics["rmse_mean"] > 0, "RMSE should be positive"
    assert metrics["n_forecast"] == T, f"Expected n_forecast={T}, got {metrics['n_forecast']}"
    
    print(f"  ✓ R² = {metrics['r2']:.4f}")
    print(f"  ✓ RMSE = {metrics['rmse_mean']:.6f}")
    print(f"  ✓ All keys present")
    print()


def test_compute_relative_errors_timeseries():
    """Test timeseries error computation."""
    print("Testing compute_relative_errors_timeseries...")
    
    T, Ny, Nx = 50, 16, 16
    np.random.seed(42)
    
    density_true = np.random.rand(T, Ny, Nx) + 1.0
    density_pred = density_true + 0.05 * np.random.randn(T, Ny, Nx)
    
    errors = compute_relative_errors_timeseries(density_true, density_pred)
    
    # Check keys
    expected_keys = {"e1", "e2", "einf", "rel_e2", "mass_error"}
    assert set(errors.keys()) == expected_keys
    
    # Check shapes
    for key in expected_keys:
        assert errors[key].shape == (T,), f"{key} has wrong shape: {errors[key].shape}"
    
    # Check all positive
    for key in expected_keys:
        assert np.all(errors[key] >= 0), f"{key} has negative values"
    
    print(f"  ✓ All error arrays shape = ({T},)")
    print(f"  ✓ All errors non-negative")
    print()


def test_perfect_reconstruction():
    """Test that perfect reconstruction gives zero error."""
    print("Testing perfect reconstruction...")
    
    T, Ny, Nx = 20, 16, 16
    np.random.seed(42)
    
    density = np.random.rand(T, Ny, Nx) + 1.0
    
    metrics = compute_forecast_metrics(
        density,
        density,  # Perfect reconstruction
        tol=0.1,
    )
    
    assert metrics["r2"] > 0.999, f"R² should be ~1.0 for perfect match, got {metrics['r2']}"
    assert metrics["rmse_mean"] < 1e-10, f"RMSE should be ~0 for perfect match, got {metrics['rmse_mean']}"
    assert metrics["mass_error_mean"] < 1e-10, "Mass error should be ~0"
    assert metrics["tau"] is None, "tau should be None (error never exceeds threshold)"
    
    print(f"  ✓ R² = {metrics['r2']:.8f} (near 1.0)")
    print(f"  ✓ RMSE = {metrics['rmse_mean']:.2e} (near 0)")
    print(f"  ✓ tau = {metrics['tau']} (no threshold crossing)")
    print()


def test_mass_conservation():
    """Test mass error computation."""
    print("Testing mass conservation...")
    
    T, Ny, Nx = 30, 16, 16
    np.random.seed(42)
    
    density_true = np.random.rand(T, Ny, Nx) + 1.0
    
    # Create prediction with 5% mass error
    density_pred = density_true * 1.05
    
    metrics = compute_forecast_metrics(density_true, density_pred)
    
    # Mass error should be ~5%
    assert 0.04 < metrics["mass_error_mean"] < 0.06, \
        f"Expected ~5% mass error, got {metrics['mass_error_mean']:.4f}"
    
    print(f"  ✓ Mass error = {metrics['mass_error_mean']:.4f} (expected ~0.05)")
    print()


def test_tau_detection():
    """Test tau (threshold crossing) detection."""
    print("Testing tau detection...")
    
    T, Ny, Nx = 100, 16, 16
    times = np.linspace(0, 10, T)
    
    # Create data where error grows linearly
    density_true = np.ones((T, Ny, Nx))
    
    # Error grows from 0% to 50%
    error_levels = np.linspace(0, 0.5, T)
    density_pred = density_true * (1 + error_levels[:, None, None])
    
    metrics = compute_forecast_metrics(
        density_true,
        density_pred,
        times=times,
        tol=0.1,  # 10% threshold
    )
    
    # Should cross threshold around 20% of the way through
    assert metrics["tau"] is not None, "tau should be detected"
    assert 1.0 < metrics["tau"] < 3.0, f"tau should be ~2.0, got {metrics['tau']}"
    
    print(f"  ✓ tau = {metrics['tau']:.2f} (detected threshold crossing)")
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("ROM/MVAR Evaluation Metrics Unit Tests")
    print("=" * 70)
    print()
    
    test_compute_forecast_metrics()
    test_compute_relative_errors_timeseries()
    test_perfect_reconstruction()
    test_mass_conservation()
    test_tau_detection()
    
    print("=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
