"""Unit tests for ROM/MVAR evaluation visualization utilities.

Tests best run selection and plotting utilities without requiring real data.

Author: Maria
Date: November 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from rectsim.rom_eval_metrics import SimulationMetrics
from rectsim.rom_eval_data import SimulationSample
from rectsim.rom_eval_viz import (
    select_best_runs,
    compute_order_params_from_sample,
)


def test_select_best_runs():
    """Test best run selection."""
    print("Testing select_best_runs...")
    
    # Create synthetic metrics
    metrics = [
        SimulationMetrics(
            ic_type="ring",
            name="sim_000",
            r2=0.95,
            rmse_mean=0.05,
            e1_median=0.03,
            e2_median=0.04,
            einf_median=0.10,
            mass_error_mean=0.01,
            mass_error_max=0.02,
            tau=None,
            n_forecast=100,
            train_frac=0.8,
        ),
        SimulationMetrics(
            ic_type="ring",
            name="sim_001",
            r2=0.90,  # Worse
            rmse_mean=0.08,
            e1_median=0.05,
            e2_median=0.07,
            einf_median=0.15,
            mass_error_mean=0.02,
            mass_error_max=0.03,
            tau=5.0,
            n_forecast=100,
            train_frac=0.8,
        ),
        SimulationMetrics(
            ic_type="gaussian",
            name="sim_000",
            r2=0.92,
            rmse_mean=0.06,
            e1_median=0.04,
            e2_median=0.05,
            einf_median=0.12,
            mass_error_mean=0.015,
            mass_error_max=0.025,
            tau=None,
            n_forecast=100,
            train_frac=0.8,
        ),
    ]
    
    # Select best by R² (maximize)
    best = select_best_runs(metrics, key="r2", maximize=True)
    
    assert len(best) == 2, f"Expected 2 IC types, got {len(best)}"
    assert "ring" in best and "gaussian" in best
    assert best["ring"].name == "sim_000", "Should select sim_000 (R²=0.95)"
    assert best["ring"].r2 == 0.95
    assert best["gaussian"].name == "sim_000"
    
    print(f"  ✓ Selected best runs: ring={best['ring'].name}, gaussian={best['gaussian'].name}")
    
    # Select best by RMSE (minimize)
    best = select_best_runs(metrics, key="rmse_mean", maximize=False)
    
    assert best["ring"].name == "sim_000", "Should select sim_000 (RMSE=0.05)"
    assert best["ring"].rmse_mean == 0.05
    
    print(f"  ✓ Minimize RMSE works correctly")
    print()


def test_compute_order_params():
    """Test order parameter computation."""
    print("Testing compute_order_params_from_sample...")
    
    T, N = 50, 100
    np.random.seed(42)
    
    # Create synthetic trajectories
    x = np.random.randn(T, N, 2)
    
    # Create velocities with some alignment
    v_base = np.array([1.0, 0.5])  # Common direction
    v = v_base + 0.2 * np.random.randn(T, N, 2)
    
    times = np.linspace(0, 10, T)
    
    # Create sample
    sample = SimulationSample(
        ic_type="test",
        name="sim_000",
        density_true=np.random.rand(T, 16, 16),
        traj_true={"x": x, "v": v, "times": times},
        meta={},
        path=Path("/tmp/test"),
    )
    
    # Compute order params
    df = compute_order_params_from_sample(sample)
    
    assert len(df) == T, f"Expected {T} rows, got {len(df)}"
    assert "time" in df.columns
    assert "polarization" in df.columns
    assert "speed_mean" in df.columns
    assert "speed_std" in df.columns
    
    # Check polarization is in [0, 1]
    assert df["polarization"].min() >= 0
    assert df["polarization"].max() <= 1
    
    # Check speeds are positive
    assert df["speed_mean"].min() > 0
    assert df["speed_std"].min() >= 0
    
    print(f"  ✓ Computed order params: polarization={df['polarization'].mean():.3f}")
    print(f"  ✓ Speed: {df['speed_mean'].mean():.3f} ± {df['speed_std'].mean():.3f}")
    print()
    
    # Test with T0 (forecast horizon only)
    T0 = 40
    df_forecast = compute_order_params_from_sample(sample, T0=T0)
    
    expected_len = T - T0
    assert len(df_forecast) == expected_len, f"Expected {expected_len} rows, got {len(df_forecast)}"
    
    print(f"  ✓ T0={T0} works: {len(df_forecast)} forecast steps")
    print()


def test_plotting_functions():
    """Test plotting functions (just check they run without errors)."""
    print("Testing plotting functions...")
    
    from rectsim.rom_eval_viz import plot_error_time_series, plot_order_params
    
    T = 100
    times = np.linspace(0, 10, T)
    
    # Create synthetic errors
    errors = {
        "e1": np.linspace(0.01, 0.05, T),
        "e2": np.linspace(0.01, 0.05, T),
        "einf": np.linspace(0.02, 0.10, T),
        "rel_e2": np.linspace(0.01, 0.15, T),
        "mass_error": 0.01 * np.random.rand(T),
    }
    
    # Create temp directory
    tmpdir = Path(tempfile.mkdtemp())
    
    try:
        # Test error plot
        out_path = tmpdir / "test_error.png"
        plot_error_time_series(
            times,
            errors,
            out_path,
            title="Test Error Plot",
            ic_type="test",
        )
        
        assert out_path.exists(), "Error plot not created"
        print(f"  ✓ Error plot created: {out_path.stat().st_size} bytes")
        
        # Test order param plot
        df = pd.DataFrame({
            "time": times,
            "polarization": 0.5 + 0.2 * np.sin(2 * np.pi * times / 10),
            "speed_mean": 1.0 + 0.1 * np.random.randn(T),
            "speed_std": 0.2 + 0.05 * np.random.randn(T),
        })
        
        out_path = tmpdir / "test_order.png"
        plot_order_params(
            df,
            out_path,
            title="Test Order Params",
            ic_type="test",
            T0=80,
        )
        
        assert out_path.exists(), "Order param plot not created"
        print(f"  ✓ Order param plot created: {out_path.stat().st_size} bytes")
        
    finally:
        # Clean up
        shutil.rmtree(tmpdir)
    
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("ROM/MVAR Visualization Unit Tests")
    print("=" * 70)
    print()
    
    test_select_best_runs()
    test_compute_order_params()
    test_plotting_functions()
    
    print("=" * 70)
    print("✓ All visualization tests passed!")
    print("=" * 70)
