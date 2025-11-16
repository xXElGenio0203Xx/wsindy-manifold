"""Test enhanced MVAR implementation with lag selection and forecasting."""

import numpy as np
import pytest

from wsindy_manifold.latent.mvar import (
    MVARModel,
    fit_mvar_auto,
    horizon_test,
    plot_lag_selection,
    plot_forecast_trajectories,
    plot_horizon_test,
)


def generate_synthetic_latent(T=500, d=3, noise_level=0.1, seed=42):
    """Generate synthetic latent trajectory with oscillatory components."""
    np.random.seed(seed)
    t = np.linspace(0, 10, T)
    
    # Create oscillatory latent dynamics
    Y = np.zeros((T, d))
    Y[:, 0] = np.sin(2 * np.pi * 0.5 * t) + noise_level * np.random.randn(T)
    Y[:, 1] = np.cos(2 * np.pi * 0.5 * t) + noise_level * np.random.randn(T)
    if d > 2:
        Y[:, 2] = 0.5 * np.sin(2 * np.pi * 0.3 * t) + noise_level * np.random.randn(T)
    
    return Y


def test_mvar_model_basic():
    """Test basic MVARModel fitting and forecasting."""
    Y = generate_synthetic_latent(T=300, d=3)
    
    model = MVARModel(max_lag=3, criterion="AIC")
    model.fit(Y)
    
    # Check model was fitted
    assert model.best_lag is not None
    assert 1 <= model.best_lag <= 3
    assert model.A0 is not None
    assert model.A_list is not None
    assert len(model.A_list) == model.best_lag
    assert model.cov is not None
    
    # Check dimensions
    assert model.A0.shape == (3,)
    for A_j in model.A_list:
        assert A_j.shape == (3, 3)
    assert model.cov.shape == (3, 3)
    
    print(f"✓ Model fitted: lag={model.best_lag}, RMSE={model.train_rmse:.4f}")


def test_mvar_forecasting():
    """Test MVAR forecasting functionality."""
    Y = generate_synthetic_latent(T=300, d=3)
    
    model = MVARModel(max_lag=3, train_fraction=0.8)
    model.fit(Y)
    
    # Forecast
    T0 = int(0.8 * 300)
    Y_init = Y[:T0][-model.best_lag:]
    Y_forecast = model.forecast(Y_init, steps=50, add_noise=False)
    
    # Check output shape
    assert Y_forecast.shape == (50, 3)
    
    # Evaluate
    Y_true = Y[T0:T0+50]
    metrics = model.evaluate_forecast(Y_true, Y_forecast)
    
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "corr_mean" in metrics
    assert "spectral_ratio" in metrics
    
    print(f"✓ Forecast RMSE: {metrics['rmse']:.4f}")
    print(f"✓ Forecast Corr: {metrics['corr_mean']:.4f}")


def test_mvar_lag_selection():
    """Test automatic lag selection with different criteria."""
    Y = generate_synthetic_latent(T=300, d=3)
    
    for criterion in ["AIC", "BIC", None]:
        model = MVARModel(max_lag=4, criterion=criterion)
        model.fit(Y)
        
        assert model.best_lag is not None
        assert len(model.lag_scores) > 0
        
        print(f"✓ {criterion or 'LL'}: Selected lag = {model.best_lag}")


def test_fit_mvar_auto():
    """Test convenience function for automatic fitting."""
    Y = generate_synthetic_latent(T=300, d=3)
    
    model = fit_mvar_auto(Y, max_lag=3, criterion="BIC")
    
    assert model.best_lag is not None
    assert model.A0 is not None
    
    print(f"✓ fit_mvar_auto: lag={model.best_lag}")


def test_horizon_test_function():
    """Test horizon testing functionality."""
    Y = generate_synthetic_latent(T=800, d=3, noise_level=0.05)  # Longer trajectory
    
    model, results = horizon_test(
        Y,
        max_lag=3,
        horizon_ratios=[0.3, 0.5],  # Smaller ratios to fit in data
        n_trials=2,
        train_fraction=0.6  # Use less for training to have room for forecasting
    )
    
    # Check model was fitted
    assert model.best_lag is not None
    
    # Check results structure
    assert len(results) > 0
    for ratio, metrics in results.items():
        assert "rmse" in metrics
        assert "rmse_std" in metrics
        assert "corr" in metrics
        assert "corr_std" in metrics
        
        print(f"✓ Ratio {ratio}: RMSE={metrics['rmse']:.4f}, Corr={metrics['corr']:.4f}")


def test_noise_sampling():
    """Test different noise types."""
    Y = generate_synthetic_latent(T=300, d=3)
    
    for noise_type in ["gaussian", "uniform"]:
        model = MVARModel(max_lag=2, noise_type=noise_type)
        model.fit(Y)
        
        Y_init = Y[-model.best_lag:]
        Y_forecast = model.forecast(Y_init, steps=10, add_noise=True)
        
        assert Y_forecast.shape == (10, 3)
        print(f"✓ Noise type '{noise_type}' works")


def test_model_to_dict():
    """Test conversion to dictionary format (API compatibility)."""
    Y = generate_synthetic_latent(T=300, d=3)
    
    model = MVARModel(max_lag=3)
    model.fit(Y)
    
    model_dict = model.to_dict()
    
    # Check dictionary structure
    assert "A0" in model_dict
    assert "A" in model_dict
    assert "w" in model_dict
    assert "ridge_lambda" in model_dict
    assert "cov" in model_dict
    
    # Check shapes
    assert model_dict["A"].shape == (model.best_lag, 3, 3)
    assert model_dict["w"] == model.best_lag
    
    print(f"✓ Model dict: w={model_dict['w']}, lambda={model_dict['ridge_lambda']}")


def test_ensemble_fitting():
    """Test fitting on ensemble data (C, T, d)."""
    C, T, d = 3, 200, 3
    Y_ensemble = np.random.randn(C, T, d)
    
    model = MVARModel(max_lag=2)
    model.fit(Y_ensemble)
    
    assert model.best_lag is not None
    assert model.A0.shape == (d,)
    
    print(f"✓ Ensemble fitting: {C} cases, lag={model.best_lag}")


def test_plotting_functions():
    """Test all plotting functions."""
    Y = generate_synthetic_latent(T=600, d=3)  # Longer for horizon test
    
    # Fit model
    model = MVARModel(max_lag=3, criterion="AIC")
    model.fit(Y)
    
    # Test lag selection plot
    fig = plot_lag_selection(model)
    assert fig is not None
    print("✓ Lag selection plot created")
    
    # Test forecast trajectories plot
    T0 = int(0.8 * 600)
    Y_init = Y[:T0][-model.best_lag:]
    Y_forecast = model.forecast(Y_init, steps=100, add_noise=False)
    Y_true = Y[:T0]
    
    fig = plot_forecast_trajectories(Y_true, Y_forecast, dims_to_plot=[0, 1])
    assert fig is not None
    print("✓ Forecast trajectories plot created")
    
    # Test horizon test plot
    _, results = horizon_test(Y, max_lag=3, horizon_ratios=[0.2, 0.4], n_trials=2, train_fraction=0.6)
    fig = plot_horizon_test(results)
    assert fig is not None
    print("✓ Horizon test plot created")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Testing Enhanced MVAR Implementation")
    print("="*70 + "\n")
    
    test_mvar_model_basic()
    print()
    
    test_mvar_forecasting()
    print()
    
    test_mvar_lag_selection()
    print()
    
    test_fit_mvar_auto()
    print()
    
    test_horizon_test_function()
    print()
    
    test_noise_sampling()
    print()
    
    test_model_to_dict()
    print()
    
    test_ensemble_fitting()
    print()
    
    test_plotting_functions()
    print()
    
    print("="*70)
    print("✓ All tests passed!")
    print("="*70)
