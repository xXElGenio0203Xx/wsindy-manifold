"""
Forecast function utilities for ROM evaluation.

This module provides wrappers to convert model-specific forecasting logic
into a generic forecast_fn interface that can be used by evaluate_test_runs().
"""

import numpy as np


def mvar_forecast_fn_factory(mvar_model, lag):
    """
    Create a forecast function closure for MVAR model.
    
    Parameters
    ----------
    mvar_model : sklearn.linear_model.Ridge
        Trained MVAR model with .predict() method
    lag : int
        Lookback window size
    
    Returns
    -------
    forecast_fn : callable
        Function with signature: forecast_fn(y_init_window, n_steps) -> ys_pred
        - y_init_window: [lag, d] initial condition window
        - n_steps: number of steps to forecast
        - ys_pred: [n_steps, d] predictions in latent space
    
    Example
    -------
    >>> mvar_model = Ridge(alpha=1e-6).fit(X, Y)
    >>> forecast_fn = mvar_forecast_fn_factory(mvar_model, lag=5)
    >>> predictions = forecast_fn(ic_window, n_steps=100)
    """
    def forecast_fn(y_init_window, n_steps):
        """
        MVAR closed-loop forecast.
        
        Parameters
        ----------
        y_init_window : np.ndarray
            Initial condition window [lag, d]
        n_steps : int
            Number of steps to forecast
        
        Returns
        -------
        ys_pred : np.ndarray
            Predictions [n_steps, d] in latent space
        """
        # Validate input shape
        if y_init_window.shape[0] != lag:
            raise ValueError(
                f"Expected IC window with {lag} timesteps, got {y_init_window.shape[0]}"
            )
        
        d = y_init_window.shape[1]  # Latent dimension
        
        # Autoregressive prediction
        ys_pred = []
        current_history = y_init_window.copy()
        
        for _ in range(n_steps):
            # Prepare feature vector (flatten history)
            x_hist = current_history[-lag:].flatten()
            
            # Predict next step
            y_next = mvar_model.predict(x_hist.reshape(1, -1))[0]
            ys_pred.append(y_next)
            
            # Update history (sliding window)
            current_history = np.vstack([current_history[1:], y_next])
        
        return np.array(ys_pred)  # Shape: [n_steps, d]
    
    return forecast_fn


def validate_forecast_fn(forecast_fn, lag=5, d=10, n_steps=20):
    """
    Validate that a forecast function has the correct interface.
    
    Parameters
    ----------
    forecast_fn : callable
        Forecast function to validate
    lag : int, optional
        Lag to use for test (default: 5)
    d : int, optional
        Latent dimension for test (default: 10)
    n_steps : int, optional
        Number of steps to forecast in test (default: 20)
    
    Returns
    -------
    bool
        True if validation passes
    
    Raises
    ------
    AssertionError
        If validation fails with detailed error message
    """
    # Create test IC window
    y_init = np.random.randn(lag, d)
    
    # Test forecast
    try:
        ys_pred = forecast_fn(y_init, n_steps)
    except Exception as e:
        raise AssertionError(f"Forecast function raised exception: {e}")
    
    # Check output shape
    expected_shape = (n_steps, d)
    assert ys_pred.shape == expected_shape, (
        f"Expected output shape {expected_shape}, got {ys_pred.shape}"
    )
    
    # Check output type
    assert isinstance(ys_pred, np.ndarray), (
        f"Expected np.ndarray output, got {type(ys_pred)}"
    )
    
    # Check for NaNs or Infs
    assert not np.any(np.isnan(ys_pred)), "Output contains NaN values"
    assert not np.any(np.isinf(ys_pred)), "Output contains Inf values"
    
    print(f"✓ Forecast function validation passed:")
    print(f"  Input: ({lag}, {d}) → Output: ({n_steps}, {d})")
    print(f"  No NaN or Inf values detected")
    
    return True


if __name__ == "__main__":
    """
    Test MVAR forecast function factory.
    """
    from sklearn.linear_model import Ridge
    
    print("Testing MVAR forecast function factory...")
    print("="*80)
    
    # Create synthetic training data
    lag = 5
    d = 10
    n_samples = 100
    
    X = np.random.randn(n_samples, lag * d)
    Y = np.random.randn(n_samples, d)
    
    # Train MVAR model
    mvar_model = Ridge(alpha=1e-6)
    mvar_model.fit(X, Y)
    print(f"✓ Trained MVAR model: {n_samples} samples, lag={lag}, d={d}")
    
    # Create forecast function
    forecast_fn = mvar_forecast_fn_factory(mvar_model, lag)
    print(f"✓ Created forecast function closure")
    
    # Validate
    validate_forecast_fn(forecast_fn, lag=lag, d=d, n_steps=20)
    
    # Test actual forecast
    y_init = np.random.randn(lag, d)
    ys_pred = forecast_fn(y_init, n_steps=50)
    print(f"✓ Generated forecast: {ys_pred.shape}")
    
    print("\n" + "="*80)
    print("All tests passed!")
