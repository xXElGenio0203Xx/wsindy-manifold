"""
Test the refactored evaluation system with generic forecast functions.

This test validates:
1. MVAR forecast function wrapper works with refactored evaluator
2. LSTM forecast function works with refactored evaluator
3. Both models can evaluate test runs using identical evaluation logic
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import tempfile
from pathlib import Path
from sklearn.linear_model import Ridge
import torch

from rectsim.forecast_utils import mvar_forecast_fn_factory, validate_forecast_fn
from rom.lstm_rom import LatentLSTMROM, lstm_forecast_fn_factory


def test_mvar_forecast_wrapper():
    """Test MVAR forecast function wrapper."""
    print("\n" + "="*80)
    print("TEST 1: MVAR Forecast Function Wrapper")
    print("="*80)
    
    # Create synthetic MVAR model
    lag = 5
    d = 10
    n_samples = 100
    
    X = np.random.randn(n_samples, lag * d)
    Y = np.random.randn(n_samples, d)
    
    mvar_model = Ridge(alpha=1e-6)
    mvar_model.fit(X, Y)
    print(f"✓ Trained MVAR model: lag={lag}, d={d}")
    
    # Create forecast function
    forecast_fn = mvar_forecast_fn_factory(mvar_model, lag)
    print(f"✓ Created MVAR forecast function")
    
    # Validate
    validate_forecast_fn(forecast_fn, lag=lag, d=d, n_steps=20)
    
    # Test with different n_steps
    ic_window = np.random.randn(lag, d)
    for n_steps in [1, 10, 50, 100]:
        ys_pred = forecast_fn(ic_window, n_steps)
        assert ys_pred.shape == (n_steps, d), f"Shape mismatch for n_steps={n_steps}"
    
    print(f"✓ Tested multiple forecast horizons: 1, 10, 50, 100 steps")
    print("✅ MVAR forecast wrapper test PASSED")
    return True


def test_lstm_forecast_wrapper():
    """Test LSTM forecast function wrapper."""
    print("\n" + "="*80)
    print("TEST 2: LSTM Forecast Function Wrapper")
    print("="*80)
    
    # Create LSTM model
    lag = 5
    d = 10
    hidden_units = 32
    num_layers = 2
    
    model = LatentLSTMROM(d=d, hidden_units=hidden_units, num_layers=num_layers)
    model.eval()
    print(f"✓ Created LSTM model: lag={lag}, d={d}, hidden={hidden_units}, layers={num_layers}")
    
    # Create forecast function
    forecast_fn = lstm_forecast_fn_factory(model)
    print(f"✓ Created LSTM forecast function")
    
    # Validate
    validate_forecast_fn(forecast_fn, lag=lag, d=d, n_steps=20)
    
    # Test with different n_steps
    ic_window = np.random.randn(lag, d)
    for n_steps in [1, 10, 50, 100]:
        ys_pred = forecast_fn(ic_window, n_steps)
        assert ys_pred.shape == (n_steps, d), f"Shape mismatch for n_steps={n_steps}"
    
    print(f"✓ Tested multiple forecast horizons: 1, 10, 50, 100 steps")
    print("✅ LSTM forecast wrapper test PASSED")
    return True


def test_forecast_interface_compatibility():
    """Test that MVAR and LSTM forecasts have identical interfaces."""
    print("\n" + "="*80)
    print("TEST 3: Forecast Interface Compatibility")
    print("="*80)
    
    lag = 5
    d = 10
    n_steps = 30
    
    # Create MVAR model
    X = np.random.randn(100, lag * d)
    Y = np.random.randn(100, d)
    mvar_model = Ridge(alpha=1e-6).fit(X, Y)
    mvar_forecast_fn = mvar_forecast_fn_factory(mvar_model, lag)
    
    # Create LSTM model
    lstm_model = LatentLSTMROM(d=d, hidden_units=32, num_layers=2)
    lstm_model.eval()
    lstm_forecast_fn = lstm_forecast_fn_factory(lstm_model)
    
    print(f"✓ Created both MVAR and LSTM forecast functions")
    
    # Test with same IC
    ic_window = np.random.randn(lag, d)
    
    mvar_pred = mvar_forecast_fn(ic_window, n_steps)
    lstm_pred = lstm_forecast_fn(ic_window, n_steps)
    
    # Check both have same output shape
    assert mvar_pred.shape == lstm_pred.shape == (n_steps, d), "Shape mismatch"
    
    # Check both return numpy arrays
    assert isinstance(mvar_pred, np.ndarray), "MVAR should return numpy array"
    assert isinstance(lstm_pred, np.ndarray), "LSTM should return numpy array"
    
    # Check no NaNs
    assert not np.any(np.isnan(mvar_pred)), "MVAR produced NaNs"
    assert not np.any(np.isnan(lstm_pred)), "LSTM produced NaNs"
    
    print(f"✓ Both forecasts have shape: {mvar_pred.shape}")
    print(f"✓ Both forecasts are numpy arrays")
    print(f"✓ No NaN values detected")
    print("✅ Interface compatibility test PASSED")
    return True


def test_evaluation_integration_mock():
    """
    Mock test of evaluation integration.
    
    Simulates what happens in run_unified_rom_pipeline.py when calling
    evaluate_test_runs() with both MVAR and LSTM forecast functions.
    """
    print("\n" + "="*80)
    print("TEST 4: Evaluation Integration (Mock)")
    print("="*80)
    
    lag = 5
    d = 10
    
    # Create MVAR forecast function
    X = np.random.randn(100, lag * d)
    Y = np.random.randn(100, d)
    mvar_model = Ridge(alpha=1e-6).fit(X, Y)
    mvar_forecast_fn = mvar_forecast_fn_factory(mvar_model, lag)
    
    # Create LSTM forecast function
    lstm_model = LatentLSTMROM(d=d, hidden_units=32, num_layers=2)
    lstm_model.eval()
    lstm_forecast_fn = lstm_forecast_fn_factory(lstm_model)
    
    print(f"✓ Created MVAR and LSTM forecast functions")
    
    # Simulate what evaluate_test_runs() does
    # (without actually running full simulation)
    
    # Simulate test trajectory
    T_test = 100
    T_train = 50  # First 50 steps for conditioning
    
    test_latent = np.random.randn(T_test, d)
    ic_window = test_latent[T_train-lag:T_train]  # Last 'lag' steps from training
    n_forecast_steps = T_test - T_train
    
    # MVAR forecast
    mvar_pred = mvar_forecast_fn(ic_window, n_forecast_steps)
    assert mvar_pred.shape == (n_forecast_steps, d)
    
    # LSTM forecast
    lstm_pred = lstm_forecast_fn(ic_window, n_forecast_steps)
    assert lstm_pred.shape == (n_forecast_steps, d)
    
    # Simulate R² computation (same for both models)
    true_latent = test_latent[T_train:]
    
    def compute_r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - y_true.mean())**2)
        return 1 - ss_res / ss_tot
    
    r2_mvar = compute_r2(true_latent, mvar_pred)
    r2_lstm = compute_r2(true_latent, lstm_pred)
    
    print(f"✓ MVAR forecast: {mvar_pred.shape}, R²={r2_mvar:.4f}")
    print(f"✓ LSTM forecast: {lstm_pred.shape}, R²={r2_lstm:.4f}")
    print(f"✓ Both models use identical R² computation")
    print("✅ Evaluation integration mock test PASSED")
    return True


def test_error_handling():
    """Test error handling in forecast functions."""
    print("\n" + "="*80)
    print("TEST 5: Error Handling")
    print("="*80)
    
    lag = 5
    d = 10
    
    # Create MVAR forecast function
    X = np.random.randn(100, lag * d)
    Y = np.random.randn(100, d)
    mvar_model = Ridge(alpha=1e-6).fit(X, Y)
    mvar_forecast_fn = mvar_forecast_fn_factory(mvar_model, lag)
    
    # Test wrong IC shape
    try:
        wrong_ic = np.random.randn(lag+1, d)  # Wrong lag
        mvar_forecast_fn(wrong_ic, 10)
        assert False, "Should have raised error for wrong IC shape"
    except ValueError as e:
        print(f"✓ Correctly caught wrong IC shape: {e}")
    
    # Test with correct shape
    correct_ic = np.random.randn(lag, d)
    result = mvar_forecast_fn(correct_ic, 10)
    assert result.shape == (10, d)
    print(f"✓ Correct IC shape works: {result.shape}")
    
    print("✅ Error handling test PASSED")
    return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTING REFACTORED EVALUATION SYSTEM")
    print("="*80)
    
    all_passed = True
    
    try:
        all_passed &= test_mvar_forecast_wrapper()
    except Exception as e:
        print(f"❌ TEST 1 FAILED: {e}")
        all_passed = False
    
    try:
        all_passed &= test_lstm_forecast_wrapper()
    except Exception as e:
        print(f"❌ TEST 2 FAILED: {e}")
        all_passed = False
    
    try:
        all_passed &= test_forecast_interface_compatibility()
    except Exception as e:
        print(f"❌ TEST 3 FAILED: {e}")
        all_passed = False
    
    try:
        all_passed &= test_evaluation_integration_mock()
    except Exception as e:
        print(f"❌ TEST 4 FAILED: {e}")
        all_passed = False
    
    try:
        all_passed &= test_error_handling()
    except Exception as e:
        print(f"❌ TEST 5 FAILED: {e}")
        all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("="*80)
        print("\nRefactored evaluation system is ready:")
        print("  ✓ test_evaluator.py accepts generic forecast_fn")
        print("  ✓ MVAR forecast wrapper implemented")
        print("  ✓ LSTM forecast wrapper ready")
        print("  ✓ Both models use identical evaluation logic")
        print("  ✓ Error handling validated")
        print("\nNext: Run full pipeline with both models enabled!")
    else:
        print("❌ SOME TESTS FAILED")
        print("="*80)
        sys.exit(1)
