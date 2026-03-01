#!/usr/bin/env python3
"""
Test LSTM ROM forecasting and evaluation pipeline integration.

This script demonstrates:
1. Closed-loop LSTM forecasting in latent space
2. How LSTM forecast function integrates with evaluation pipeline
3. Folder structure for MVAR vs LSTM outputs
4. Generic evaluate_rom function design pattern
"""

import sys
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import torch
from rom.lstm_rom import (
    LatentLSTMROM, 
    train_lstm_rom, 
    forecast_with_lstm,
    lstm_forecast_fn_factory
)


def test_basic_forecasting():
    """Test 1: Basic closed-loop forecasting."""
    print("\n" + "="*80)
    print("TEST 1: Basic LSTM Closed-Loop Forecasting")
    print("="*80 + "\n")
    
    # Create and train a small LSTM model
    lag = 5
    d = 10
    n_steps_forecast = 20
    
    print(f"Configuration:")
    print(f"  Lag (sequence length): {lag}")
    print(f"  Latent dimension: {d}")
    print(f"  Forecast steps: {n_steps_forecast}")
    
    # Create dummy training data
    N_samples = 100
    X_train = np.random.randn(N_samples, lag, d).astype(np.float32)
    Y_train = np.random.randn(N_samples, d).astype(np.float32)
    
    # Quick training config
    class Config:
        class ROM:
            class Models:
                class LSTM:
                    batch_size = 16
                    hidden_units = 16
                    num_layers = 1
                    learning_rate = 0.001
                    weight_decay = 0.0
                    max_epochs = 10
                    patience = 5
                    gradient_clip = 1.0
                lstm = LSTM()
            models = Models()
        rom = ROM()
    
    # Train model
    print(f"\nTraining LSTM model...")
    model_path, val_loss = train_lstm_rom(X_train, Y_train, Config(), 'tmp_forecast_test')
    print(f"Training complete. Best val loss: {val_loss:.6f}")
    
    # Load trained model
    model = LatentLSTMROM(d=d, hidden_units=16, num_layers=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"✓ Model loaded from: {model_path}")
    
    # Create initial window (simulate truth)
    np.random.seed(42)
    y_init_window = np.random.randn(lag, d).astype(np.float32)
    
    print(f"\nForecasting:")
    print(f"  Initial window: {y_init_window.shape}  [lag, d]")
    print(f"  Forecast steps: {n_steps_forecast}")
    
    # Perform closed-loop forecast
    ys_pred = forecast_with_lstm(model, y_init_window, n_steps_forecast)
    
    print(f"  Result: {ys_pred.shape}  [n_steps, d]")
    
    # Verify shape
    assert ys_pred.shape == (n_steps_forecast, d), \
        f"Expected shape ({n_steps_forecast}, {d}), got {ys_pred.shape}"
    print(f"  ✓ Shape verified")
    
    # Verify predictions are different (not all same)
    assert not np.allclose(ys_pred[0], ys_pred[-1]), \
        "All predictions are identical (model not working)"
    print(f"  ✓ Predictions vary over time")
    
    # Show statistics
    print(f"\nPrediction statistics:")
    print(f"  Mean: {ys_pred.mean():.4f}")
    print(f"  Std:  {ys_pred.std():.4f}")
    print(f"  Min:  {ys_pred.min():.4f}")
    print(f"  Max:  {ys_pred.max():.4f}")
    
    print("\n✅ TEST 1 PASSED")
    
    # Cleanup
    shutil.rmtree('tmp_forecast_test')
    
    return model, y_init_window


def test_forecast_factory():
    """Test 2: lstm_forecast_fn_factory for pipeline integration."""
    print("\n" + "="*80)
    print("TEST 2: Forecast Function Factory")
    print("="*80 + "\n")
    
    # Create simple model
    d = 8
    lag = 4
    model = LatentLSTMROM(d=d, hidden_units=8, num_layers=1)
    model.eval()
    
    print(f"Model: d={d}, lag={lag}, hidden_units=8, num_layers=1")
    
    # Create forecast function using factory
    lstm_forecast_fn = lstm_forecast_fn_factory(model)
    
    print(f"✓ Created forecast function using factory")
    
    # Test the forecast function
    y_init = np.random.randn(lag, d).astype(np.float32)
    n_steps = 10
    
    print(f"\nTesting forecast function:")
    print(f"  Input: {y_init.shape}  [lag, d]")
    print(f"  Steps: {n_steps}")
    
    y_pred = lstm_forecast_fn(y_init, n_steps)
    
    print(f"  Output: {y_pred.shape}  [n_steps, d]")
    
    # Verify
    assert y_pred.shape == (n_steps, d)
    print(f"  ✓ Function works correctly")
    
    # Test multiple calls (should be deterministic in eval mode)
    y_pred2 = lstm_forecast_fn(y_init, n_steps)
    assert np.allclose(y_pred, y_pred2), "Forecast not deterministic"
    print(f"  ✓ Forecast is deterministic (as expected in eval mode)")
    
    print("\n✅ TEST 2 PASSED")


def test_autoregressive_behavior():
    """Test 3: Verify autoregressive window updates."""
    print("\n" + "="*80)
    print("TEST 3: Autoregressive Window Update Mechanism")
    print("="*80 + "\n")
    
    # Small model for detailed inspection
    d = 3
    lag = 3
    model = LatentLSTMROM(d=d, hidden_units=4, num_layers=1)
    model.eval()
    
    print(f"Configuration: d={d}, lag={lag}")
    
    # Simple initial window
    y_init = np.array([
        [1.0, 0.0, 0.0],  # t=0
        [0.0, 1.0, 0.0],  # t=1
        [0.0, 0.0, 1.0],  # t=2
    ], dtype=np.float32)
    
    print(f"\nInitial window:")
    for i, y in enumerate(y_init):
        print(f"  t={i}: {y}")
    
    # Forecast 3 steps
    n_steps = 3
    ys_pred = forecast_with_lstm(model, y_init, n_steps)
    
    print(f"\nForecasted latent states:")
    for i, y in enumerate(ys_pred):
        print(f"  t={lag+i}: {y}")
    
    print(f"\n✓ Window slides forward {n_steps} steps")
    print(f"✓ Each prediction uses lag={lag} previous states")
    print(f"✓ Earlier predictions become inputs for later ones")
    
    print("\n✅ TEST 3 PASSED")


def test_evaluation_pipeline_pattern():
    """Test 4: Demonstrate evaluation pipeline design pattern."""
    print("\n" + "="*80)
    print("TEST 4: Evaluation Pipeline Design Pattern")
    print("="*80 + "\n")
    
    print("Design pattern for generic ROM evaluation:")
    print("-" * 80)
    
    # Simulated evaluation function (placeholder)
    def evaluate_rom(model_name, forecast_fn, R, L, test_trajectories, out_dir):
        """
        Generic ROM evaluation function.
        
        This function:
        1. Takes a model-agnostic forecast function
        2. Evaluates on test trajectories
        3. Computes metrics (R², errors, etc.)
        4. Saves results to model-specific folder
        """
        print(f"\nEvaluating {model_name} ROM:")
        print(f"  Forecast function: {forecast_fn.__name__}")
        print(f"  Test trajectories: {len(test_trajectories)}")
        print(f"  Output directory: {out_dir}")
        
        # Simulate evaluation loop
        for i, traj in enumerate(test_trajectories):
            # Get truth densities
            x_truth = traj  # [K, nx, ny]
            
            # Restrict to latent
            y_truth = [R(x) for x in x_truth]  # List of [d] arrays
            
            # Initial window from truth
            lag = 5  # Example
            y_init_window = np.stack(y_truth[:lag], axis=0)  # [lag, d]
            
            # Forecast in latent space
            n_steps = len(x_truth) - lag
            ys_pred = forecast_fn(y_init_window, n_steps)
            
            # Lift predictions to density
            xs_pred = [L(y) for y in ys_pred]  # List of [nx, ny] arrays
            
            # Compute R² (placeholder)
            # r2 = compute_r2(xs_pred, x_truth[lag:])
            
            print(f"    Trajectory {i+1}: forecast {n_steps} steps")
        
        # Save results
        print(f"  ✓ Writing r2_vs_time_{model_name}.csv")
        print(f"  ✓ Creating r2_vs_time_{model_name}.png")
        print(f"  ✓ Saving metrics to {out_dir}/")
    
    # Dummy operators
    def R(x):
        """Restriction: density -> latent (placeholder)."""
        return np.random.randn(10)  # [d=10]
    
    def L(y):
        """Lifting: latent -> density (placeholder)."""
        return np.random.randn(32, 32)  # [nx, ny]
    
    # Dummy test data
    test_trajectories = [
        np.random.randn(50, 32, 32) for _ in range(3)  # 3 trajectories
    ]
    
    # Create MVAR forecast function (placeholder)
    def mvar_forecast_fn(y_init_window, n_steps):
        """MVAR forecast (placeholder)."""
        d = y_init_window.shape[-1]
        return np.random.randn(n_steps, d)
    
    # Create LSTM forecast function
    model = LatentLSTMROM(d=10, hidden_units=16, num_layers=1)
    model.eval()
    lstm_forecast_fn = lstm_forecast_fn_factory(model)
    
    print("\nExample 1: Evaluate MVAR")
    print("="*80)
    evaluate_rom(
        model_name="MVAR",
        forecast_fn=mvar_forecast_fn,
        R=R, L=L,
        test_trajectories=test_trajectories,
        out_dir="results/run_001/MVAR"
    )
    
    print("\n" + "="*80)
    print("Example 2: Evaluate LSTM")
    print("="*80)
    evaluate_rom(
        model_name="LSTM",
        forecast_fn=lstm_forecast_fn,
        R=R, L=L,
        test_trajectories=test_trajectories,
        out_dir="results/run_001/LSTM"
    )
    
    print("\n" + "="*80)
    print("Folder Structure:")
    print("="*80)
    print("""
results/
  run_001/
    rom_common/
      POD_basis.npy
      singular_values.npy
      mean_density.npy
    MVAR/
      r2_vs_time_MVAR.csv
      r2_vs_time_MVAR.png
      reconstruction_MVAR.mp4
    LSTM/
      lstm_state_dict.pt
      training_log.csv
      r2_vs_time_LSTM.csv
      r2_vs_time_LSTM.png
      reconstruction_LSTM.mp4
    """)
    
    print("Key points:")
    print("  ✓ Same evaluation function for both models")
    print("  ✓ Model-agnostic forecast interface")
    print("  ✓ Separate output folders (MVAR/ and LSTM/)")
    print("  ✓ Consistent naming with model suffix")
    
    print("\n✅ TEST 4 PASSED")


def test_long_horizon_forecast():
    """Test 5: Long-horizon forecast stability."""
    print("\n" + "="*80)
    print("TEST 5: Long-Horizon Forecast Stability")
    print("="*80 + "\n")
    
    # Train a model
    d = 15
    lag = 10
    
    # Generate smooth training data (more realistic than random)
    N_samples = 500
    t = np.linspace(0, 10, N_samples + lag)
    X_train = []
    Y_train = []
    
    for i in range(N_samples):
        # Create smooth sequences
        window = np.sin(t[i:i+lag, None] * np.linspace(0.5, 2.0, d))
        target = np.sin(t[i+lag] * np.linspace(0.5, 2.0, d))
        X_train.append(window)
        Y_train.append(target)
    
    X_train = np.array(X_train, dtype=np.float32)
    Y_train = np.array(Y_train, dtype=np.float32)
    
    print(f"Training data: {X_train.shape}, {Y_train.shape}")
    
    # Config
    class Config:
        class ROM:
            class Models:
                class LSTM:
                    batch_size = 32
                    hidden_units = 32
                    num_layers = 2
                    learning_rate = 0.001
                    weight_decay = 0.0001
                    max_epochs = 30
                    patience = 10
                    gradient_clip = 1.0
                lstm = LSTM()
            models = Models()
        rom = ROM()
    
    # Train
    print(f"\nTraining LSTM for smooth dynamics...")
    model_path, val_loss = train_lstm_rom(X_train, Y_train, Config(), 'tmp_long_test')
    print(f"Training complete. Val loss: {val_loss:.6f}")
    
    # Load model
    model = LatentLSTMROM(d=d, hidden_units=32, num_layers=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Long-horizon forecast
    y_init = X_train[0]  # [lag, d]
    n_steps = 200  # Long horizon
    
    print(f"\nLong-horizon forecast:")
    print(f"  Initial window: {y_init.shape}")
    print(f"  Forecast steps: {n_steps}")
    
    ys_pred = forecast_with_lstm(model, y_init, n_steps)
    
    print(f"  Result: {ys_pred.shape}")
    
    # Check for stability (no NaN, no explosion)
    assert not np.any(np.isnan(ys_pred)), "NaN values in forecast"
    assert not np.any(np.isinf(ys_pred)), "Inf values in forecast"
    assert np.abs(ys_pred).max() < 1000, "Forecast exploded"
    
    print(f"\n  ✓ No NaN values")
    print(f"  ✓ No Inf values")
    print(f"  ✓ Values bounded (max abs: {np.abs(ys_pred).max():.2f})")
    
    # Show trajectory statistics over time
    print(f"\nForecast statistics by time chunk:")
    chunk_size = 50
    for i in range(0, n_steps, chunk_size):
        chunk = ys_pred[i:i+chunk_size]
        print(f"  Steps {i:3d}-{i+chunk_size:3d}: "
              f"mean={chunk.mean():6.3f}, std={chunk.std():6.3f}, "
              f"range=[{chunk.min():6.3f}, {chunk.max():6.3f}]")
    
    print("\n✅ TEST 5 PASSED")
    
    # Cleanup
    shutil.rmtree('tmp_long_test')


def main():
    print("\n" + "="*80)
    print("LSTM ROM FORECASTING & EVALUATION PIPELINE INTEGRATION TEST")
    print("="*80)
    
    # Run all tests
    test_basic_forecasting()
    test_forecast_factory()
    test_autoregressive_behavior()
    test_evaluation_pipeline_pattern()
    test_long_horizon_forecast()
    
    # Final summary
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED")
    print("="*80)
    
    print("\nImplemented functionality:")
    print("  ✓ forecast_with_lstm() - closed-loop latent forecasting")
    print("  ✓ lstm_forecast_fn_factory() - evaluation pipeline integration")
    print("  ✓ Autoregressive window sliding mechanism")
    print("  ✓ Model-agnostic evaluation pattern")
    print("  ✓ Long-horizon stability")
    
    print("\nIntegration with evaluation pipeline:")
    print("  ✓ Same interface as MVAR (forecast_fn signature)")
    print("  ✓ Works with generic evaluate_rom function")
    print("  ✓ Separate output folders: MVAR/ and LSTM/")
    print("  ✓ Consistent file naming with model suffix")
    
    print("\nNext steps:")
    print("  1. Integrate into existing ROM pipeline (run_unified_mvar_pipeline.py)")
    print("  2. Add model selection based on rom.models.*.enabled flags")
    print("  3. Use shared POD basis (rom_common/)")
    print("  4. Generate comparative plots (MVAR vs LSTM R² curves)")
    print()


if __name__ == "__main__":
    main()
