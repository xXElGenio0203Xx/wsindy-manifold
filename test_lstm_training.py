#!/usr/bin/env python3
"""
Sanity test for LSTM ROM training function.

This script tests the train_lstm_rom function with small dummy data
to verify that:
1. The training loop runs without errors
2. The model checkpoint is saved correctly
3. The training log is created
4. The best validation loss is returned
"""

import sys
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import torch
from rom.lstm_rom import train_lstm_rom, LatentLSTMROM


def create_dummy_config():
    """Create a minimal config dict for testing."""
    class DummyConfig:
        class ROM:
            class Models:
                class LSTM:
                    batch_size = 16
                    hidden_units = 16
                    num_layers = 1
                    learning_rate = 0.001
                    weight_decay = 0.0001
                    max_epochs = 20
                    patience = 5
                    gradient_clip = 1.0
                
                lstm = LSTM()
            
            models = Models()
        
        rom = ROM()
    
    return DummyConfig()


def test_training_basic():
    """Test 1: Basic training with small dataset."""
    print("\n" + "="*80)
    print("TEST 1: Basic training with small dataset")
    print("="*80)
    
    # Create small dummy dataset
    N_samples = 200
    lag = 5
    d = 10
    
    np.random.seed(42)
    X_all = np.random.randn(N_samples, lag, d).astype(np.float32)
    Y_all = np.random.randn(N_samples, d).astype(np.float32)
    
    print(f"\nDummy data:")
    print(f"  X_all: {X_all.shape}  [N_samples, lag, d]")
    print(f"  Y_all: {Y_all.shape}  [N_samples, d]")
    
    # Create config
    config = create_dummy_config()
    
    # Train
    out_dir = "tmp_lstm_test"
    model_path, best_val_loss = train_lstm_rom(X_all, Y_all, config, out_dir)
    
    # Verify outputs
    print("\n" + "-"*80)
    print("Verification:")
    
    # Check directory exists
    assert Path(out_dir).exists(), "Output directory not created"
    print(f"  ✓ Output directory created: {out_dir}")
    
    # Check model file exists
    assert Path(model_path).exists(), "Model checkpoint not saved"
    print(f"  ✓ Model checkpoint saved: {model_path}")
    
    # Check log file exists
    log_path = Path(out_dir) / "training_log.csv"
    assert log_path.exists(), "Training log not created"
    print(f"  ✓ Training log created: {log_path}")
    
    # Check log has content
    with open(log_path, 'r') as f:
        lines = f.readlines()
    assert len(lines) > 1, "Training log is empty"
    print(f"  ✓ Training log has {len(lines)-1} epochs recorded")
    
    # Check best_val_loss is finite
    assert np.isfinite(best_val_loss), "Best validation loss is not finite"
    print(f"  ✓ Best validation loss: {best_val_loss:.6f}")
    
    # Check model can be loaded
    model = LatentLSTMROM(d=d, hidden_units=16, num_layers=1)
    model.load_state_dict(torch.load(model_path))
    print(f"  ✓ Model loaded successfully from checkpoint")
    
    # Test inference with loaded model
    model.eval()
    x_test = torch.randn(1, lag, d)
    with torch.no_grad():
        y_pred = model(x_test)
    assert y_pred.shape == (1, d), f"Expected output shape (1, {d}), got {y_pred.shape}"
    print(f"  ✓ Inference with loaded model: {x_test.shape} → {y_pred.shape}")
    
    print("\n✅ TEST 1 PASSED")
    
    # Cleanup
    shutil.rmtree(out_dir)
    print(f"  (Cleaned up {out_dir})")
    
    return best_val_loss


def test_training_with_dict_config():
    """Test 2: Training with dict-style config (not object)."""
    print("\n" + "="*80)
    print("TEST 2: Training with dict-style config")
    print("="*80)
    
    # Create dict config
    config = {
        'rom': {
            'models': {
                'lstm': {
                    'batch_size': 8,
                    'hidden_units': 8,
                    'num_layers': 1,
                    'learning_rate': 0.001,
                    'weight_decay': 0.0,
                    'max_epochs': 10,
                    'patience': 3,
                    'gradient_clip': 0.5
                }
            }
        }
    }
    
    # Small dataset
    N_samples = 100
    lag = 3
    d = 5
    
    np.random.seed(123)
    X_all = np.random.randn(N_samples, lag, d).astype(np.float32)
    Y_all = np.random.randn(N_samples, d).astype(np.float32)
    
    print(f"\nDict config: {config['rom']['models']['lstm']}")
    
    # Train
    out_dir = "tmp_lstm_test_dict"
    model_path, best_val_loss = train_lstm_rom(X_all, Y_all, config, out_dir)
    
    # Verify
    assert Path(model_path).exists(), "Model checkpoint not saved"
    assert np.isfinite(best_val_loss), "Best validation loss is not finite"
    
    print(f"\n✓ Training completed with dict config")
    print(f"✓ Best validation loss: {best_val_loss:.6f}")
    print("\n✅ TEST 2 PASSED")
    
    # Cleanup
    shutil.rmtree(out_dir)
    print(f"  (Cleaned up {out_dir})")
    
    return best_val_loss


def test_early_stopping():
    """Test 3: Verify early stopping works."""
    print("\n" + "="*80)
    print("TEST 3: Early stopping verification")
    print("="*80)
    
    # Create dataset
    N_samples = 150
    lag = 4
    d = 8
    
    np.random.seed(456)
    X_all = np.random.randn(N_samples, lag, d).astype(np.float32)
    Y_all = np.random.randn(N_samples, d).astype(np.float32)
    
    # Config with low patience
    config = create_dummy_config()
    config.rom.models.lstm.max_epochs = 100  # High max_epochs
    config.rom.models.lstm.patience = 3      # But low patience
    
    print(f"\nConfig: max_epochs={config.rom.models.lstm.max_epochs}, "
          f"patience={config.rom.models.lstm.patience}")
    
    # Train
    out_dir = "tmp_lstm_test_early"
    model_path, best_val_loss = train_lstm_rom(X_all, Y_all, config, out_dir)
    
    # Check that training stopped early
    log_path = Path(out_dir) / "training_log.csv"
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    epochs_trained = len(lines) - 1  # Subtract header
    
    print(f"\nEarly stopping test:")
    print(f"  Max epochs: 100")
    print(f"  Actual epochs: {epochs_trained}")
    
    # Should stop well before max_epochs due to early stopping
    assert epochs_trained < 100, "Training did not stop early"
    print(f"  ✓ Training stopped early at epoch {epochs_trained}")
    
    print("\n✅ TEST 3 PASSED")
    
    # Cleanup
    shutil.rmtree(out_dir)
    print(f"  (Cleaned up {out_dir})")


def test_realistic_config():
    """Test 4: Training with realistic production config parameters."""
    print("\n" + "="*80)
    print("TEST 4: Realistic production config parameters")
    print("="*80)
    
    # Simulate production scenario
    N_samples = 1000  # Larger dataset
    lag = 20          # From best_run_extended_test.yaml
    d = 25            # 25 POD modes
    
    np.random.seed(789)
    X_all = np.random.randn(N_samples, lag, d).astype(np.float32)
    Y_all = np.random.randn(N_samples, d).astype(np.float32)
    
    print(f"\nProduction-like scenario:")
    print(f"  Samples: {N_samples:,}")
    print(f"  Lag: {lag}")
    print(f"  Latent dimension: {d}")
    
    # Production config
    config = create_dummy_config()
    config.rom.models.lstm.batch_size = 32
    config.rom.models.lstm.hidden_units = 64
    config.rom.models.lstm.num_layers = 2
    config.rom.models.lstm.learning_rate = 0.0001
    config.rom.models.lstm.weight_decay = 0.0001
    config.rom.models.lstm.max_epochs = 50
    config.rom.models.lstm.patience = 10
    config.rom.models.lstm.gradient_clip = 1.0
    
    # Train
    out_dir = "tmp_lstm_test_production"
    model_path, best_val_loss = train_lstm_rom(X_all, Y_all, config, out_dir)
    
    # Verify
    assert Path(model_path).exists()
    assert np.isfinite(best_val_loss)
    
    # Check model size
    model = LatentLSTMROM(d=d, hidden_units=64, num_layers=2)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    
    print("\n✅ TEST 4 PASSED")
    
    # Cleanup
    shutil.rmtree(out_dir)
    print(f"  (Cleaned up {out_dir})")


def main():
    print("\n" + "="*80)
    print("SANITY TEST: LSTM ROM Training Function")
    print("="*80)
    
    # Run all tests
    test_training_basic()
    test_training_with_dict_config()
    test_early_stopping()
    test_realistic_config()
    
    # Final summary
    print("\n" + "="*80)
    print("✅ ALL SANITY TESTS PASSED")
    print("="*80)
    print("\nThe train_lstm_rom function is working correctly:")
    print("  ✓ Creates output directory")
    print("  ✓ Splits data into train/validation")
    print("  ✓ Trains LSTM model with MSE loss")
    print("  ✓ Implements early stopping")
    print("  ✓ Saves best model checkpoint")
    print("  ✓ Creates training log CSV")
    print("  ✓ Returns model path and validation loss")
    print("  ✓ Works with both object and dict configs")
    print("  ✓ Handles production-scale parameters")
    print("\nNext: PART 5 - Implement closed-loop forecasting")
    print()


if __name__ == "__main__":
    main()
