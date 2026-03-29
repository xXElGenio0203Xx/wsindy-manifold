#!/usr/bin/env python3
"""
Quick local test of the fixed LSTM: residual + normalization + scheduled sampling.
Verifies end-to-end: synthetic data → train → forecast → evaluate.
"""
import sys
sys.path.insert(0, '/Users/maria_1/Desktop/wsindy-manifold/src')
sys.path.insert(0, '/Users/maria_1/Desktop/wsindy-manifold')

import numpy as np
import torch
import tempfile
import os

from rom.lstm_rom import LatentLSTMROM, train_lstm_rom, forecast_with_lstm, lstm_forecast_fn_factory, load_lstm_model

np.random.seed(42)

# ── Create synthetic latent data with known dynamics ──
# Simple: damped oscillation + linear drift in 5D latent space
d = 5
T = 200  # timesteps per trajectory
n_traj = 30
dt = 0.1

print("=" * 70)
print("LSTM FIX VERIFICATION TEST")
print("=" * 70)

# Generate trajectories with nonlinear dynamics
trajs = []
for _ in range(n_traj):
    y = np.zeros((T, d))
    y[0] = np.random.randn(d) * 2  # random IC, moderate scale
    for t in range(T - 1):
        # Nonlinear dynamics: rotation + damping + bounded nonlinearity
        A = np.array([
            [0.98, -0.1, 0, 0, 0],
            [0.1, 0.98, 0, 0, 0],
            [0, 0, 0.95, -0.08, 0],
            [0, 0, 0.08, 0.95, 0],
            [0, 0, 0, 0, 0.92]
        ])
        # Use tanh for bounded nonlinearity
        y[t+1] = A @ y[t] + 0.1 * np.tanh(y[t]) + np.random.randn(d) * 0.02
    trajs.append(y)

trajs = np.array(trajs)  # [n_traj, T, d]
print(f"\nSynthetic data: {n_traj} trajectories × {T} timesteps × {d}D")
print(f"  State range: [{trajs.min():.1f}, {trajs.max():.1f}]")
print(f"  Δ range: [{np.diff(trajs, axis=1).min():.3f}, {np.diff(trajs, axis=1).max():.3f}]")

# ── Build windowed dataset (same as rom_data_utils.build_latent_dataset) ──
lag = 3
X_all, Y_all = [], []
for traj in trajs:
    for t in range(lag, T):
        X_all.append(traj[t-lag:t])  # [lag, d]
        Y_all.append(traj[t])        # [d]
X_all = np.array(X_all)  # [N, lag, d]
Y_all = np.array(Y_all)  # [N, d]
print(f"\nDataset: {X_all.shape[0]} samples, lag={lag}, d={d}")

# ── Test 1: Train with ALL fixes enabled (new defaults) ──
print("\n" + "=" * 70)
print("TEST 1: Train with residual + normalization + scheduled sampling")
print("=" * 70)

with tempfile.TemporaryDirectory() as tmpdir:
    config = {
        'rom': {
            'models': {
                'lstm': {
                    'batch_size': 64,
                    'hidden_units': 16,
                    'num_layers': 1,
                    'learning_rate': 1e-3,
                    'weight_decay': 1e-4,
                    'max_epochs': 200,  # Quick test
                    'patience': 50,
                    'gradient_clip': 1.0,
                    'dropout': 0.0,
                    'residual': True,
                    'normalize_input': True,
                    'scheduled_sampling': True,
                    'ss_start_epoch': 20,
                    'ss_end_epoch': 150,
                    'ss_max_ratio': 0.5,
                }
            }
        }
    }
    
    model_path, val_loss = train_lstm_rom(X_all, Y_all, config, tmpdir)
    print(f"\n✅ Training complete: val_loss = {val_loss:.6f}")
    
    # Verify saved checkpoint has all metadata
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    assert isinstance(checkpoint, dict) and 'state_dict' in checkpoint, "Checkpoint should have metadata"
    assert checkpoint['residual'] == True, "residual flag should be True"
    assert checkpoint['normalize_input'] == True, "normalize_input should be True"
    assert checkpoint['input_mean'] is not None, "input_mean should be saved"
    print(f"✅ Checkpoint has all metadata (residual={checkpoint['residual']}, norm={checkpoint['normalize_input']})")
    
    # ── Test 2: Load model with load_lstm_model ──
    print("\n" + "-" * 70)
    print("TEST 2: load_lstm_model")
    print("-" * 70)
    
    model, input_mean, input_std = load_lstm_model(tmpdir)
    print(f"  Model: {model}")
    print(f"  Normalization: mean shape={input_mean.shape}, std shape={input_std.shape}")
    print(f"✅ load_lstm_model works")
    
    # ── Test 3: Forecast and compare with old-style (no fixes) ──
    print("\n" + "-" * 70)
    print("TEST 3: Forecast comparison")
    print("-" * 70)
    
    # Pick a test trajectory
    test_traj = trajs[0]  # [T, d]
    y_init = test_traj[:lag]  # [lag, d]
    n_steps = T - lag
    
    # Forecast with normalization
    y_pred = forecast_with_lstm(model, y_init, n_steps, input_mean, input_std)
    y_truth = test_traj[lag:]
    
    # Compute R²
    ss_res = np.sum((y_truth - y_pred) ** 2)
    ss_tot = np.sum((y_truth - y_truth.mean(axis=0)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"  Rollout R² (with fixes): {r2:.4f}")
    
    # Also test factory function
    forecast_fn = lstm_forecast_fn_factory(model, input_mean, input_std)
    y_pred2 = forecast_fn(y_init, n_steps)
    assert np.allclose(y_pred, y_pred2), "Factory function should give same results"
    print(f"✅ lstm_forecast_fn_factory consistent")

# ── Test 4: Train WITHOUT fixes (old behavior) to compare ──
print("\n" + "=" * 70)
print("TEST 4: Train WITHOUT fixes (old behavior baseline)")
print("=" * 70)

with tempfile.TemporaryDirectory() as tmpdir:
    config_old = {
        'rom': {
            'models': {
                'lstm': {
                    'batch_size': 64,
                    'hidden_units': 16,
                    'num_layers': 1,
                    'learning_rate': 1e-3,
                    'weight_decay': 1e-4,
                    'max_epochs': 200,
                    'patience': 50,
                    'gradient_clip': 1.0,
                    'dropout': 0.0,
                    'residual': False,       # OLD behavior
                    'normalize_input': False, # OLD behavior  
                    'scheduled_sampling': False, # OLD behavior
                }
            }
        }
    }
    
    model_path_old, val_loss_old = train_lstm_rom(X_all, Y_all, config_old, tmpdir)
    print(f"\n✅ Old-style training complete: val_loss = {val_loss_old:.6f}")
    
    # Load old model manually (no normalization)
    checkpoint_old = torch.load(model_path_old, map_location='cpu', weights_only=False)
    model_old = LatentLSTMROM(d=d, hidden_units=16, num_layers=1, residual=False)
    model_old.load_state_dict(checkpoint_old['state_dict'])
    model_old.eval()
    
    y_pred_old = forecast_with_lstm(model_old, y_init, n_steps)
    ss_res_old = np.sum((y_truth - y_pred_old) ** 2)
    r2_old = 1 - ss_res_old / ss_tot
    print(f"  Rollout R² (no fixes): {r2_old:.4f}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  LSTM with fixes (residual+norm+SS):  R² = {r2:.4f}")
print(f"  LSTM without fixes (old behavior):   R² = {r2_old:.4f}")
print(f"  Improvement: {r2 - r2_old:+.4f}")
print("=" * 70)
