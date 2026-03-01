#!/usr/bin/env python3
"""
Integration test: Build latent dataset from POD trajectories.

This script demonstrates how build_latent_dataset() would be used
in the actual ROM training pipeline with POD latent trajectories.
"""

import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from rectsim.rom_data_utils import build_latent_dataset, print_dataset_info


def simulate_pod_latent_trajectories(n_trajectories: int, n_timesteps: int, 
                                     n_modes: int, seed: int = 42):
    """
    Simulate what POD latent trajectories would look like.
    
    In reality, these come from:
        Y_c = Phi^T @ (rho_c - rho_mean)
    
    where Phi is the POD basis and rho_c are density snapshots.
    """
    np.random.seed(seed)
    
    trajectories = []
    for i in range(n_trajectories):
        # Simulate smooth latent trajectories with some dynamics
        t = np.linspace(0, 10, n_timesteps)
        
        # Each mode has different dynamics (oscillations, exponentials, etc.)
        y = np.zeros((n_timesteps, n_modes))
        for mode in range(n_modes):
            freq = 0.5 + mode * 0.3
            decay = 0.95 ** mode
            phase = np.random.rand() * 2 * np.pi
            
            # Combine oscillation with decay and noise
            y[:, mode] = (
                decay * np.sin(freq * t + phase) + 
                0.1 * np.random.randn(n_timesteps)
            )
        
        trajectories.append(y)
    
    return trajectories


def main():
    print("\n" + "="*80)
    print("INTEGRATION TEST: Build Latent Dataset from POD Trajectories")
    print("="*80 + "\n")
    
    # Simulate typical training scenario
    print("Simulating POD training scenario:")
    print("  - 400 training trajectories")
    print("  - Each trajectory: 20 timesteps (T=2.0s, dt=0.1s)")
    print("  - Latent dimension: d=25 POD modes")
    print("  - Window length: lag=20\n")
    
    # Parameters matching best_run_extended_test.yaml
    n_trajectories = 400
    n_timesteps = 20  # T=2.0s / dt=0.1s
    n_modes = 25
    lag = 20
    
    # Generate simulated latent trajectories
    print("Generating simulated latent trajectories...")
    y_trajs = simulate_pod_latent_trajectories(n_trajectories, n_timesteps, n_modes)
    print(f"✓ Generated {len(y_trajs)} trajectories")
    print(f"  Example trajectory shape: {y_trajs[0].shape}  [timesteps, modes]\n")
    
    # Build dataset
    print("Building windowed dataset...")
    try:
        X_all, Y_all = build_latent_dataset(y_trajs, lag)
    except ValueError as e:
        print(f"\n⚠️  Dataset building failed: {e}")
        print("\nNote: With lag=20 and only 20 timesteps, we get 0 samples per trajectory!")
        print("This is why we need T_train > lag * dt for MVAR training.\n")
        
        # Retry with valid parameters
        print("Retrying with lag=5 (more realistic for T=2.0s)...")
        lag = 5
        X_all, Y_all = build_latent_dataset(y_trajs, lag)
    
    # Print dataset info
    print_dataset_info(X_all, Y_all, lag)
    
    # Show how this would be used for MVAR
    print("="*80)
    print("MVAR Usage:")
    print("="*80)
    X_mvar = X_all.reshape(X_all.shape[0], -1)  # Flatten lag dimension
    print(f"  1. Reshape X: {X_all.shape} → {X_mvar.shape}")
    print(f"  2. Fit Ridge regression: Y ~ X_mvar @ A^T + b")
    print(f"  3. Parameters: A is [{n_modes} × {lag * n_modes}] matrix")
    print(f"     Total parameters: {lag * n_modes * n_modes:,}\n")
    
    # Show how this would be used for LSTM
    print("="*80)
    print("LSTM Usage:")
    print("="*80)
    print(f"  1. Use X directly: {X_all.shape}  [batch, seq_len, features]")
    print(f"  2. Target Y: {Y_all.shape}  [batch, features]")
    print(f"  3. LSTM layers process sequences of length {lag}")
    print(f"  4. Final layer outputs: {n_modes}-dimensional prediction\n")
    
    # Demonstrate train/val split
    print("="*80)
    print("Train/Validation Split Example:")
    print("="*80)
    n_samples = X_all.shape[0]
    train_frac = 0.8
    n_train = int(n_samples * train_frac)
    
    # Shuffle indices
    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    X_train, Y_train = X_all[train_idx], Y_all[train_idx]
    X_val, Y_val = X_all[val_idx], Y_all[val_idx]
    
    print(f"  Total samples: {n_samples:,}")
    print(f"  Train split: {len(train_idx):,} samples ({train_frac*100:.0f}%)")
    print(f"  Val split:   {len(val_idx):,} samples ({(1-train_frac)*100:.0f}%)")
    print(f"\n  X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"  X_val:   {X_val.shape}, Y_val:   {Y_val.shape}\n")
    
    # Show identifiability analysis
    print("="*80)
    print("Identifiability Analysis:")
    print("="*80)
    n_params_mvar = lag * n_modes * n_modes
    n_samples_train = len(train_idx)
    ratio = n_samples_train / n_params_mvar
    
    print(f"  Training samples: {n_samples_train:,}")
    print(f"  MVAR parameters:  {n_params_mvar:,}")
    print(f"  Ratio (samples/params): {ratio:.2f}")
    
    if ratio > 10:
        print(f"  ✓ Well-conditioned (ratio > 10)")
    elif ratio > 5:
        print(f"  ⚠️  Moderately conditioned (5 < ratio < 10)")
    else:
        print(f"  ✗ Poorly conditioned (ratio < 5)")
    print()
    
    print("="*80)
    print("✅ Integration test complete!")
    print("="*80)
    print("\nNext steps:")
    print("  - Integrate build_latent_dataset() into mvar_trainer.py")
    print("  - Use same dataset for LSTM training")
    print("  - Implement model selection based on rom.models.*.enabled")
    print()


if __name__ == "__main__":
    main()
