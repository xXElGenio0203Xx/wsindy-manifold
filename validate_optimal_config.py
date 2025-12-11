#!/usr/bin/env python3
"""
Validate Optimal Configuration
===============================

Quick validation script to verify the vicsek_rom_joint_optimal.yaml
configuration before submitting to Oscar.

Checks:
- YAML syntax
- Parameter counts and data/parameter ratio
- Training/test setup
- ROM configuration consistency
"""

import yaml
import sys
from pathlib import Path

def validate_config(config_path):
    """Validate configuration and print analysis."""
    
    print("=" * 80)
    print("CONFIGURATION VALIDATION: vicsek_rom_joint_optimal.yaml")
    print("=" * 80)
    
    # Load config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("\n✓ YAML syntax valid")
    except Exception as e:
        print(f"\n✗ YAML syntax error: {e}")
        return False
    
    # Extract key parameters
    sim = config.get('sim', {})
    rom = config.get('rom', {})
    train_ic = config.get('train_ic', {})
    test_ic = config.get('test_ic', {})
    
    T = sim.get('T', 0)
    dt = sim.get('dt', 0.1)
    subsample = rom.get('subsample', 1)
    fixed_modes = rom.get('fixed_modes', 0)
    
    mvar_config = rom.get('models', {}).get('mvar', {})
    lstm_config = rom.get('models', {}).get('lstm', {})
    
    mvar_lag = mvar_config.get('lag', 0)
    lstm_lag = lstm_config.get('lag', 0)
    ridge_alpha = mvar_config.get('ridge_alpha', 0)
    
    # Count training runs
    n_train = 0
    if train_ic.get('gaussian', {}).get('enabled', False):
        n_train += train_ic['gaussian'].get('count', 0)
    if train_ic.get('uniform', {}).get('enabled', False):
        n_train += train_ic['uniform'].get('count', 0)
    if train_ic.get('ring', {}).get('enabled', False):
        n_train += train_ic['ring'].get('count', 0)
    if train_ic.get('two_clusters', {}).get('enabled', False):
        n_train += train_ic['two_clusters'].get('count', 0)
    
    # Count test runs
    n_test = 0
    if test_ic.get('gaussian', {}).get('enabled', False):
        n_test += test_ic['gaussian'].get('count', 0)
    if test_ic.get('uniform', {}).get('enabled', False):
        n_test += test_ic['uniform'].get('count', 0)
    if test_ic.get('ring', {}).get('enabled', False):
        n_test += test_ic['ring'].get('count', 0)
    if test_ic.get('two_clusters', {}).get('enabled', False):
        n_test += test_ic['two_clusters'].get('count', 0)
    
    # Calculate derived quantities
    raw_steps = int(T / dt)
    rom_steps = raw_steps // subsample
    windows_per_traj = rom_steps - mvar_lag
    total_windows = n_train * windows_per_traj
    mvar_params = fixed_modes ** 2 * mvar_lag
    data_param_ratio = total_windows / mvar_params if mvar_params > 0 else 0
    
    print("\n" + "=" * 80)
    print("SIMULATION PARAMETERS")
    print("=" * 80)
    print(f"  Duration (T):           {T:.1f} s")
    print(f"  Time step (dt):         {dt:.3f} s")
    print(f"  Raw steps per traj:     {raw_steps}")
    print(f"  ROM subsample:          {subsample}")
    print(f"  ROM steps per traj:     {rom_steps}")
    
    print("\n" + "=" * 80)
    print("TRAINING DATA")
    print("=" * 80)
    print(f"  Total training runs:    {n_train}")
    print(f"  Gaussian clusters:      {train_ic.get('gaussian', {}).get('count', 0)}")
    print(f"  Uniform random:         {train_ic.get('uniform', {}).get('count', 0)}")
    print(f"  Ring configurations:    {train_ic.get('ring', {}).get('count', 0)}")
    print(f"  Two-cluster configs:    {train_ic.get('two_clusters', {}).get('count', 0)}")
    
    print("\n" + "=" * 80)
    print("TEST DATA")
    print("=" * 80)
    print(f"  Total test runs:        {n_test}")
    print(f"  Test duration:          {test_ic.get('test_T', T):.1f} s")
    
    print("\n" + "=" * 80)
    print("ROM CONFIGURATION")
    print("=" * 80)
    print(f"  POD modes (d):          {fixed_modes}")
    print(f"  MVAR enabled:           {mvar_config.get('enabled', False)}")
    print(f"  LSTM enabled:           {lstm_config.get('enabled', False)}")
    
    print("\n" + "=" * 80)
    print("MVAR ANALYSIS")
    print("=" * 80)
    print(f"  Lag (w):                {mvar_lag}")
    print(f"  Ridge alpha:            {ridge_alpha:.2e}")
    print(f"  Windows per traj:       {windows_per_traj}")
    print(f"  Total windows:          {total_windows:,}")
    print(f"  MVAR parameters:        {mvar_params:,} (d² × w = {fixed_modes}² × {mvar_lag})")
    print(f"  Data/param ratio:       {data_param_ratio:.2f}")
    
    # Validation checks
    print("\n" + "=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)
    
    checks_passed = True
    
    # Check 1: Lag alignment
    if mvar_lag == lstm_lag:
        print(f"  ✓ Lags aligned: MVAR={mvar_lag}, LSTM={lstm_lag}")
    else:
        print(f"  ✗ Lags misaligned: MVAR={mvar_lag}, LSTM={lstm_lag}")
        checks_passed = False
    
    # Check 2: Data/parameter ratio
    if data_param_ratio >= 2.0:
        print(f"  ✓ Data/param ratio adequate: {data_param_ratio:.2f} ≥ 2.0")
    else:
        print(f"  ⚠ Data/param ratio low: {data_param_ratio:.2f} < 2.0 (risk of overfitting)")
        checks_passed = False
    
    # Check 3: Training data sufficiency
    if n_train >= 300:
        print(f"  ✓ Training data sufficient: {n_train} runs")
    else:
        print(f"  ⚠ Training data may be insufficient: {n_train} runs < 300")
    
    # Check 4: Test data
    if n_test >= 20:
        print(f"  ✓ Test data adequate: {n_test} runs")
    else:
        print(f"  ⚠ Test data limited: {n_test} runs < 20")
    
    # Check 5: Ridge regularization
    if 1e-5 <= ridge_alpha <= 1e-3:
        print(f"  ✓ Ridge alpha in good range: {ridge_alpha:.2e}")
    elif ridge_alpha < 1e-5:
        print(f"  ⚠ Ridge alpha very weak: {ridge_alpha:.2e} (risk of overfitting)")
    else:
        print(f"  ⚠ Ridge alpha very strong: {ridge_alpha:.2e} (may underfit)")
    
    # Check 6: LSTM configuration
    lstm_hidden = lstm_config.get('hidden_units', 0)
    lstm_layers = lstm_config.get('num_layers', 0)
    if lstm_hidden > 0 and lstm_layers > 0:
        print(f"  ✓ LSTM configured: {lstm_hidden} hidden units, {lstm_layers} layer(s)")
    else:
        print(f"  ⚠ LSTM configuration incomplete")
    
    print("\n" + "=" * 80)
    print("EXPECTED COMPUTATIONAL COST")
    print("=" * 80)
    
    # Rough estimates
    training_sims_minutes = n_train * T / 60  # assume ~1s real time per 1s sim
    test_sims_minutes = n_test * T / 60
    rom_training_minutes = 5  # POD + MVAR + LSTM training
    total_minutes = training_sims_minutes + test_sims_minutes + rom_training_minutes
    
    print(f"  Training simulations:   ~{training_sims_minutes:.1f} minutes")
    print(f"  Test simulations:       ~{test_sims_minutes:.1f} minutes")
    print(f"  ROM training:           ~{rom_training_minutes:.1f} minutes")
    print(f"  Total estimated:        ~{total_minutes:.1f} minutes ({total_minutes/60:.1f} hours)")
    
    print("\n" + "=" * 80)
    if checks_passed:
        print("✓ ALL CHECKS PASSED - CONFIGURATION READY FOR OSCAR")
    else:
        print("⚠ SOME CHECKS FAILED - REVIEW CONFIGURATION")
    print("=" * 80)
    
    return checks_passed


if __name__ == "__main__":
    config_path = Path("configs/vicsek_rom_joint_optimal.yaml")
    
    if not config_path.exists():
        print(f"✗ Configuration file not found: {config_path}")
        sys.exit(1)
    
    success = validate_config(config_path)
    sys.exit(0 if success else 1)
