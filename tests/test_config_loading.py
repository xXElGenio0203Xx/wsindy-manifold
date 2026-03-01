#!/usr/bin/env python3
"""
Test script to verify new ROM config structure is loaded correctly.
"""

import yaml
from pathlib import Path

def test_config_loading(config_path):
    """Test loading the new ROM config structure."""
    
    print(f"\n{'='*80}")
    print(f"Testing Config Loading: {config_path}")
    print(f"{'='*80}\n")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    rom_config = config.get('rom', {})
    
    # Test general ROM settings
    print("General ROM Settings:")
    print(f"  subsample: {rom_config.get('subsample', 'NOT FOUND')}")
    print(f"  pod_energy: {rom_config.get('pod_energy', 'NOT FOUND')}")
    print(f"  fixed_modes: {rom_config.get('fixed_modes', 'NOT FOUND')}")
    print(f"  eigenvalue_threshold: {rom_config.get('eigenvalue_threshold', 'NOT FOUND')}")
    
    # Test models section
    if 'models' in rom_config:
        print("\n✓ Models section found!")
        models = rom_config['models']
        
        # Test MVAR config
        if 'mvar' in models:
            print("\nMVAR Configuration:")
            mvar = models['mvar']
            print(f"  enabled: {mvar.get('enabled', 'NOT FOUND')}")
            print(f"  lag: {mvar.get('lag', 'NOT FOUND')}")
            print(f"  ridge_alpha: {mvar.get('ridge_alpha', 'NOT FOUND')}")
        else:
            print("\n✗ MVAR section NOT FOUND")
        
        # Test LSTM config
        if 'lstm' in models:
            print("\nLSTM Configuration:")
            lstm = models['lstm']
            print(f"  enabled: {lstm.get('enabled', 'NOT FOUND')}")
            print(f"  lag: {lstm.get('lag', 'NOT FOUND')}")
            print(f"  hidden_units: {lstm.get('hidden_units', 'NOT FOUND')}")
            print(f"  num_layers: {lstm.get('num_layers', 'NOT FOUND')}")
            print(f"  activation: {lstm.get('activation', 'NOT FOUND')}")
            print(f"  batch_size: {lstm.get('batch_size', 'NOT FOUND')}")
            print(f"  learning_rate: {lstm.get('learning_rate', 'NOT FOUND')}")
            print(f"  max_epochs: {lstm.get('max_epochs', 'NOT FOUND')}")
            print(f"  patience: {lstm.get('patience', 'NOT FOUND')}")
            print(f"  weight_decay: {lstm.get('weight_decay', 'NOT FOUND')}")
            print(f"  gradient_clip: {lstm.get('gradient_clip', 'NOT FOUND')}")
            print(f"  loss: {lstm.get('loss', 'NOT FOUND')}")
        else:
            print("\n✗ LSTM section NOT FOUND")
    else:
        print("\n✗ Models section NOT FOUND")
    
    # Test backward compatibility path
    print("\n" + "="*80)
    print("Testing Backward Compatibility Access:")
    print("="*80)
    
    if 'models' in rom_config and 'mvar' in rom_config['models']:
        mvar_config = rom_config['models']['mvar']
        lag = mvar_config.get('lag', 'DEFAULT')
        ridge = mvar_config.get('ridge_alpha', 'DEFAULT')
        print(f"✓ New path: rom.models.mvar.lag = {lag}")
        print(f"✓ New path: rom.models.mvar.ridge_alpha = {ridge}")
    else:
        lag = rom_config.get('mvar_lag', 'DEFAULT')
        ridge = rom_config.get('ridge_alpha', 'DEFAULT')
        print(f"✓ Old path: rom.mvar_lag = {lag}")
        print(f"✓ Old path: rom.ridge_alpha = {ridge}")
    
    print("\n✅ Config loading test complete!\n")

if __name__ == "__main__":
    config_path = Path("configs/best_run_extended_test.yaml")
    if config_path.exists():
        test_config_loading(config_path)
    else:
        print(f"❌ Config file not found: {config_path}")
