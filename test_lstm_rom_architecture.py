#!/usr/bin/env python3
"""
Validation test for LatentLSTMROM model architecture.

This script confirms that the model:
1. Instantiates correctly with specified parameters
2. Accepts input of shape [batch_size, lag, d]
3. Produces output of shape [batch_size, d]
4. Works with various configurations (layers, hidden units, lag)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
from rom.lstm_rom import LatentLSTMROM


def main():
    print("\n" + "="*80)
    print("VALIDATION TEST: LatentLSTMROM Model Architecture")
    print("="*80 + "\n")
    
    # Configuration matching best_run_extended_test.yaml
    print("Configuration from best_run_extended_test.yaml:")
    print("  - Latent dimension (d): 25 POD modes")
    print("  - Sequence length (lag): 20")
    print("  - Hidden units: 64")
    print("  - Number of layers: 2\n")
    
    d = 25
    lag = 20
    hidden_units = 64
    num_layers = 2
    
    # Create model
    print("Creating LatentLSTMROM model...")
    model = LatentLSTMROM(d=d, hidden_units=hidden_units, num_layers=num_layers)
    print(f"✓ Model created:\n{model}\n")
    
    # Test with realistic batch sizes
    print("="*80)
    print("Testing with realistic batch sizes:")
    print("="*80 + "\n")
    
    for batch_size in [1, 16, 32, 64, 128]:
        x_dummy = torch.randn(batch_size, lag, d)
        y_pred = model(x_dummy)
        
        print(f"Batch size: {batch_size:3d}")
        print(f"  Input:  {str(tuple(x_dummy.shape)):20s} [batch, lag, d]")
        print(f"  Output: {str(tuple(y_pred.shape)):20s} [batch, d]")
        
        # Verify output shape
        assert y_pred.shape == (batch_size, d), \
            f"Expected output shape ({batch_size}, {d}), got {y_pred.shape}"
        print(f"  ✓ Shape verified\n")
    
    # Test with configurations from all production configs
    print("="*80)
    print("Testing with all production config parameters:")
    print("="*80 + "\n")
    
    configs = [
        {
            "name": "best_run_extended_test.yaml",
            "d": 25,
            "lag": 20,
            "hidden_units": 64,
            "num_layers": 2
        },
        {
            "name": "alvarez_style_production.yaml",
            "d": 35,
            "lag": 5,
            "hidden_units": 64,
            "num_layers": 2
        },
        {
            "name": "high_capacity_production.yaml",
            "d": 192,  # POD energy=0.99 typically gives ~150-200 modes
            "lag": 4,
            "hidden_units": 64,
            "num_layers": 2
        }
    ]
    
    for config in configs:
        print(f"Config: {config['name']}")
        print(f"  d={config['d']}, lag={config['lag']}, "
              f"hidden_units={config['hidden_units']}, num_layers={config['num_layers']}")
        
        # Create model with config parameters
        model_config = LatentLSTMROM(
            d=config['d'],
            hidden_units=config['hidden_units'],
            num_layers=config['num_layers']
        )
        
        # Test forward pass
        batch_size = 32
        x_test = torch.randn(batch_size, config['lag'], config['d'])
        y_test = model_config(x_test)
        
        # Count parameters
        total_params = sum(p.numel() for p in model_config.parameters())
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Input shape:  {tuple(x_test.shape)}")
        print(f"  Output shape: {tuple(y_test.shape)}")
        print(f"  ✓ Forward pass successful\n")
    
    # Test autoregressive forecasting simulation
    print("="*80)
    print("Testing autoregressive forecasting (closed-loop):")
    print("="*80 + "\n")
    
    model_forecast = LatentLSTMROM(d=25, hidden_units=64, num_layers=2)
    model_forecast.eval()  # Set to evaluation mode
    
    # Initial window (from truth)
    lag = 20
    y_init = torch.randn(1, lag, 25)  # [1, lag, d]
    
    print(f"Initial truth window: {tuple(y_init.shape)}")
    print(f"Forecasting 10 steps ahead in closed-loop...\n")
    
    # Simulate closed-loop forecasting
    forecast_steps = 10
    y_window = y_init.clone()
    predictions = []
    
    with torch.no_grad():
        for step in range(forecast_steps):
            # Predict next state
            y_next = model_forecast(y_window)  # [1, d]
            predictions.append(y_next)
            
            # Update window: remove oldest, append newest
            y_window = torch.cat([
                y_window[:, 1:, :],      # [1, lag-1, d]
                y_next.unsqueeze(1)      # [1, 1, d]
            ], dim=1)                     # [1, lag, d]
            
            print(f"  Step {step+1:2d}: predicted {tuple(y_next.shape)}, "
                  f"updated window {tuple(y_window.shape)}")
    
    print(f"\n✓ Closed-loop forecast successful ({forecast_steps} steps)\n")
    
    # Final summary
    print("="*80)
    print("✅ ALL VALIDATION TESTS PASSED")
    print("="*80)
    print("\nLatentLSTMROM model is ready for:")
    print("  1. Training on windowed latent datasets")
    print("  2. One-step-ahead prediction")
    print("  3. Closed-loop autoregressive forecasting")
    print("\nNext: Implement training loop (PART 4)")
    print()


if __name__ == "__main__":
    main()
