#!/usr/bin/env python3
"""
Runtime Analyzer Test Script
=============================

Demonstrates the usage of the runtime analysis module with mock MVAR and LSTM models.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from rectsim.runtime_analyzer import (
    RuntimeAnalyzer,
    compute_mvar_params,
    quick_benchmark
)


def create_mock_mvar_model(d: int, p: int):
    """Create a mock MVAR model for testing."""
    A_0 = np.random.randn(d)
    A_j = np.random.randn(p, d, d)
    
    return {
        'A_0': A_0,
        'A_j': A_j,
        'order': p,
        'latent_dim': d
    }


def mvar_forecast_mock(mvar_model):
    """Create mock MVAR forecast function."""
    A_0 = mvar_model['A_0']
    A_j = mvar_model['A_j']
    p = mvar_model['order']
    
    def forecast(z0: np.ndarray, n_steps: int) -> np.ndarray:
        """Mock MVAR forecast."""
        d = len(A_0)
        z_history = np.zeros((p + n_steps, d))
        
        # Initialize with z0 (assuming z0 is a window of length p)
        if len(z0.shape) == 1:
            z_history[0] = z0
        else:
            z_history[:p] = z0[:p]
        
        # Forecast
        for t in range(p, p + n_steps):
            z_new = A_0.copy()
            for j in range(p):
                z_new += A_j[j] @ z_history[t - j - 1]
            z_history[t] = z_new
        
        return z_history[p:]
    
    return forecast


def main():
    print("="*80)
    print("Runtime Analyzer Test")
    print("="*80)
    
    # Setup
    d = 20  # Latent dimension
    p_mvar = 5  # MVAR lag
    
    print(f"\nSetup:")
    print(f"   Latent dimension: {d}")
    print(f"   MVAR lag: {p_mvar}")
    
    # Create mock models
    print(f"\n{'='*80}")
    print("Creating Mock Models")
    print("="*80)
    
    mvar_model = create_mock_mvar_model(d, p_mvar)
    mvar_params = compute_mvar_params(mvar_model)
    print(f"✓ MVAR model created: {mvar_params:,} parameters")
    
    # Create forecast functions
    mvar_forecast = mvar_forecast_mock(mvar_model)
    
    # Initialize analyzer
    analyzer = RuntimeAnalyzer()
    
    # Mock training time
    print(f"\n{'='*80}")
    print("Simulating Training")
    print("="*80)
    
    import time
    mvar_train_start = time.perf_counter()
    _ = mvar_forecast(np.random.randn(d), 100)  # Warmup
    mvar_train_time = time.perf_counter() - mvar_train_start + 2.5  # Mock 2.5s training
    
    print(f"✓ Mock MVAR training: {mvar_train_time:.2f}s")
    
    # Benchmark inference
    print(f"\n{'='*80}")
    print("Benchmarking Inference")
    print("="*80)
    
    z0 = np.random.randn(d)
    
    profile = quick_benchmark(
        model_name='MVAR',
        forecast_fn=mvar_forecast,
        z0=z0,
        training_time=mvar_train_time,
        model_params=mvar_params,
        latent_dim=d,
        n_steps=100,
        n_trials=50,
        lag=p_mvar
    )
    
    # Display results
    print(f"\n{'='*80}")
    print("Runtime Profile Results")
    print("="*80)
    
    print(f"\nTraining:")
    print(f"   Total time: {profile.training.total_seconds:.2f}s")
    
    print(f"\nInference (single step):")
    print(f"   Mean: {profile.inference_single_step.mean_seconds*1000:.3f}ms")
    print(f"   Std:  {profile.inference_single_step.std_seconds*1000:.3f}ms")
    print(f"   Min:  {profile.inference_single_step.min_seconds*1000:.3f}ms")
    print(f"   Max:  {profile.inference_single_step.max_seconds*1000:.3f}ms")
    
    print(f"\nInference (100 steps):")
    print(f"   Mean: {profile.inference_full_trajectory.mean_seconds:.3f}s")
    print(f"   Std:  {profile.inference_full_trajectory.std_seconds:.3f}s")
    
    print(f"\nMemory:")
    print(f"   Model parameters: {profile.memory.model_parameters:,}")
    print(f"   Parameter memory: {profile.memory.parameter_memory_mb:.2f} MB")
    print(f"   Current process: {profile.memory.current_mb:.2f} MB")
    
    print(f"\nThroughput:")
    print(f"   Steps per second: {profile.throughput['steps_per_second']:.1f}")
    print(f"   Predictions per second: {profile.throughput['predictions_per_second']:.1f}")
    print(f"   Microseconds per step: {profile.throughput['microseconds_per_step']:.1f}μs")
    
    print(f"\nComplexity:")
    print(f"   Total parameters: {profile.complexity['total_parameters']:,}")
    print(f"   Lag order: {profile.complexity['lag_order']}")
    print(f"   Training complexity: {profile.complexity['training_complexity']}")
    print(f"   Inference complexity: {profile.complexity['inference_complexity']}")
    print(f"   Inference FLOPs per step: {profile.complexity['inference_flops_per_step']:,}")
    
    # Save profile
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    analyzer.save_profile(profile, output_dir / "test_runtime_profile.json")
    
    print(f"\n{'='*80}")
    print("✅ Test Complete")
    print("="*80)
    print(f"\nProfile saved: {output_dir}/test_runtime_profile.json")


if __name__ == "__main__":
    main()
