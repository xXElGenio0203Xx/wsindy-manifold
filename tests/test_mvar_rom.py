"""
Test and demonstrate MVAR-ROM evaluation pipeline.

Generates synthetic density evolution, runs complete POD→MVAR→lift pipeline,
and produces all evaluation metrics and visualizations.
"""

import numpy as np
from pathlib import Path

from wsindy_manifold.mvar_rom import (
    MVARROMConfig,
    run_mvar_rom_evaluation,
    fit_pod,
    restrict,
    lift,
    fit_mvar,
    forecast_closed_loop,
    evaluate,
)


def generate_synthetic_density_evolution(
    T=500,
    nx=40,
    ny=40,
    n_blobs=2,
    noise_level=0.01,
    seed=42
):
    """
    Generate synthetic density evolution with moving Gaussian blobs.
    
    Simulates density fields that evolve over time with drift and diffusion.
    
    Args:
        T: Number of time steps
        nx, ny: Grid dimensions
        n_blobs: Number of moving Gaussian blobs
        noise_level: Observation noise level
        seed: Random seed
        
    Returns:
        densities: Array (T, nx, ny)
    """
    np.random.seed(seed)
    
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    densities = np.zeros((T, nx, ny))
    
    # Initialize blob positions and velocities
    blob_pos = np.random.rand(n_blobs, 2)  # (n_blobs, 2)
    blob_vel = 0.002 * np.random.randn(n_blobs, 2)  # Slow drift
    blob_sigma = 0.05 + 0.02 * np.random.rand(n_blobs)  # Width
    
    for t in range(T):
        density = np.zeros((nx, ny))
        
        # Update blob positions (with periodic boundaries)
        blob_pos += blob_vel
        blob_pos = blob_pos % 1.0
        
        # Add some dynamics (sinusoidal modulation)
        for i in range(n_blobs):
            # Gaussian blob
            dx = X - blob_pos[i, 0]
            dy = Y - blob_pos[i, 1]
            
            # Handle periodic boundaries
            dx = np.minimum(dx, 1.0 - dx)
            dy = np.minimum(dy, 1.0 - dy)
            
            sigma = blob_sigma[i] * (1.0 + 0.2 * np.sin(2 * np.pi * t / 100))
            blob = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
            density += blob
        
        # Normalize
        density /= (density.sum() + 1e-16)
        
        # Add noise
        density += noise_level * np.random.randn(nx, ny)
        density = np.maximum(density, 0)  # Keep positive
        density /= (density.sum() + 1e-16)  # Renormalize
        
        densities[t] = density
    
    return densities


def test_basic_pipeline():
    """Test basic POD-MVAR-lift pipeline."""
    print("\n" + "="*80)
    print("Test 1: Basic POD-MVAR-Lift Pipeline")
    print("="*80)
    
    # Generate synthetic data
    T, nx, ny = 200, 30, 30
    densities = generate_synthetic_density_evolution(T=T, nx=nx, ny=ny, n_blobs=2)
    
    # Flatten
    X = densities.reshape(T, nx * ny)
    
    # Split
    T0 = int(0.8 * T)
    X_train = X[:T0]
    X_test = X[T0:]
    
    # POD
    Ud, xbar, d, energy_curve = fit_pod(X_train, energy=0.95)
    print(f"✓ POD: d = {d}")
    
    # Restrict
    Y_train = restrict(X_train, Ud, xbar)
    Y_test = restrict(X_test, Ud, xbar)
    print(f"✓ Restrict: Y_train.shape = {Y_train.shape}")
    
    # MVAR
    A0, A = fit_mvar(Y_train, w=3, ridge=1e-6)
    print(f"✓ MVAR: A0.shape = {A0.shape}, A.shape = {A.shape}")
    
    # Forecast
    Y_seed = Y_train[-3:]
    Y_forecast = forecast_closed_loop(Y_seed, A0, A, steps=len(Y_test))
    print(f"✓ Forecast: Y_forecast.shape = {Y_forecast.shape}")
    
    # Lift
    X_forecast = lift(Y_forecast, Ud, xbar)
    print(f"✓ Lift: X_forecast.shape = {X_forecast.shape}")
    
    # Evaluate
    frame_metrics, summary = evaluate(X_test, X_forecast, xbar, T0)
    print(f"✓ Evaluate: R² = {summary['r2']:.4f}, median L² = {summary['median_e2']:.4f}")
    
    print("\n✓ Basic pipeline test passed!\n")


def test_complete_evaluation():
    """Test complete evaluation pipeline with all outputs."""
    print("\n" + "="*80)
    print("Test 2: Complete Evaluation Pipeline")
    print("="*80)
    
    # Generate synthetic data
    T, nx, ny = 400, 40, 40
    print(f"\nGenerating synthetic density evolution: T={T}, nx={nx}, ny={ny}")
    densities = generate_synthetic_density_evolution(T=T, nx=nx, ny=ny, n_blobs=3)
    
    # Configure
    config = MVARROMConfig(
        pod_energy=0.99,
        mvar_order=4,
        ridge=1e-6,
        train_frac=0.8,
        tolerance_threshold=0.10,
        output_dir="outputs/test_mvar_rom",
        save_videos=False,  # Skip videos for test
        save_snapshots=True,
    )
    
    # Run evaluation
    results = run_mvar_rom_evaluation(densities, nx, ny, config)
    
    # Check outputs
    assert results["summary"]["r2"] is not None
    assert "frame_metrics" in results
    assert results["output_dir"].exists()
    
    print("\n✓ Complete evaluation test passed!")
    print(f"✓ Results saved to: {results['output_dir']}")
    

def test_different_configurations():
    """Test with different MVAR orders and POD energies."""
    print("\n" + "="*80)
    print("Test 3: Different Configurations")
    print("="*80)
    
    # Generate data once
    T, nx, ny = 300, 30, 30
    densities = generate_synthetic_density_evolution(T=T, nx=nx, ny=ny, n_blobs=2)
    
    configs = [
        {"pod_energy": 0.95, "mvar_order": 2, "name": "low_order"},
        {"pod_energy": 0.99, "mvar_order": 4, "name": "default"},
        {"pod_energy": 0.99, "mvar_order": 6, "name": "high_order"},
    ]
    
    results_all = []
    
    for cfg in configs:
        print(f"\n--- Configuration: {cfg['name']} (POD={cfg['pod_energy']}, w={cfg['mvar_order']}) ---")
        
        config = MVARROMConfig(
            pod_energy=cfg["pod_energy"],
            mvar_order=cfg["mvar_order"],
            train_frac=0.8,
            output_dir=f"outputs/test_mvar_rom_{cfg['name']}",
            save_videos=False,
            save_snapshots=False,
        )
        
        results = run_mvar_rom_evaluation(densities, nx, ny, config)
        results_all.append((cfg["name"], results["summary"]))
    
    # Compare results
    print("\n" + "-"*80)
    print("Comparison:")
    print("-"*80)
    print(f"{'Config':<15} {'d':<5} {'w':<5} {'R²':<10} {'Median L²':<12} {'τ_tol':<10}")
    print("-"*80)
    for name, summary in results_all:
        print(f"{name:<15} {summary['d']:<5} {summary['w']:<5} "
              f"{summary['r2']:<10.4f} {summary['median_e2']:<12.4f} {summary['tau_tol']:<10}")
    print("-"*80)
    
    print("\n✓ Configuration comparison test passed!")


def demo_full_evaluation():
    """Demonstration with comprehensive output."""
    print("\n" + "="*80)
    print("DEMONSTRATION: Full MVAR-ROM Evaluation")
    print("="*80)
    
    # Generate realistic-looking density evolution
    T, nx, ny = 600, 50, 50
    print(f"\nGenerating density evolution: T={T}, grid={nx}×{ny}")
    print("(3 moving Gaussian blobs with drift and diffusion)")
    
    densities = generate_synthetic_density_evolution(
        T=T, nx=nx, ny=ny, n_blobs=3, noise_level=0.005, seed=123
    )
    
    # Configure for thorough evaluation
    config = MVARROMConfig(
        pod_energy=0.99,
        mvar_order=4,
        ridge=1e-6,
        train_frac=0.75,  # More data for testing
        tolerance_threshold=0.10,
        output_dir="outputs/demo_mvar_rom",
        save_videos=False,
        save_snapshots=True,
        fps=20,
    )
    
    print(f"\nConfiguration:")
    print(f"  POD energy threshold: {config.pod_energy}")
    print(f"  MVAR order: w = {config.mvar_order}")
    print(f"  Ridge regularization: λ = {config.ridge}")
    print(f"  Train/test split: {config.train_frac*100:.0f}% / {(1-config.train_frac)*100:.0f}%")
    
    # Run full evaluation
    results = run_mvar_rom_evaluation(densities, nx, ny, config)
    
    # Display detailed results
    summary = results["summary"]
    
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    
    print("\nModel Configuration:")
    print(f"  POD modes retained: d = {summary['d']}")
    print(f"  MVAR lag order: w = {summary['w']}")
    print(f"  Ridge penalty: λ = {summary['lambda']}")
    
    print("\nData Split:")
    print(f"  Training frames: T0 = {summary['T0']}")
    print(f"  Test frames: T1 = {summary['T1']}")
    
    print("\nForecast Accuracy:")
    print(f"  R² score: {summary['r2']:.4f}")
    print(f"  Median L² error: {summary['median_e2']:.4f}")
    print(f"  10th percentile L²: {summary['p10_e2']:.4f}")
    print(f"  90th percentile L²: {summary['p90_e2']:.4f}")
    print(f"  Tolerance horizon (10%): {summary['tau_tol']} frames")
    
    print("\nMass Conservation:")
    print(f"  Mean mass error: {summary['mean_mass_error']:.6f}")
    print(f"  Max mass error: {summary['max_mass_error']:.6f}")
    
    print("\nComputational Performance:")
    print(f"  Total training time: {summary['train_time_s']:.2f} seconds")
    print(f"  Forecast time: {summary['forecast_time_s']:.2f} seconds")
    print(f"  Forecast speed: {summary['forecast_fps']:.1f} frames/second")
    
    print("\nGenerated Outputs:")
    output_dir = results["output_dir"]
    files = [
        "summary.json",
        "metrics_over_time.csv",
        "errors_timeseries.png",
        "snapshots.png",
        "latent_scatter.png",
    ]
    for fname in files:
        fpath = output_dir / fname
        if fpath.exists():
            print(f"  ✓ {fname}")
    
    print(f"\n  Directory: {output_dir}")
    print("="*80)
    
    print("\n✓ Full demonstration complete!")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("MVAR-ROM EVALUATION PIPELINE - TEST SUITE")
    print("="*80)
    
    # Run tests
    test_basic_pipeline()
    test_complete_evaluation()
    test_different_configurations()
    
    # Run demonstration
    demo_full_evaluation()
    
    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED!")
    print("="*80 + "\n")
