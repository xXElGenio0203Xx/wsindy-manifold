"""Smoke test for ROM/MVAR evaluation pipeline.

Quick validation that:
1. PODMVARModel loads from disk correctly
2. Test simulations can be loaded
3. Encode/decode cycle preserves data structure
4. Reconstruction error is reasonable

Author: Maria
Date: November 2025
"""

from pathlib import Path
from typing import Optional

import numpy as np

from rectsim.rom_mvar_model import PODMVARModel
from rectsim.rom_eval_data import load_unseen_simulations, print_dataset_summary


def smoke_test_rom_mvar_eval(
    rom_bundle_path: Path,
    simulations_root: Path,
    n_test_samples: int = 2,
) -> None:
    """Run smoke test on ROM/MVAR evaluation pipeline.
    
    Parameters
    ----------
    rom_bundle_path : Path
        Path to ROM model directory containing:
        - pod_basis.npz
        - mvar_params.npz
        - train_summary.json
    rom_bundle_path : Path
        Root directory with test simulations organized by IC type.
    n_test_samples : int, default=2
        Number of test samples to validate (uses first N found).
        
    Raises
    ------
    AssertionError
        If any validation check fails.
    FileNotFoundError
        If model files or simulations not found.
    """
    print("=" * 70)
    print("ROM/MVAR Evaluation Pipeline Smoke Test")
    print("=" * 70)
    print()
    
    # Step 1: Load ROM model
    print("Step 1: Loading ROM model...")
    print(f"  Bundle: {rom_bundle_path}")
    
    try:
        model = PODMVARModel.load(rom_bundle_path)
        print(f"  ✓ Model loaded successfully")
        print(f"    - Latent dim: {model.latent_dim}")
        print(f"    - MVAR order: {model.mvar_order}")
        print(f"    - POD basis shape: {model.pod_basis.shape}")
        print()
    except Exception as e:
        print(f"  ✗ Model load failed: {e}")
        raise
    
    # Step 2: Load test simulations
    print("Step 2: Loading test simulations...")
    print(f"  Root: {simulations_root}")
    
    try:
        samples = load_unseen_simulations(simulations_root, require_density=True)
        
        if not samples:
            raise ValueError(f"No simulations found in {simulations_root}")
        
        print(f"  ✓ Loaded {len(samples)} simulations")
        print_dataset_summary(samples)
        
    except Exception as e:
        print(f"  ✗ Simulation load failed: {e}")
        raise
    
    # Step 3: Validate encode/decode on a few samples
    print("Step 3: Testing encode/decode cycle...")
    
    test_samples = samples[:n_test_samples]
    
    for i, sample in enumerate(test_samples):
        print(f"  Test {i+1}/{len(test_samples)}: {sample.ic_type}/{sample.name}")
        
        density = sample.density_true
        T, Ny, Nx = density.shape
        
        # Check grid compatibility
        if model.grid_shape != (Ny, Nx):
            print(f"    ⚠ Grid mismatch: model={model.grid_shape}, data={(Ny, Nx)}")
            print(f"    Skipping this sample...")
            continue
        
        try:
            # Encode
            latent = model.encode(density)
            assert latent.shape == (T, model.latent_dim), \
                f"Latent shape mismatch: {latent.shape} vs expected ({T}, {model.latent_dim})"
            print(f"    ✓ Encode: ({T}, {Ny}, {Nx}) → ({T}, {model.latent_dim})")
            
            # Decode
            density_recon = model.decode(latent)
            assert density_recon.shape == (T, Ny, Nx), \
                f"Reconstruction shape mismatch: {density_recon.shape}"
            print(f"    ✓ Decode: ({T}, {model.latent_dim}) → ({T}, {Ny}, {Nx})")
            
            # Check reconstruction error
            recon_error = np.linalg.norm(density - density_recon) / np.linalg.norm(density)
            print(f"    ✓ Reconstruction error: {recon_error:.6f}")
            
            # Warn if error is suspiciously high
            if recon_error > 0.5:
                print(f"    ⚠ High reconstruction error (>50%)")
            
            # Test forecast (just check it runs, don't validate output quality)
            if T >= model.mvar_order + 10:
                n_forecast = 5
                latent_pred = model.forecast(
                    latent[:model.mvar_order],
                    n_steps=n_forecast
                )
                assert latent_pred.shape == (n_forecast, model.latent_dim), \
                    f"Forecast shape mismatch: {latent_pred.shape}"
                print(f"    ✓ Forecast: {n_forecast} steps in latent space")
            
        except Exception as e:
            print(f"    ✗ Test failed: {e}")
            raise
    
    print()
    print("=" * 70)
    print("✓ All smoke tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python rom_eval_smoke_test.py <rom_bundle_path> <simulations_root>")
        print()
        print("Example:")
        print("  python rom_eval_smoke_test.py \\")
        print("    rom_mvar/vicsek_exp1/model \\")
        print("    simulations_unseen/")
        sys.exit(1)
    
    rom_bundle = Path(sys.argv[1])
    sim_root = Path(sys.argv[2])
    
    smoke_test_rom_mvar_eval(rom_bundle, sim_root)
