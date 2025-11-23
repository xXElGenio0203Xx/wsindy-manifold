#!/usr/bin/env python3
"""
Generate ROM-MVAR predictions for test data.
Run this after data generation completes if predictions weren't made on Oscar.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import json


def mvar_forecast(y_init, A_matrices, T_forecast):
    """
    Forecast future latent states using MVAR model.
    
    Args:
        y_init: Initial conditions (P_lag, r)
        A_matrices: List of lag coefficient matrices [(r,r), (r,r), ...]
        T_forecast: Number of steps to forecast
        
    Returns:
        y_pred: Predicted latent states (T_forecast, r)
    """
    P_lag = len(A_matrices)
    r = y_init.shape[1]
    
    # Store predictions
    y_pred = np.zeros((T_forecast, r))
    
    # Buffer for lag terms (P_lag, r)
    y_buffer = y_init.copy()
    
    for t in range(T_forecast):
        # Compute next state: y(t) = A1*y(t-1) + A2*y(t-2) + ... + Ap*y(t-p)
        y_next = np.zeros(r)
        for lag_idx, A in enumerate(A_matrices):
            y_next += y_buffer[-(lag_idx+1)] @ A.T
        
        y_pred[t] = y_next
        
        # Update buffer
        y_buffer = np.vstack([y_buffer[1:], y_next.reshape(1, -1)])
    
    return y_pred


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate ROM-MVAR predictions for test data")
    parser.add_argument("--experiment_name", type=str, default="oscar_production",
                        help="Name of experiment (subdirectory in oscar_output/)")
    args = parser.parse_args()
    
    # Paths
    BASE_DIR = Path("oscar_output") / args.experiment_name
    MVAR_DIR = BASE_DIR / "mvar"
    TEST_DIR = BASE_DIR / "test"
    
    print(f"Loading MVAR model from {MVAR_DIR}...")
    
    # Load POD basis
    pod_data = np.load(MVAR_DIR / "pod_basis.npz")
    U = pod_data["U"]  # (n_space, r)
    
    # Load MVAR model
    mvar_data = np.load(MVAR_DIR / "mvar_model.npz")
    A_matrices_array = mvar_data["A_matrices"]  # (P_lag, r, r)
    A_matrices = [A_matrices_array[i] for i in range(A_matrices_array.shape[0])]
    P_LAG = len(A_matrices)
    
    print(f"POD basis: {U.shape}")
    print(f"MVAR model: {len(A_matrices)} lag matrices, each {A_matrices[0].shape}")
    
    # Load test metadata
    with open(TEST_DIR / "metadata.json", "r") as f:
        test_metadata = json.load(f)
    
    print(f"\nGenerating predictions for {len(test_metadata)} test runs...")
    
    for meta in tqdm(test_metadata, desc="Predictions"):
        run_name = meta["run_name"]
        run_dir = TEST_DIR / run_name
        
        # Load true density
        data = np.load(run_dir / "density_true.npz")
        rho_true = data["rho"]
        times = data["times"]
        T, ny, nx = rho_true.shape
        n_space = nx * ny
        
        # Project to latent space
        rho_flat = rho_true.reshape(T, n_space)
        y_true = rho_flat @ U
        
        # Initialize MVAR with first P_LAG timesteps
        y_init = y_true[:P_LAG]
        
        # Forecast remaining timesteps
        T_forecast = T - P_LAG
        y_pred = mvar_forecast(y_init, A_matrices, T_forecast)
        
        # Combine init + forecast
        y_full = np.vstack([y_init, y_pred])
        
        # Reconstruct density
        rho_pred_flat = y_full @ U.T
        rho_pred = rho_pred_flat.reshape(T, ny, nx)
        
        # Save predicted density
        np.savez(
            run_dir / "density_pred.npz",
            rho=rho_pred,
            xgrid=data["xgrid"],
            ygrid=data["ygrid"],
            times=times
        )
    
    print(f"\n✓ Predictions saved to {TEST_DIR}/test_*/density_pred.npz")
    print(f"\nNow run: python run_visualizations.py --experiment_name {args.experiment_name}")


if __name__ == "__main__":
    main()
