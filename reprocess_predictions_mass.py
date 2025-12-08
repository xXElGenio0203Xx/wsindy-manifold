#!/usr/bin/env python3
"""
Reprocess existing predictions to add:
1. Mass conservation enforcement
2. Order parameters from density
3. Mass violation metrics

This script updates predictions in-place without re-running simulations.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

def reprocess_experiment(experiment_name):
    """Reprocess predictions for a given experiment."""
    
    print("="*80)
    print(f"Reprocessing: {experiment_name}")
    print("="*80)
    
    test_dir = Path(f"oscar_output/{experiment_name}/test")
    
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        return
    
    # Find all test runs
    test_runs = sorted([d for d in test_dir.iterdir() if d.is_dir() and d.name.startswith("test_")])
    
    print(f"\nFound {len(test_runs)} test runs")
    
    all_mass_violations = []
    reprocessed = 0
    
    for run_dir in tqdm(test_runs, desc="Reprocessing"):
        try:
            # Load true and predicted densities
            true_path = run_dir / "density_true.npz"
            pred_path = run_dir / "density_pred.npz"
            
            if not true_path.exists() or not pred_path.exists():
                print(f"  Skipping {run_dir.name}: missing density files")
                continue
            
            true_data = np.load(true_path)
            pred_data = np.load(pred_path)
            
            rho_true = true_data['rho']
            rho_pred = pred_data['rho']
            xgrid = true_data['xgrid']
            ygrid = true_data['ygrid']
            times = true_data['times']
            
            T = rho_true.shape[0]
            
            # Compute cell area
            dx = xgrid[1] - xgrid[0] if len(xgrid) > 1 else 1.0
            dy = ygrid[1] - ygrid[0] if len(ygrid) > 1 else 1.0
            cell_area = dx * dy
            
            # ENFORCE MASS CONSERVATION
            mass_true_initial = np.sum(rho_true[0]) * cell_area
            
            # Rescale each predicted timestep
            for t in range(T):
                mass_pred_t = np.sum(rho_pred[t]) * cell_area
                if mass_pred_t > 1e-10:
                    rho_pred[t] *= (mass_true_initial / mass_pred_t)
            
            # Compute mass conservation metrics
            mass_true_t = np.sum(rho_true, axis=(1, 2)) * cell_area
            mass_pred_t = np.sum(rho_pred, axis=(1, 2)) * cell_area
            rel_mass_error = np.abs(mass_pred_t - mass_true_initial) / mass_true_initial
            max_mass_violation = np.max(rel_mass_error)
            all_mass_violations.append(max_mass_violation)
            
            # Compute order parameters
            order_true = np.array([np.std(rho_true[t]) for t in range(T)])
            order_pred = np.array([np.std(rho_pred[t]) for t in range(T)])
            
            # Save updated prediction
            np.savez_compressed(
                pred_path,
                rho=rho_pred,
                xgrid=pred_data['xgrid'],
                ygrid=pred_data['ygrid'],
                times=pred_data['times']
            )
            
            # Save order parameters
            order_df = pd.DataFrame({
                't': times,
                'order_true': order_true,
                'order_pred': order_pred,
                'mass_true': mass_true_t,
                'mass_pred': mass_pred_t,
                'mass_error_rel': rel_mass_error
            })
            order_df.to_csv(run_dir / "order_params_density.csv", index=False)
            
            reprocessed += 1
            
        except Exception as e:
            print(f"  Error processing {run_dir.name}: {e}")
            continue
    
    # Print summary
    print(f"\n✓ Reprocessed {reprocessed}/{len(test_runs)} runs")
    
    if all_mass_violations:
        mean_violation = np.mean(all_mass_violations)
        max_violation = np.max(all_mass_violations)
        print(f"\nMass conservation after correction:")
        print(f"  Mean violation: {mean_violation*100:.4f}%")
        print(f"  Max violation: {max_violation*100:.4f}%")
        print(f"  (These should be ~0% after rescaling)")
    
    print(f"\n✓ Order parameters saved to: order_params_density.csv")
    print(f"✓ Predictions updated with mass conservation")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reprocess predictions with mass conservation')
    parser.add_argument('--experiment_name', type=str, required=True,
                       help='Experiment name (e.g., robust_mvar_v1)')
    args = parser.parse_args()
    
    reprocess_experiment(args.experiment_name)
    
    print("\n" + "="*80)
    print("Now run visualizations to see order parameters and mass plots:")
    print(f"  python run_visualizations.py --experiment_name {args.experiment_name}")
    print("="*80)
