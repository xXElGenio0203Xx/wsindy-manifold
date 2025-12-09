#!/usr/bin/env python3
"""
Verification script for unified pipeline output format.
Checks that output matches stable pipeline structure.
"""

import numpy as np
from pathlib import Path
import json
import sys

def check_file_exists(filepath, required=True):
    """Check if file exists"""
    exists = filepath.exists()
    status = "✓" if exists else ("✗ MISSING" if required else "○ optional")
    print(f"{status} {filepath.relative_to(filepath.parents[3])}")
    return exists

def check_npz_keys(filepath, expected_keys, required=True):
    """Check keys in .npz file"""
    if not filepath.exists():
        if required:
            print(f"  ✗ File missing: {filepath.name}")
        return False
    
    data = np.load(filepath)
    actual_keys = set(data.keys())
    expected_set = set(expected_keys)
    
    if actual_keys == expected_set:
        print(f"  ✓ Keys correct: {sorted(expected_keys)}")
        return True
    else:
        missing = expected_set - actual_keys
        extra = actual_keys - expected_set
        print(f"  ✗ Key mismatch in {filepath.name}:")
        if missing:
            print(f"    Missing: {sorted(missing)}")
        if extra:
            print(f"    Extra: {sorted(extra)}")
        return False

def check_json_keys(filepath, expected_keys):
    """Check keys in JSON file"""
    if not filepath.exists():
        print(f"  ✗ File missing: {filepath.name}")
        return False
    
    with open(filepath) as f:
        data = json.load(f)
    
    actual_keys = set(data.keys())
    expected_set = set(expected_keys)
    
    if actual_keys == expected_set:
        print(f"  ✓ Keys correct: {sorted(expected_keys)}")
        return True
    else:
        missing = expected_set - actual_keys
        extra = actual_keys - expected_set
        print(f"  ✗ Key mismatch in {filepath.name}:")
        if missing:
            print(f"    Missing: {sorted(missing)}")
        if extra:
            print(f"    Extra: {sorted(extra)}")
        return False

def verify_unified_output(experiment_name):
    """Verify unified pipeline output structure"""
    
    output_dir = Path("oscar_output") / experiment_name
    
    if not output_dir.exists():
        print(f"✗ Output directory not found: {output_dir}")
        return False
    
    print(f"\n{'='*80}")
    print(f"Verifying output format for: {experiment_name}")
    print(f"{'='*80}\n")
    
    all_checks_passed = True
    
    # Check training run structure
    print("Training Run (train_000):")
    print("-" * 40)
    train_dir = output_dir / "train" / "train_000"
    
    if not train_dir.exists():
        print(f"✗ Training directory not found: {train_dir}")
        all_checks_passed = False
    else:
        # Required files
        all_checks_passed &= check_file_exists(train_dir / "trajectory.npz")
        all_checks_passed &= check_file_exists(train_dir / "density.npz")
        
        # Check trajectory.npz keys
        all_checks_passed &= check_npz_keys(
            train_dir / "trajectory.npz",
            ['traj', 'vel', 'times']
        )
        
        # Check density.npz keys
        all_checks_passed &= check_npz_keys(
            train_dir / "density.npz",
            ['rho', 'xgrid', 'ygrid', 'times']
        )
    
    # Check test run structure
    print("\nTest Run (test_000):")
    print("-" * 40)
    test_dir = output_dir / "test" / "test_000"
    
    if not test_dir.exists():
        print(f"✗ Test directory not found: {test_dir}")
        all_checks_passed = False
    else:
        # Required files
        all_checks_passed &= check_file_exists(test_dir / "trajectory.npz")
        all_checks_passed &= check_file_exists(test_dir / "density_true.npz")
        all_checks_passed &= check_file_exists(test_dir / "density_pred.npz")
        all_checks_passed &= check_file_exists(test_dir / "metrics_summary.json")
        
        # Optional files
        check_file_exists(test_dir / "r2_vs_time.csv", required=False)
        
        # Check trajectory.npz keys
        all_checks_passed &= check_npz_keys(
            test_dir / "trajectory.npz",
            ['traj', 'vel', 'times']
        )
        
        # Check density_true.npz keys
        all_checks_passed &= check_npz_keys(
            test_dir / "density_true.npz",
            ['rho', 'xgrid', 'ygrid', 'times']
        )
        
        # Check density_pred.npz keys
        all_checks_passed &= check_npz_keys(
            test_dir / "density_pred.npz",
            ['rho', 'xgrid', 'ygrid', 'times']
        )
        
        # Check metrics_summary.json keys
        expected_metrics = [
            'r2_recon', 'r2_latent', 'r2_pod',
            'rmse_recon', 'rmse_latent', 'rmse_pod',
            'rel_error_recon', 'rel_error_pod',
            'max_mass_violation'
        ]
        all_checks_passed &= check_json_keys(
            test_dir / "metrics_summary.json",
            expected_metrics
        )
    
    # Check MVAR directory structure
    print("\nMVAR Directory:")
    print("-" * 40)
    mvar_dir = output_dir / "mvar"
    
    if not mvar_dir.exists():
        print(f"✗ MVAR directory not found: {mvar_dir}")
        all_checks_passed = False
    else:
        # Required files
        all_checks_passed &= check_file_exists(mvar_dir / "X_train_mean.npy")
        all_checks_passed &= check_file_exists(mvar_dir / "pod_basis.npz")
        all_checks_passed &= check_file_exists(mvar_dir / "mvar_model.npz")
        
        # Check pod_basis.npz keys
        expected_pod_keys = [
            'U', 'singular_values', 'all_singular_values',
            'total_energy', 'explained_energy', 'energy_ratio', 'cumulative_ratio'
        ]
        all_checks_passed &= check_npz_keys(
            mvar_dir / "pod_basis.npz",
            expected_pod_keys
        )
        
        # Check mvar_model.npz keys
        expected_mvar_keys = [
            'A_matrices', 'A_companion', 'p', 'r', 'alpha',
            'train_r2', 'train_rmse', 'rho_before', 'rho_after'
        ]
        all_checks_passed &= check_npz_keys(
            mvar_dir / "mvar_model.npz",
            expected_mvar_keys
        )
    
    # Summary
    print(f"\n{'='*80}")
    if all_checks_passed:
        print("✓ All checks PASSED - Output format matches stable pipeline")
        print("='*80}\n")
        return True
    else:
        print("✗ Some checks FAILED - Output format needs corrections")
        print(f"{'='*80}\n")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_unified_output.py <experiment_name>")
        print("Example: python verify_unified_output.py unified_format_test")
        sys.exit(1)
    
    experiment_name = sys.argv[1]
    success = verify_unified_output(experiment_name)
    sys.exit(0 if success else 1)
