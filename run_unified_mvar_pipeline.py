#!/usr/bin/env python3
"""
Unified ROM-MVAR Pipeline
==========================

Comprehensive pipeline supporting ALL experiment types:
- Multiple IC distributions (Gaussian, uniform, ring, two-cluster)
- Custom Gaussian experiments (variance/center variations)
- Flexible ROM configuration (fixed modes or energy threshold)
- Optional stability enforcement (eigenvalue scaling)
- Time-resolved evaluation
- Strong regularization options

This unified pipeline replaces:
- run_stable_mvar_pipeline.py
- run_robust_mvar_pipeline.py
- run_gaussians_pipeline.py
"""

import numpy as np
from pathlib import Path
import json
import time
import argparse
import yaml
import shutil

# Import pipeline modules
from rectsim.config_loader import load_config
from rectsim.ic_generator import generate_training_configs, generate_test_configs
from rectsim.simulation_runner import run_simulations_parallel
from rectsim.pod_builder import build_pod_basis, save_pod_basis
from rectsim.mvar_trainer import train_mvar_model, save_mvar_model
from rectsim.test_evaluator import evaluate_test_runs


def main():
    parser = argparse.ArgumentParser(description='Unified ROM-MVAR pipeline')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name')
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("="*80)
    print("UNIFIED ROM-MVAR PIPELINE")
    print("="*80)
    print(f"\nExperiment: {args.experiment_name}")
    print(f"Config: {args.config}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load configuration
    (BASE_CONFIG, DENSITY_NX, DENSITY_NY, DENSITY_BANDWIDTH,
     train_ic_config, test_ic_config, test_sim_config, rom_config, eval_config) = load_config(args.config)
    
    OUTPUT_DIR = Path(f"oscar_output/{args.experiment_name}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save config for reference
    shutil.copy(args.config, OUTPUT_DIR / "config_used.yaml")
    
    # Extract MVAR config (support new nested structure and old flat structure)
    if 'models' in rom_config and 'mvar' in rom_config['models']:
        mvar_lag = rom_config['models']['mvar'].get('lag', 'auto')
        ridge_alpha = rom_config['models']['mvar'].get('ridge_alpha', 1e-6)
    else:
        # Backward compatibility
        mvar_lag = rom_config.get('mvar_lag', 'auto')
        ridge_alpha = rom_config.get('ridge_alpha', 1e-6)
    
    print(f"\nConfiguration:")
    print(f"   N: {BASE_CONFIG['sim']['N']}")
    print(f"   T: {BASE_CONFIG['sim']['T']}s")
    print(f"   dt: {BASE_CONFIG['sim']['dt']}s")
    print(f"   Domain: {BASE_CONFIG['sim']['Lx']}×{BASE_CONFIG['sim']['Ly']}")
    print(f"   Density: {DENSITY_NX}×{DENSITY_NY}")
    print(f"   ROM lag: {mvar_lag}")
    print(f"   Ridge α: {ridge_alpha}")
    
    # =========================================================================
    # STEP 1: Generate Training Data
    # =========================================================================
    
    print(f"\n{'='*80}")
    print("STEP 1: Generating Training Data")
    print("="*80)
    
    train_configs = generate_training_configs(train_ic_config, BASE_CONFIG)
    n_train = len(train_configs)
    
    print(f"\nTraining configurations:")
    print(f"   Total runs: {n_train}")
    
    # Count by distribution
    dist_counts = {}
    for cfg in train_configs:
        dist = cfg['distribution']
        dist_counts[dist] = dist_counts.get(dist, 0) + 1
    for dist, count in dist_counts.items():
        print(f"   {dist}: {count} runs")
    
    train_metadata, train_time = run_simulations_parallel(
        configs=train_configs,
        base_config=BASE_CONFIG,
        output_dir=OUTPUT_DIR,
        density_nx=DENSITY_NX,
        density_ny=DENSITY_NY,
        density_bandwidth=DENSITY_BANDWIDTH,
        is_test=False
    )
    
    print(f"\n✓ Generated {n_train} training runs")
    print(f"   Time: {train_time/60:.1f}m")
    
    # =========================================================================
    # STEP 2: Build POD Basis and Train MVAR
    # =========================================================================
    
    print(f"\n{'='*80}")
    print("STEP 2: Global POD and MVAR Training")
    print("="*80)
    
    TRAIN_DIR = OUTPUT_DIR / "train"
    MVAR_DIR = OUTPUT_DIR / "mvar"
    
    # Build POD basis
    pod_data = build_pod_basis(TRAIN_DIR, n_train, rom_config)
    save_pod_basis(pod_data, MVAR_DIR)
    
    # Train MVAR model
    mvar_data = train_mvar_model(pod_data, rom_config)
    save_mvar_model(mvar_data, MVAR_DIR)
    
    print(f"\n✓ ROM artifacts saved to {MVAR_DIR}/")
    
    # =========================================================================
    # STEP 3: Generate Test Data
    # =========================================================================
    
    print(f"\n{'='*80}")
    print("STEP 3: Generating Test Data")
    print("="*80)
    
    test_configs = generate_test_configs(test_ic_config, BASE_CONFIG)
    n_test = len(test_configs)
    
    if n_test == 0:
        print("\n⚠️  No test configurations specified. Skipping test evaluation.")
    else:
        print(f"\nTest configurations:")
        print(f"   Total runs: {n_test}")
        
        # Count by distribution
        test_dist_counts = {}
        for cfg in test_configs:
            dist = cfg['distribution']
            test_dist_counts[dist] = test_dist_counts.get(dist, 0) + 1
        for dist, count in test_dist_counts.items():
            print(f"   {dist}: {count} runs")
        
        # Get test duration (may differ from training)
        # Check test_sim.T first, then test_ic.test_T, then default to training T
        test_T = test_sim_config.get('T', test_ic_config.get('test_T', BASE_CONFIG['sim']['T']))
        
        # Temporarily override T for test runs
        BASE_CONFIG_TEST = BASE_CONFIG.copy()
        BASE_CONFIG_TEST['sim'] = BASE_CONFIG['sim'].copy()
        BASE_CONFIG_TEST['sim']['T'] = test_T
        
        print(f"\nTest duration: {test_T}s (train was {BASE_CONFIG['sim']['T']}s)")
        
        test_metadata, test_time = run_simulations_parallel(
            configs=test_configs,
            base_config=BASE_CONFIG_TEST,
            output_dir=OUTPUT_DIR,
            density_nx=DENSITY_NX,
            density_ny=DENSITY_NY,
            density_bandwidth=DENSITY_BANDWIDTH,
            is_test=True
        )
        
        print(f"\n✓ Generated {n_test} test runs")
        print(f"   Time: {test_time/60:.1f}m")
        
        # =====================================================================
        # STEP 4: Test Evaluation
        # =====================================================================
        
        print(f"\n{'='*80}")
        print("STEP 4: ROM-MVAR Test Evaluation")
        print("="*80)
        
        TEST_DIR = OUTPUT_DIR / "test"
        ROM_SUBSAMPLE = rom_config.get('subsample', rom_config.get('rom_subsample', 1))
        
        test_results_df = evaluate_test_runs(
            test_dir=TEST_DIR,
            n_test=n_test,
            base_config_test=BASE_CONFIG_TEST,
            pod_data=pod_data,
            mvar_model=mvar_data['model'],
            density_nx=DENSITY_NX,
            density_ny=DENSITY_NY,
            rom_subsample=ROM_SUBSAMPLE,
            eval_config=eval_config,
            train_T=BASE_CONFIG['sim']['T']
        )
        
        mean_r2_recon = test_results_df['r2_reconstructed'].mean()
        mean_r2_latent = test_results_df['r2_latent'].mean()
        mean_r2_pod = test_results_df['r2_pod'].mean()
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\nTotal time: {total_time/60:.1f}m")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nKey files:")
    print(f"   {OUTPUT_DIR}/config_used.yaml")
    print(f"   {OUTPUT_DIR}/train/metadata.json")
    print(f"   {OUTPUT_DIR}/mvar/pod_basis.npz")
    print(f"   {OUTPUT_DIR}/mvar/mvar_model.npz")
    if n_test > 0:
        print(f"   {OUTPUT_DIR}/test/test_results.csv")
    
    # Save final summary
    summary = {
        'experiment_name': args.experiment_name,
        'config': args.config,
        'n_train': n_train,
        'n_test': n_test if n_test > 0 else 0,
        'r_pod': int(pod_data['R_POD']),
        'p_lag': int(mvar_data['P_LAG']),
        'ridge_alpha': float(mvar_data['RIDGE_ALPHA']),
        'r2_train': float(mvar_data['r2_train']),
        'total_time_minutes': total_time / 60
    }
    
    if n_test > 0:
        summary.update({
            'mean_r2_reconstructed': float(mean_r2_recon),
            'mean_r2_latent': float(mean_r2_latent),
            'mean_r2_pod': float(mean_r2_pod)
        })
    
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Pipeline completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
