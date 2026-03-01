#!/usr/bin/env python3
"""
R² Degradation Experiment: Reuse Existing Models
=================================================

Copies pre-trained model + POD basis from a base experiment,
generates new (very long) test simulations, and evaluates.
This avoids retraining when only the test horizon changes.

Usage:
    python run_degradation_experiment.py \
        --base_experiment ABL8_N200_sqrt_simplex_align_H300_v2 \
        --config configs/DEG1_long_horizon_200s.yaml \
        --experiment_name DEG1_long_horizon_200s
"""

import argparse
import json
import shutil
import time
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from rectsim.config_loader import load_config
from rectsim.ic_generator import generate_test_configs
from rectsim.simulation_runner import run_simulations_parallel
from rectsim.pod_builder import build_pod_basis, save_pod_basis
from rectsim.mvar_trainer import train_mvar_model, save_mvar_model
from rectsim.test_evaluator import evaluate_test_runs
from rectsim.forecast_utils import mvar_forecast_fn_factory


def main():
    parser = argparse.ArgumentParser(description='R² degradation experiment (reuse existing models)')
    parser.add_argument('--base_experiment', type=str, required=True,
                       help='Name of base experiment to copy model from (in oscar_output/)')
    parser.add_argument('--config', type=str, required=True,
                       help='Config YAML (only test_ic and test_sim sections are used)')
    parser.add_argument('--experiment_name', type=str, required=True,
                       help='Name for this degradation experiment')
    parser.add_argument('--retrain', action='store_true',
                       help='Re-train from scratch instead of copying model (for verification)')
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("="*80)
    print("R² DEGRADATION EXPERIMENT")
    print("="*80)
    print(f"\nExperiment: {args.experiment_name}")
    print(f"Base experiment: {args.base_experiment}")
    print(f"Config: {args.config}")
    print(f"Mode: {'RETRAIN' if args.retrain else 'REUSE model from base'}")
    
    # ---------- Load config ----------
    (BASE_CONFIG, DENSITY_NX, DENSITY_NY, DENSITY_BANDWIDTH,
     train_ic_config, test_ic_config, test_sim_config, rom_config, eval_config) = load_config(args.config)
    
    # ---------- Directories ----------
    BASE_DIR = Path(f"oscar_output/{args.base_experiment}")
    OUTPUT_DIR = Path(f"oscar_output/{args.experiment_name}")
    ROM_COMMON_DIR = OUTPUT_DIR / "rom_common"
    MVAR_DIR = OUTPUT_DIR / "MVAR"
    TEST_DIR = OUTPUT_DIR / "test"
    TRAIN_DIR = OUTPUT_DIR / "train"  # Only needed if retraining
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ROM_COMMON_DIR.mkdir(exist_ok=True)
    MVAR_DIR.mkdir(exist_ok=True)
    
    # Save config
    shutil.copy(args.config, OUTPUT_DIR / "config_used.yaml")
    
    # ========================================================================
    # STEP 1: Get model (copy or retrain)
    # ========================================================================
    
    if not args.retrain:
        # ---------- Copy existing model artifacts ----------
        print(f"\n{'='*80}")
        print("STEP 1: Copying model from base experiment")
        print("="*80)
        
        required_files = {
            'rom_common/pod_basis.npz': ROM_COMMON_DIR / 'pod_basis.npz',
            'rom_common/X_train_mean.npy': ROM_COMMON_DIR / 'X_train_mean.npy',
            'MVAR/mvar_model.npz': MVAR_DIR / 'mvar_model.npz',
            'MVAR/test_results.csv': MVAR_DIR / 'test_results_base.csv',  # Keep for reference
        }
        
        # Optional files
        optional_files = {
            'rom_common/shift_align.npz': ROM_COMMON_DIR / 'shift_align.npz',
            'rom_common/latent_dataset.npz': ROM_COMMON_DIR / 'latent_dataset.npz',
            'train/metadata.json': TRAIN_DIR / 'metadata.json',
        }
        
        for rel_src, dst in required_files.items():
            src = BASE_DIR / rel_src
            if not src.exists():
                raise FileNotFoundError(f"Required file not found: {src}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"   ✓ {rel_src}")
        
        for rel_src, dst in optional_files.items():
            src = BASE_DIR / rel_src
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                print(f"   ✓ {rel_src} (optional)")
        
        # Load POD data
        pod_basis = np.load(ROM_COMMON_DIR / 'pod_basis.npz')
        X_mean = np.load(ROM_COMMON_DIR / 'X_train_mean.npy')
        
        R_POD = pod_basis['U'].shape[1]
        pod_data = {
            'U_r': pod_basis['U'],
            'X_mean': X_mean,
            'R_POD': R_POD,
            'shift_align': False,
            'shift_align_data': None,
        }
        
        # Also need density_transform info from config
        pod_data['density_transform'] = rom_config.get('density_transform', 'raw')
        pod_data['density_transform_eps'] = rom_config.get('density_transform_eps', 1e-8)
        
        # Check if shift_align data exists
        shift_align_path = ROM_COMMON_DIR / 'shift_align.npz'
        if shift_align_path.exists():
            sa_data = np.load(shift_align_path, allow_pickle=True)
            pod_data['shift_align'] = True
            pod_data['shift_align_data'] = {
                'ref': sa_data['ref'],
                'shifts': sa_data['shifts'],
                'ref_method': str(sa_data['ref_method']),
                'density_shape_2d': tuple(sa_data['density_shape_2d']),
            }
            print(f"   ✓ Shift alignment loaded (ref_method={sa_data['ref_method']})")
        
        # Load MVAR model
        mvar_model_data = np.load(MVAR_DIR / 'mvar_model.npz')
        P_LAG = int(mvar_model_data['p'])
        
        # Create forecast function
        class MVARPredictor:
            def __init__(self, A_companion, intercept=None):
                self.coef_ = A_companion
                self.intercept_ = intercept
            def predict(self, X):
                result = X @ self.coef_.T
                if self.intercept_ is not None:
                    result += self.intercept_
                return result
        
        intercept = mvar_model_data['intercept'] if 'intercept' in mvar_model_data else None
        mvar_predictor = MVARPredictor(mvar_model_data['A_companion'], intercept)
        forecast_fn = mvar_forecast_fn_factory(mvar_predictor, P_LAG)
        
        rho_before = float(mvar_model_data['rho_before'])
        print(f"\n   Model: MVAR(p={P_LAG}, d={R_POD})")
        print(f"   Spectral radius: {rho_before:.4f}")
        print(f"   Train R²: {float(mvar_model_data['train_r2']):.4f}")
        
    else:
        # ---------- Retrain from scratch ----------
        print(f"\n{'='*80}")
        print("STEP 1: Training from scratch")
        print("="*80)
        
        from rectsim.ic_generator import generate_training_configs
        
        train_configs = generate_training_configs(train_ic_config, BASE_CONFIG)
        n_train = len(train_configs)
        print(f"   Generating {n_train} training runs...")
        
        TRAIN_DIR.mkdir(parents=True, exist_ok=True)
        train_metadata, train_time = run_simulations_parallel(
            configs=train_configs, base_config=BASE_CONFIG,
            output_dir=OUTPUT_DIR, density_nx=DENSITY_NX, density_ny=DENSITY_NY,
            density_bandwidth=DENSITY_BANDWIDTH, is_test=False
        )
        print(f"   ✓ Training sims: {train_time/60:.1f}m")
        
        pod_data = build_pod_basis(TRAIN_DIR, n_train, rom_config)
        save_pod_basis(pod_data, ROM_COMMON_DIR)
        
        mvar_data = train_mvar_model(pod_data, rom_config)
        save_mvar_model(mvar_data, MVAR_DIR)
        
        P_LAG = mvar_data['P_LAG']
        forecast_fn = mvar_forecast_fn_factory(mvar_data['model'], P_LAG)
        
        print(f"   ✓ MVAR trained: R²={mvar_data['r2_train']:.4f}")
    
    # ========================================================================
    # STEP 2: Generate long test simulations
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("STEP 2: Generating Long Test Simulations")
    print("="*80)
    
    test_configs = generate_test_configs(test_ic_config, BASE_CONFIG)
    n_test = len(test_configs)
    
    test_T = test_sim_config.get('T', BASE_CONFIG['sim']['T'])
    train_T = BASE_CONFIG['sim']['T']
    ROM_SUBSAMPLE = rom_config.get('subsample', 1)
    dt = BASE_CONFIG['sim']['dt']
    rom_dt = dt * ROM_SUBSAMPLE
    n_rom_steps = int(test_T / rom_dt)
    n_forecast_steps = int((test_T - train_T) / rom_dt)
    
    print(f"\n   Test duration: {test_T}s")
    print(f"   Training duration: {train_T}s")
    print(f"   ROM timestep: {rom_dt}s (subsample={ROM_SUBSAMPLE})")
    print(f"   Total ROM steps: {n_rom_steps}")
    print(f"   Forecast ROM steps: {n_forecast_steps}")
    print(f"   Test runs: {n_test}")
    for cfg in test_configs:
        print(f"      {cfg['distribution']}: {cfg['label']}")
    
    BASE_CONFIG_TEST = BASE_CONFIG.copy()
    BASE_CONFIG_TEST['sim'] = BASE_CONFIG['sim'].copy()
    BASE_CONFIG_TEST['sim']['T'] = test_T
    
    test_metadata, test_time = run_simulations_parallel(
        configs=test_configs, base_config=BASE_CONFIG_TEST,
        output_dir=OUTPUT_DIR, density_nx=DENSITY_NX, density_ny=DENSITY_NY,
        density_bandwidth=DENSITY_BANDWIDTH, is_test=True
    )
    
    print(f"\n   ✓ Generated {n_test} test runs ({test_time/60:.1f}m)")
    
    # ========================================================================
    # STEP 3: Evaluate MVAR on long test trajectories
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("STEP 3: Evaluating MVAR on Long Horizons")
    print("="*80)
    
    test_results_df = evaluate_test_runs(
        test_dir=TEST_DIR,
        n_test=n_test,
        base_config_test=BASE_CONFIG_TEST,
        pod_data=pod_data,
        forecast_fn=forecast_fn,
        lag=P_LAG,
        density_nx=DENSITY_NX,
        density_ny=DENSITY_NY,
        rom_subsample=ROM_SUBSAMPLE,
        eval_config=eval_config,
        train_T=train_T,
        model_name="MVAR"
    )
    
    # Save results
    test_results_df.to_csv(MVAR_DIR / "test_results.csv", index=False)
    
    mean_r2 = test_results_df['r2_reconstructed'].mean()
    print(f"\n   ✓ Mean R² (reconstructed): {mean_r2:.4f}")
    
    # ========================================================================
    # STEP 4: Summary
    # ========================================================================
    
    total_time = time.time() - start_time
    
    summary = {
        'experiment_name': args.experiment_name,
        'base_experiment': args.base_experiment if not args.retrain else None,
        'mode': 'reuse' if not args.retrain else 'retrain',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_test': n_test,
        'test_T': test_T,
        'train_T': train_T,
        'forecast_rom_steps': n_forecast_steps,
        'rom_subsample': ROM_SUBSAMPLE,
        'rom_dt': rom_dt,
        'mean_r2_test': float(mean_r2),
        'per_test_r2': {
            row['run_name']: float(row['r2_reconstructed']) 
            for _, row in test_results_df.iterrows()
        },
        'total_time_minutes': total_time / 60,
        'mvar': {
            'p_lag': int(P_LAG),
            'spectral_radius': float(mvar_model_data['rho_before']) if not args.retrain else None,
        },
        'pod': {
            'r_pod': int(pod_data['R_POD']),
        },
    }
    
    with open(OUTPUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print("DEGRADATION EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\n   Output: {OUTPUT_DIR}")
    print(f"   Mean R²: {mean_r2:.4f}")
    print(f"   Time: {total_time/60:.1f}m")
    
    print(f"\n   Per-test results:")
    for _, row in test_results_df.iterrows():
        print(f"      {row.get('ic_type', row.get('run_name','?'))}: R²={row['r2_reconstructed']:.4f}")
    
    print(f"\n   Key files:")
    print(f"      {OUTPUT_DIR}/summary.json")
    print(f"      {MVAR_DIR}/test_results.csv")
    print(f"      test/test_XXX/r2_vs_time_mvar.csv  (time-resolved degradation)")


if __name__ == "__main__":
    main()
