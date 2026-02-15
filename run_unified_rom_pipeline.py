#!/usr/bin/env python3
"""
Unified ROM Pipeline with MVAR and LSTM Support
================================================

This pipeline supports training and evaluating BOTH MVAR and LSTM ROMs:
- Computes microsims, densities, and POD ONCE
- Builds shared latent dataset ONCE
- Trains MVAR if enabled → outputs to MVAR/
- Trains LSTM if enabled → outputs to LSTM/
- Both use identical evaluation metrics and visualizations

Key Features:
- Model-agnostic evaluation pipeline
- Shared POD basis (rom_common/)
- Parallel model outputs (MVAR/ and LSTM/)
- Fair comparison with identical data and metrics
"""

import numpy as np
import torch
from pathlib import Path
import json
import time
import argparse
import yaml
import shutil
import os
import sys

# Add src to path for rom modules
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import existing pipeline modules
from rectsim.config_loader import load_config
from rectsim.ic_generator import generate_training_configs, generate_test_configs
from rectsim.simulation_runner import run_simulations_parallel
from rectsim.pod_builder import build_pod_basis, save_pod_basis
from rectsim.mvar_trainer import train_mvar_model, save_mvar_model
from rectsim.test_evaluator import evaluate_test_runs
from rectsim.forecast_utils import mvar_forecast_fn_factory

# Import ROM utilities
from rectsim.rom_data_utils import build_latent_dataset, build_multistep_latent_dataset
from rom.lstm_rom import LatentLSTMROM, train_lstm_rom, lstm_forecast_fn_factory, load_lstm_model

# Import runtime analysis
from rectsim.runtime_analyzer import RuntimeAnalyzer, compute_mvar_params, compute_lstm_params


def main():
    parser = argparse.ArgumentParser(description='Unified ROM pipeline (MVAR + LSTM)')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name')
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("="*80)
    print("UNIFIED ROM PIPELINE (MVAR + LSTM)")
    print("="*80)
    print(f"\nExperiment: {args.experiment_name}")
    print(f"Config: {args.config}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load configuration
    (BASE_CONFIG, DENSITY_NX, DENSITY_NY, DENSITY_BANDWIDTH,
     train_ic_config, test_ic_config, test_sim_config, rom_config, eval_config) = load_config(args.config)
    
    # Setup directory structure
    OUTPUT_DIR = Path(f"oscar_output/{args.experiment_name}")
    ROM_COMMON_DIR = OUTPUT_DIR / "rom_common"
    MVAR_DIR = OUTPUT_DIR / "MVAR"
    LSTM_DIR = OUTPUT_DIR / "LSTM"
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ROM_COMMON_DIR.mkdir(exist_ok=True)
    MVAR_DIR.mkdir(exist_ok=True)
    LSTM_DIR.mkdir(exist_ok=True)
    
    # Save config for reference
    shutil.copy(args.config, OUTPUT_DIR / "config_used.yaml")
    
    # Initialize runtime analyzer
    runtime_analyzer = RuntimeAnalyzer()
    runtime_profiles = []
    
    # Check which models are enabled
    models_cfg = rom_config.get('models', {})
    mvar_enabled = models_cfg.get('mvar', {}).get('enabled', True)  # Default True for backward compat
    lstm_enabled = models_cfg.get('lstm', {}).get('enabled', False)
    
    print(f"\nModel Configuration:")
    print(f"   MVAR: {'ENABLED' if mvar_enabled else 'DISABLED'}")
    print(f"   LSTM: {'ENABLED' if lstm_enabled else 'DISABLED'}")
    
    if not mvar_enabled and not lstm_enabled:
        raise ValueError("At least one ROM model (MVAR or LSTM) must be enabled!")
    
    # Extract configuration
    print(f"\nSimulation Configuration:")
    print(f"   N: {BASE_CONFIG['sim']['N']}")
    print(f"   T: {BASE_CONFIG['sim']['T']}s")
    print(f"   dt: {BASE_CONFIG['sim']['dt']}s")
    print(f"   Domain: {BASE_CONFIG['sim']['Lx']}×{BASE_CONFIG['sim']['Ly']}")
    print(f"   Density: {DENSITY_NX}×{DENSITY_NY}")
    
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
    # STEP 2: Build POD Basis (Shared by MVAR and LSTM)
    # =========================================================================
    
    print(f"\n{'='*80}")
    print("STEP 2: Building Shared POD Basis")
    print("="*80)
    
    TRAIN_DIR = OUTPUT_DIR / "train"
    
    # Build POD basis
    pod_data = build_pod_basis(TRAIN_DIR, n_train, rom_config)
    
    # Save to rom_common directory
    save_pod_basis(pod_data, ROM_COMMON_DIR)
    
    print(f"\n✓ POD basis saved to {ROM_COMMON_DIR}/")
    print(f"   Latent dimension: {pod_data['R_POD']}")
    print(f"   Training timesteps: {pod_data['T_rom']}")
    
    # =========================================================================
    # STEP 3: Build Shared Latent Dataset
    # =========================================================================
    
    print(f"\n{'='*80}")
    print("STEP 3: Building Shared Latent Dataset")
    print("="*80)
    
    # Extract latent trajectories from POD data
    X_latent = pod_data['X_latent']  # [M * T_rom, R_POD]
    M = pod_data['M']
    T_rom = pod_data['T_rom']
    R_POD = pod_data['R_POD']
    
    # Reshape to list of trajectories
    y_trajs = []
    for m in range(M):
        start_idx = m * T_rom
        end_idx = (m + 1) * T_rom
        y_m = X_latent[start_idx:end_idx, :]  # [T_rom, R_POD]
        y_trajs.append(y_m)
    
    # Determine lag (use MVAR lag if enabled, else LSTM lag)
    if mvar_enabled:
        lag = models_cfg['mvar'].get('lag', 5)
    else:
        lag = models_cfg['lstm'].get('lag', 20)
    
    print(f"\nUsing lag={lag} for windowed dataset")
    print(f"   (This will be shared by both MVAR and LSTM)")
    
    # Build windowed dataset
    X_all, Y_all = build_latent_dataset(y_trajs, lag=lag)
    
    print(f"\n✓ Windowed dataset built:")
    print(f"   X_all: {X_all.shape}  [N_samples, lag, d]")
    print(f"   Y_all: {Y_all.shape}  [N_samples, d]")
    print(f"   Total samples: {X_all.shape[0]:,}")

    # Build multi-step targets for LSTM supervised rollout loss
    Y_multi = None
    X_lstm = X_all  # default: same dataset for LSTM and MVAR
    Y_lstm = Y_all
    if lstm_enabled:
        lstm_ms_cfg = models_cfg.get('lstm', {})
        ms_enabled = lstm_ms_cfg.get('multistep_loss', False)
        ms_k = lstm_ms_cfg.get('multistep_k', 5)
        if ms_enabled and ms_k > 1:
            X_lstm, Y_multi = build_multistep_latent_dataset(y_trajs, lag=lag, k_steps=ms_k)
            Y_lstm = Y_multi[:, 0, :]  # 1-step target aligned with X_lstm
            print(f"\n✓ Multi-step targets built:")
            print(f"   X_lstm:  {X_lstm.shape}  (aligned with Y_multi)")
            print(f"   Y_multi: {Y_multi.shape}  [N_samples, k={ms_k}, d]")
            print(f"   (Dropped {X_all.shape[0] - X_lstm.shape[0]} samples near trajectory ends)")
    
    # Save dataset to rom_common
    np.savez(
        ROM_COMMON_DIR / "latent_dataset.npz",
        X_all=X_all,
        Y_all=Y_all,
        lag=lag
    )
    print(f"   Saved to {ROM_COMMON_DIR}/latent_dataset.npz")
    
    # =========================================================================
    # STEP 4: Train MVAR ROM (if enabled)
    # =========================================================================
    
    mvar_data = None
    mvar_training_time = None
    if mvar_enabled:
        print(f"\n{'='*80}")
        print("STEP 4a: Training MVAR ROM")
        print("="*80)
        
        # Train MVAR using existing trainer (with timing)
        with runtime_analyzer.time_operation('mvar_training') as timer:
            mvar_data = train_mvar_model(pod_data, rom_config)
        mvar_training_time = timer.elapsed
        
        save_mvar_model(mvar_data, MVAR_DIR)
        
        print(f"\n✓ MVAR model saved to {MVAR_DIR}/")
        print(f"   Lag: {mvar_data['P_LAG']}")
        print(f"   Ridge α: {mvar_data['RIDGE_ALPHA']}")
        print(f"   Training R²: {mvar_data['r2_train']:.4f}")
        print(f"   Training time: {mvar_training_time:.2f}s")
    lstm_data = None
    lstm_training_time = None
    if lstm_enabled:
        print(f"\n{'='*80}")
        print("STEP 4b: Training LSTM ROM")
        print("="*80)
        
        # Extract LSTM config
        lstm_config = rom_config['models']['lstm']
        print(f"   Hidden units: {lstm_config['hidden_units']}, Num layers: {lstm_config['num_layers']}")
        print(f"   Lag: {lstm_config['lag']}")
        
        # Train LSTM (with timing) - pass rom_config dict directly
        with runtime_analyzer.time_operation('lstm_training') as timer:
            lstm_model_path, lstm_val_loss = train_lstm_rom(
                X_all=X_lstm,
                Y_all=Y_lstm,
                config={'rom': rom_config},  # Wrap in expected structure
                out_dir=str(LSTM_DIR),
                Y_multi=Y_multi,
            )
        lstm_training_time = timer.elapsed
        
        # Store LSTM data for later use
        lstm_data = {
            'model_path': lstm_model_path,
            'val_loss': lstm_val_loss,
            'lag': lstm_config['lag'],
            'hidden_units': lstm_config['hidden_units'],
            'num_layers': lstm_config['num_layers']
        }
        
        print(f"\n✓ LSTM model saved to {LSTM_DIR}/")
        print(f"   Validation loss: {lstm_val_loss:.6f}")
        print(f"   Training time: {lstm_training_time:.2f}s")
    
    # =========================================================================
    # STEP 4.5: Runtime Benchmarking
    # =========================================================================
    
    if mvar_enabled or lstm_enabled:
        print(f"\n{'='*80}")
        print("STEP 4.5: Runtime Benchmarking")
        print("="*80)
        
        # Get a sample initial condition window for benchmarking
        # Need lag timesteps to initialize the forecast
        z0_window = X_latent[:lag, :]  # First lag timesteps from training data
        benchmark_steps = 100
        benchmark_trials = 50
        
        # Benchmark MVAR
        if mvar_enabled:
            print(f"\n🔬 Benchmarking MVAR inference...")
            
            # Load MVAR model for benchmarking
            mvar_model_data = np.load(MVAR_DIR / "mvar_model.npz")
            
            # Create a simple predictor class for the forecast function
            class MVARPredictor:
                def __init__(self, A_companion):
                    self.coef_ = A_companion
                
                def predict(self, X):
                    """Predict next step(s) using MVAR coefficients."""
                    return X @ self.coef_.T
            
            mvar_predictor = MVARPredictor(mvar_model_data['A_companion'])
            
            # Create forecast function with lag
            mvar_forecast_fn = mvar_forecast_fn_factory(mvar_predictor, lag=lag)
            
            # Count parameters
            mvar_params = compute_mvar_params(mvar_model_data)
            
            # Benchmark inference
            mvar_inference_times = runtime_analyzer.benchmark_inference(
                mvar_forecast_fn,
                z0_window,
                n_steps=benchmark_steps,
                n_trials=benchmark_trials
            )
            
            # Build profile
            mvar_profile = runtime_analyzer.build_profile(
                model_name='MVAR',
                training_time=mvar_training_time,
                inference_times=mvar_inference_times,
                model_params=mvar_params,
                latent_dim=R_POD,
                n_forecast_steps=benchmark_steps,
                lag=int(mvar_data['P_LAG'])
            )
            
            runtime_profiles.append(mvar_profile)
            runtime_analyzer.save_profile(mvar_profile, MVAR_DIR / "runtime_profile.json")
            
            print(f"✓ MVAR runtime profile saved")
            print(f"   Parameters: {mvar_params:,}")
            print(f"   Inference: {mvar_inference_times['single_step'].mean_seconds*1000:.2f}ms per step")
        
        # Benchmark LSTM
        if lstm_enabled:
            print(f"\n🔬 Benchmarking LSTM inference...")
            
            # Load LSTM model for benchmarking (new format with normalization)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            try:
                lstm_model, lstm_input_mean, lstm_input_std = load_lstm_model(
                    str(LSTM_DIR), device=str(device))
            except (ValueError, Exception):
                # Fallback for legacy format
                lstm_model = LatentLSTMROM(
                    d=R_POD,
                    hidden_units=lstm_data['hidden_units'],
                    num_layers=lstm_data['num_layers']
                )
                state = torch.load(lstm_data['model_path'], map_location=device, weights_only=False)
                if isinstance(state, dict) and 'state_dict' in state:
                    lstm_model.load_state_dict(state['state_dict'])
                else:
                    lstm_model.load_state_dict(state)
                lstm_model.to(device)
                lstm_model.eval()
                lstm_input_mean, lstm_input_std = None, None
            
            # Create forecast function
            lstm_forecast_fn = lstm_forecast_fn_factory(lstm_model, lstm_input_mean, lstm_input_std)
            
            # Count parameters
            lstm_params = compute_lstm_params(lstm_model)
            
            # Benchmark inference
            lstm_inference_times = runtime_analyzer.benchmark_inference(
                lstm_forecast_fn,
                z0_window,
                n_steps=benchmark_steps,
                n_trials=benchmark_trials
            )
            
            # Build profile
            lstm_profile = runtime_analyzer.build_profile(
                model_name='LSTM',
                training_time=lstm_training_time,
                inference_times=lstm_inference_times,
                model_params=lstm_params,
                latent_dim=R_POD,
                n_forecast_steps=benchmark_steps,
                lag=int(lstm_data['lag']),
                hidden_dim=lstm_data['hidden_units'],
                n_layers=lstm_data['num_layers']
            )
            
            runtime_profiles.append(lstm_profile)
            runtime_analyzer.save_profile(lstm_profile, LSTM_DIR / "runtime_profile.json")
            
            print(f"✓ LSTM runtime profile saved")
            print(f"   Parameters: {lstm_params:,}")
            print(f"   Inference: {lstm_inference_times['single_step'].mean_seconds*1000:.2f}ms per step")
        
        # Generate comparison if both models are enabled
        if mvar_enabled and lstm_enabled:
            comparison = runtime_analyzer.compare_models(runtime_profiles)
            runtime_analyzer.save_comparison(comparison, OUTPUT_DIR / "runtime_comparison.json")
            
            print(f"\n✓ Runtime comparison saved: {OUTPUT_DIR}/runtime_comparison.json")
            print(f"\n📊 Quick Comparison:")
            print(f"   Training time ratio (LSTM/MVAR): {comparison['training_time_ratio']['LSTM_vs_MVAR']:.2f}x")
            print(f"   Inference speedup (LSTM vs MVAR): {comparison['inference_speedup']['LSTM_vs_MVAR']:.2f}x")
            print(f"   Parameter ratio (LSTM/MVAR): {comparison['parameter_ratio']['LSTM_vs_MVAR']:.2f}x")
            print(f"   Winners: Training={comparison['winners']['fastest_training']}, "
                  f"Inference={comparison['winners']['fastest_inference']}, "
                  f"Memory={comparison['winners']['smallest_memory']}")
    
    # =========================================================================
    # STEP 5: Generate Test Data
    # =========================================================================
    
    print(f"\n{'='*80}")
    print("STEP 5: Generating Test Data")
    print("="*80)
    
    test_configs = generate_test_configs(test_ic_config, BASE_CONFIG)
    n_test = len(test_configs)
    
    # Initialize test results variables (will be set if evaluation runs)
    mean_r2_mvar = None
    mean_r2_lstm = None
    n_test_evaluated = 0
    
    if n_test == 0:
        print("\n⚠️  No test configurations specified. Skipping evaluation.")
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
        
        # Get test duration
        test_T = test_sim_config.get('T', test_ic_config.get('test_T', BASE_CONFIG['sim']['T']))
        
        # Create test config
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
        # STEP 7: Evaluate MVAR ROM (if enabled)
        # =====================================================================
        
        if mvar_enabled:
            print(f"\n{'='*80}")
            print("STEP 6a: Evaluating MVAR ROM")
            print("="*80)
            
            TEST_DIR = OUTPUT_DIR / "test"
            ROM_SUBSAMPLE = rom_config.get('subsample', rom_config.get('rom_subsample', 1))
            
            # Create MVAR forecast function
            mvar_lag = mvar_data['P_LAG']
            mvar_forecast_fn = mvar_forecast_fn_factory(mvar_data['model'], mvar_lag)
            
            test_results_df = evaluate_test_runs(
                test_dir=TEST_DIR,
                n_test=n_test,
                base_config_test=BASE_CONFIG_TEST,
                pod_data=pod_data,
                forecast_fn=mvar_forecast_fn,
                lag=mvar_lag,
                density_nx=DENSITY_NX,
                density_ny=DENSITY_NY,
                rom_subsample=ROM_SUBSAMPLE,
                eval_config=eval_config,
                train_T=BASE_CONFIG['sim']['T'],
                model_name="MVAR"
            )
            
            # Save results to MVAR directory
            test_results_df.to_csv(MVAR_DIR / "test_results.csv", index=False)
            
            mean_r2_mvar = test_results_df['r2_reconstructed'].mean()
            print(f"\n✓ MVAR evaluation complete")
            print(f"   Mean R² (reconstructed): {mean_r2_mvar:.4f}")
            print(f"   Results: {MVAR_DIR}/test_results.csv")
        
        # =====================================================================
        # STEP 6b: Evaluate LSTM ROM (if enabled)
        # =====================================================================
        
        if lstm_enabled:
            print(f"\n{'='*80}")
            print("STEP 6b: Evaluating LSTM ROM")
            print("="*80)
            
            TEST_DIR = OUTPUT_DIR / "test"
            ROM_SUBSAMPLE = rom_config.get('subsample', rom_config.get('rom_subsample', 1))
            
            # Load LSTM model (new format with normalization + residual)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            lstm_lag = lstm_data['lag']
            
            try:
                lstm_model, lstm_input_mean, lstm_input_std = load_lstm_model(
                    str(LSTM_DIR), device=str(device))
            except (ValueError, Exception):
                # Fallback for legacy format
                lstm_hidden = lstm_data['hidden_units']
                lstm_layers = lstm_data['num_layers']
                lstm_model = LatentLSTMROM(d=R_POD, hidden_units=lstm_hidden, num_layers=lstm_layers)
                state = torch.load(lstm_data['model_path'], map_location=device, weights_only=False)
                if isinstance(state, dict) and 'state_dict' in state:
                    lstm_model.load_state_dict(state['state_dict'])
                else:
                    lstm_model.load_state_dict(state)
                lstm_model.to(device)
                lstm_model.eval()
                lstm_input_mean, lstm_input_std = None, None
            
            # Create LSTM forecast function
            lstm_forecast_fn = lstm_forecast_fn_factory(lstm_model, lstm_input_mean, lstm_input_std)
            
            test_results_df = evaluate_test_runs(
                test_dir=TEST_DIR,
                n_test=n_test,
                base_config_test=BASE_CONFIG_TEST,
                pod_data=pod_data,
                forecast_fn=lstm_forecast_fn,
                lag=lstm_lag,
                density_nx=DENSITY_NX,
                density_ny=DENSITY_NY,
                rom_subsample=ROM_SUBSAMPLE,
                eval_config=eval_config,
                train_T=BASE_CONFIG['sim']['T'],
                model_name="LSTM"
            )
            
            # Save results to LSTM directory
            test_results_df.to_csv(LSTM_DIR / "test_results.csv", index=False)
            
            mean_r2_lstm = test_results_df['r2_reconstructed'].mean()
            print(f"\n✓ LSTM evaluation complete")
            print(f"   Mean R² (reconstructed): {mean_r2_lstm:.4f}")
            print(f"   Results: {LSTM_DIR}/test_results.csv")
        
        n_test_evaluated = n_test
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\nTotal time: {total_time/60:.1f}m")
    print(f"Output directory: {OUTPUT_DIR}")
    
    print(f"\nDirectory structure:")
    print(f"   {ROM_COMMON_DIR}/ - Shared POD basis and dataset")
    if mvar_enabled:
        print(f"   {MVAR_DIR}/ - MVAR model and results")
    if lstm_enabled:
        print(f"   {LSTM_DIR}/ - LSTM model and results")
    
    print(f"\nKey files:")
    print(f"   {OUTPUT_DIR}/config_used.yaml")
    print(f"   {ROM_COMMON_DIR}/pod_basis.npz")
    print(f"   {ROM_COMMON_DIR}/latent_dataset.npz")
    
    if mvar_enabled:
        print(f"   {MVAR_DIR}/mvar_model.npz")
        if n_test > 0:
            print(f"   {MVAR_DIR}/test_results.csv")
    
    if lstm_enabled:
        print(f"   {LSTM_DIR}/lstm_state_dict.pt")
        print(f"   {LSTM_DIR}/training_log.csv")
    
    # Save final summary
    summary = {
        'experiment_name': args.experiment_name,
        'config': args.config,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_train': n_train,
        'n_test': n_test if n_test > 0 else 0,
        'r_pod': int(pod_data['R_POD']),
        'lag': int(lag),
        'models_enabled': {
            'mvar': mvar_enabled,
            'lstm': lstm_enabled
        },
        'total_time_minutes': total_time / 60
    }
    
    # Add runtime analysis to summary
    if runtime_profiles:
        summary['runtime_analysis'] = {
            'profiles': [profile.to_dict() for profile in runtime_profiles]
        }
        
        # Add comparison if both models were profiled
        if len(runtime_profiles) == 2:
            comparison = runtime_analyzer.compare_models(runtime_profiles)
            summary['runtime_analysis']['comparison'] = comparison
    
    if mvar_enabled and mvar_data is not None:
        summary['mvar'] = {
            'p_lag': int(mvar_data['P_LAG']),
            'ridge_alpha': float(mvar_data['RIDGE_ALPHA']),
            'r2_train': float(mvar_data['r2_train'])
        }
        if mean_r2_mvar is not None:
            summary['mvar']['mean_r2_test'] = float(mean_r2_mvar)
    
    if lstm_enabled and lstm_data is not None:
        summary['lstm'] = {
            'hidden_units': lstm_data['hidden_units'],
            'num_layers': lstm_data['num_layers'],
            'val_loss': float(lstm_data['val_loss'])
        }
        if mean_r2_lstm is not None:
            summary['lstm']['mean_r2_test'] = float(mean_r2_lstm)
    
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Pipeline completed successfully!")
    print(f"   Summary: {OUTPUT_DIR}/summary.json")
    if runtime_profiles:
        print(f"   Runtime profiles: {len(runtime_profiles)} model(s) benchmarked")
        if len(runtime_profiles) == 2:
            print(f"   Runtime comparison: {OUTPUT_DIR}/runtime_comparison.json")
    print("="*80)


if __name__ == "__main__":
    main()
