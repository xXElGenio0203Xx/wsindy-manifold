#!/usr/bin/env python3
"""
Extend test horizon to 1.5x training (12s instead of 10s).
Only regenerates test runs and re-evaluates. Uses existing training data and ROM.
"""

import yaml
import time
from pathlib import Path
from config_loader import load_config
from ic_generator import generate_test_configs
from simulation_runner import run_simulations_parallel
from rectsim.test_evaluator import evaluate_test_runs
from rectsim.rom_mvar import load_mvar_model
import numpy as np

# Configuration
EXPERIMENT_NAME = "alvarez_production"
CONFIG_FILE = "configs/alvarez_style_production.yaml"
OUTPUT_DIR = Path(f"oscar_output/{EXPERIMENT_NAME}")

print("="*80)
print(f"EXTENDING TEST HORIZON TO 1.5x TRAINING")
print("="*80)

# Load configuration
(BASE_CONFIG, DENSITY_NX, DENSITY_NY, DENSITY_BANDWIDTH,
 train_ic_config, test_ic_config, test_sim_config, rom_config, eval_config) = load_config(CONFIG_FILE)

train_T = BASE_CONFIG['sim']['T']
test_T = test_sim_config.get('T', train_T * 1.5)

print(f"\nTraining horizon: {train_T}s")
print(f"New test horizon: {test_T}s (1.5x)")
print(f"Forecast window: {train_T}s → {test_T}s ({test_T - train_T}s)")

# =========================================================================
# STEP 1: Regenerate Test Data with Extended Horizon
# =========================================================================

print(f"\n{'='*80}")
print("STEP 1: Regenerating Test Data (Extended to 12s)")
print("="*80)

test_configs = generate_test_configs(test_ic_config, BASE_CONFIG)
n_test = len(test_configs)

# Override test duration
BASE_CONFIG_TEST = BASE_CONFIG.copy()
BASE_CONFIG_TEST['sim'] = BASE_CONFIG['sim'].copy()
BASE_CONFIG_TEST['sim']['T'] = test_T

print(f"\nTest configurations: {n_test} runs")

# Count by distribution
test_dist_counts = {}
for cfg in test_configs:
    dist = cfg['distribution']
    test_dist_counts[dist] = test_dist_counts.get(dist, 0) + 1
for dist, count in test_dist_counts.items():
    print(f"   {dist}: {count} runs")

start_time = time.time()

test_metadata, test_time = run_simulations_parallel(
    configs=test_configs,
    base_config=BASE_CONFIG_TEST,
    output_dir=OUTPUT_DIR,
    density_nx=DENSITY_NX,
    density_ny=DENSITY_NY,
    density_bandwidth=DENSITY_BANDWIDTH,
    is_test=True
)

print(f"\n✓ Generated {n_test} test runs (T={test_T}s)")
print(f"   Time: {test_time/60:.1f}m")

# =========================================================================
# STEP 2: Re-evaluate with Extended Forecast Window
# =========================================================================

print(f"\n{'='*80}")
print("STEP 2: ROM-MVAR Evaluation (Extended Forecast)")
print("="*80)

# Load existing ROM artifacts
print("\nLoading existing ROM artifacts...")
MVAR_DIR = OUTPUT_DIR / "mvar"
pod_data = dict(np.load(MVAR_DIR / "pod_basis.npz"))
mvar_data = load_mvar_model(MVAR_DIR)

print(f"   POD: d={pod_data['R_POD']} modes")
print(f"   MVAR: order={mvar_data['order']}")

# Evaluate
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
    train_T=train_T
)

mean_r2 = test_results_df['r2_reconstructed'].mean()
std_r2 = test_results_df['r2_reconstructed'].std()

# =========================================================================
# SUMMARY
# =========================================================================

total_time = time.time() - start_time

print(f"\n{'='*80}")
print("EXTENSION COMPLETE")
print("="*80)
print(f"\nTest horizon: {test_T}s (was 10s)")
print(f"Forecast window: {test_T - train_T}s (was 2s)")
print(f"\nMean R²: {mean_r2:.4f} ± {std_r2:.4f}")
print(f"\nTop 5 runs by R²:")
top_5 = test_results_df.nlargest(5, 'r2_reconstructed')[['run', 'r2_reconstructed', 'r2_latent', 'r2_pod']]
print(top_5.to_string(index=False))

print(f"\nTotal time: {total_time/60:.1f}m")
print(f"\n✓ Ready for visualization: python run_visualizations.py --experiment_name {EXPERIMENT_NAME}")

# Check sample prediction
test_005_pred = np.load(TEST_DIR / "test_005" / "density_pred.npz")
print(f"\nSample prediction (test_005):")
print(f"   Shape: {test_005_pred['rho'].shape}")
print(f"   Expected: (~40, 64, 64) for 4s forecast")
