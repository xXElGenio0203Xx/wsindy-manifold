#!/usr/bin/env python3
"""
Regenerate test simulations with T=20s and re-evaluate.
Reuses existing training data, POD basis, and MVAR model.
"""

import yaml
import time
from pathlib import Path
from rectsim.ic_generator import generate_ic_configs
from rectsim.sim_runner import run_simulations_parallel
from rectsim.test_evaluator import evaluate_test_runs
from sklearn.linear_model import Ridge
import numpy as np

# Configuration
EXPERIMENT_NAME = "alvarez_production"
CONFIG_PATH = Path("configs/alvarez_style_production.yaml")
OUTPUT_DIR = Path("oscar_output") / EXPERIMENT_NAME

print("="*80)
print("REGENERATING TEST DATA WITH T=20s")
print("="*80)

# Load config
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Extract configs
BASE_CONFIG = config
test_ic_config = config['test_ic']
test_sim_config = config['test_sim']
rom_config = config['rom']
eval_config = config.get('eval', {'save_time_resolved': True})

DENSITY_NX = config['density']['nx']
DENSITY_NY = config['density']['ny']
DENSITY_BANDWIDTH = config['density']['bandwidth']

# Base config for test (merge test_sim overrides)
BASE_CONFIG_TEST = config.copy()
BASE_CONFIG_TEST['sim'] = config['sim'].copy()
BASE_CONFIG_TEST['sim'].update(test_sim_config)

print(f"\nConfiguration:")
print(f"  Training T: {config['sim']['T']}s")
print(f"  Test T: {BASE_CONFIG_TEST['sim']['T']}s")
print(f"  Forecast window: {config['sim']['T']}s → {BASE_CONFIG_TEST['sim']['T']}s")
print(f"  Extrapolation: {BASE_CONFIG_TEST['sim']['T'] - config['sim']['T']}s ({100*(BASE_CONFIG_TEST['sim']['T']/config['sim']['T'] - 1):.0f}% beyond training)")

# =========================================================================
# STEP 1: Generate test configurations
# =========================================================================
print(f"\n{'='*80}")
print("STEP 1: Generating Test Configurations")
print("="*80)

test_configs = generate_ic_configs(
    train_ic_config={},  # Empty - we only want test
    test_ic_config=test_ic_config,
    is_test=True
)

n_test = len(test_configs)
print(f"\n✓ Generated {n_test} test configurations")

# Count by distribution
test_dist_counts = {}
for cfg in test_configs:
    dist = cfg['distribution']
    test_dist_counts[dist] = test_dist_counts.get(dist, 0) + 1
for dist, count in test_dist_counts.items():
    print(f"   {dist}: {count} runs")

# =========================================================================
# STEP 2: Run test simulations (20 seconds each!)
# =========================================================================
print(f"\n{'='*80}")
print("STEP 2: Running Test Simulations (T=20s)")
print("="*80)

# Delete old test directory
test_dir = OUTPUT_DIR / "test"
if test_dir.exists():
    print(f"Removing old test data: {test_dir}")
    import shutil
    shutil.rmtree(test_dir)

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
print(f"\n✓ Generated {n_test} test runs in {test_time/60:.1f}m")

# =========================================================================
# STEP 3: Load POD and MVAR model
# =========================================================================
print(f"\n{'='*80}")
print("STEP 3: Loading Trained ROM-MVAR Model")
print("="*80)

# Load POD
pod_npz = np.load(OUTPUT_DIR / "mvar" / "pod_basis.npz")
X_mean = np.load(OUTPUT_DIR / "mvar" / "X_train_mean.npy")
pod_data = {
    'U_r': pod_npz['U'],
    'X_mean': X_mean,
    'R_POD': pod_npz['U'].shape[1],
    'singular_values': pod_npz['singular_values']
}
print(f"✓ POD: d={pod_data['R_POD']} modes")

# Load MVAR
mvar_npz = np.load(OUTPUT_DIR / "mvar" / "mvar_model.npz")
mvar_model = Ridge(alpha=float(mvar_npz['alpha']))
mvar_model.coef_ = mvar_npz['A_companion']
mvar_model.intercept_ = 0.0
mvar_model.n_features_in_ = mvar_npz['A_companion'].shape[1]
print(f"✓ MVAR: p={mvar_npz['p']}, alpha={mvar_npz['alpha']}, train R²={mvar_npz['train_r2']:.6f}")

# =========================================================================
# STEP 4: Evaluate on 20s test data
# =========================================================================
print(f"\n{'='*80}")
print("STEP 4: Evaluating ROM-MVAR (8s → 20s forecast)")
print("="*80)

ROM_SUBSAMPLE = rom_config.get('subsample', rom_config.get('rom_subsample', 1))

test_results_df = evaluate_test_runs(
    test_dir=test_dir,
    n_test=n_test,
    base_config_test=BASE_CONFIG_TEST,
    pod_data=pod_data,
    mvar_model=mvar_model,
    density_nx=DENSITY_NX,
    density_ny=DENSITY_NY,
    rom_subsample=ROM_SUBSAMPLE,
    eval_config=eval_config,
    train_T=config['sim']['T']
)

mean_r2 = test_results_df['r2_reconstructed'].mean()
std_r2 = test_results_df['r2_reconstructed'].std()

# =========================================================================
# FINAL SUMMARY
# =========================================================================
total_time = time.time() - start_time

print(f"\n{'='*80}")
print("COMPLETE: 20s TEST DATA REGENERATED")
print("="*80)
print(f"Total time: {total_time/60:.1f}m")
print(f"\nTest Results (12s forecast window):")
print(f"  Mean R²: {mean_r2:.4f} ± {std_r2:.4f}")
print(f"  Best R²: {test_results_df['r2_reconstructed'].max():.4f}")
print(f"  Worst R²: {test_results_df['r2_reconstructed'].min():.4f}")
print(f"\n✓ Results saved to: {OUTPUT_DIR}/test/")
print(f"✓ Ready for visualization: python run_visualizations.py --experiment_name {EXPERIMENT_NAME}")
