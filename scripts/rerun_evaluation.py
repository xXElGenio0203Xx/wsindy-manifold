#!/usr/bin/env python3
"""
Re-run just the test evaluation step with fixed forecasting.
Uses existing training data, POD basis, and MVAR model.
"""

import numpy as np
import yaml
from pathlib import Path
from sklearn.linear_model import Ridge
from rectsim.test_evaluator import evaluate_test_runs

# Load configuration and data from alvarez_production
OUTPUT_DIR = Path("oscar_output/alvarez_production")

# Load config
with open(OUTPUT_DIR / "config_used.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Extract relevant configs
BASE_CONFIG_TRAIN = config  # Full config with training T
BASE_CONFIG_TEST = config.copy()
# Merge test_sim into sim (test_sim only overrides T)
BASE_CONFIG_TEST['sim'] = config['sim'].copy()
BASE_CONFIG_TEST['sim'].update(config['test_sim'])

rom_config = config['rom']
eval_config = config.get('eval', {'save_time_resolved': True})

# Load POD data
print("Loading POD basis...")
pod_npz = np.load(OUTPUT_DIR / "mvar" / "pod_basis.npz")
X_mean = np.load(OUTPUT_DIR / "mvar" / "X_train_mean.npy")

# Reconstruct pod_data dict with expected keys
pod_data = {
    'U_r': pod_npz['U'],
    'X_mean': X_mean,
    'R_POD': pod_npz['U'].shape[1],
    'singular_values': pod_npz['singular_values']
}

# Load MVAR model
print("Loading MVAR model...")
mvar_npz = np.load(OUTPUT_DIR / "mvar" / "mvar_model.npz")

# Reconstruct sklearn model
mvar_model = Ridge(alpha=float(mvar_npz['alpha']))
mvar_model.coef_ = mvar_npz['A_companion']
mvar_model.intercept_ = 0.0
mvar_model.n_features_in_ = mvar_npz['A_companion'].shape[1]

print(f"  Loaded MVAR: p={mvar_npz['p']}, r={mvar_npz['r']}, alpha={mvar_npz['alpha']}")
print(f"  Train R²: {mvar_npz['train_r2']:.6f}")

# Get parameters
DENSITY_NX = config['density']['nx']
DENSITY_NY = config['density']['ny']
ROM_SUBSAMPLE = rom_config.get('subsample', rom_config.get('rom_subsample', 1))

# Count test runs
test_dir = OUTPUT_DIR / "test"
n_test = len([d for d in test_dir.iterdir() if d.is_dir() and d.name.startswith('test_')])

print(f"\nRe-evaluating {n_test} test runs...")
print(f"Training T: {BASE_CONFIG_TRAIN['sim']['T']}s")
print(f"Test T: {BASE_CONFIG_TEST['sim']['T']}s")
print(f"Forecast window: {BASE_CONFIG_TRAIN['sim']['T']}s → {BASE_CONFIG_TEST['sim']['T']}s")

# Run evaluation with fixed forecasting
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
    train_T=BASE_CONFIG_TRAIN['sim']['T']
)

# Print results
mean_r2 = test_results_df['r2_reconstructed'].mean()
std_r2 = test_results_df['r2_reconstructed'].std()

print(f"\n{'='*80}")
print("EVALUATION COMPLETE")
print("="*80)
print(f"Mean R²: {mean_r2:.4f} ± {std_r2:.4f}")
print(f"\nTop 5 runs by R²:")
top_5 = test_results_df.nlargest(5, 'r2_reconstructed')[['r2_reconstructed', 'r2_latent', 'r2_pod']]
print(top_5.to_string())

# Check one prediction file
test_005_pred = np.load(test_dir / "test_005" / "density_pred.npz")
print(f"\nSample prediction shape (test_005):")
print(f"  rho: {test_005_pred['rho'].shape}")
print(f"  times: {test_005_pred['times'].shape}")
print(f"  Expected: (~20, 64, 64) for 2.0s forecast (8.0s → 10.0s)")
print(f"\n✓ Videos should now have {test_005_pred['rho'].shape[0]} frames")
