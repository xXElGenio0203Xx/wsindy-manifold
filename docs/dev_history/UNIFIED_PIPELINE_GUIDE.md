# Unified ROM-MVAR Pipeline Guide

## Overview

The **unified pipeline** (`run_unified_mvar_pipeline.py`) replaces three separate pipelines:
- ~~`run_stable_mvar_pipeline.py`~~ (1198 lines)
- ~~`run_robust_mvar_pipeline.py`~~ (729 lines)
- ~~`run_gaussians_pipeline.py`~~ (642 lines)

**Benefits:**
- ✅ Single pipeline for all experiments
- ✅ Consistent behavior across all IC types
- ✅ All features from all three pipelines
- ✅ Fixed `fixed_modes` priority issue
- ✅ No more confusion about which pipeline to use

## Features Inherited from Each Pipeline

### From `run_stable_mvar_pipeline.py`:
- ✅ Eigenvalue stability enforcement (optional)
- ✅ Time-resolved R² evaluation
- ✅ Comprehensive evaluation config support

### From `run_robust_mvar_pipeline.py`:
- ✅ Mixed distribution support (gaussian, uniform, ring, two_clusters)
- ✅ Interpolation/extrapolation test configurations
- ✅ Strong regularization options

### From `run_gaussians_pipeline.py`:
- ✅ Custom Gaussian experiments (variance/center variations)
- ✅ Simple config format support
- ✅ Spatial translation generalization tests

## Usage

```bash
python run_unified_mvar_pipeline.py \
    --config configs/your_config.yaml \
    --experiment_name your_experiment_name
```

## Configuration Format

The unified pipeline supports **both** config formats:

### Format 1: Mixed Distributions (most experiments)

```yaml
# Simulation parameters
sim:
  N: 40
  Lx: 15.0
  Ly: 15.0
  T: 2.0
  dt: 0.1
  # ... other sim params

# ROM configuration
rom:
  fixed_modes: 25           # PRIORITY 1: Use exactly 25 modes
  pod_energy: 0.995         # PRIORITY 2: Fallback if fixed_modes not specified
  mvar_lag: 20
  ridge_alpha: 1.0e-6
  subsample: 1              # Temporal subsampling (also accepts 'rom_subsample')
  eigenvalue_threshold: 0.95  # OPTIONAL: Scale eigenvalues if max > threshold

# Training IC configurations
train_ic:
  type: "mixed_comprehensive"
  
  gaussian:
    enabled: true
    n_runs: 200
    positions_x: [3.75, 7.5, 11.25]
    positions_y: [3.75, 7.5, 11.25]
    variances: [0.5, 1.0, 2.0]
    n_samples_per_config: 2
  
  uniform:
    enabled: true
    n_runs: 100              # Can use 'n_runs' or 'n_samples'
  
  ring:
    enabled: true
    n_runs: 50
    radii: [2.0, 3.0, 4.0]
    widths: [0.3, 0.6]
    n_samples_per_config: 5
  
  two_clusters:
    enabled: true
    n_runs: 50
    separations: [3.0, 4.5, 6.0]
    sigmas: [0.8, 1.5]
    n_samples_per_config: 5

# Test IC configurations
test_ic:
  type: "mixed_test_comprehensive"
  test_T: 10.0              # Test duration (can differ from training)
  
  gaussian:
    enabled: true
    n_runs: 20
    test_positions_x: [5.5, 9.0]      # Interpolation positions
    test_positions_y: [5.5, 9.0]
    test_variances: [1.5]
    extrapolation_positions: [[2.0, 2.0], [13.0, 13.0]]  # Extrapolation
    extrapolation_variance: [1.5]
  
  uniform:
    enabled: true
    n_runs: 5
  
  ring:
    enabled: true
    n_runs: 8
    test_radii: [2.5, 3.5]
    test_widths: [0.45]
    n_samples_per_config: 2
  
  two_clusters:
    enabled: true
    n_runs: 7
    test_separations: [3.75, 5.25]
    test_sigmas: [1.1]
    extrapolation_separations: [2.0, 10.5]
    extrapolation_sigma: [1.1]

# Evaluation configuration (optional)
evaluation:
  save_time_resolved: true   # Generate r2_vs_time.csv for each test run
  forecast_start: 2.0         # Start time of forecast evaluation
  forecast_end: 10.0          # End time of forecast evaluation
```

### Format 2: Custom Gaussian Experiments

```yaml
# Simulation parameters (same as above)
sim:
  N: 100
  Lx: 20.0
  Ly: 20.0
  T: 10.0
  dt: 0.1

# ROM configuration (same as above)
rom:
  fixed_modes: 50
  mvar_lag: 5
  ridge_alpha: 1e-4

# Training: Same center, varying variances
train_ic:
  center: [10.0, 10.0]
  variances: [0.5, 1.0, 2.0, 4.0, 8.0]
  n_samples_per_variance: 3

# Testing: Different centers, fixed variance
test_ic:
  centers: [[5.0, 5.0], [15.0, 5.0], [5.0, 15.0], [15.0, 15.0]]
  variance: 2.0
  n_samples_per_center: 2
```

## Parameter Naming Compatibility

The pipeline accepts **both names** for maximum compatibility:

| Parameter | Standard Name | Alternative Name | Priority |
|-----------|---------------|------------------|----------|
| POD modes | `fixed_modes` | `fixed_d` | `fixed_modes` first |
| POD energy | `pod_energy` | - | - |
| Subsampling | `subsample` | `rom_subsample` | `subsample` first |
| Test runs | `n_runs` | `n_samples` | `n_runs` first |

## ROM Mode Selection Priority

**CRITICAL**: The pipeline now correctly prioritizes mode selection:

1. **`fixed_modes: N`** (highest priority)
   - Use exactly N modes, regardless of energy captured
   - Example: `fixed_modes: 25` → Always use 25 modes

2. **`fixed_d: N`** (backward compatibility)
   - Same as `fixed_modes`, checked if `fixed_modes` not present

3. **`pod_energy: 0.XX`** (fallback)
   - Use enough modes to capture XX% energy
   - Only used if neither `fixed_modes` nor `fixed_d` specified

**Example:**
```yaml
rom:
  fixed_modes: 25      # ← This will be used (exactly 25 modes)
  pod_energy: 0.995    # ← This will be IGNORED
```

## Output Structure

```
oscar_output/
└── your_experiment_name/
    ├── config_used.yaml            # Config file copy
    ├── summary.json                # Experiment summary
    ├── train/
    │   ├── metadata.json           # Training run metadata
    │   ├── index_mapping.csv       # Run ID mappings
    │   ├── train_000/
    │   │   ├── density.npz         # Density field time series
    │   │   └── order_parameters.csv
    │   ├── train_001/
    │   └── ...
    ├── mvar/
    │   ├── pod_basis.npz           # POD basis (U_r, S, mean)
    │   └── mvar_model.npz          # MVAR coefficients
    └── test/
        ├── metadata.json
        ├── index_mapping.csv
        ├── test_results.csv        # Summary R² metrics
        ├── test_000/
        │   ├── density.npz
        │   ├── order_parameters.csv
        │   ├── predictions.npz     # Predictions + ground truth
        │   └── r2_vs_time.csv      # Time-resolved R² (if enabled)
        ├── test_001/
        └── ...
```

## Example Workflows

### 1. Standard Mixed Distribution Experiment

```bash
# Config: configs/best_run_extended_test.yaml
python run_unified_mvar_pipeline.py \
    --config configs/best_run_extended_test.yaml \
    --experiment_name best_run_v2
```

**Config highlights:**
- 400 training runs (mixed gaussian/uniform/ring/two_cluster)
- 40 test runs with interpolation/extrapolation
- `fixed_modes: 25` (exactly 25 POD modes)
- `mvar_lag: 20`
- Time-resolved evaluation enabled

### 2. Stability-Enforced Experiment

```bash
# Config: configs/stable_mvar_v2.yaml
python run_unified_mvar_pipeline.py \
    --config configs/stable_mvar_v2.yaml \
    --experiment_name stable_v3
```

**Config highlights:**
- `eigenvalue_threshold: 0.95` (scale eigenvalues if > 0.95)
- `ridge_alpha: 0.5` (very strong regularization)
- `mvar_lag: 2` (reduced complexity)

### 3. Custom Gaussian Experiment

```bash
# Config: configs/gaussian_variance_study.yaml
python run_unified_mvar_pipeline.py \
    --config configs/gaussian_variance_study.yaml \
    --experiment_name gauss_variance
```

**Config highlights:**
- Training: Same center, 5 different variances
- Testing: 4 different centers, fixed variance
- Tests spatial translation generalization

## Migration from Old Pipelines

If you have existing configs for the old pipelines, they should work **as-is** with the unified pipeline:

```bash
# Old command:
# python run_stable_mvar_pipeline.py --config configs/stable_mvar_v2.yaml --experiment_name test

# New command (same config):
python run_unified_mvar_pipeline.py --config configs/stable_mvar_v2.yaml --experiment_name test
```

**No config changes needed** - the unified pipeline handles all formats!

## Differences from Old Pipelines

### What Changed:
1. **Fixed `fixed_modes` bug**: Now correctly prioritizes `fixed_modes` over `pod_energy`
2. **Unified naming**: Accepts both naming conventions for compatibility
3. **Flexible IC loading**: Handles both mixed distributions and custom Gaussian formats
4. **All features enabled**: Time-resolved, eigenvalue scaling, etc. all available

### What Stayed the Same:
- Config file format (both formats supported)
- Output directory structure
- File naming conventions
- All functionality preserved

## Troubleshooting

### Issue: Too many/too few POD modes selected

**Cause**: Config has `pod_energy: 0.995` but you want a fixed number of modes.

**Solution**: Add `fixed_modes: N` to your config:
```yaml
rom:
  fixed_modes: 25        # ← Add this line
  pod_energy: 0.995      # ← Keep or remove (will be ignored)
```

### Issue: "No training samples" warning

**Cause**: `mvar_lag` ≥ number of timesteps in training trajectories.

**Solution**: Either:
- Increase training duration: `sim.T: 10.0` (instead of 2.0)
- Decrease lag: `rom.mvar_lag: 5` (instead of 20)
- Reduce subsampling: `rom.subsample: 1` (instead of 2)

**Target**: Lag window should be ≤ 20% of training duration:
- Good: lag=5, T=10s → window = 0.5s / 10s = 5% ✓
- Bad: lag=20, T=2s → window = 2.0s / 2s = 100% ❌

### Issue: Model unstable (eigenvalues > 1)

**Cause**: Weak regularization or high-dimensional ROM.

**Solution**: Add stability enforcement:
```yaml
rom:
  eigenvalue_threshold: 0.95   # Scale eigenvalues to max 0.95
  ridge_alpha: 0.1             # Increase regularization
```

## Performance Tips

1. **Parallelization**: Pipeline automatically uses up to 16 CPU cores
2. **Memory**: Each run needs ~50-100MB RAM (estimate: N_runs × 100MB)
3. **Time**: Typical experiment (400 train + 40 test) takes ~30-60 minutes
4. **Subsampling**: Use `subsample: 2` to reduce memory/time by 2×

## Next Steps

After running the unified pipeline, use the **visualization pipeline**:

```bash
python run_visualizations.py --experiment_name your_experiment_name
```

This will generate:
- POD energy spectrum plots
- Best run animations
- R² summary plots
- Time-resolved degradation analysis (if enabled)

## Support

**Old pipelines will still work** but are now deprecated. We recommend:
1. Test the unified pipeline with your existing configs
2. Verify outputs match expected results
3. Switch to using the unified pipeline for all future experiments
4. Eventually delete the old pipeline files once confident

**All bug fixes and new features will only be added to the unified pipeline.**
