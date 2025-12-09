# Unified Pipeline Output Format Updates

## Overview

Updated `run_unified_mvar_pipeline.py` to produce **identical output format** as `run_stable_mvar_pipeline.py` for backward compatibility with visualization tools and downstream analysis scripts.

## Changes Made

### 1. Training/Test Simulation Output (lines ~380-440)

**File Structure Changes:**
- ✅ Added `trajectory.npz` saving (required by visualization at line 239)
- ✅ Changed density key from `'density'` → `'rho'` (stable format)
- ✅ Added spatial grids: `xgrid`, `ygrid` to all density files
- ✅ Test runs now save as `density_true.npz` (not `density.npz`)

**Code Changes:**
```python
# Before:
result = simulate_backend(config)
np.savez_compressed(run_dir / "density.npz", density=density_movie, times=times)

# After:
result = simulate_backend(config, rng)
np.savez_compressed(run_dir / "trajectory.npz", traj=traj, vel=vel, times=times)
np.savez_compressed(run_dir / filename, rho=density_movie, xgrid=xgrid, ygrid=ygrid, times=times)
# filename = "density_true.npz" for test, "density.npz" for training
```

**Impact:**
- Visualization pipeline can now load trajectory data
- Density files use consistent naming convention
- Spatial coordinates available for plotting

---

### 2. POD/MVAR Model Saving (lines ~670-730)

**File Structure Changes:**
- ✅ Added separate `X_train_mean.npy` file (was embedded in pod_basis.npz)
- ✅ Changed POD basis keys to match stable pipeline
- ✅ Changed MVAR model keys to match stable pipeline

**POD Basis Key Changes:**
```python
# Before:
np.savez(MVAR_DIR / "pod_basis.npz", 
    U_r=U_r, 
    X_mean=X_mean)

# After:
np.save(MVAR_DIR / "X_train_mean.npy", X_mean)  # Separate file
np.savez_compressed(MVAR_DIR / "pod_basis.npz",
    U=U_r,                              # U_r → U
    singular_values=S[:R_POD],          # Added
    all_singular_values=S,              # Added
    total_energy=total_energy,          # Added
    explained_energy=energy_captured,   # Added
    energy_ratio=energy_captured,       # Added
    cumulative_ratio=cumulative_energy[:R_POD]  # Added
)
```

**MVAR Model Key Changes:**
```python
# Before:
np.savez(MVAR_DIR / "mvar_model.npz",
    coef=mvar_model.coef_,
    p_lag=P_LAG,
    R_POD=R_POD,
    ridge_alpha=RIDGE_ALPHA,
    train_r2=train_r2)

# After:
A_matrices = mvar_model.coef_.T.reshape(P_LAG, R_POD, R_POD)  # Reshape to (p, d, d)
np.savez_compressed(MVAR_DIR / "mvar_model.npz",
    A_matrices=A_matrices,              # coef → A_matrices (reshaped)
    A_companion=A_companion,            # Added companion form
    p=P_LAG,                            # p_lag → p
    r=R_POD,                            # R_POD → r
    alpha=RIDGE_ALPHA,                  # ridge_alpha → alpha
    train_r2=train_r2,                  # Kept
    train_rmse=train_rmse,              # Added
    rho_before=rho_before,              # Added for diagnostics
    rho_after=rho_after                 # Added for diagnostics
)
```

**Impact:**
- POD basis now includes full energy spectrum analysis
- MVAR model coefficients in standard (p, d, d) tensor format
- Additional diagnostic information preserved

---

### 3. Training Data Loading (lines ~550-563)

**Key Change:**
```python
# Before:
density = data['density']

# After:
density = data['rho']  # Match new training output format
```

**Impact:**
- Consistent with new density.npz format from training runs

---

### 4. Test Evaluation Output (lines ~820-980)

**File Structure Changes:**
- ✅ Load test data from `density_true.npz` (not `density.npz`)
- ✅ Use `'rho'` key (not `'density'`)
- ✅ Save predictions as `density_pred.npz` (not `predictions.npz`)
- ✅ Added `metrics_summary.json` file (required by visualization)
- ✅ Optional: `r2_vs_time.csv` for time-resolved analysis

**Metrics Saved:**
```json
{
  "r2_recon": 0.xxxx,      // Reconstructed physical space R²
  "r2_latent": 0.xxxx,     // Latent space R²
  "r2_pod": 0.xxxx,        // POD reconstruction quality R²
  "rmse_recon": 0.xxxx,    // Reconstruction RMSE
  "rmse_latent": 0.xxxx,   // Latent RMSE
  "rmse_pod": 0.xxxx,      // POD RMSE
  "rel_error_recon": 0.xxxx,  // Relative error (normalized)
  "rel_error_pod": 0.xxxx,    // POD relative error
  "max_mass_violation": 0.xxxx  // Conservation violation
}
```

**Code Changes:**
```python
# Before:
test_data = np.load(test_run_dir / "density.npz")
test_density = test_data['density']
np.savez_compressed(test_run_dir / "predictions.npz", ...)

# After:
test_data = np.load(test_run_dir / "density_true.npz")
test_density = test_data['rho']
with open(test_run_dir / "metrics_summary.json", 'w') as f:
    json.dump(metrics_dict, f, indent=2)
np.savez_compressed(test_run_dir / "density_pred.npz",
    rho=pred_physical, xgrid=xgrid, ygrid=ygrid, times=forecast_times)
```

**Impact:**
- Visualization pipeline can load test predictions (line 204)
- Metrics available in both JSON (summary) and CSV (time-resolved)
- Consistent naming: density_true.npz vs density_pred.npz

---

## File Structure Comparison

### Stable Pipeline (Original):
```
oscar_output/{experiment}/
├── train/
│   └── train_000/
│       ├── trajectory.npz         (traj, vel, times)
│       └── density.npz            (rho, xgrid, ygrid, times)
├── test/
│   └── test_000/
│       ├── trajectory.npz         (traj, vel, times)
│       ├── density_true.npz       (rho, xgrid, ygrid, times)
│       ├── density_pred.npz       (rho, xgrid, ygrid, times)
│       ├── metrics_summary.json   (9 metrics)
│       └── r2_vs_time.csv         (optional)
└── mvar/
    ├── X_train_mean.npy           (standalone)
    ├── pod_basis.npz              (U, singular_values, ...)
    └── mvar_model.npz             (A_matrices, p, r, ...)
```

### Unified Pipeline (Now Updated):
```
✓ IDENTICAL to stable pipeline structure above
```

---

## Verification

Use the provided verification script:

```bash
# Run unified pipeline (with small subset for quick test)
python run_unified_mvar_pipeline.py \
    --config configs/best_run_extended_test.yaml \
    --experiment_name unified_format_test

# Verify output format
python verify_unified_output.py unified_format_test

# Test visualization compatibility
python run_visualizations.py --experiment_name unified_format_test
```

Expected output:
```
================================================================================
Verifying output format for: unified_format_test
================================================================================

Training Run (train_000):
----------------------------------------
✓ train/train_000/trajectory.npz
✓ train/train_000/density.npz
  ✓ Keys correct: ['rho', 'times', 'xgrid', 'ygrid']

Test Run (test_000):
----------------------------------------
✓ test/test_000/trajectory.npz
✓ test/test_000/density_true.npz
✓ test/test_000/density_pred.npz
✓ test/test_000/metrics_summary.json
  ✓ Keys correct: ['max_mass_violation', 'r2_latent', 'r2_pod', ...]

MVAR Directory:
----------------------------------------
✓ mvar/X_train_mean.npy
✓ mvar/pod_basis.npz
✓ mvar/mvar_model.npz
  ✓ Keys correct: ['U', 'all_singular_values', ...]

================================================================================
✓ All checks PASSED - Output format matches stable pipeline
================================================================================
```

---

## Breaking Changes

None! The unified pipeline now produces **identical output** to the stable pipeline, ensuring:
- ✅ Visualization tools work without modification
- ✅ Analysis scripts can read output directly
- ✅ Backward compatibility with existing workflows
- ✅ SLURM job scripts can migrate seamlessly

---

## Migration Path

Once verified:

1. Test unified pipeline with representative configs:
   ```bash
   python run_unified_mvar_pipeline.py --config configs/best_run_extended_test.yaml
   python run_unified_mvar_pipeline.py --config configs/extreme_clustering.yaml
   python run_unified_mvar_pipeline.py --config configs/rom_mvar_example.yaml
   ```

2. Rename old pipelines (backup):
   ```bash
   ./rename_old_pipelines.sh
   ```

3. Update SLURM files to use unified pipeline:
   ```bash
   # Replace:
   python run_stable_mvar_pipeline.py ...
   # With:
   python run_unified_mvar_pipeline.py ...
   ```

4. Update documentation references.

---

## Summary

All output format updates complete:
- ✅ Training runs save trajectory + density with correct keys
- ✅ Test runs use density_true/density_pred naming
- ✅ POD/MVAR files use stable pipeline key names
- ✅ Metrics saved in both JSON and CSV formats
- ✅ Spatial grids included in all density files
- ✅ Full backward compatibility achieved

**Status:** Ready for testing and verification.
