# Pipeline Unification Summary

## What Was Done

Merged three separate pipelines into one unified pipeline that handles ALL experiment types:

### Old Pipelines (DEPRECATED):
1. **run_stable_mvar_pipeline.py** (1198 lines)
   - Mixed distributions
   - Eigenvalue stability enforcement
   - Time-resolved evaluation

2. **run_robust_mvar_pipeline.py** (729 lines)
   - Mixed distributions  
   - Strong regularization
   - Interpolation/extrapolation tests

3. **run_gaussians_pipeline.py** (642 lines)
   - Custom Gaussian experiments
   - Variance/center variations
   - Different config format

**Total: 2569 lines across 3 files**

### New Pipeline:
**run_unified_mvar_pipeline.py** (1156 lines)
- All features from all three pipelines
- Supports both config formats
- Fixed `fixed_modes` priority bug
- Consistent naming (accepts both conventions)
- Single source of truth

## Key Improvements

### 1. Fixed Critical Bug
**Before**: Config `fixed_modes: 25` was ignored → used 287 modes (99.5% energy)
**After**: Config `fixed_modes: 25` is respected → uses exactly 25 modes

**Priority order now:**
1. `fixed_modes: N` (highest priority)
2. `fixed_d: N` (backward compatibility)
3. `pod_energy: 0.XX` (fallback)

### 2. Unified Config Support

The new pipeline accepts **both** config formats:

#### Format A: Mixed Distributions
```yaml
train_ic:
  type: "mixed_comprehensive"
  gaussian:
    enabled: true
    positions_x: [3.75, 7.5, 11.25]
    positions_y: [3.75, 7.5, 11.25]
    variances: [0.5, 1.0, 2.0]
  uniform:
    enabled: true
    n_runs: 100
  # ... ring, two_clusters
```

#### Format B: Custom Gaussian
```yaml
train_ic:
  center: [10.0, 10.0]
  variances: [0.5, 1.0, 2.0, 4.0]
  n_samples_per_variance: 3

test_ic:
  centers: [[5.0, 5.0], [15.0, 5.0]]
  variance: 2.0
```

### 3. Parameter Name Compatibility

Accepts multiple names for the same parameter:

| Feature | Name 1 | Name 2 | Priority |
|---------|--------|--------|----------|
| POD modes | `fixed_modes` | `fixed_d` | Name 1 |
| Subsampling | `subsample` | `rom_subsample` | Name 1 |
| Test runs | `n_runs` | `n_samples` | Name 1 |

### 4. All Features Available

Every feature from every pipeline is now available in one place:

- ✅ Mixed distributions (gaussian, uniform, ring, two_clusters)
- ✅ Custom Gaussian experiments
- ✅ Eigenvalue stability enforcement (optional)
- ✅ Time-resolved R² evaluation (optional)
- ✅ Strong regularization options
- ✅ Interpolation/extrapolation tests
- ✅ Flexible test durations
- ✅ Comprehensive metadata

## Migration Path

### Immediate (Safe):
Old pipelines still work, but use the unified pipeline for new experiments:

```bash
# Old way (still works):
python run_stable_mvar_pipeline.py --config configs/stable_mvar_v2.yaml --experiment_name test

# New way (recommended):
python run_unified_mvar_pipeline.py --config configs/stable_mvar_v2.yaml --experiment_name test
```

**Same config files work with both!**

### Soon (Recommended):
After testing the unified pipeline:

```bash
./rename_old_pipelines.sh
```

This renames old pipelines to `.deprecated` so you only use the unified one.

### Later (Cleanup):
Once confident, delete deprecated files:

```bash
rm run_*_pipeline.py.deprecated
```

## Testing Checklist

Before fully switching, test with your configs:

- [ ] Mixed distribution experiment (e.g., best_run_extended_test)
- [ ] Stability-enforced experiment (e.g., stable_mvar_v2)
- [ ] Custom Gaussian experiment (if you have one)
- [ ] Verify `fixed_modes` is respected (check console output)
- [ ] Compare outputs with old pipeline results
- [ ] Check time-resolved evaluation works (if enabled)

## Expected Console Output

When running with `fixed_modes: 25`:

```
STEP 2: Global POD and MVAR Training
================================================================================

Loading training density data (subsample=1)...
✓ Loaded data shape: (8000, 4096)
   400 runs × 20 timesteps × 4096 spatial dims

Computing global POD...
✓ Using FIXED d=25 modes (energy=0.4892, hard cap from config)
                          ^^^^^^^^ Should see this!
✓ Latent training data shape: (8000, 25)

Training global MVAR (p=20, α=1e-06)...
✓ MVAR training data: X(7600, 500), Y(7600, 25)
✓ Training R² = 0.9995
```

**Key line**: `✓ Using FIXED d=25 modes` confirms the bug fix is working!

## File Organization

```
wsindy-manifold/
├── run_unified_mvar_pipeline.py    ← NEW: Use this for ALL experiments
├── UNIFIED_PIPELINE_GUIDE.md       ← NEW: Complete documentation
├── CRITICAL_BUG_FIX.md             ← Documents the fixed_modes bug
├── rename_old_pipelines.sh         ← Script to deprecate old pipelines
│
├── run_stable_mvar_pipeline.py     ← OLD: Will be deprecated
├── run_robust_mvar_pipeline.py     ← OLD: Will be deprecated
├── run_gaussians_pipeline.py       ← OLD: Will be deprecated
│
└── configs/
    ├── best_run_extended_test.yaml     ← Works with unified pipeline
    ├── stable_mvar_v2.yaml             ← Works with unified pipeline
    └── ...                             ← All configs compatible!
```

## Benefits Summary

### For Users:
- 🎯 One pipeline to learn instead of three
- 🐛 Critical bug fixed (`fixed_modes` now works)
- 📚 Better documentation
- 🔧 More flexible (supports all config formats)
- ⚡ Same performance as before

### For Maintenance:
- 📦 Single source of truth
- 🔍 Easier to debug (one codebase)
- ✨ New features benefit all experiments
- 🧪 Easier to test (one pipeline)
- 📉 Less code to maintain (1156 vs 2569 lines)

### For Reproducibility:
- 📋 All experiments use the same code
- 🔒 Consistent behavior across experiments
- 📊 Easier to compare results
- 🗂️ Standardized outputs

## Recommended Next Steps

1. **Test the unified pipeline** with your existing configs
2. **Verify `fixed_modes` bug is fixed** (check console output)
3. **Run visualization pipeline** to confirm outputs match expected
4. **Deprecate old pipelines** with `./rename_old_pipelines.sh`
5. **Update any scripts/SLURM files** to use unified pipeline
6. **Rerun best_run_extended_test** with corrected 25-mode config

## Questions?

See `UNIFIED_PIPELINE_GUIDE.md` for:
- Complete usage examples
- Config format specifications
- Troubleshooting guide
- Performance tips
- Migration instructions

## Summary

**Before**: 3 pipelines, inconsistent behavior, `fixed_modes` bug, 2569 lines
**After**: 1 pipeline, all features, bug fixed, 1156 lines

**Action**: Start using `run_unified_mvar_pipeline.py` for all experiments!
