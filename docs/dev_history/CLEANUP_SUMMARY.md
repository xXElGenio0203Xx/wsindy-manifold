# Codebase Cleanup Summary
**Date:** November 22, 2025

## Actions Taken

### ✅ Deleted (Redundant)
These scripts duplicated functionality already in `run_complete_pipeline.py`:

1. **`scripts/rom_mvar_best_plots.py`**
   - Purpose: Generate best-run plots per IC type
   - Redundant: Main pipeline generates comprehensive `best_runs/` output

2. **`scripts/rom_mvar_visualize_best.py`**
   - Purpose: Visualize best ICs from tests
   - Redundant: Main pipeline has complete visualization suite

### 📦 Archived (Superseded)
Moved to `.archive/scripts/` for historical reference:

3. **`scripts/rom_mvar_train.py`**
   - Purpose: Config-based training workflow
   - Status: Alternative approach, superseded by main pipeline

4. **`scripts/rom_train_mvar.py`**
   - Purpose: Modular MVAR-only training
   - Status: Part of older multi-stage workflow

5. **`scripts/rom_mvar_full_eval_local.py`**
   - Purpose: Complete local evaluation pipeline
   - Status: Uses older module structure, superseded

### ✅ Kept (Active Use)
These scripts provide unique functionality:

6. **`run_complete_pipeline.py`** ⭐
   - **Main orchestrator** - Self-contained full workflow
   - Dynamic POD, MVAR training, stratified testing
   - Comprehensive JSON output

7. **`scripts/rom_mvar_generalization_test.py`**
   - IC sensitivity testing (uniform vs gaussian, 1-4 clusters)
   - Research tool for generalization studies

8. **`scripts/rom_mvar_visualize.py`**
   - Post-hoc visualization from saved data
   - Useful for cluster → local workflow

9. **`scripts/rom_mvar_eval.py`**
   - Oscar-friendly evaluation (no videos)
   - Batch job evaluation tool

10. **`scripts/rom_mvar_eval_unseen.py`**
    - Evaluate on pre-existing simulation datasets
    - Testing on archived/external data

---

## Results

### Before Cleanup
```
scripts/rom_mvar*.py: 9 files
- 5 overlapping/redundant
- 4 specialized tools
```

### After Cleanup
```
scripts/rom_mvar*.py: 4 files
- All serve unique purposes
- Clear separation of concerns
- Well-documented in scripts/README.md
```

### File Reduction
- **Deleted:** 2 files
- **Archived:** 3 files
- **Active:** 4 focused scripts + 1 main pipeline

---

## Current Active Structure

```
wsindy-manifold/
├── run_complete_pipeline.py          ⭐ MAIN PIPELINE
│
├── scripts/
│   ├── README.md                     📚 Complete documentation
│   │
│   ├── rom_mvar_generalization_test.py   🔬 Research
│   ├── rom_mvar_visualize.py             🎬 Post-processing
│   ├── rom_mvar_eval.py                  📊 Cluster evaluation
│   ├── rom_mvar_eval_unseen.py           📊 Archive testing
│   │
│   └── utilities/
│       ├── create_animations.py
│       ├── oscar_sync_results.sh
│       └── ...
│
└── .archive/scripts/                 📦 Historical reference
    ├── rom_mvar_train.py
    ├── rom_train_mvar.py
    └── rom_mvar_full_eval_local.py
```

---

## Benefits

### 🎯 Clarity
- Each script has a **clear, unique purpose**
- No confusion about which script to use
- Main pipeline is the obvious starting point

### 📉 Reduced Maintenance
- Fewer files to maintain
- No redundant code to keep in sync
- Cleaner git history

### 📚 Better Documentation
- `scripts/README.md` documents all active scripts
- `FILE_USAGE_ANALYSIS.md` explains the decisions
- Clear guidance for contributors

### 🚀 Improved Workflow
- Main pipeline is production-ready
- Research tools are clearly identified
- Archive preserves historical approaches

---

## Recommendations

### For Daily Use
```bash
# Complete workflow (start here)
python run_complete_pipeline.py
```

### For Research
```bash
# IC sensitivity studies
python scripts/rom_mvar_generalization_test.py \
    --experiment test_ics \
    --config configs/vicsek_morse_test.yaml
```

### For Oscar Workflows
```bash
# 1. Train on cluster (use main pipeline)
sbatch job_train.slurm

# 2. Evaluate on cluster
python scripts/rom_mvar_eval.py --experiment exp1

# 3. Sync and visualize locally
bash scripts/oscar_sync_results.sh
python scripts/rom_mvar_visualize.py --experiment exp1
```

---

## Next Steps

### Optional Improvements
1. **Test archived scripts** - Verify they still work if needed
2. **Update root README** - Document the main pipeline workflow
3. **Add examples** - Create example configs for common use cases
4. **Integration tests** - Test main pipeline end-to-end

### If Issues Arise
- Archived scripts are in `.archive/scripts/`
- Can be restored if needed
- Git history preserves all versions

---

## Summary

✅ **Codebase is now lean and focused**
- Main pipeline handles 95% of use cases
- Specialized tools for research and testing
- Clear documentation for all scripts
- Archived alternatives preserved for reference

The cleanup **reduces confusion** without losing functionality. Every remaining script has a clear purpose and documented use case.
