# Complete Pipeline File Usage Analysis
**Date:** November 22, 2025  
**Analyzed Pipeline:** `run_complete_pipeline.py`

## Executive Summary

The main pipeline `run_complete_pipeline.py` is **SELF-CONTAINED** and does NOT use any of the `rom_mvar_*.py` scripts. It implements all POD and MVAR functionality inline.

---

## ✅ ACTIVELY USED FILES

### Main Pipeline
- **`run_complete_pipeline.py`** (1060 lines)
  - Self-contained orchestrator for full MVAR-ROM workflow
  - Implements POD, MVAR training, and evaluation inline
  - Uses only `rectsim` module functions

### Core Dependencies (from `src/rectsim/`)
1. **`vicsek_discrete.py`** - Simulation backend
   - `simulate_backend()` - Run particle simulations

2. **`legacy_functions.py`** - Visualization & metrics
   - `kde_density_movie()` - Generate density from trajectories
   - `trajectory_video()` - Create particle trajectory videos
   - `side_by_side_video()` - Truth vs prediction comparisons
   - `compute_frame_metrics()` - Per-frame error metrics
   - `compute_summary_metrics()` - Aggregate metrics (R², errors)
   - `plot_errors_timeseries()` - Error plots over time
   - `compute_order_params()` - Polarization, speed, nematic order

### Inline Implementations (in main pipeline)
```python
# Lines 344-390: fit_mvar()
# Lines 524-543: mvar_forecast()
# Lines 215-285: POD computation (SVD-based)
```

---

## 🔧 UTILITY SCRIPTS (May Be Useful for Testing/Development)

### ROM/MVAR Modular Workflow Scripts
These scripts provide an **alternative modular approach** using the `src/rectsim/rom_mvar.py` module:

1. **`rom_mvar_train.py`** (321 lines)
   - **Purpose:** Standalone training script with config files
   - **Use case:** Training with YAML configs, CLI overrides
   - **Status:** NOT used by main pipeline (redundant)
   - **Keep?** Maybe - useful for command-line training experiments

2. **`rom_train_mvar.py`** (225 lines)
   - **Purpose:** Stage 3 of modular ROM pipeline (MVAR only)
   - **Workflow:** Load POD → Train MVAR → Save model
   - **Status:** NOT used by main pipeline
   - **Keep?** Maybe - useful if splitting POD/MVAR into separate stages

3. **`rom_mvar_eval.py`** (411 lines)
   - **Purpose:** Standalone evaluation on unseen ICs
   - **Use case:** Oscar batch evaluation without videos
   - **Status:** NOT used by main pipeline
   - **Keep?** Maybe - useful for cluster evaluation jobs

4. **`rom_mvar_visualize.py`** (375 lines)
   - **Purpose:** Post-hoc visualization from saved results
   - **Use case:** Generate videos locally after Oscar rsync
   - **Status:** NOT used by main pipeline
   - **Keep?** Yes - useful for post-processing

5. **`rom_mvar_generalization_test.py`** (416 lines)
   - **Purpose:** Test generalization across IC distributions
   - **Use case:** Systematic testing: uniform vs gaussian (1-4 clusters)
   - **Status:** NOT used by main pipeline
   - **Keep?** Yes - useful for research/testing different IC types

### Potentially Redundant Scripts

6. **`rom_mvar_best_plots.py`**
   - **Status:** Likely redundant (main pipeline generates best plots)
   - **Action:** Review and possibly delete

7. **`rom_mvar_visualize_best.py`**
   - **Status:** Likely redundant (main pipeline has best_runs/ outputs)
   - **Action:** Review and possibly delete

8. **`rom_mvar_eval_unseen.py`**
   - **Status:** Likely overlaps with rom_mvar_eval.py
   - **Action:** Review and possibly consolidate

9. **`rom_mvar_full_eval_local.py`**
   - **Status:** Unclear purpose
   - **Action:** Review and document or delete

---

## 📦 MODULE STRUCTURE

### `src/rectsim/rom_mvar.py`
This module provides reusable ROM/MVAR functions but is **NOT imported** by the main pipeline:

**Functions available:**
- `compute_global_pod()` - POD computation
- `project_to_pod()` / `reconstruct_from_pod()` - Projection/reconstruction
- `fit_mvar()` - MVAR training
- `forecast_mvar()` - MVAR forecasting
- `load_pod_model()` / `save_pod_model()` - I/O
- `load_mvar_model()` / `save_mvar_model()` - I/O
- `compute_summary_metrics()` - Metrics

**Status:** Available but unused by main pipeline (inline implementations instead)

### `src/rectsim/legacy_functions.py`
This module is **ACTIVELY USED** by main pipeline:
- All visualization functions
- All metrics computation
- Order parameter calculations

---

## 🎯 RECOMMENDATIONS

### Keep (Essential)
1. ✅ `run_complete_pipeline.py` - Main orchestrator
2. ✅ `src/rectsim/vicsek_discrete.py` - Simulation backend
3. ✅ `src/rectsim/legacy_functions.py` - Viz & metrics

### Keep (Useful for Testing/Research)
4. ✅ `rom_mvar_generalization_test.py` - IC distribution testing
5. ✅ `rom_mvar_visualize.py` - Post-processing visualization
6. ✅ `src/rectsim/rom_mvar.py` - Reusable module (may use later)

### Review & Decide
7. ❓ `rom_mvar_train.py` - Config-based training (alternative workflow)
8. ❓ `rom_train_mvar.py` - Modular MVAR-only training
9. ❓ `rom_mvar_eval.py` - Standalone evaluation
10. ❓ `rom_mvar_best_plots.py` - May be redundant
11. ❓ `rom_mvar_visualize_best.py` - May be redundant
12. ❓ `rom_mvar_eval_unseen.py` - May overlap with #9
13. ❓ `rom_mvar_full_eval_local.py` - Unclear purpose

### Consider Consolidating
- The modular workflow scripts (2, 3, 9, 12) could potentially be consolidated into a single CLI tool
- The visualization scripts (4, 10, 11) could be merged

---

## 💡 KEY INSIGHTS

### Why doesn't the main pipeline use `rom_mvar.py`?
The main pipeline implements POD/MVAR **inline** for maximum clarity and control:
- Direct SVD computation (lines 215-285)
- Custom `fit_mvar()` with ridge regression (lines 344-390)
- Custom `mvar_forecast()` for prediction (lines 524-543)

This makes the pipeline **self-documenting** and easier to understand as a complete workflow.

### Are the modular scripts useful?
**Yes, for different use cases:**
- **Oscar batch jobs:** Use `rom_mvar_eval.py` (no videos, cluster-friendly)
- **Config-based experiments:** Use `rom_mvar_train.py` with YAML configs
- **Post-processing:** Use `rom_mvar_visualize.py` after rsync from cluster
- **Research testing:** Use `rom_mvar_generalization_test.py` for IC studies

### Should we refactor to use `rom_mvar.py`?
**Not necessarily.** The inline implementation has benefits:
- ✅ Single-file workflow is easier to understand
- ✅ No hidden dependencies
- ✅ Clear data flow
- ✅ Easier to debug

However, if you plan to:
- Run many different experiments with varying configs
- Deploy to Oscar with job arrays
- Reuse ROM/MVAR code in other projects

Then refactoring to use the `rom_mvar.py` module could be beneficial.

---

## 📊 CURRENT PIPELINE OUTPUTS

The main pipeline generates:
```
outputs/complete_pipeline/
├── pipeline_summary.json           # Comprehensive metadata
├── training/                       # 100 training sims
├── test/                          # 20 test sims
├── pod/                           # POD model (U, S, mean)
├── mvar/                          # MVAR model (A matrices)
├── plots/
│   ├── pod_singular_values.png
│   ├── pod_energy_spectrum.png
│   ├── r2_by_ic_type.png
│   └── error_by_ic_type.png
└── best_runs/
    ├── uniform/
    ├── gaussian_cluster/
    ├── ring/
    └── two_clusters/
        ├── traj_truth.mp4
        ├── density_truth_vs_pred.mp4
        ├── error_time.png
        ├── error_hist.png
        └── order_parameters.png
```

This structure is **complete** and **production-ready**.

---

## ✅ ACTION ITEMS

1. **Review each "❓" script** - Determine keep/delete/consolidate
2. **Document script purposes** - Add clear docstrings to kept scripts
3. **Consider archiving** - Move unused scripts to `.archive/` if unsure
4. **Update README** - Document which scripts are for what use cases
5. **Test modular workflow** - Verify `rom_mvar_*.py` scripts still work if kept


---

## 🔍 DETAILED SCRIPT REVIEW

### Scripts to DELETE (Redundant with main pipeline)

**`scripts/rom_mvar_best_plots.py`** ❌
- **Purpose:** Generate best-run plots per IC type
- **Redundant:** Main pipeline already generates this in `best_runs/`
- **Action:** DELETE - functionality fully covered

**`scripts/rom_mvar_visualize_best.py`** ❌
- **Purpose:** Visualize best ICs from generalization test
- **Redundant:** Main pipeline has comprehensive best_runs/ structure
- **Action:** DELETE - functionality fully covered

### Scripts to KEEP (Different use cases)

**`scripts/rom_mvar_generalization_test.py`** ✅
- **Purpose:** Systematic IC distribution testing (uniform vs gaussian, 1-4 clusters)
- **Unique value:** Tests generalization across IC parameter space
- **Action:** KEEP - research tool for IC sensitivity analysis

**`scripts/rom_mvar_visualize.py`** ✅
- **Purpose:** Post-hoc visualization from saved NPZ/CSV
- **Unique value:** Separates computation (Oscar) from visualization (local)
- **Action:** KEEP - useful for cluster workflows

**`scripts/rom_mvar_eval_unseen.py`** ✅
- **Purpose:** Evaluate on pre-generated unseen IC simulations
- **Unique value:** Tests on existing simulation datasets
- **Action:** KEEP - useful for testing on archived data

### Scripts to CONSOLIDATE (Overlapping functionality)

**Modular Training Scripts:**
- `rom_mvar_train.py` (config-based, all-in-one)
- `rom_train_mvar.py` (MVAR-only stage)

**Recommendation:** Pick ONE approach:
- **Option A:** Keep `rom_mvar_train.py` (comprehensive), delete `rom_train_mvar.py`
- **Option B:** Delete both (main pipeline is self-contained)

**Evaluation Scripts:**
- `rom_mvar_eval.py` (generate new test sims + evaluate)
- `rom_mvar_eval_unseen.py` (evaluate existing sims)

**Recommendation:** Keep both - they serve different purposes:
- `rom_mvar_eval.py` → Oscar batch jobs with fresh simulations
- `rom_mvar_eval_unseen.py` → Evaluate on archived/external datasets

### Scripts Needing Investigation

**`scripts/rom_mvar_full_eval_local.py`** ❓
- **Status:** Incomplete header, need full review
- **Action:** Read full file and determine purpose

---

## 📋 FINAL ACTION PLAN

### Immediate Actions (High Confidence)

```bash
# Delete redundant scripts
rm scripts/rom_mvar_best_plots.py
rm scripts/rom_mvar_visualize_best.py

# Keep essential utilities
# ✓ rom_mvar_generalization_test.py
# ✓ rom_mvar_visualize.py
# ✓ rom_mvar_eval_unseen.py
```

### Review & Decide (Need Input)

1. **Training Scripts Decision:**
   ```bash
   # Option A: Keep comprehensive training script
   # ✓ rom_mvar_train.py
   rm scripts/rom_train_mvar.py
   
   # Option B: Delete both (use main pipeline only)
   rm scripts/rom_mvar_train.py
   rm scripts/rom_train_mvar.py
   ```

2. **Evaluation Scripts:**
   ```bash
   # Keep both (different purposes)
   # ✓ rom_mvar_eval.py
   # ✓ rom_mvar_eval_unseen.py
   ```

3. **Investigate:**
   ```bash
   # Review full file first
   cat scripts/rom_mvar_full_eval_local.py
   ```

### Documentation Updates

1. Update `README.md` with script usage guide
2. Add docstrings to kept scripts
3. Create `scripts/README.md` explaining each tool

---

## 🎯 RECOMMENDED FINAL STRUCTURE

```
scripts/
├── README.md                           # Usage guide for all scripts
│
├── run_complete_pipeline.py            # ⭐ MAIN PIPELINE (self-contained)
│
├── rom_mvar_generalization_test.py    # Research: IC sensitivity testing
├── rom_mvar_visualize.py               # Post-processing: Generate videos/plots
├── rom_mvar_eval_unseen.py             # Evaluate on existing datasets
│
├── rom_mvar_eval.py                    # (Optional) Oscar batch evaluation
├── rom_mvar_train.py                   # (Optional) Config-based training
│
└── utilities/                          # Helper scripts
    ├── create_animations.py
    ├── oscar_sync_results.sh
    └── ...
```

---

## 💬 MY RECOMMENDATION

**Delete immediately:**
- ❌ `rom_mvar_best_plots.py`
- ❌ `rom_mvar_visualize_best.py`

**Archive (move to `.archive/scripts/`):**
- 📦 `rom_train_mvar.py` - Superseded by main pipeline
- 📦 `rom_mvar_train.py` - Alternative approach, keep as reference

**Keep active:**
- ✅ `run_complete_pipeline.py` - Main workflow
- ✅ `rom_mvar_generalization_test.py` - Research tool
- ✅ `rom_mvar_visualize.py` - Post-processing utility
- ✅ `rom_mvar_eval.py` - Oscar evaluation tool
- ✅ `rom_mvar_eval_unseen.py` - Archived data evaluation

This keeps your codebase **lean and focused** while preserving useful utilities for different workflows.

