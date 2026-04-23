# Pipeline Migration Checklist

## Phase 1: Validation (Do This First)

### Test with best_run_extended_test config:

```bash
# 1. Test the unified pipeline
python run_unified_mvar_pipeline.py \
    --config configs/best_run_extended_test.yaml \
    --experiment_name unified_test

# 2. Check console output for the fix
# Look for: "✓ Using FIXED d=25 modes (energy=0.XXXX, hard cap from config)"
# Should see 25, not 287!

# 3. Run visualization
python run_visualizations.py --experiment_name unified_test

# 4. Compare results
# - Check: predictions/unified_test/summary_plots/
# - Verify: POD modes = 25 (not 287)
# - Compare: R² values with previous runs
```

### Expected Changes:

- **POD modes**: 287 → **25** ✓
- **Energy captured**: 99.5% → ~49% ✓
- **Test R²**: -3.85 → hopefully > 0.3 ✓

---

## Phase 2: Broader Testing

### Test each experiment type:

- [ ] **Mixed distributions** (best_run_extended_test)
  ```bash
  python run_unified_mvar_pipeline.py --config configs/best_run_extended_test.yaml --experiment_name test_mixed
  ```

- [ ] **Stability enforced** (stable_mvar_v2)
  ```bash
  python run_unified_mvar_pipeline.py --config configs/stable_mvar_v2.yaml --experiment_name test_stable
  ```

- [ ] **Any custom Gaussian configs** (if you have them)
  ```bash
  python run_unified_mvar_pipeline.py --config configs/your_gaussian_config.yaml --experiment_name test_gauss
  ```

### Verification for each:

- [ ] Pipeline runs without errors
- [ ] Console shows correct number of POD modes
- [ ] Output directory structure matches expected
- [ ] Visualization pipeline works
- [ ] Results are reasonable (R² > baseline)

---

## Phase 3: Update Scripts

### Find all references to old pipelines:

```bash
# Search for old pipeline usage
grep -r "run_stable_mvar_pipeline" . --include="*.sh" --include="*.slurm" --include="*.md"
grep -r "run_robust_mvar_pipeline" . --include="*.sh" --include="*.slurm" --include="*.md"
grep -r "run_gaussians_pipeline" . --include="*.sh" --include="*.slurm" --include="*.md"
```

### Update SLURM files:

Example change in `run_stable_v2.slurm`:

```bash
# OLD:
python run_stable_mvar_pipeline.py \
    --config configs/stable_mvar_v2.yaml \
    --experiment_name stable_mvar_v2

# NEW:
python run_unified_mvar_pipeline.py \
    --config configs/stable_mvar_v2.yaml \
    --experiment_name stable_mvar_v2
```

Files to check:
- [ ] `run_stable_v2.slurm`
- [ ] `run_robust_mvar_v1.slurm`
- [ ] Any other `.slurm` files
- [ ] Shell scripts in `scripts/`
- [ ] Documentation in `docs/`

---

## Phase 4: Deprecate Old Pipelines

### Once confident in unified pipeline:

```bash
# Rename old pipelines to .deprecated
./rename_old_pipelines.sh
```

This will:
- ✓ Rename `run_stable_mvar_pipeline.py` → `.deprecated`
- ✓ Rename `run_robust_mvar_pipeline.py` → `.deprecated`
- ✓ Rename `run_gaussians_pipeline.py` → `.deprecated`

Files are preserved but hidden from normal use.

**Revert if needed:**
```bash
mv run_stable_mvar_pipeline.py.deprecated run_stable_mvar_pipeline.py
mv run_robust_mvar_pipeline.py.deprecated run_robust_mvar_pipeline.py
mv run_gaussians_pipeline.py.deprecated run_gaussians_pipeline.py
```

---

## Phase 5: Rerun Critical Experiments

Now that `fixed_modes` bug is fixed, rerun key experiments:

### 1. Rerun best_run_extended_test (with correct 25 modes):

```bash
# On Oscar:
ssh emaciaso@ssh.ccv.brown.edu
cd ~/wsindy-manifold

# Update code
git pull

# Submit job
sbatch --export=CONFIG=configs/best_run_extended_test.yaml \
       run_unified_v1.slurm  # Create this SLURM file

# Monitor
squeue -u emaciaso
tail -f slurm_logs/unified_*_JOBID.out
```

### 2. Compare results:

| Metric | Old (287 modes) | New (25 modes) | Target |
|--------|----------------|----------------|--------|
| POD modes | 287 | 25 | 25 ✓ |
| Energy | 99.5% | ~49% | ~49% ✓ |
| Training R² | 1.000 | ~0.95-0.99 | < 1.0 ✓ |
| Test R² (mean) | -3.85 | ? | > 0.3 |
| R² at t=2.0s | 0.78 | ? | > 0.7 |
| R² at t=3.1s | 0.14 | ? | > 0.5 |

Expected improvement:
- **Old**: Severe overfitting (287 modes), R² = -3.85
- **New**: Better generalization (25 modes), R² ≈ 0.3-0.5

### 3. Time-resolved analysis:

```bash
# After job completes, download and visualize
python run_visualizations.py --experiment_name best_run_v2_corrected
```

Check if degradation improves:
- Does model maintain R² > 0.5 for longer?
- Is cascade failure delayed beyond t=4.4s?
- Do 25 modes generalize better than 287?

---

## Phase 6: Documentation Update

### Update project documentation:

- [ ] Update `README.md` to reference unified pipeline
- [ ] Update any `QUICKSTART.md` or tutorial docs
- [ ] Add note in experiment configs about pipeline change
- [ ] Update Oscar workflow docs

### Example README update:

```markdown
## Running Experiments

Use the unified pipeline for all ROM-MVAR experiments:

```bash
python run_unified_mvar_pipeline.py \
    --config configs/your_config.yaml \
    --experiment_name your_experiment
```

See `UNIFIED_PIPELINE_GUIDE.md` for complete documentation.
```

---

## Phase 7: Cleanup (Optional)

### After everything works:

```bash
# Delete deprecated pipelines (point of no return!)
rm run_stable_mvar_pipeline.py.deprecated
rm run_robust_mvar_pipeline.py.deprecated
rm run_gaussians_pipeline.py.deprecated

# Commit changes
git add run_unified_mvar_pipeline.py
git add UNIFIED_PIPELINE_GUIDE.md
git add PIPELINE_UNIFICATION_SUMMARY.md
git commit -m "Unified ROM-MVAR pipeline: merge 3 pipelines, fix fixed_modes bug"
git push
```

---

## Rollback Plan (If Needed)

If unified pipeline has issues:

```bash
# 1. Restore old pipelines
mv run_*_pipeline.py.deprecated run_*_pipeline.py

# 2. Revert SLURM files to old pipeline names

# 3. Continue using old pipelines while debugging

# 4. Report issues (check console output, error logs)
```

---

## Success Criteria

Migration is complete when:

- ✅ Unified pipeline tested with all config types
- ✅ `fixed_modes` bug verified fixed (25 modes, not 287)
- ✅ All SLURM/scripts updated to use unified pipeline
- ✅ Old pipelines deprecated (.deprecated extension)
- ✅ Documentation updated
- ✅ Critical experiments rerun with corrected parameters
- ✅ Results validated and improved

---

## Timeline Estimate

- **Phase 1** (Validation): 1-2 hours
- **Phase 2** (Testing): 2-4 hours
- **Phase 3** (Update scripts): 30 minutes
- **Phase 4** (Deprecate): 5 minutes
- **Phase 5** (Rerun experiments): 1-2 days (job runtime)
- **Phase 6** (Documentation): 1 hour
- **Phase 7** (Cleanup): 15 minutes

**Total**: ~1 week (including Oscar job time)

---

## Questions During Migration?

1. **Check documentation**: `UNIFIED_PIPELINE_GUIDE.md`
2. **Compare outputs**: Old vs new pipeline results
3. **Verify console logs**: Look for "FIXED d=25 modes" message
4. **Test incrementally**: One config type at a time

**Remember**: Old configs work as-is with unified pipeline!
