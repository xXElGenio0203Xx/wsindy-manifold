# Thesis-Final OSCAR Submission — Session Log

**Date:** 1 April 2026  
**OSCAR Job ID:** 1154756 (Phase A)

---

## What We Did (Chronological)

### Prior Sessions (completed before today)

1. **WSINDy Reframing** — changed WSINDy from forecaster to interpretable PDE identifier:
   - Added `identification_only` flag to `ROM_WSINDY_pipeline.py` (Step 8 if/else split)
   - Added dominant balance ratio computation to diagnostics
   - Outputs: `identification_results.csv`, `identification_summary.json`
   - No `density_pred_wsindy.npz` in identification-only mode (no ETDRK4 rollout)

2. **Visualization Updates** in `3_models_plots.py`:
   - `plot_dominant_balance()`, `plot_regime_comparison_heatmap()`, `plot_condition_numbers()`
   - Enhanced `plot_wsindy_coefficients()` with bootstrap CI
   - WSINDy excluded from R²(t) plots (no forecast to compare)

3. **6 old wsindy_probe configs** updated with `identification_only: true`

4. **Thesis closure section** + `callaham2021dominant` BibTeX entry

5. **Submission plan v2** finalized (saved to session memory)

### Today's Session

6. **Created 7 Phase-A configs** — sim + POD only  
   - Source: `configs/systematic/main_regimes/*.yaml`
   - Output: `configs/systematic/thesis_final/phase_a/*.yaml`
   - Changes: `mvar.enabled=false`, `lstm.enabled=false`, `wsindy.enabled=false`
   - Experiment names: `{regime}_thesis_final`

7. **Created support files:**
   - `configs/systematic/thesis_final/phase_a/manifest.txt` (7 lines)
   - `slurm_scripts/run_thesis_final_phase_a.slurm` (7 tasks, 4h, 96GB)
   - `shell_scripts/verify_phase_a.sh` (checks train/metadata.json + rom_common/pod_basis.npz)
   - `shell_scripts/submit_thesis_final_phase_a.sh` (rsync + sbatch)

8. **Submitted Phase A to OSCAR** — job array 1154756, all 7 tasks running

---

## Files Created / Modified

| File | Status | Purpose |
|------|--------|---------|
| `configs/systematic/thesis_final/phase_a/NDYN04_gas.yaml` | NEW | Phase A config |
| `configs/systematic/thesis_final/phase_a/NDYN04_gas_VS.yaml` | NEW | Phase A config |
| `configs/systematic/thesis_final/phase_a/NDYN05_blackhole.yaml` | NEW | Phase A config |
| `configs/systematic/thesis_final/phase_a/NDYN05_blackhole_VS.yaml` | NEW | Phase A config |
| `configs/systematic/thesis_final/phase_a/NDYN06_supernova.yaml` | NEW | Phase A config |
| `configs/systematic/thesis_final/phase_a/NDYN06_supernova_VS.yaml` | NEW | Phase A config |
| `configs/systematic/thesis_final/phase_a/NDYN08_pure_vicsek.yaml` | NEW | Phase A config |
| `configs/systematic/thesis_final/phase_a/manifest.txt` | NEW | SLURM manifest (7 lines) |
| `slurm_scripts/run_thesis_final_phase_a.slurm` | NEW | Phase A SLURM script |
| `shell_scripts/verify_phase_a.sh` | NEW | Post-Phase-A verification |
| `shell_scripts/submit_thesis_final_phase_a.sh` | NEW | Upload + submit to OSCAR |

---

## The 7 Regimes

| # | Name | Speed Mode | Key Physics |
|---|------|-----------|-------------|
| 0 | NDYN04_gas | constant | Disordered gas (η=1.5, weak forces) |
| 1 | NDYN04_gas_VS | variable | Same, variable speed |
| 2 | NDYN05_blackhole | constant | Extreme attraction (Ca=20) |
| 3 | NDYN05_blackhole_VS | variable | Same, variable speed |
| 4 | NDYN06_supernova | constant | Extreme repulsion (Cr=15) |
| 5 | NDYN06_supernova_VS | variable | Same, variable speed |
| 6 | NDYN08_pure_vicsek | constant | Alignment only, no forces |

---

## Pipeline Fixes (All 6 Hardcoded — No Config Risk)

| Fix | File:Line | Key |
|-----|-----------|-----|
| center_flux=True | ROM_WSINDY_pipeline.py:1006 | Hardcoded arg |
| bilap removal | multifield.py:403-404 | Never called |
| ell_t≥7 floor | multifield.py:1117 | `min_temporal_ell = 7` |
| Sign constraints | multifield.py:64-68, 747-807 | `_NONPOSITIVE_POSTFIT_TERMS` |
| MAX_COEFF=5.0 | multifield.py:68, 772 | `_MAX_ABS_POSTFIT_COEFF` |
| Regime-aware library | multifield.py:345-376 | `resolve_regime_aware_library_settings()` |

---

## Two-Phase OSCAR Plan

### Phase A (SUBMITTED — Job 1154756)
- **What:** Simulation + POD only (all models disabled)
- **Tasks:** 7 (one per regime)
- **Resources:** 4h walltime, 96GB RAM, 4 cores, batch partition
- **Outputs:** `oscar_output/{regime}_thesis_final/train/` + `rom_common/`

### Between Phases (LOCAL)
```bash
# 1. Verify Phase A
ssh oscar 'cd ~/wsindy-manifold && bash shell_scripts/verify_phase_a.sh'

# 2. Download rom_common for lag selection
# (rsync oscar_output from OSCAR to local)

# 3. Run lag selection locally
python -m rom_hyperparameters.alvarez_lag_selection \
    --experiment_dir oscar_output/NDYN04_gas_thesis_final \
                     oscar_output/NDYN04_gas_VS_thesis_final \
                     oscar_output/NDYN05_blackhole_thesis_final \
                     oscar_output/NDYN05_blackhole_VS_thesis_final \
                     oscar_output/NDYN06_supernova_thesis_final \
                     oscar_output/NDYN06_supernova_VS_thesis_final \
                     oscar_output/NDYN08_pure_vicsek_thesis_final \
    --w_max 100

# 4. Read lag_bic from each summary.json
```

### Phase B (NOT YET CREATED — blocked on lag_bic values)
- **What:** Full pipeline: MVAR + LSTM + WSINDy (identification-only)
- **Tasks:** 14 (7 regimes × 2 MVAR lags: w=5 + w=BIC)
- **Key upgrades from existing configs:**

| Parameter | Old | New |
|-----------|-----|-----|
| LSTM max_epochs | 300 | 500 |
| LSTM patience | 40 | 50 |
| WSINDy identification_only | (absent) | true |
| WSINDy n_ell | 12 | 20 |
| WSINDy bootstrap B | 0 | 200 |
| MVAR lag | 5 | 5 and BIC |

- **Resources:** 10h walltime, 96GB RAM, 4 cores, `--array=0-13%7`
- **Expected outputs per task:**
  - `MVAR/test_results.csv`
  - `LSTM/best_model.pt`, `LSTM/test_results.csv`
  - `WSINDy/identification_results.csv`, `WSINDy/identification_summary.json`
  - `WSINDy/bootstrap_{rho,px,py}.npz` (200 replicates each)

---

## Monitoring Commands

```bash
# Check job status
ssh oscar 'squeue -u $USER'

# Watch logs
ssh oscar 'tail -f ~/wsindy-manifold/slurm_logs/thesis_phA_1154756_*.out'

# After completion — verify
ssh oscar 'cd ~/wsindy-manifold && bash shell_scripts/verify_phase_a.sh'
```

---

## Known Gaps (Documented Intentionally)

1. **Bootstrap ≠ stability selection** — B=200 measures coefficient robustness to resampling at fixed ℓ*, not cross-scale robustness. `stability.py:stability_selection()` exists but is not wired in.

2. **WSINDy has no forecast** — intentionally asymmetric comparison with MVAR/LSTM. Ch9 closure section explains why (unclosed PDE → blowup).

3. **LSTM redundancy** — 14 runs produce only 7 unique LSTM models (only MVAR lag changes between w5/wBIC pairs). Accepted as negligible cost.

---

## What's Next (from 1 April — now superseded by tiered system below)

1. ~~Wait for Phase A to complete~~ ✅
2. ~~Run `verify_phase_a.sh`~~ ✅
3. ~~Lag selection~~ ✅ (BIC lags computed)
4. ~~Phase B configs~~ ✅ (superseded by tier1-4 system)

---
---

# Full Experiment Tracking — 8 April 2026

## Architecture

All experiments now use a 4-tier system with N=300 particles, sqrt+none preprocessing:

| Tier | Purpose | Tasks | Models | Time limit | SLURM script |
|------|---------|-------|--------|------------|--------------|
| 1 | Main regimes × 2 lags | 18 | MVAR+LSTM+WSINDy | 18h | `run_tier1.slurm` |
| 2 | Noise sweep (4 regimes × 5 η) | 20 | MVAR+LSTM+WSINDy | 18h | `run_tier2_noise.slurm` |
| 3 | Extended catalogue (5 NDYN + 14 DO) | 19 | MVAR+LSTM only | 12h | `run_tier3_extended.slurm` |
| 4 | N-convergence (N=50..1000) | 6 | MVAR+LSTM+WSINDy | 24h | `run_tier4_convergence.slurm` |

## Tier 1 — Main Regimes (Job 1384148, 18 tasks)

9 regimes × 2 lag variants (w=5 baseline + w=BIC data-driven).
Manifest: `configs/systematic/tier1/manifest.txt`

| Task | Config | Experiment Name | Status | Notes |
|------|--------|-----------------|--------|-------|
| 0 | NDYN04_gas_tier1_w5 | NDYN04_gas_tier1_w5 | 🔁 RESUBMITTED | Old job 1345300_0 TIMEOUT at Step 5 (test gen) |
| 1 | NDYN04_gas_tier1_bic | NDYN04_gas_tier1_bic | ⏳ RUNNING (old 1345300_1) | ~2h in, 12h limit |
| 2 | NDYN04_gas_VS_tier1_w5 | NDYN04_gas_VS_tier1_w5 | ⏳ RUNNING (old 1345300_2) | ~2h in, 12h limit |
| 3 | NDYN04_gas_VS_tier1_bic | NDYN04_gas_VS_tier1_bic | 🔁 RESUBMITTED | |
| 4 | NDYN05_blackhole_tier1_w5 | — | 🔁 RESUBMITTED | |
| 5 | NDYN05_blackhole_tier1_bic | — | 🔁 RESUBMITTED | |
| 6 | NDYN05_blackhole_VS_tier1_w5 | — | 🔁 RESUBMITTED | |
| 7 | NDYN05_blackhole_VS_tier1_bic | — | 🔁 RESUBMITTED | |
| 8 | NDYN06_supernova_tier1_w5 | — | 🔁 RESUBMITTED | |
| 9 | NDYN06_supernova_tier1_bic | — | 🔁 RESUBMITTED | |
| 10 | NDYN06_supernova_VS_tier1_w5 | — | 🔁 RESUBMITTED | |
| 11 | NDYN06_supernova_VS_tier1_bic | — | 🔁 RESUBMITTED | |
| 12 | NDYN07_crystal_tier1_w5 | — | 🔁 RESUBMITTED | |
| 13 | NDYN07_crystal_tier1_bic | — | 🔁 RESUBMITTED | |
| 14 | NDYN07_crystal_VS_tier1_w5 | — | 🔁 RESUBMITTED | |
| 15 | NDYN07_crystal_VS_tier1_bic | — | 🔁 RESUBMITTED | |
| 16 | NDYN08_pure_vicsek_tier1_w5 | — | 🔁 RESUBMITTED | |
| 17 | NDYN08_pure_vicsek_tier1_bic | — | 🔁 RESUBMITTED | |

## Tier 2 — Noise Sweep (Job 1384149, 20 tasks)

4 regimes × 5 noise levels (η ∈ {0.05, 0.15, 0.50, 1.00, 1.50}).
Manifest: `configs/noise_sweep/manifest.txt`

| Task | Config | Status |
|------|--------|--------|
| 0 | NDYN04_gas_eta0p15 | 🔁 SUBMITTED |
| 1 | NDYN04_gas_eta0p50 | 🔁 SUBMITTED |
| 2 | NDYN04_gas_eta1p00 | 🔁 SUBMITTED |
| 3 | NDYN04_gas_eta1p50 | 🔁 SUBMITTED |
| 4 | NDYN04_gas_eta2p00 | 🔁 SUBMITTED |
| 5 | NDYN05_blackhole_eta0p05 | 🔁 SUBMITTED |
| 6 | NDYN05_blackhole_eta0p15 | 🔁 SUBMITTED |
| 7 | NDYN05_blackhole_eta0p50 | 🔁 SUBMITTED |
| 8 | NDYN05_blackhole_eta1p00 | 🔁 SUBMITTED |
| 9 | NDYN05_blackhole_eta1p50 | 🔁 SUBMITTED |
| 10 | NDYN06_supernova_eta0p05 | 🔁 SUBMITTED |
| 11 | NDYN06_supernova_eta0p15 | 🔁 SUBMITTED |
| 12 | NDYN06_supernova_eta0p50 | 🔁 SUBMITTED |
| 13 | NDYN06_supernova_eta1p00 | 🔁 SUBMITTED |
| 14 | NDYN06_supernova_eta1p50 | 🔁 SUBMITTED |
| 15 | NDYN07_crystal_eta0p05 | 🔁 SUBMITTED |
| 16 | NDYN07_crystal_eta0p10 | 🔁 SUBMITTED |
| 17 | NDYN07_crystal_eta0p50 | 🔁 SUBMITTED |
| 18 | NDYN07_crystal_eta1p00 | 🔁 SUBMITTED |
| 19 | NDYN07_crystal_eta1p50 | 🔁 SUBMITTED |

## Tier 3 — Extended Catalogue (Job 1384150, 19 tasks)

5 residual NDYN + 14 D'Orsogna outcomes. MVAR+LSTM only (no WSINDy).
Manifest: `configs/extended_catalogue/manifest.txt`

| Task | Config | Status | Notes |
|------|--------|--------|-------|
| 0 | NDYN01_crawl | 🔁 RESUBMITTED | Old 1336908_0 TIMEOUT at 6h (models trained, no eval) |
| 1 | NDYN02_flock | 🔁 RESUBMITTED | Same |
| 2 | NDYN09_longrange | 🔁 RESUBMITTED | Same |
| 3 | NDYN10_shortrange | 🔁 RESUBMITTED | Same |
| 4 | NDYN13_chaos | 🔁 RESUBMITTED | Same |
| 5 | DO_CS01_swarm_C01_l05 | 🔁 RESUBMITTED | Same |
| 6 | DO_CS02_swarm_C05_l3 | 🔁 RESUBMITTED | Same |
| 7 | DO_CS03_swarm_C09_l3 | 🔁 RESUBMITTED | Same |
| 8 | DO_DM01_dmill_C09_l05 | 🔁 RESUBMITTED | Same |
| 9 | DO_DR01_dring_C01_l01 | 🔁 RESUBMITTED | Same |
| 10 | DO_DR02_dring_C09_l09 | 🔁 RESUBMITTED | Same |
| 11 | DO_EC01_esccol_C2_l3 | 🔁 RESUBMITTED | Same |
| 12 | DO_EC02_esccol_C3_l05 | 🔁 RESUBMITTED | Same |
| 13 | DO_ES01_escsym_C3_l09 | ✅ COMPLETED | MVAR R²=0.81, LSTM R²=-0.49 |
| 14 | DO_EU01_escuns_C2_l2 | ⏳ RUNNING (old 1336908_14) | ~3h in, 6h limit |
| 15 | DO_EU02_escuns_C3_l3 | 🔁 RESUBMITTED | |
| 16 | DO_SM01_mill_C05_l01 | 🔁 RESUBMITTED | |
| 17 | DO_SM02_mill_C3_l01 | 🔁 RESUBMITTED | |
| 18 | DO_SM03_mill_C2_l05 | 🔁 RESUBMITTED | |

## Tier 4 — N-Convergence (Job 1384152, 6 tasks)

Pure Vicsek at N ∈ {50, 100, 200, 300, 500, 1000}.
Manifest: `configs/n_convergence/manifest.txt`

| Task | Config | Status | MVAR R² | LSTM R² | WSINDy |
|------|--------|--------|---------|---------|--------|
| 0 | N0050 | 🔁 RESUBMITTED (24h) | 0.982 | 0.644 | ❌ not run (timeout) |
| 1 | N0100 | 🔁 RESUBMITTED (24h) | 0.992 | 0.496 | ❌ not run (timeout) |
| 2 | N0200 | 🔁 RESUBMITTED (24h) | 0.995 | 0.360 | ❌ not run (timeout) |
| 3 | N0300 | 🔁 RESUBMITTED (24h) | 0.992 | 0.208 | ❌ not run (timeout) |
| 4 | N0500 | ⏳ RUNNING (old, 12h) + 🔁 RESUBMITTED (24h) | — | — | — |
| 5 | N1000 | ⏳ RUNNING (old, 12h) + 🔁 RESUBMITTED (24h) | — | — | — |

## Re-evaluation Jobs

| Job ID | Script | Purpose | Status |
|--------|--------|---------|--------|
| 1348876 | `reeval_phB` | Re-evaluate thesis_final Phase B experiments | ✅ COMPLETED (1h28m) |
| 1382781 | `reeval_scale` | LSTM re-eval with mass_postprocess=scale | ✅ COMPLETED (6m) |
| 1384155 | `reeval_all_scale` | Re-eval ALL with mass_postprocess=scale | 🔁 SUBMITTED (pending) |

## Completed experiments (from older runs, in `scratch/oscar_output/`)

These are from thesis_final Phase A/B and earlier probes. Available for analysis:

| Experiment | Location | MVAR | LSTM | WSINDy |
|-----------|----------|------|------|--------|
| NDYN04_gas_thesis_final | scratch | ✅ | ✅ | ✅ (identification only) |
| NDYN04_gas_VS_thesis_final | scratch | ✅ | ✅ | ✅ |
| NDYN05_blackhole_thesis_final | scratch | ✅ | ✅ | ✅ |
| NDYN05_blackhole_VS_thesis_final | scratch | ✅ | ✅ | ✅ |
| NDYN06_supernova_thesis_final | scratch | ✅ | ✅ | ✅ |
| NDYN06_supernova_VS_thesis_final | scratch | ✅ | ✅ | ✅ |
| NDYN08_pure_vicsek_thesis_final | scratch | ✅ | ✅ | ✅ |

## Timeout Root Causes and Fixes

| Tier | Old limit | Problem | New limit |
|------|-----------|---------|-----------|
| 1 | 12h | Task 0 timed out at Step 5 (test data gen after LSTM training) | **18h** |
| 2 | 12h | No failures yet (all were pending) | **18h** (preventive) |
| 3 | 6h | 12/13 tasks timed out — models trained but no test data or eval | **12h** |
| 4 | 12h | All 4 completed tasks timed out during LSTM eval (Step 6b); WSINDy (Step 7) never ran | **24h** |

## LSTM Performance Issue

LSTM R² degrades badly with none mass postprocessing:
- Tier4 N0050: 0.644, N0100: 0.496, N0200: 0.360, N0300: 0.208
- Tier3 DO_ES01: -0.49

Re-evaluation with `mass_postprocess=scale` submitted (job 1384155).
Previous `reeval_scale` (job 1382781) completed in 6 min but found no experiments to process
(output dirs had no test results at the time).

## MVAR R² by IC Family (from N-convergence data)

| IC Family | Mean R² | Notes |
|-----------|---------|-------|
| Gaussian cluster | 0.990 | Consistently excellent across all N |
| Two clusters | 0.718 | Decent |
| Ring | 0.453 | Poor |
| Uniform | 0.315 | Worst — most mixing |

## Disk Quota Status

- HOME: 92.35 GB / 100 GB (OK)
- Scratch: ~167 GB / 512 GB
- Running jobs write to HOME (`oscar_output/`), must migrate to scratch after completion

## What's Next

1. **Monitor running jobs** — tier1 tasks 1,2 and tier4 tasks 4,5 (old 12h limit) will likely timeout; new resubmissions (18h/24h) will replace them
2. **After tier3 completes** — run `reeval_all_scale` to test scale postprocessing on all LSTM results
3. **After tier4 completes** — verify WSINDy identification ran, extract coefficient tables for N-convergence plot
4. **After tier1 completes** — full 9-regime × 2-lag comparison table for thesis Chapter 7
5. **After tier2 completes** — noise robustness curves for Chapter 9
6. **Migrate results** — `scripts/finalize_oscar_output_symlink.sh` to move completed dirs to scratch
7. **Create N-convergence plotting script** — log-log coefficient error vs N
8. **Thesis tables/figures** — populate 44 placeholders (26 figures, 18 tables)
