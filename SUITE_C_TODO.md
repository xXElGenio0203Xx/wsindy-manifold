# Suite C — Experiment TODO Checklist

**Created:** Session implementing the "BEST NEXT SUITE" plan  
**Total experiments:** 19 configs across 5 blocks (C0–C4)  
**Fastest path:** 7 experiments in `run_fastest_path.slurm`

---

## Pre-Flight (Before OSCAR Submit)

- [ ] **Local syntax check** — run quick import test:
  ```bash
  cd ~/wsindy-manifold
  PYTHONPATH=src python -c "from rectsim.mvar_trainer import train_mvar_kstep; print('kstep OK')"
  PYTHONPATH=src python -c "from rectsim.test_evaluator import _project_simplex; print('simplex OK')"
  ```
- [ ] **Git commit + push** — stage all new/modified files:
  - `src/rectsim/mvar_trainer.py` (added `train_mvar_kstep`)
  - `src/rectsim/test_evaluator.py` (added `_project_simplex`, `clamp_mode='simplex'`)
  - `ROM_pipeline.py` (import + kstep conditional + summary field)
  - 19 config YAMLs in `configs/suite_C*.yaml`
  - 5 SLURM scripts in `slurm_scripts/run_suite_C{0,1,2_C3,4}.slurm` + `run_fastest_path.slurm`
  - This file (`SUITE_C_TODO.md`)
- [ ] **Pull on OSCAR** — `ssh oscar` → `cd ~/wsindy-manifold && git pull`
- [ ] **Verify OSCAR env** — `module load python/3.11.11-5e66 && source ~/wsindy_env_new/bin/activate && python -c "import torch; print(torch.__version__)"`
- [ ] **Create slurm_logs dir on OSCAR** — `mkdir -p slurm_logs`

---

## Phase 1: Fastest Path (7 runs, ~12-14h)

Submit: `sbatch slurm_scripts/run_fastest_path.slurm`

### C0 — Sqrt vs Dimension Isolation (3 runs)

| # | Config | Transform | d | Question | Status |
|---|--------|-----------|---|----------|--------|
| 1 | `suite_C0_raw_D11_p5_V1` | none | 11 | Raw at d=11 (sqrt's dimension) | ☐ |
| 2 | `suite_C0_sqrt_D19_p5_V1` | sqrt | 19 | Sqrt at d=19 (raw's dimension) | ☐ |
| 3 | `suite_C0_sqrt_D11_p5_V1` | sqrt | 11 | Baseline sqrt (should match R²≈0.730) | ☐ |

**After C0 completes:**
- [ ] Record R² for all 3
- [ ] If raw_D11 ≈ sqrt_D11 → transform irrelevant, it's all about lower d
- [ ] If raw_D11 ≪ sqrt_D11 → sqrt genuinely helps beyond dimension
- [ ] Confirm or update `raw_best` and `sqrt_best` values. Placeholders:
  - raw_best = d=19 (from pod_energy=0.90 on raw)
  - sqrt_best = d=11 (from pod_energy=0.90 on sqrt)
- [ ] **If winners differ from placeholders**, update C1/C2/C3/C4 config `fixed_modes` values

### C2-3 — k-Step MVAR (1 run)

| # | Config | kstep_k | d | Horizon | Status |
|---|--------|---------|---|---------|--------|
| 4 | `suite_C2_mvar_kstepTrain_sqrtBest_k5_p5_H150` | 5 | 11 | H150 | ☐ |

### C4-3 — LSTM Re-Entry (1 run)

| # | Config | window | k | α | Horizon | Status |
|---|--------|--------|---|---|---------|--------|
| 5 | `suite_C4_lstm_sqrtBest_w20_k5_alpha0p3_H150` | 20 | 5 | 0.3 | H150 | ☐ |

### C3-2 — Simplex Projection (1 run)

| # | Config | clamp_mode | d | Horizon | Status |
|---|--------|------------|---|---------|--------|
| 6 | `suite_C3_projSimplex_sqrtBest_H150` | simplex | 11 | H150 | ☐ |

### Baseline for comparison (1 run)

| # | Config | Notes | Status |
|---|--------|-------|--------|
| 7 | `suite_C1_sqrtBest_H150` | Standard MVAR sqrt d=11 @ H150 (clampC2) | ☐ |

**After Phase 1 completes:**
- [ ] Compare R² rollout: C2-3 vs baseline (run 4 vs 7) → does k-step help?
- [ ] Compare R² rollout: C4-3 vs baseline (run 5 vs 7) → does LSTM close gap?
- [ ] Compare R² rollout + negativity: C3-2 vs baseline (run 6 vs 7) → simplex vs clampC2?
- [ ] Decide which methods to expand to H300 and full sweeps

---

## Phase 2: Expand Winners (conditional)

### C2 Full — k-Step Sweep (3 remaining runs)

Submit: `sbatch slurm_scripts/run_suite_C2_C3.slurm` (runs all 6 C2+C3)

| # | Config | kstep_k | Transform | d | Status |
|---|--------|---------|-----------|---|--------|
| — | `suite_C2_mvar_kstepTrain_rawBest_k5_p5_H150` | 5 | none | 19 | ☐ |
| — | `suite_C2_mvar_kstepTrain_rawBest_k10_p5_H150` | 10 | none | 19 | ☐ |
| — | `suite_C2_mvar_kstepTrain_sqrtBest_k10_p5_H150` | 10 | sqrt | 11 | ☐ |

### C3 Full — Simplex on Raw (1 remaining run)

| — | `suite_C3_projSimplex_rawBest_H150` | simplex | none | 19 | ☐ |

### C4 Full — All 4 LSTM Configs

Submit: `sbatch slurm_scripts/run_suite_C4.slurm`

| # | Config | window | k | α | Transform | d | Status |
|---|--------|--------|---|---|-----------|---|--------|
| — | `suite_C4_lstm_rawBest_w20_k5_alpha0p3_H150` | 20 | 5 | 0.3 | none | 19 | ☐ |
| — | `suite_C4_lstm_rawBest_w40_k10_alpha0p5_H150` | 40 | 10 | 0.5 | none | 19 | ☐ |
| — | `suite_C4_lstm_sqrtBest_w40_k10_alpha0p5_H150` | 40 | 10 | 0.5 | sqrt | 11 | ☐ |

### C1 — Full Horizon Curves (6 runs)

Submit: `sbatch slurm_scripts/run_suite_C1.slurm`

| # | Config | Transform | d | Horizon | Status |
|---|--------|-----------|---|---------|--------|
| — | `suite_C1_rawBest_H37` | none | 19 | H37 | ☐ |
| — | `suite_C1_rawBest_H150` | none | 19 | H150 | ☐ |
| — | `suite_C1_rawBest_H300` | none | 19 | H300 | ☐ |
| — | `suite_C1_sqrtBest_H37` | sqrt | 11 | H37 | ☐ |
| — | `suite_C1_sqrtBest_H150` | sqrt | 11 | H150 | ☐ (done in Phase 1) |
| — | `suite_C1_sqrtBest_H300` | sqrt | 11 | H300 | ☐ |

---

## Phase 3: Follow-Up Configs (create AFTER Phase 1/2 results)

These are not yet created — create them only if the method proves worthwhile:

- [ ] **C2-5/C2-6**: Best k-step MVAR winner at H300
- [ ] **C3-3**: Best simplex config at H300
- [ ] **C4-5**: Best LSTM config at H300
- [ ] **Hybrid**: If both k-step MVAR and LSTM improve, test ensemble/hybrid

---

## New Code Features Implemented

### 1. `train_mvar_kstep()` in `mvar_trainer.py`
- **Phase 1:** Warm-start with 1-step Ridge regression
- **Phase 2:** Build k-step windows from training trajectories
- **Phase 3:** Differentiable VAR(p) model (PyTorch tensors A, b)
- **Phase 4:** Adam optimizer, mini-batch, early stopping on val loss
- **Config keys:** `kstep_k`, `kstep_lr`, `kstep_epochs`, `kstep_patience`, `kstep_weights`
- **Wired in:** `ROM_pipeline.py` checks `kstep_k > 1` → calls `train_mvar_kstep` instead of `train_mvar_model`

### 2. `_project_simplex()` in `test_evaluator.py`
- **Algorithm:** Duchi et al. (2008) — O(n log n) Euclidean projection
- **Ensures:** `rho >= 0` AND `sum(rho) = mass_target`
- **Wired in:** `clamp_mode='simplex'` in evaluation post-processing

---

## File Inventory

### Modified Source Files
| File | Change |
|------|--------|
| `src/rectsim/mvar_trainer.py` | Added `train_mvar_kstep()` function |
| `src/rectsim/test_evaluator.py` | Added `_project_simplex()` + `clamp_mode='simplex'` |
| `ROM_pipeline.py` | Import + kstep conditional + summary field |

### New Config YAMLs (19 total)
| Block | Count | Pattern |
|-------|-------|---------|
| C0 | 3 | `configs/suite_C0_*.yaml` |
| C1 | 6 | `configs/suite_C1_*.yaml` |
| C2 | 4 | `configs/suite_C2_*.yaml` |
| C3 | 2 | `configs/suite_C3_*.yaml` |
| C4 | 4 | `configs/suite_C4_*.yaml` |

### New SLURM Scripts (5 total)
| Script | Experiments | Est. Time |
|--------|-------------|-----------|
| `run_fastest_path.slurm` | 7 (C0 + priority picks) | ~14h |
| `run_suite_C0.slurm` | 3 | ~8h |
| `run_suite_C1.slurm` | 6 | ~18h |
| `run_suite_C2_C3.slurm` | 6 | ~16h |
| `run_suite_C4.slurm` | 4 | ~16h |

---

## Horizon Reference

| Label | test_sim.T | Formula | Approx Steps |
|-------|------------|---------|--------------|
| H37 | 5.0 (default) | (5.0-0.60)/0.12 | ~37 |
| H150 | 20.0 | (20.0-0.60)/0.12 | ~162 |
| H300 | 38.0 | (38.0-0.60)/0.12 | ~312 |

---

## Decision Tree After Results

```
C0 results:
├── raw_D11 ≈ sqrt_D11 → "It's the dimension, not the transform"
│   └── Use raw for simplicity, lower d for quality
├── raw_D11 ≪ sqrt_D11 → "Sqrt genuinely helps"
│   └── Always use sqrt going forward
└── sqrt_D19 > sqrt_D11 → "More modes help even with sqrt"
    └── Reconsider pod_energy threshold

Phase 1 comparison (all vs baseline @ H150):
├── k-step MVAR >> baseline → Priority method, expand to H300
├── LSTM ≈ baseline → Professor's fixes work, expand sweep
├── Simplex > clampC2 → Switch default clamp mode
└── Nothing helps → Focus on horizon curves (C1) for thesis plots
```
