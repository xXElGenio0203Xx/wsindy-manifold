# Bad LSTM & MVAR Tracking — 9 Apr 2026

## Overview
Comprehensive tracking of poor-performing ROM forecasters across all completed experiments.
Test protocol: 4 held-out ICs per experiment, N=100 (thesis_final) or N=300 (tier1).

---

## BAD LSTM Runs (mean R²_recon < 0.5)

| Experiment | mean R² | min R² | max R² | n_tests | Notes |
|---|---|---|---|---|---|
| gas_VS_tier1_bic | **-1.225** | -4.615 | 0.304 | 4 | Worst overall; catastrophic on test_001 |
| gas_VS_tier1_w5 | **-1.019** | -3.029 | 0.148 | 4 | Similarly bad, all negative except one marginal |
| blackhole_VS_thesis_final | **-0.656** | -1.282 | -0.376 | 4 | All 4 test ICs negative |
| gas_VS_thesis_final | **-0.533** | -2.383 | 0.657 | 4 | Highly variable; 1 OK, 3 bad |
| gas_tier1_w5 | **0.401** | 0.229 | 0.489 | 4 | Consistently mediocre (N=300) |

### Patterns
- **Variable-speed (VS) regimes are worst**: All VS experiments have bad LSTMs
- **N=300 hurts LSTM**: gas_tier1 (N=300) LSTM = 0.40 vs gas_thesis_final (N=100) LSTM = 0.57
- **gas_VS is systematically terrible**: LSTM R² < -1 across both lag variants
- **LSTM r2_1step is always decent**: 0.88-0.99 even when rollout r2_recon is catastrophic
  → Suggests LSTM learns local dynamics but diverges on multi-step rollout

### Possible Causes
1. **Density field complexity**: VS adds velocity-speed coupling → higher dimensional manifold
2. **POD truncation inadequacy**: r2_pod is always >0.95, so POD is fine → issue is in latent forecasting
3. **Rollout instability**: Good 1-step but bad multi-step = error accumulation / no stability regularization
4. **N=300 → larger POD basis?**: More particles may need more POD modes, amplifying LSTM errors

---

## BAD MVAR Runs (mean R²_recon < 0.5)

| Experiment | mean R² | min R² | max R² | n_tests | Notes |
|---|---|---|---|---|---|
| blackhole_VS_thesis_final | **-0.433** | -0.959 | -0.121 | 4 | All 4 negative, worst MVAR |
| supernova_VS_thesis_final | **0.120** | -0.030 | 0.252 | 4 | Poor across all ICs |
| gas_VS_tier1_w5 | **0.410** | -0.120 | 0.700 | 4 | Variable; one negative |
| pure_vicsek_thesis_final | **0.465** | -0.192 | 0.992 | 4 | Extreme variance: best IC = 0.99, worst = -0.19 |
| gas_VS_tier1_bic | **0.491** | 0.141 | 0.752 | 4 | Marginal |

### Patterns
- **VS regimes also hard for MVAR**: blackhole_VS and supernova_VS both fail
- **Pure Vicsek is IC-dependent**: 1 test IC works perfectly (0.99), 3 fail
- **gas_VS MVAR mediocre**: w5 slightly better than bic variant

---

## MISSING LSTM Results

| Experiment | Reason | Fix |
|---|---|---|
| supernova_VS_thesis_final | Config had `lstm: enabled: false` (Phase A) | Tier1 will cover (lstm=true) |
| pure_vicsek_thesis_final | Config had `lstm: enabled: false` (Phase A) | Tier1 will cover (lstm=true) |

---

## GOOD Experiments (for reference)

| Experiment | MVAR mean R² | LSTM mean R² | Status |
|---|---|---|---|
| blackhole_thesis_final | 0.990 | 0.989 | ✅ Best — both models excellent |
| gas_thesis_final | 0.995 | 0.574 | ✅ MVAR excellent, LSTM variable |
| gas_tier1_bic | 0.989 | 0.562 | ✅ Similar to thesis_final |
| blackhole_tier1_w5 | 0.993 | 0.785 | ✅ MVAR excellent, LSTM decent |
| supernova_thesis_final | 0.545 | 0.508 | ⚠️ Both mediocre |

---

## Action Items
1. [ ] Wait for tier1 tasks 5-17 (crystal, supernova_tier1, pure_vicsek_tier1) to fill remaining gaps
2. [ ] Investigate LSTM rollout stability — consider teacher forcing ratio or stability penalty
3. [ ] Consider LSTM hyperparameter sweep for VS regimes (lower learning rate? more regularization?)
4. [ ] pure_vicsek: investigate why test_000 (R²=0.99) vs test_001-003 (R²<0.5) — IC sensitivity
5. [ ] For thesis: present LSTM limitations honestly; focus narrative on MVAR as primary forecaster

---

## OSCAR Job Status (9 Apr ~19:00 EDT)
- **Tier1 (1426337)**: tasks 0-4 RUNNING (sims 39-78%), tasks 5-17 PENDING
- **Tier4 (1426340)**: PENDING (N-convergence)
- **Reeval simplex (1433234)**: PENDING
- **Reeval none (1433469)**: PENDING
- **Tier2 (1426338)**: CANCELLED
- **Tier3 (1426339)**: CANCELLED
