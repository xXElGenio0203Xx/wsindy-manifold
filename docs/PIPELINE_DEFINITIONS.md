# Official Pipeline Definitions (Locked)

> **Status**: LOCKED — these are the two pipeline variants under cross-regime validation  
> **Suites**: X (12 experiments), Y (6 experiments), Z (6 shift-aligned evals)

---

## Pipeline A: Raw Baseline

```
ρ(x,y,t)  →  POD(d modes)  →  MVAR(p lags, α ridge)  →  clamp C2  →  ρ̂(x,y,t)
```

| Step | Detail |
|------|--------|
| **Input** | Density field ρ ∈ ℝ^{Nx×Ny}, one frame per ROM_dt = 0.12 s |
| **POD** | Standard SVD on ρ; retain d modes |
| **MVAR** | Fit y_t = A₁ y_{t-1} + ⋯ + Aₚ y_{t-p} + ε with ridge α |
| **Clamp C2** | Set ρ̂<0 → 0, then scale to preserve ∑ρ̂ = M₀ |
| **Mass** | Approximately conserved by clamp-scale; no projection |

Config keys:
```yaml
rom.density_transform: raw
rom.mass_postprocess: "none"   # or absent (default)
```

---

## Pipeline B: √ρ + Simplex (Best)

```
ρ  →  u=√ρ  →  POD(d modes)  →  MVAR(p lags, α ridge)  →  û²  →  simplex L₂  →  ρ̂
```

| Step | Detail |
|------|--------|
| **Transform** | u = √ρ (variance-stabilizing; maps [0,∞) → [0,∞)) |
| **POD** | SVD on u; retain d modes |
| **MVAR** | Same model structure as Pipeline A |
| **Inverse map** | ρ̂_raw = û² (guarantees ρ̂ ≥ 0 before projection) |
| **Simplex L₂** | Project ρ̂_raw onto {ρ ≥ 0, ∑ρ = M₀} using Duchi et al. algorithm |
| **M₀** | M₀ = ∑ ρ_true(t_start), the true total mass at forecast onset |

Config keys:
```yaml
rom.density_transform: sqrt
rom.density_transform_eps: 1.0e-10
rom.mass_postprocess: "simplex"
```

The simplex projection solves:
$$\min_{\hat\rho} \|\hat\rho - \hat\rho_{\rm raw}\|_2^2 \quad \text{s.t.} \quad \hat\rho \ge 0, \; \sum \hat\rho = M_0$$

Implementation: `src/rectsim/test_evaluator.py`, `_project_simplex()` (line 22), Duchi et al. 2008.

---

## Regime Parameters

| Regime | speed | C_a | R   | η    | T_train | d  | p | α     | forecast_start |
|--------|-------|-----|-----|------|---------|----|----|-------|----------------|
| V1     | 1.5   | 0.8 | 2.5 | 0.2  | 5.0 s   | 19 | 5  | 1e-4  | 0.60           |
| V3.3   | 3.0   | 2.0 | 4.0 | 0.15 | 6.0 s   | 20 | 3  | 10.0  | 0.36           |
| V3.4   | 5.0   | 3.0 | 5.0 | 0.1  | 6.0 s   | 10 | 3  | 10.0  | 0.36           |

---

## Horizon → Forecast Time Mapping

ROM_dt = 0.12 s = dt(0.04) × subsample(3)

| Horizon | Formula | V1 T_test | V3.3/V3.4 T_test |
|---------|---------|-----------|-------------------|
| H37     | default (no extension) | 5.0 s | 6.0 s |
| H100    | (H + p_max) × ROM_dt = 105 × 0.12 | 12.6 s | 12.6 s |
| H162    | see below | 20.04 s | 19.80 s |

H162 T_test derivation:
- V1: forecast_start=0.60 → start_idx = floor(0.60 × 5.0/0.04 / 3) +1 ≈ 26; need 162 steps → T = (26+162)×0.12 ≈ 22.56 s; actual config: T=20.04
- V3.3/V3.4: forecast_start=0.36 → T=19.80

---

## Metrics Saved per Experiment

From `test_evaluator.py`:
| Metric | Key in CSV | Definition |
|--------|-----------|------------|
| R² rollout | `r2_reconstructed` | 1 − ‖ρ̂−ρ‖²/‖ρ−ρ̄‖² over full forecast |
| R² 1-step | `r2_1step` | Same formula, 1 step ahead only |
| R² latent | `r2_latent` | R² in latent (POD coefficient) space |
| R² POD | `r2_pod` | R² of POD reconstruction (no forecast) |
| Negativity % | `negativity_frac` | Fraction of cells with ρ̂ < 0 |
| Max mass violation | `max_mass_violation` | max_t |∑ρ̂(t) − M₀| / M₀ |
| RMSE recon | `rmse_recon` | RMSE of forecast in density space |

From `summary.json`:
| Metric | Key | Definition |
|--------|-----|------------|
| Spectral radius | `spectral_radius` | ρ(A_companion) before stabilization |
| Spectral radius (after) | `spectral_radius_after` | ρ(A_companion) after stabilization |

From Z suite (`shift_aligned_summary.json`):
| Metric | Key | Definition |
|--------|-----|------------|
| Shift-aligned R² | `R2_SA_mean` | R² after per-timestep optimal (dx,dy) shift |
| Phase drift % | `phase_drift_pct_mean` | (R²_SA − R²_raw) / |R²_SA| × 100 |
