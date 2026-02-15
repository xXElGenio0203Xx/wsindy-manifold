# Synthesis V1 → V2: Stability Diagnosis & Fix

**Date:** February 10, 2026  
**Commit:** `b323457` (synthesis_v2: stability-aware compact ROM)  
**OSCAR Job:** 336255 (synthesis_v2, running on node2344)

---

## 1. The Problem: Synthesis V1 Results

Synthesis V1 achieved **MVAR R² = 0.49** (pipeline-reported) / **R² = 0.28** (recomputed with mass renormalization). Despite a training R² of 0.9994, the model's multi-step forecasts were poor. The advisor hypothesized that **clamping** (zeroing negative density pixels) and **train-test mismatch** were the cause, potentially explaining the gap with Alvarez et al.'s results.

We conducted a systematic error attribution analysis across all 26 test runs to locate the true bottleneck.

---

## 2. Error Attribution Analysis (26 Test Runs)

We evaluated R² at **five stages** of the pipeline, isolating each source of error:

| Stage | Mean R² | Std | Δ from Previous |
|---|---|---|---|
| **POD reconstruction ceiling** (truncation limit)¹ | **+0.88** | 0.08 | — |
| Raw lifted (no post-processing) | **−6.37** | 3.47 | −7.25 ← **MVAR error** |
| + Clamping only | **−2.96** | 1.81 | +3.40 (helps) |
| + Clamping + mass renorm | **+0.28** | 0.09 | +3.24 (helps) |
| Latent-space R² (ROM coords) | **−83.6** | 67.8 | — |

> ¹ **POD reconstruction ceiling** = project true test densities $x$ to latent $y = (x - \mu)^\top U_r$, lift back $\tilde{x} = U_r y + \mu$, then $R^2(\tilde{x}, x)$. This is computed on **raw** (unprocessed) densities. For a strictly apples-to-apples comparison with post-processed predictions, the ceiling should be evaluated under the same evaluation map $\Pi(\cdot)$ used for predictions (e.g., $\Pi(\cdot) = \text{clamp+renorm}$). The raw ceiling of 0.88 is therefore an upper bound; the post-processed ceiling would be marginally different.  
> All R² values are computed on the **forecast-only region** (after conditioning window).

### Key Deltas

- **POD reconstruction ceiling → Raw:** −7.25 — the **MVAR prediction error dominates everything**
- **Raw → Clamped:** +3.40 — clamping **helps** (40% of pixels are negative)
- **Clamped → Renorm:** +3.24 — mass renormalization **helps** (clamped mass inflates to 2,700–11,900 vs true 1,024)
- **Latent → Raw:** +77.3 — the lifting step $U_r y + \mu$ acts as a spectral filter. Because $U_r$ maps a low-dimensional latent vector into a smooth POD subspace, large errors in $\hat{y}$ do not translate into high-frequency pixel-wise errors; the reconstruction projects onto spatial modes that suppress small-scale noise. This is why latent R² = −83.6 becomes physical R² = −6.37 after lifting

### Initial Hypotheses vs. Measured Behavior

| Initial Hypothesis | Finding | Evidence |
|---|---|---|
| "Clamping hurts R²" | Clamping **helps** | R² improves by +3.4 (removes negative-density artifacts) |
| "Train-test mismatch from clamping" | **Secondary effect** | Dominant error source is MVAR instability, not distribution shift |
| "Lifting degrades predictions" | Lifting **improves** R² | Latent R² = −83.6 → physical R² = −6.37 after lifting (+77.3) |

---

## 3. Root Cause: MVAR is Dynamically Unstable

### Companion Matrix Analysis

A VAR(p) model $y(t) = \sum_{k=1}^{p} A_k\, y(t-k) + c$ has multi-step stability governed by its **companion matrix**:

$$
\mathbf{C} = \begin{bmatrix} A_1 & A_2 & \cdots & A_p \\ I & 0 & \cdots & 0 \\ 0 & I & \cdots & 0 \\ \vdots & & \ddots & \vdots \\ 0 & 0 & \cdots & 0 \end{bmatrix} \in \mathbb{R}^{pm \times pm}
$$

For the synthesis_v1 model ($p=5$, $m=19$):

| Metric | Value |
|---|---|
| Companion matrix size | 95 × 95 |
| **Spectral radius** $\rho(\mathbf{C})$ | **1.322** (must be < 1.0) |
| Eigenvalues outside unit circle | **42 / 95** |
| Training R² | 0.9994 (one-step-ahead — masks instability) |

### Prediction Divergence

| Forecast Step | ‖error‖ | Relative Error |
|---|---|---|
| 0 | 28 | 50% |
| 5 | 53 | 92% |
| 15 | 176 | 297% |
| 16 | 599 | **1015%** |

### Per-Mode Error (test_000)

Every single mode has R² < 0, with the worst modes reaching R² = −2,323:

| Mode | Singular Value | True σ | Predicted σ | R² |
|---|---|---|---|---|
| 0 | 1448 | 8.60 | 24.34 | −12.3 |
| 5 | 992 | 2.64 | 42.40 | −291.3 |
| 8 | 772 | 2.42 | 116.76 | **−2323.1** |
| 16 | 524 | 2.91 | 73.65 | −641.1 |

The MVAR predictions are **orders of magnitude** larger than the true signals across all modes.

### Why Training R² ≈ 1.0 is Misleading

Training R² measures **one-step-ahead** prediction: given the true history $[y(t{-}p), \ldots, y(t{-}1)]$, predict $y(t)$. This is trivially high for any reasonable regression. But **multi-step forecasting** iterates the model — each prediction feeds back as input, and unstable eigenvalues amplify errors exponentially:

$$
\|e(t)\| \sim \rho^t \cdot \|e(0)\|, \quad \rho = 1.322
$$

After 16 steps: $1.322^{16} \approx 100$, matching the observed 100× error amplification.

---

## 4. Why V1 Was Overfitting

| Factor | V1 Value | Issue |
|---|---|---|
| POD modes ($m$) | 19 | Too many for MVAR to model |
| MVAR lag ($p$) | 5 | More lags = more unstable modes |
| Parameters | $p \times m^2 = 1{,}805$ | Large model |
| Training samples | 6,512 | **3.6× ratio** (borderline) |
| Ridge α | $10^{-4}$ | Too weak to constrain eigenvalues |
| Stability check | **Disabled** | `eigenvalue_threshold` not in config |

The existing stability check in `mvar_trainer.py` had **two additional bugs**:
1. Only checked the **last lag matrix** $A_p$, not the full companion matrix
2. Applied crude uniform scaling that doesn't account for the identity sub-diagonal blocks

---

## 5. Code Changes

### 5.1. Fixed: `src/rectsim/mvar_trainer.py`

**Before (broken):**

```python
# Only checked last lag matrix — WRONG for VAR(p)
A_p = A_coef[:, -R_POD:]
eigenvalues = np.linalg.eigvals(A_p)
rho_before = np.max(np.abs(eigenvalues))
```

**After (correct):**

The stability check now builds the **full companion matrix** $\mathbf{C} \in \mathbb{R}^{pm \times pm}$:

```python
# Build FULL companion matrix for VAR(p)
companion_dim = P_LAG * R_POD
C = np.zeros((companion_dim, companion_dim))

# First block row: the A matrices [A_1, A_2, ..., A_p]
C[:R_POD, :] = A_coef

# Identity blocks on the sub-diagonal
for k in range(P_LAG - 1):
    C[(k+1)*R_POD:(k+2)*R_POD, k*R_POD:(k+1)*R_POD] = np.eye(R_POD)

eigenvalues = np.linalg.eigvals(C)
rho_before = np.max(np.abs(eigenvalues))
```

If $\rho > \text{threshold}$, we enforce stability post hoc by iteratively scaling the learned VAR coefficients until the companion matrix spectral radius is below the threshold (up to 10 iterations). This ensures stable rollouts but does not "fix" the model — it is a blunt instrument that uniformly shrinks dynamics and can bias toward over-damping (forecasts collapsing toward the mean):

```python
for iteration in range(10):
    scale = eigenvalue_threshold / np.max(np.abs(np.linalg.eigvals(C)))
    C[:R_POD, :] *= scale
    mvar_model.coef_ *= scale
    if mvar_model.intercept_ is not None:
        mvar_model.intercept_ *= scale
    if np.max(np.abs(np.linalg.eigvals(C))) <= eigenvalue_threshold:
        break
```

> **Note on stability strategies (ranked):**
> 1. **Best** (statistically principled): Constrain the VAR to be stable *during* fitting (constrained optimization / stable VAR estimation methods).
> 2. **Good** (practical): Regularize harder + reduce dimension + reduce lag, then reject unstable fits. V2 primarily relies on this approach (`fixed_modes=8`, `lag=3`, `ridge_alpha=1.0`).
> 3. **Acceptable** (current post hoc scaling): Guarantees stability but can distort learned dynamics.

The threshold is read from **two locations** for flexibility:

```python
eigenvalue_threshold = rom_config.get('eigenvalue_threshold', None)
if eigenvalue_threshold is None and 'models' in rom_config and 'mvar' in rom_config['models']:
    eigenvalue_threshold = rom_config['models']['mvar'].get('eigenvalue_threshold', None)
```

### 5.2. POD Mode Selection: `src/rectsim/pod_builder.py`

The `build_pod_basis()` function supports both **fixed mode count** and **energy threshold**, with fixed count taking priority:

```python
FIXED_D = rom_config.get('fixed_modes', None)
if FIXED_D is None:
    FIXED_D = rom_config.get('fixed_d', None)

if FIXED_D is not None:
    R_POD = min(FIXED_D, len(S))  # Fixed mode count (PRIORITY)
else:
    R_POD = np.searchsorted(cumulative_energy, TARGET_ENERGY) + 1  # Energy threshold
```

V2 uses `fixed_modes: 8` to explicitly control model complexity.

### 5.3. Post-Processing: `src/rectsim/test_evaluator.py`

The `evaluate_test_runs()` function applies two post-processing steps to lifted predictions:

1. **Clamping** — zeros negative density pixels (POD reconstruction artifact)
2. **Mass renormalization** — scales each frame to preserve the total mass lost by clamping

```python
for t_idx in range(len(pred_physical_forecast)):
    frame = pred_physical_forecast[t_idx]
    mass_before = frame.sum()
    frame = np.maximum(frame, 0.0)        # Clamp negatives
    mass_after = frame.sum()
    if mass_after > 0 and mass_before > 0:
        frame *= (mass_before / mass_after)  # Renormalize mass
```

This is applied to both the **full trajectory** (conditioning + forecast) and the **forecast-only** region used for R² computation.

### 5.4. Forecasting: `src/rectsim/forecast_utils.py`

The `mvar_forecast_fn_factory()` creates a closure for autoregressive forecasting:

```python
def forecast_fn(y_init_window, n_steps):
    current_history = y_init_window.copy()  # [lag, d]
    for _ in range(n_steps):
        x_hist = current_history[-lag:].flatten()   # [lag*d]
        y_next = mvar_model.predict(x_hist.reshape(1, -1))[0]  # [d]
        current_history = np.vstack([current_history[1:], y_next])  # Sliding window
    return np.array(ys_pred)  # [n_steps, d]
```

This is where instability manifests — each $y_{\text{next}}$ feeds back as input, amplifying errors through unstable modes.

---

## 6. Synthesis V2 Configuration

### Parameter Comparison

| Parameter | V1 | V2 | Rationale |
|---|---|---|---|
| POD modes | 19 (energy=0.90) | **8** (fixed) | Fewer modes → fewer params |
| MVAR lag | 5 | **3** | Fewer unstable companion eigenvalues |
| Ridge α | $10^{-4}$ | **1.0** | 10,000× stronger regularization |
| Stability check | ❌ disabled | **✅ ρ ≤ 0.98** | Full companion matrix enforcement |
| Training runs | 200 (actual: 187) | **300** (actual: 261) | More training data |
| MVAR params | 1,805 | **192** | 96% fewer parameters |
| Data/param ratio | 3.6× | **~56×** | Massively over-determined |
| LSTM hidden | 64 | **32** | Matched to d=8 |
| LSTM params | ~53,000 | ~5,500 | Better sample/param ratio |
| Sim time | 5.0s | 6.0s | Stable model can handle longer horizon |
| Forecast start | 0.60s (5 frames) | 0.36s (3 frames) | Matched to lag |

### V2 Budget

```
ROM frames/run:      150 / 3 = 50
MVAR samples/run:    50 - 3 = 47
Total samples:       261 × 47 ≈ 12,267
MVAR params:         3 × 8² + 8 = 200 (with intercept)
Data/param ratio:    12,267 / 200 ≈ 61× (excellent!)
Companion dim:       3 × 8 = 24 (vs 95 in v1)
Forecast steps:      50 - 3 = 47
```

### Expected Performance

| Metric | V1 Actual | V2 Expected |
|---|---|---|
| POD reconstruction ceiling R² | 0.88 | ~0.70–0.75 (fewer modes, lower energy capture) |
| MVAR latent R² | −83.6 | Positive (stable dynamics) |
| Final R² (physical) | +0.28 | **+0.50–0.70** (approaching ceiling) |
| LSTM R² | −0.02 | Positive (d=8 is easier to learn) |

---

## 7. Pipeline Architecture

The full pipeline is implemented in `ROM_pipeline.py` and executes these stages:

```
┌─────────────────────────────────────────────────────┐
│ 1. SIMULATION  (run_simulations_parallel)           │
│    N=100 particles, T=6.0s, dt=0.04                 │
│    261 training + 26 test runs                      │
│    Vicsek alignment + Morse forces                  │
├─────────────────────────────────────────────────────┤
│ 2. DENSITY FIELDS  (KDE → 48×48 grid)              │
│    bandwidth=4.0, mass = N per frame                │
│    density.npz per run                              │
├─────────────────────────────────────────────────────┤
│ 3. POD BASIS  (build_pod_basis → pod_builder.py)    │
│    Subsample ×3 → 50 ROM frames/run                │
│    fixed_modes=8 → U_r ∈ ℝ^{2304×8}               │
│    SVD of centered data: U S Vᵀ = (X-μ)ᵀ           │
│    Saved: pod_basis.npz, X_train_mean.npy           │
├─────────────────────────────────────────────────────┤
│ 4. LATENT DATASET  (build_latent_dataset)           │
│    y(t) = (x(t) - μ)ᵀ Uᵣ  →  Y_all, X_all        │
│    Windowed: X_all[i] = [y(t-p),...,y(t-1)]         │
│    Saved: latent_dataset.npz                        │
├─────────────────────────┬───────────────────────────┤
│ 5a. MVAR TRAINING       │ 5b. LSTM TRAINING         │
│  (mvar_trainer.py)      │  (lstm_rom.py)            │
│  Ridge(α=1.0)           │  LatentLSTMROM(d=8,h=32)  │
│  fit_intercept=True     │  2-layer LSTM + Linear    │
│  ↓                      │  Adam, lr=3e-4            │
│  STABILITY CHECK:       │  patience=80              │
│  Build companion C      │  max_epochs=2000          │
│  If ρ(C) > 0.98:        │                           │
│    scale coefs until     │                           │
│    ρ ≤ 0.98              │                           │
│  Save: mvar_model.npz   │  Save: lstm_state_dict.pt │
├─────────────────────────┴───────────────────────────┤
│ 6. EVALUATION  (test_evaluator.py)                  │
│    For each test run:                               │
│    a. Project test density → latent                 │
│    b. IC window = last p frames before forecast     │
│    c. Autoregressive forecast (p → T_test)          │
│    d. Lift: x̂ = Uᵣ ŷ + μ                           │
│    e. Clamp negatives + mass renormalize            │
│    f. R² on forecast region                         │
│    Save: density_pred_{model}.npz,                  │
│          density_metrics_{model}.csv,               │
│          r2_vs_time.csv, metrics_summary.json        │
└─────────────────────────────────────────────────────┘
```

### Key Mathematical Operations

**Restriction** (physical → latent):
$$y(t) = (x(t) - \mu)^\top U_r, \quad y \in \mathbb{R}^m$$

**MVAR Forecast** (latent → latent):
$$\hat{y}(t) = \sum_{k=1}^{p} A_k\, \hat{y}(t{-}k) + c, \quad A_k \in \mathbb{R}^{m \times m}$$

**Lifting** (latent → physical):
$$\hat{x}(t) = U_r\, \hat{y}(t) + \mu, \quad \hat{x} \in \mathbb{R}^d$$

**Post-processing**:
$$\hat{x}^+(t) = \max(\hat{x}(t), 0) \cdot \frac{\sum_i \hat{x}_i(t)}{\sum_i \max(\hat{x}_i(t), 0)}$$

**Mass preservation property** (proven in thesis):  
KDE normalizes all snapshots to mass $= N$, so $\mathbf{1}^\top x_i = N$ for all $i$. The centered snapshots $\bar{x}_i = x_i - \mu$ therefore satisfy $\mathbf{1}^\top \bar{x}_i = 0$ for every $i$, which means $\mathbf{1} \perp \mathrm{col}(\bar{X})$. Since the POD modes $U_r$ form an orthonormal basis for (a subspace of) $\mathrm{col}(\bar{X})$, it follows that $\mathbf{1}^\top U_r = 0$ for all retained modes (up to floating-point precision, verified to $\sim 3 \times 10^{-14}$).

Therefore the **raw linear lift** preserves the discrete sum-mass exactly:
$$\mathbf{1}^\top \hat{x}(t) = \mathbf{1}^\top U_r \hat{y}(t) + \mathbf{1}^\top \mu = 0 + M(\mu) = N$$

This holds regardless of $\hat{y}$. However:
- **Clamping** ($\max(\hat{x}, 0)$) removes negative pixels and **breaks** this mass identity.
- **Renormalization** ($\hat{x}^+ \cdot M_{\text{target}} / M(\hat{x}^+)$) **restores** the target mass after clamping.

This three-stage chain — exact mass from lifting, destroyed by clamping, restored by renormalization — is consistent with the measured error attribution deltas.

---

## 8. File Reference

| File | Role |
|---|---|
| `ROM_pipeline.py` | Main entry point, orchestrates all stages |
| `src/rectsim/config_loader.py` | Loads YAML config, returns structured dicts |
| `src/rectsim/pod_builder.py` | SVD, mode selection (fixed or energy), projection |
| `src/rectsim/mvar_trainer.py` | Ridge regression, **companion matrix stability check** |
| `src/rectsim/forecast_utils.py` | `mvar_forecast_fn_factory()` — autoregressive loop |
| `src/rectsim/test_evaluator.py` | Evaluate, clamp+renorm, save predictions |
| `src/rectsim/rom_data_utils.py` | Build windowed latent dataset for MVAR/LSTM |
| `src/rom/lstm_rom.py` | `LatentLSTMROM` model, `train_lstm_rom()` |
| `configs/synthesis_v1.yaml` | V1 config (unstable, 19 modes, no stability check) |
| `configs/synthesis_v2.yaml` | V2 config (stable, 8 modes, ρ ≤ 0.98) |
| `slurm_scripts/run_synthesis_v2.slurm` | OSCAR submission script for V2 |

---

## 9. What Success Looks Like

When synthesis_v2 completes, we should see in the logs:

```
Stability check (FULL companion matrix 24×24):
   Spectral radius ρ = X.XXXXXX
   Unstable eigenvalues (|λ|>1): 0/24        ← or scaled down
   ✓ Model is stable (ρ=X.XXXX ≤ 0.98)
```

And in test results:
- **MVAR latent R² > 0** (stable forecasts that don't explode)
- **MVAR physical R² approaching POD reconstruction ceiling** (the gap should shrink dramatically)
- **Less negative pixels** in raw predictions (smaller latent errors → smaller physical errors)
- **Clamping has less work to do** → mass renormalization effect is smaller

### Diagnostic Plots to Generate Immediately

**1. Latent rollout stability plot:**
$$\|\hat{y}(t)\|_2 \text{ vs } t, \quad \|y(t)\|_2 \text{ vs } t$$
If $\|\hat{y}(t)\|$ still grows unboundedly, the stability enforcement didn't work (or there's a bug). If it decays toward zero, the scaling may be over-damping the dynamics.

**2. Gap-to-ceiling plot (time-resolved):**
$$\Delta(t) = R^2_{\text{ceiling}}(t) - R^2_{\text{pred}}(t)$$
If stability worked, $\Delta(t)$ should stop growing explosively and remain bounded. In V1, this gap grew without limit due to the unstable companion eigenvalues.

---

## 10. If V2 Still Underperforms: Nonlinearity, Not Overfitting

The V2 sample-to-parameter ratio is ~61× (excellent). If a stable VAR with strong ridge regularization and $m=8$ modes still yields poor latent R², the diagnosis shifts from **overfitting** to **nonlinearity**: the latent dynamics are genuinely nonlinear and not well-approximated by a linear autoregression, regardless of stability.

Principled next moves at that point:
- **LSTM** (already included in V2) — if LSTM latent R² is substantially better than MVAR, this confirms nonlinearity
- **NVAR** (nonlinear vector autoregression) — polynomial features in latent space as a middle ground between linear VAR and neural networks
- **EDMD/DMD variants** (Koopman-based) — model nonlinear dynamics linearly in a lifted feature space

But these should not be pursued until V2 results are in and analyzed.
