# Alvarez et al. (2025) vs. Our Pipeline: Detailed Comparison

**Date:** February 10, 2026  
**Reference:** Alvarez, Kevrekidis, Bhatt, Gallos — *Next Generation Equation-Free Multiscale Modelling of Crowd Dynamics via Machine Learning* (2025)  
**Our pipeline:** Vicsek + Morse → KDE → POD → MVAR/LSTM (synthesis_v1/v2)

---

## 1. Physical Models

| Aspect | Alvarez | Ours |
|---|---|---|
| **Model** | Social Force Model (SFM, Helbing 1995) | Vicsek alignment + Morse forces |
| **Particles** | 100 pedestrians | 100 particles |
| **Domain** | 48m × 12m corridor with obstacle | 15m × 15m periodic box |
| **Boundary** | Walls + obstacle | Periodic |
| **Dynamics** | Continuous ODE (Newton's 2nd law) | Discrete heading update (Vicsek) + Morse force |
| **Timescale** | Slow (τ=0.5s relaxation), δt=0.025s | Fast (constant speed v₀=1.5), dt=0.04s |
| **Behavior** | Goal-directed corridor flow, splits around obstacle, recombines | Self-organizing: clustering, alignment, dispersion |
| **Stationarity** | Verified ADF stationarity (p<0.001) on latent series | **Not verified** — Vicsek+Morse may be nonstationary |

### Why This Matters
Alvarez's SFM produces **quasi-stationary** corridor flow: pedestrians enter, split around an obstacle, and recombine. The density field evolves smoothly toward a repeatable pattern. VAR models **require stationarity** — this is a core assumption of the method.

Our Vicsek+Morse dynamics are fundamentally different: particles self-organize from diverse ICs through alignment and attraction/repulsion. The transient from IC → steady state is the dominant dynamic, and the steady state itself may not be stationary in the VAR sense.

---

## 2. Training Data Structure

| Aspect | Alvarez | Ours (V1) | Ours (V2) |
|---|---|---|---|
| **Number of ICs** | 10 | ~187 | ~261 |
| **Timesteps per IC** | 1,100 (T=275s, δt=0.25s) | ~37 (T=5s, 50 ROM frames − 13 lag) | ~47 (T=6s, 50 ROM frames − 3 lag) |
| **Total snapshots** | 11,000 | ~6,900 | ~12,267 |
| **IC diversity** | 10 types: 3 Gaussian, 1 Uniform, 3 Double Gaussian, 2 Piecewise Linear, 1 Cosine | 3 types: Gaussian, Uniform, Two-clusters | 3 types: Gaussian, Uniform, Two-clusters |
| **Trajectory character** | Long — each IC shows full lifecycle (transient → steady state → variations) | Short — mostly transient dynamics, may not reach steady state |
| **Stacking method** | Concatenate 10 long trajectories into one tall matrix, SVD on combined data | Stack 261 short windows into tall matrix, SVD on combined data |

### Why This Matters — **THE SINGLE BIGGEST DIFFERENCE**
Alvarez uses **few ICs × long trajectories**. Each 1,100-step trajectory lets the MVAR learn the **temporal autocorrelation structure** of the dynamics — how mode coefficients evolve over hundreds of steps. The MVAR sees the system's attractor behavior, oscillations, and damping.

We use **many ICs × short trajectories**. Each 47-step trajectory barely captures the initial transient. The MVAR learns correlations within 47-step snippets from 261 different ICs, but never sees long-term dynamics. This is like trying to learn a song by hearing 261 different 2-second clips vs. hearing 10 complete 4-minute songs.

**Alvarez's effective MVAR training window per IC:**
- 1,100 timesteps − lag 4 = 1,096 training samples per IC
- Each sample has full temporal context from a coherent, long trajectory

**Our effective MVAR training window per IC:**
- 50 timesteps − lag 3 = 47 training samples per IC
- Each sample comes from a short burst with minimal temporal context

---

## 3. Density Field Construction

| Aspect | Alvarez | Ours |
|---|---|---|
| **Method** | Gaussian KDE | Gaussian KDE |
| **Grid** | 80 × 20 = 1,600 pixels | 48 × 48 = 2,304 pixels |
| **Bandwidth** | (σ_x, σ_y) = (3, 2) grid cells | σ = 4.0 grid cells (isotropic) |
| **Normalization** | Each frame integrates to 1 (probability) | Each frame sums to N=100 (count) |
| **Domain shape** | Rectangular (4:1 aspect ratio) | Square (1:1 aspect ratio) |

### Why This Matters
Alvarez's 80×20 grid is actually **fewer pixels** (1,600) than ours (2,304), despite the larger physical domain. Their anisotropic bandwidth (3,2) matches the corridor geometry. Our isotropic bandwidth=4.0 on a square grid is reasonable but produces higher-dimensional data that the SVD/POD must compress harder.

---

## 4. POD / SVD Dimensionality Reduction

| Aspect | Alvarez | Ours (V1) | Ours (V2) |
|---|---|---|---|
| **Data matrix** | 11,000 × 1,600 | ~9,350 × 2,304 | ~13,050 × 2,304 |
| **Energy threshold** | 99% | 90% (V1) | Fixed 8 modes |
| **Modes retained** | d = 13 | d = 19 | d = 8 |
| **Selection method** | Energy threshold (data-driven) | Energy threshold | Fixed (forced) |
| **Energy captured** | 99% | 90% | ~70-75% |

### Why This Matters
Alvarez captures 99% energy with only 13 modes — their smooth corridor dynamics have very fast spectral decay. We need 19 modes for only 90% energy (V1) because our diverse Vicsek+Morse dynamics span a richer function space. V2 forces 8 modes for stability at the cost of reconstruction quality (~70-75% energy).

**The ideal approach** (Alvarez's) is to let the data tell you how many modes are needed via an energy threshold, and get a small number because the dynamics are smooth. Forcing modes (V2) is a symptom of the underlying problem — our dynamics are harder for POD to compress.

---

## 5. MVAR Model

| Aspect | Alvarez | Ours (V1) | Ours (V2) |
|---|---|---|---|
| **Lag** | w = 4 (selected by BIC from w ∈ {1,...,100}) | p = 5 (manual) | p = 3 (manual) |
| **Lag selection** | **BIC/AIC** (principled, data-driven) | Manual | Manual |
| **Ridge α** | 1 × 10⁻⁶ | 1 × 10⁻⁴ | **1.0** |
| **Parameters** | 4 × 13² + 13 = 689 | 5 × 19² + 19 = 1,824 | 3 × 8² + 8 = 200 |
| **Training samples** | ~10,960 | ~6,512 | ~12,267 |
| **Sample/param ratio** | **15.9×** | 3.6× | 61× |
| **Spectral radius ρ** | Not reported (stable at α=1e-6) | **1.322** (42/95 unstable) | ≤ 0.98 (enforced) |
| **Stability check** | Not needed (naturally stable) | ❌ Disabled | ✅ Full companion matrix |
| **Eigenvalue enforcement** | None | None | Scale coefficients if ρ > 0.98 |

### Why This Matters
Alvarez's MVAR is stable **naturally** with α=1e-6 — one million times weaker regularization than our V2. This is because:

1. **BIC-selected lag**: The optimal lag is found objectively. BIC penalizes over-parameterization, naturally selecting the simplest adequate model.
2. **Long trajectories**: 1,096 samples per IC means the autocorrelation structure is well-estimated.
3. **Stationary dynamics**: The SFM dynamics satisfy VAR assumptions.
4. **Moderate dimension**: d=13 with w=4 = 689 params, comfortably over-determined.

We need α=1.0 (brute force) plus explicit eigenvalue clamping because:
1. **Manual lag**: No principled selection; may be too large or too small.
2. **Short trajectories**: 47 samples per IC — autocorrelations poorly estimated.
3. **Possibly nonstationary**: Vicsek+Morse transients violate VAR assumptions.
4. **Forced dimension**: V2's d=8 sacrifices reconstruction to make MVAR tractable.

---

## 6. LSTM Model

| Aspect | Alvarez | Ours (V2) |
|---|---|---|
| **Architecture** | 1-layer LSTM, 16 hidden units | 2-layer LSTM, 32 hidden units |
| **Input dimension** | d = 13 | d = 8 |
| **Lag (sequence length)** | w = 4 or 9 (from AIC) | 3 |
| **Learning rate** | 1 × 10⁻³ | 3 × 10⁻⁴ |
| **Batch size** | 32 | 256 |
| **Max epochs** | 100 | 2,000 |
| **Early stopping patience** | 5 | 80 |
| **Weight decay** | Not specified | 1 × 10⁻³ |
| **Dropout** | Not specified | 0.15 |
| **Gradient clipping** | Not specified | 1.0 |
| **Total params** | ~1,200 | ~5,500 |
| **Optimizer** | Adam | Adam |

### Why This Matters
Alvarez uses a **tiny** LSTM (16 hidden, 1 layer, ~1,200 params) and trains for only 100 epochs with patience=5. They don't need much capacity because:
1. The dynamics are smooth and well-captured by 13 POD modes.
2. Long training trajectories provide rich temporal context.
3. The LSTM only needs to learn gentle corrections over the MVAR baseline.

Our larger LSTM (32 hidden, 2 layers, ~5,500 params) with 2,000 epochs and patience=80 suggests we're compensating for harder dynamics with more capacity — but capacity alone doesn't solve nonstationary or short-trajectory problems.

---

## 7. Evaluation Protocol

| Aspect | Alvarez | Ours |
|---|---|---|
| **Test ICs** | 10 (different from training) | 26-30 (different from training) |
| **Test trajectory length** | 1,091 autoregressive steps | 47 autoregressive steps |
| **Primary metric** | Relative L₂ error (median + 10th/90th percentile) | R² (mean ± std) |
| **Conditioning** | First w steps from true trajectory | First p ROM frames from true trajectory |
| **Post-processing** | None (predictions stay non-negative) | Clamp negatives + mass renormalization |
| **Alvarez MVAR(4) result** | Median L₂ relative error = **25.4%** (range: 16.1%–31.0%) | — |
| **Alvarez LSTM(4) result** | Median L₂ relative error = **9.6%** (range: 5.1%–21.0%) | — |
| **Our V1 result** | — | R² = 0.28 (after clamp+renorm) |
| **Our V2 expected** | — | R² = 0.50–0.70 |

### Why This Matters
1. **Alvarez forecasts 1,091 steps** — a genuinely long horizon. Their MVAR achieves 25% error over this horizon, which is actually decent. Our 47-step forecast is much shorter, yet we get worse results.
2. **No post-processing needed**: Alvarez's predictions stay non-negative naturally because the POD modes + mean produce physically valid densities. Our predictions go 40% negative (V1), requiring clamping+renormalization — a sign that the MVAR predictions are far from the POD subspace.
3. **Different metrics**: Their relative L₂ error = 0.254 roughly translates to R² ≈ 1 − 0.254² ≈ 0.94 if errors were uniform (crude estimate). Our R² = 0.28 is dramatically worse.

---

## 8. Root Cause Summary: Why Their Pipeline Works and Ours Struggles

### Ranked by Impact (1 = most critical)

| # | Difference | Alvarez | Ours | Impact |
|---|---|---|---|---|
| 1 | **Trajectory length** | 1,100 steps/IC | 47-50 steps/IC | 🔴 Critical — MVAR never sees long-term dynamics |
| 2 | **Stationarity** | Verified (ADF p<0.001) | Not tested; likely nonstationary | 🔴 Critical — VAR requires stationarity |
| 3 | **Lag selection** | BIC/AIC (principled) | Manual guess | 🟡 Significant — wrong lag → over/underfitting |
| 4 | **Ridge α** | 1e-6 (data speaks) | 1.0 (brute force suppression) | 🟡 Symptom — massive α compensates for other issues |
| 5 | **POD energy** | 99% (13 modes) | 70-90% (8-19 modes) | 🟡 Trade-off — we sacrifice accuracy for tractability |
| 6 | **Post-processing** | Not needed | Clamp + renormalize (40% neg pixels) | 🟠 Symptom of unstable/inaccurate predictions |
| 7 | **Physics** | SFM (smooth, quasi-steady corridor flow) | Vicsek+Morse (complex self-organization) | 🟠 Harder dynamics ≠ impossible, but different |
| 8 | **LSTM capacity** | 16 hidden, 1 layer | 32 hidden, 2 layers | 🟢 Minor — more capacity isn't the bottleneck |
| 9 | **Density grid** | 80×20 = 1,600 px | 48×48 = 2,304 px | 🟢 Minor — similar effective dimensionality |

---

## 9. Recommendations: An Alvarez-Faithful Configuration

To perform a fair comparison, we should create a configuration that **mimics Alvarez's methodology as closely as possible** within our pipeline:

1. **Run fewer, longer simulations**: 10-20 ICs × T=60-100s (instead of 261 ICs × T=6s)
   - Each trajectory should show the full dynamical lifecycle
   - MVAR sees 500+ autoregressive steps per IC

2. **Verify stationarity**: Run ADF test on each POD coefficient time series
   - If nonstationary, consider differencing or detrending

3. **Use BIC/AIC for lag selection**: Search w ∈ {1, ..., 30} and select by BIC
   - This is straightforward to implement with sklearn

4. **Use energy threshold (99%) instead of fixed modes**
   - Let the data determine the right dimensionality

5. **Start with weak regularization**: α = 1e-6, increase only if unstable
   - If the dynamics are well-conditioned, weak α should suffice

6. **Match LSTM architecture**: 1 layer, 16 hidden units, patience=5, 100 epochs
   - Don't over-build the LSTM; let the data constrain the capacity

7. **Evaluate with relative L₂ error** for direct comparison with Alvarez's Table 2

See `configs/synthesis_v3_alvarez.yaml` for the implementation.

---

## 10. What Success/Failure Would Tell Us

### If V3 (Alvarez-style) Succeeds (R² > 0.7, ρ < 1 naturally):
→ The problem was **structural** (short trajectories, manual tuning, missing stationarity checks). Our dynamics are learnable by VAR if we respect the method's assumptions.

### If V3 Still Fails Despite Long Trajectories:
→ The Vicsek+Morse dynamics are **genuinely harder** than SFM corridor flow — either nonstationary, or nonlinear in a way that VAR cannot capture. At that point:
- LSTM should outperform MVAR (confirms nonlinearity)
- Consider NVAR (polynomial features) or DMD/EDMD
- Consider transforming the latent space (e.g., time-delay embedding)

### If V3 Works But Only With Strong α:
→ The dynamics are borderline — learnable with regularization but not naturally well-conditioned. This would suggest the POD latent space has mild nonstationarity or weak nonlinear coupling.
