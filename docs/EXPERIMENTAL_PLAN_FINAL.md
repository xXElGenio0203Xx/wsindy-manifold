# Final Experimental Plan

> **Status**: Locked 3 April 2026. Execute after preproc ablation (job 1209744) and
> Phase B (job 1206781) results are in.

---

## Configs to Remove (duplicates superseded by systematic suite)

### Legacy DYN configs — all 8 will not be re-run

| Legacy config | Superseded by |
|---|---|
| `DYN1_gentle` | `NDYN01_crawl` |
| `DYN1_gentle_wsindy` | subset of DYN1 |
| `DYN2_hypervelocity` | `NDYN03_sprint` |
| `DYN3_hypernoisy` | `NDYN04_gas` |
| `DYN4_blackhole` | `NDYN05_blackhole` + `NDYN11_noisy_collapse` |
| `DYN5_supernova` | `NDYN06_supernova` + `NDYN12_fast_explosion` |
| `DYN6_varspeed` | `_VS` variants |
| `DYN7_pure_vicsek` | `NDYN08_pure_vicsek` |

### Within NDYN extended — remove 3

| Config | Reason |
|---|---|
| `NDYN03_sprint` | Identical forces to `NDYN02_flock`; only speed differs. `NDYN13_chaos` covers extreme speed. |
| `NDYN11_noisy_collapse` | Near-duplicate of `NDYN05_blackhole` (same Ca≈12-scale attraction, differs only in noise). Noise sweep covers this axis. |
| `NDYN12_fast_explosion` | Near-duplicate of `NDYN06_supernova` (same Cr≈10-scale repulsion, differs only in speed). |

---

## Main Regimes (Final Set)

9 configs: 5 base + 4 VS (pure Vicsek has no VS — no forces to couple speed to).

| Regime | Behavioral axis | CS | VS |
|---|---|:---:|:---:|
| `NDYN04_gas` | Disordered (noise-dominated) | ✓ | ✓ |
| `NDYN05_blackhole` | Attractive collapse | ✓ | ✓ |
| `NDYN06_supernova` | Repulsive dispersal | ✓ | ✓ |
| `NDYN07_crystal` | Balanced equilibrium (Ca=Cr=3) | ✓ | ✓ |
| `NDYN08_pure_vicsek` | Alignment only (control) | ✓ | — |

**Note**: `NDYN07_crystal` needs to be promoted to `main_regimes/` and a `NDYN07_crystal_VS`
config added. Before generating Tier 1 configs, verify that
`configs/variable_speed/NDYN07_crystal_VS.yaml` exists. If not, create it from the base
crystal config with `speed_mode: variable` — one-line change, do not let this block
submission.

**Note on N**: All experiments use N=300. Current `main_regimes/` configs will be
updated accordingly when Tier 1 configs are generated.

---

## Lag Selection Strategy

**MVAR**: After Phase A (simulation + POD), run `alvarez_lag_selection.py` on every regime.
Record `w*_BIC` and `w*_AIC`. Run each regime at:
- `w=5` — baseline for comparability with Alvarez et al.
- `w=w*_BIC` — data-driven selection

**LSTM**: Use the same two lags as MVAR. Rationale: information-theoretic criteria are
well-defined for VAR but not for LSTMs (likelihood is not Gaussian). Borrowing the
MVAR-selected lag is the principled choice and ensures a common forecast start across models —
this is the approach taken in Alvarez et al.

**Lag cap**: If `w*_BIC > 30` for any regime, cap it at 30. At lag 30 and dt_latent≈0.12s
the forecast window shifts by 3.6s, which is acceptable. Beyond 30 the training dataset
shrinks enough (< 137 windows/trajectory at T_rom=167) that LSTM training becomes
unreliable. Check this after lag ablation results are in.

**Output**: A lag selection table in the thesis reporting `w*_BIC` per regime and whether
BIC-optimal lag improves forecast R² over `w=5`.

---

## Tier 1 — Main Regimes (18 OSCAR tasks)

**Per-task**: MVAR + LSTM + WSINDy (identification only)  
**Preprocessing**: best `{density_transform, mass_postprocess}` from preproc ablation  
**N**: 300

> **Mass postprocessing strategy (7 Apr 2026)**: For the thesis, report BOTH
> `mass_postprocess: none` (raw model quality) and `scale` (simple postprocessing).
> This is a stronger narrative than only showing `scale`, which masks model deficiencies.
> Reporting `none` reveals the true forecasting skill of each ROM; reporting `scale`
> shows what a cheap correction buys. Since `mass_postprocess` is evaluation-only
> (does not affect training), both sets of numbers can be obtained from a single run
> via `rerun_evaluation.py` with the config patched to `scale`.

| Config | Lag A | Lag B |
|---|---|---|
| `NDYN04_gas` | w=5 | w=w\*_BIC |
| `NDYN04_gas_VS` | w=5 | w=w\*_BIC |
| `NDYN05_blackhole` | w=5 | w=w\*_BIC |
| `NDYN05_blackhole_VS` | w=5 | w=w\*_BIC |
| `NDYN06_supernova` | w=5 | w=w\*_BIC |
| `NDYN06_supernova_VS` | w=5 | w=w\*_BIC |
| `NDYN07_crystal` | w=5 | w=w\*_BIC |
| `NDYN07_crystal_VS` | w=5 | w=w\*_BIC |
| `NDYN08_pure_vicsek` | w=5 | w=w\*_BIC |

---

## Tier 2 — Noise Sweep (SLIM: 6 OSCAR tasks)

> **Updated 2026-04-XX**: Slimmed from 20 to 6 tasks. Gas + blackhole only,
> 3 η levels each. Supernova and crystal dropped. Job 1560570.

**Regimes**: gas, blackhole (2 best-characterised force-bearing regimes)  
**Noise levels**: 3 per regime, native η included (bolded)  
**N**: 100 (consistent with thesis-final)  
**Lag**: w=5 only (baseline)  
**Models**: MVAR + LSTM + WSINDy

| Regime | Native η | Sweep values |
|---|:---:|---|
| `NDYN04_gas` | 1.5 | 0.5, **1.5**, 2.0 |
| `NDYN05_blackhole` | 0.15 | 0.05, **0.15**, 1.0 |

> Gas gets η=2.0 (above native) instead of 0.05 — the interesting direction for a
> noise-dominated regime is higher noise, not less. Crystal's native η=0.1 is between
> 0.05 and 0.15; both bracketing values are included.

**Thesis output**: R² vs η curves for MVAR/LSTM, WSINDy coefficient stability vs noise,
figure showing which terms survive at high noise and which become fragile.

---

## Tier 3 — Extended Regime Catalogue (~20 OSCAR tasks)

**Configs**: remaining NDYN (01, 02, 09, 10, 13, 14) + all 14 DO configs
(NDYN03/11/12 removed as duplicates; NDYN04–08 are main regimes)  
**Lag**: w=5 only  
**Models**: MVAR + LSTM only (no WSINDy)  
**Thesis output**: Appendix F — full regime catalogue with R² values demonstrating generality.

---

## Thesis Chapter Mapping

| Tier | Chapters |
|---|---|
| Tier 1 | Ch. 7 & 9: main-text tables, R² heatmaps, WSINDy coefficient tables, bootstrap stability, CS/VS paired comparison (see below), dominant balance heatmaps, lag selection table, Rosetta Stone |
| Tier 2 | New section: noise robustness — R² vs η curves, term survival at high noise |
| Tier 3 | Appendix F: full regime catalogue |
| Tier 4 | WSINDy chapter: N-convergence log-log figure + MVAR/LSTM R² vs N panel |

### CS/VS Paired Comparison — analysis template

For each of the 4 CS/VS pairs (gas, blackhole, supernova, crystal), produce a table with
one row per library term that is active in either variant:

| Term | CS coeff ± CI | VS coeff ± CI | Sign change? | Present in CS only? | Present in VS only? |
|---|---|---|:---:|:---:|:---:|

- **Bootstrap CI**: 95%, from the pipeline's existing bootstrap stability pass.  
- **Sign change**: flag any term where `sign(CS) ≠ sign(VS)` — these are the
  scientifically interesting cases (speed coupling flips a balance).
- **Presence asymmetry**: terms that survive STLSQ in one variant but are zeroed in the
  other indicate that variable speed activates or suppresses a physical mechanism.
- This table is the primary CS/VS output. Have the LaTeX template ready before results
  arrive so you fill in numbers rather than design the analysis under time pressure.

---

## Tier 4 — Particle-Count Convergence Study (6 OSCAR tasks)

**Regime**: `NDYN08_pure_vicsek` only  
**Sweep**: N ∈ {50, 100, 200, 300, 500, 1000}  
**Fixed**: grid 64×64, bandwidth h=5, same train/test IC counts as Tier 1
*(h is held fixed deliberately to isolate the N effect; a joint (N, h) sweep is left as
future work — at N=50 the KDE will be oversmoothed, which is an expected finding, not
an error)*  
**Models**: MVAR + LSTM + WSINDy  
**Lag**: w=5 (fixed, same as Tier 1 baseline)  

### Scientific motivation

Messenger & Bortz prove an $O(N^{-1/2})$ convergence rate for the weak-form
least-squares solution as the number of particles N → ∞. This study tests whether
that theoretical rate holds in the full Vicsek-Morse pipeline where:
- the density field is estimated via KDE (additional approximation layer)
- the PDE library includes nonlinear interaction terms not covered by the original proof

**Why pure Vicsek as control**: no Morse forces means no closure problem and no
`Ca`/`Cr` terms. Any coefficient drift with N is purely a finite-particle-count effect,
not a structural modeling issue. This isolates the convergence question cleanly.

**Why also run MVAR/LSTM**: they share the same Phase A simulation and POD, so the
marginal compute cost is small. The contrast between ROM R² vs N (expected: plateau
quickly — POD compresses regardless of N) and WSINDy coefficient error vs N (expected:
systematic improvement) reinforces the "complementary tools" narrative of the thesis.

### Primary outputs

1. **Log-log plot**: relative coefficient error $\|\hat{w}(N) - \hat{w}(N=1000)\|_2 /
   \|\hat{w}(N=1000)\|_2$ vs N, with $O(N^{-1/2})$ reference line overlaid.
2. **Supplementary**: MVAR and LSTM forecast R² vs N on the same x-axis.
3. **Bootstrap CI per term**: error bars on each active term's coefficient across the
   six N values — reveals which terms converge fastest.

### Reference and thesis framing

No closed-form continuum limit exists (that is the whole point of the thesis), so
$\hat{w}(N=1000)$ serves as the best available proxy for the true coefficient vector.
The convergence plot shows $\|\hat{w}(N) - \hat{w}(1000)\|_2$ vs N on log-log axes and
checks whether the empirical slope is $\approx -1/2$.

This is the same relative-convergence approach Messenger & Bortz use in their mean-field
paper when comparing against a known kernel — except their version uses absolute error.
The thesis sentence:

> "Since no closed-form continuum limit is available, we measure relative convergence
> using $\hat{w}(N=1000)$ as a proxy and note that the slope test remains valid as a
> consistency check."

If convergence **plateaus or oscillates** before the $N^{-1/2}$ regime, that is also
informative: it would pinpoint the KDE bandwidth as the bottleneck (systematic bias
that N alone cannot remove).

### Expected result

- Alignment coefficient (div p term) should converge fastest — largest signal-to-noise.
- Diffusion-like terms (∇²ρ, ∇²**p**) may converge more slowly (smaller relative magnitude).
- MVAR R² expected to be flat or weakly increasing with N (POD compresses regardless).
- Slope $\approx -1/2$ on the log-log plot = publishable validation of the Messenger-Bortz
  bound in a complex KDE-mediated PDE setting.

---

## Execution Sequence

```
[NOW RUNNING]
  1209744  preproc ablation (24 tasks, LSTM only) → best transform + mass_postprocess
  1208610  lag ablation (16 tasks, MVAR+LSTM)     → w*_BIC per regime
  1206781  Phase B (7 tasks, full)                → baseline LSTM R² at N=100

[AFTER RESULTS]
  Decision: N=100 or N=300 for main regimes?
  Decision: promote NDYN07_crystal to main_regimes/

[TIER 1 GENERATION]
  - Create 9 base configs + 9 BIC-lag copies = 18 configs
  - NDYN07_crystal(_VS) promoted from systematic/
  - Apply best preproc settings

[TIER 2 GENERATION]
  - 4 regimes × 5 η = 20 configs
  - generate_noise_sweep.py

[TIER 3 GENERATION]
  - remaining NDYN + all DO
  - generate_extended_catalogue.py

[TIER 4 GENERATION]
  - 6 configs: NDYN08_pure_vicsek_N{50,100,200,300,500,1000}
  - generate_n_convergence.py

Total: ~64 OSCAR tasks
```

---

## Total Task Count

| Tier | Tasks | Wall time estimate |
|---|:---:|---|
| Tier 1 (18 tasks, 5h each, 7 concurrent) | 18 | ~13 h |
| Tier 2 (20 tasks, 5h each, 7 concurrent) | 20 | ~15 h |
| Tier 3 (20 tasks, 3h each, 7 concurrent) | 20 | ~9 h |
| Tier 4 (6 tasks, 4h each, 6 concurrent) | 6 | ~4 h |
| **Total** | **64** | **~41 h wall** |
