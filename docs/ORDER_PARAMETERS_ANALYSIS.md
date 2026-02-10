# Order Parameters Analysis: What We Plot vs What We Have

**Document Purpose**: Identify order parameters actually used in official pipeline vs available but unused  
**Investigation Date**: February 3, 2026  
**Official Pipeline**: `run_unified_mvar_pipeline.py`  
**Analysis Scope**: Functions in `src/rectsim/` modules  

---

## Executive Summary

The official pipeline **currently plots 5 order parameters**, but we have **6 implemented functions** available. One key order parameter (**nematic order**) is implemented but **NOT being used** in the production pipeline.

---

## 1. Order Parameters CURRENTLY PLOTTED

**Source**: `src/rectsim/test_evaluator.py` (lines 257-272)

The official pipeline computes and saves these 5 order parameters for test runs:

### 1.1 Polarization (Φ)

**Function**: `standard_metrics.polarization(velocities)`

**Formula**:
$$
\Phi(t) = \left\| \frac{1}{N} \sum_{i=1}^N \frac{\mathbf{v}_i(t)}{|\mathbf{v}_i(t)|} \right\|
$$

**Range**: [0, 1]
- Φ = 0: Completely disordered
- Φ = 1: Perfect alignment

**CSV Column**: `phi`

---

### 1.2 Mean Speed (⟨|v|⟩)

**Function**: `standard_metrics.mean_speed(velocities)`

**Formula**:
$$
\bar{v}(t) = \frac{1}{N} \sum_{i=1}^N |\mathbf{v}_i(t)|
$$

**CSV Column**: `mean_speed`

---

### 1.3 Angular Momentum (L)

**Function**: `standard_metrics.angular_momentum(positions, velocities, center=None)`

**Formula**:
$$
L(t) = \frac{\left| \sum_{i=1}^N \mathbf{r}_i \times \mathbf{v}_i \right|}{N \langle |\mathbf{r}| \rangle \langle |\mathbf{v}| \rangle}
$$

where $\mathbf{r}_i \times \mathbf{v}_i = r_{i,x} v_{i,y} - r_{i,y} v_{i,x}$ (2D cross product, z-component)

**Measures**: Collective rotation around center of mass

**CSV Column**: `angular_momentum`

---

### 1.4 Density Variance (Var(ρ))

**Function**: `standard_metrics.density_variance(positions, domain_bounds, ...)`

**Formula**:
$$
\text{Var}(\rho) = \text{Var}\left\{\rho(x_i, y_j, t)\right\}_{i,j}
$$

**Measures**: Spatial clustering/heterogeneity
- High variance: Particles clustered in specific regions
- Low variance: Uniform spatial distribution

**CSV Column**: `density_variance`

**Note**: Currently returns placeholder (0.0) until proper KDE integration

---

### 1.5 Total Mass (∫ρ)

**Function**: `standard_metrics.total_mass(positions, domain_bounds, ...)`

**Formula**:
$$
m(t) = \int_\Omega \rho(x, y, t) \, dx \, dy
$$

**Purpose**: Verify mass conservation (should always ≈ 1.0 for mass-normalized KDE)

**CSV Column**: `total_mass`

**Note**: Currently returns placeholder (1.0) until proper KDE integration

---

## 2. Order Parameters AVAILABLE BUT NOT USED

### 2.1 Nematic Order (Q) ⭐ MISSING FROM PIPELINE

**Function**: `legacy_functions.nematic_order(vel, eps=1e-10)`

**Implementation**: Lines 234-285 in `src/rectsim/legacy_functions.py`

**Formula**:

Q-tensor:
$$
\mathbf{Q} = \frac{1}{N} \sum_{i=1}^N \left( \mathbf{n}_i \otimes \mathbf{n}_i - \frac{\mathbf{I}}{d} \right)
$$

where $\mathbf{n}_i = \mathbf{v}_i / |\mathbf{v}_i|$ (unit heading vectors)

Nematic order parameter:
$$
Q = \lambda_{\max}(\mathbf{Q})
$$

**Range**: [0, 1] in 2D

**Physical Interpretation**:
- Q = 0: Isotropic (random orientations)
- Q = 1: Perfect nematic order (aligned along one axis)
- **High Q with low Φ** → Lane formation (bidirectional flow)

**Why It Matters**:

The Q-tensor measures **second-order alignment** and is **insensitive to head-tail polarity**, making it ideal for detecting:
1. **Bidirectional patterns** (lanes with opposite flow directions)
2. **Alignment without consensus** (parallel but not coordinated)
3. **Liquid crystal phases** (nematic vs polar order)

**Key Difference from Polarization**:
- **Polarization (Φ)**: Measures first-order alignment (consensus direction)
- **Nematic (Q)**: Measures second-order alignment (axis orientation, ignores polarity)

**Example Scenarios**:

| System State | Φ (Polarization) | Q (Nematic) | Interpretation |
|-------------|------------------|-------------|----------------|
| Random walk | ~0 | ~0 | Disordered |
| Flocking | ~1 | ~1 | Collective motion (polar + nematic) |
| Lane formation | ~0 | ~1 | Bidirectional flow ⭐ |
| Vortex/mill | ~0 | intermediate | Local order, no global alignment |

**This is particularly important for Vicsek systems**, which can exhibit:
- Polar ordered phase (high Φ, high Q)
- Nematic ordered phase (low Φ, high Q) ← **NOT DETECTED** by current pipeline
- Disordered phase (low Φ, low Q)

---

### 2.2 Speed Standard Deviation (σᵥ)

**Function**: `legacy_functions.speed_std(vel)`

**Formula**:
$$
\sigma_v(t) = \sqrt{\frac{1}{N} \sum_{i=1}^N \left( |\mathbf{v}_i(t)| - \bar{v}(t) \right)^2}
$$

**Measures**: Speed heterogeneity (velocity fluctuations)

**Status**: Implemented but not used in pipeline

**Rationale for exclusion**: Mean speed already tracked; std adds limited information for constant-speed models (v₀ = 1.0)

---

## 3. Comparison with Visualization Pipeline

### 3.1 `rom_eval_viz.py` Order Parameters

The visualization module (`src/rectsim/rom_eval_viz.py`) computes a **different set** of order parameters:

**Function**: `compute_order_params_from_sample()` (lines 158-217)

**Computed Metrics**:
1. `polarization`: Same as production (Φ)
2. `speed_mean`: Same as production (⟨|v|⟩)
3. `speed_std`: Standard deviation of speeds ⭐ (not in production)

**Missing from visualization**:
- Angular momentum (L)
- Density variance (Var(ρ))
- Total mass (∫ρ)
- Nematic order (Q)

**Conclusion**: Visualization pipeline is **less comprehensive** than production pipeline.

---

## 4. Legacy Functions Not Integrated

**File**: `src/rectsim/legacy_functions.py`

### 4.1 Available Order Parameter Functions

| Function | Description | Used in Pipeline? |
|----------|-------------|-------------------|
| `polarization()` | Φ(t) - velocity alignment | ✅ Yes (via `standard_metrics`) |
| `mean_speed()` | ⟨\|v\|⟩ - average speed | ✅ Yes (via `standard_metrics`) |
| `speed_std()` | σᵥ - speed heterogeneity | ❌ No |
| `nematic_order()` | Q - second-order alignment | ❌ **No (IMPORTANT)** |

### 4.2 Wrapper Function

**Function**: `compute_order_params(vel, include_nematic=False)` (lines 285+)

**Purpose**: Unified interface for all velocity-based order parameters

**Returns**:
```python
{
    'phi': polarization(vel),
    'mean_speed': mean_speed(vel),
    'speed_std': speed_std(vel),
    'nematic': nematic_order(vel)  # if include_nematic=True
}
```

**Status**: Not used by official pipeline (which calls `standard_metrics` directly)

---

## 5. Summary Table

| Order Parameter | Symbol | Function | Production | Visualization | Legacy |
|-----------------|--------|----------|------------|---------------|--------|
| **Polarization** | Φ | `standard_metrics.polarization()` | ✅ | ✅ | ✅ |
| **Mean Speed** | ⟨\|v\|⟩ | `standard_metrics.mean_speed()` | ✅ | ✅ | ✅ |
| **Angular Momentum** | L | `standard_metrics.angular_momentum()` | ✅ | ❌ | ❌ |
| **Density Variance** | Var(ρ) | `standard_metrics.density_variance()` | ✅ (placeholder) | ❌ | ❌ |
| **Total Mass** | ∫ρ | `standard_metrics.total_mass()` | ✅ (placeholder) | ❌ | ❌ |
| **Speed Std Dev** | σᵥ | `legacy_functions.speed_std()` | ❌ | ✅ | ✅ |
| **Nematic Order** | Q | `legacy_functions.nematic_order()` | ❌ ⭐ | ❌ | ✅ |

**Legend**:
- ✅ = Implemented and used
- ❌ = Not used (even if implemented)
- ⭐ = **Should be added to production pipeline**

---

## 6. Recommendations for Thesis

### 6.1 Add Nematic Order to Production Pipeline

**Why**:
- **Critical for phase diagram**: Distinguishes polar vs nematic ordered phases
- **Lane detection**: Identifies bidirectional flow patterns (low Φ, high Q)
- **Alvarez comparison**: Alvarez et al. likely computed nematic order for their phase characterization
- **Complete picture**: (Φ, Q) together fully characterize orientational order

**How to integrate**:

**Option 1**: Add to `standard_metrics.py` (recommended)

```python
def nematic_order(velocities, eps=1e-10):
    """
    Compute nematic order parameter Q.
    
    Q-tensor: 𝐐 = (1/N) Σᵢ (𝐧ᵢ ⊗ 𝐧ᵢ - 𝐈/d)
    Nematic order: Q = λₘₐₓ(𝐐)
    
    Detects second-order alignment (insensitive to head-tail polarity).
    High Q with low Φ indicates bidirectional flow (lane formation).
    
    Parameters
    ----------
    velocities : ndarray, shape (N, 2)
        Velocity vectors
    eps : float, optional
        Small constant to avoid division by zero
        
    Returns
    -------
    float
        Nematic order parameter in [0, 1]
    """
    N, d = velocities.shape
    if N == 0:
        return 0.0
    
    # Normalize to unit vectors
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    n = velocities / (speeds + eps)
    
    # Compute Q-tensor
    Q = np.zeros((d, d))
    for i in range(N):
        Q += np.outer(n[i], n[i])
    Q = Q / N - np.eye(d) / d
    
    # Max eigenvalue
    eigvals = np.linalg.eigvalsh(Q)
    return float(np.max(eigvals))
```

**Option 2**: Import from `legacy_functions` directly

Update `test_evaluator.py`:
```python
from rectsim.legacy_functions import nematic_order

# In evaluate_test_runs():
order_params = compute_metrics_series(x, v, domain_bounds, ...)
nematic_series = np.array([nematic_order(v[t]) for t in range(len(v))])

op_df = pd.DataFrame({
    't': test_times,
    'phi': order_params['polarization'],
    'mean_speed': order_params['mean_speed'],
    'angular_momentum': order_params['angular_momentum'],
    'density_variance': order_params['density_variance'],
    'total_mass': order_params['total_mass'],
    'nematic': nematic_series  # NEW
})
```

### 6.2 Document Order Parameter Definitions

**For thesis Chapter "Evaluation Metrics"**:

Add section:

```latex
\subsection{Order Parameters from Particle Trajectories}

In addition to density-based ROM evaluation metrics, we compute
order parameters directly from particle trajectories to characterize
system-level behavior.

\subsubsection{Polarization (Polar Order)}

$$
\Phi(t) = \left\| \frac{1}{N} \sum_{i=1}^N \hat{\mathbf{v}}_i(t) \right\|
$$

Measures consensus in velocity directions (first-order alignment).

\subsubsection{Nematic Order (Axial Alignment)}

$$
Q(t) = \lambda_{\max}\left( \frac{1}{N} \sum_{i=1}^N 
\left( \hat{\mathbf{v}}_i \otimes \hat{\mathbf{v}}_i - \frac{\mathbf{I}}{2} \right) \right)
$$

Measures alignment along an axis, insensitive to polarity. 
Critical for detecting lane formation (high $Q$, low $\Phi$).

\subsubsection{Combined Phase Diagram}

The $(Q, \Phi)$ plane characterizes three regimes:
\begin{itemize}
    \item \textbf{Disordered}: $Q \approx 0$, $\Phi \approx 0$
    \item \textbf{Polar}: $Q \approx 1$, $\Phi \approx 1$ (flocking)
    \item \textbf{Nematic}: $Q \approx 1$, $\Phi \approx 0$ (lanes)
\end{itemize}
```

### 6.3 Complete KDE Integration

**For density_variance and total_mass**:

Currently these return placeholders. For thesis completeness:

1. Import proper KDE function from `legacy_functions.kde_density_movie()`
2. Update `standard_metrics.density_variance()` and `total_mass()` to compute actual values
3. Verify mass conservation: should have $|m(t) - N| < 0.01 N$ for all $t$

---

## 7. Code Files Summary

### 7.1 Active Production Files

**Order Parameter Computation**:
- `src/rectsim/standard_metrics.py`: Production order parameters (5 functions)
- `src/rectsim/test_evaluator.py`: Pipeline integration (calls `compute_metrics_series()`)
- `src/rectsim/io_outputs.py`: CSV export (`save_order_parameters_csv()`)

### 7.2 Unused But Implemented

**Legacy Functions**:
- `src/rectsim/legacy_functions.py`: Contains `nematic_order()`, `speed_std()`, wrapper functions

**Visualization**:
- `src/rectsim/rom_eval_viz.py`: Simplified order parameters (3 metrics: Φ, ⟨|v|⟩, σᵥ)

---

## 8. Key Finding for Thesis

**CRITICAL GAP**: The production pipeline does **not compute nematic order (Q)**, which is essential for:

1. **Phase characterization**: Distinguishing polar vs nematic phases
2. **IC family analysis**: Ring and two-cluster ICs likely exhibit nematic ordering
3. **Completeness**: Standard in active matter literature (see Vicsek, Grégoire, Chaté, etc.)
4. **Comparison with Alvarez**: Their work on self-propelled particles likely tracked (Φ, Q)

**Recommendation**: Add nematic order to production pipeline before finalizing thesis experiments.

---

**Document Version**: 1.0  
**Last Updated**: February 3, 2026  
**Author**: Maria  
**Status**: Complete ✓
