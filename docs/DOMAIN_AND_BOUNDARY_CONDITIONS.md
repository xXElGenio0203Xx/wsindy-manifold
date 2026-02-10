# Simulation Domain and Boundary Conditions

## Overview

This document describes the spatial domain used for collective motion simulations, including geometry, boundary conditions, and their implementation. The domain configuration critically affects particle interactions, density distributions, and ROM performance.

---

## Domain Geometry

### 1. Rectangular Domain

All simulations occur in a **2D rectangular domain**:

$$
\Omega = [0, L_x] \times [0, L_y]
$$

Where:
- $L_x$ = domain width (x-direction)
- $L_y$ = domain height (y-direction)
- Particle positions: $\mathbf{x}_i = (x_i, y_i) \in \Omega$

**Standard configuration:**
```yaml
domain:
  Lx: 20.0    # Width (typical range: 10-30)
  Ly: 20.0    # Height (typically Lx = Ly for square domain)
  bc: "periodic"  # Boundary condition
```

**Typical domain sizes in experiments:**

| Experiment Type | $L_x \times L_y$ | Particle Count $N$ | Density $\rho = N/(L_x L_y)$ |
|----------------|------------------|-------------------|------------------------------|
| Standard | 20 × 20 | 100-400 | 0.25-1.0 particles/unit² |
| Low density | 30 × 30 | 100 | 0.11 particles/unit² |
| High density | 10 × 10 | 400 | 4.0 particles/unit² |

**Design considerations:**
- **Interaction radius $R$**: Typically $R = 1.0$-$3.0$
- **Rule of thumb**: $L_x, L_y \gg R$ to avoid finite-size effects
- **Typical ratio**: $L_x / R \approx 10$-$20$ ensures bulk behavior
- **Square vs. rectangular**: Most experiments use $L_x = L_y$ (square) for symmetry

---

## Boundary Conditions

### 2. Two Boundary Condition Types

**Location:** `src/rectsim/domain.py` → `apply_bc()`

Our implementation supports two fundamentally different boundary treatments:

#### **a) Periodic Boundaries (Default)**

**Description:**
- Domain wraps around like a torus
- Particles leaving through one edge re-enter through opposite edge
- No physical walls or barriers
- Preserves translational invariance

**Mathematical formulation:**
$$
\begin{aligned}
x_i &\to x_i \mod L_x \\
y_i &\to y_i \mod L_y
\end{aligned}
$$

**Code implementation:**
```python
def apply_bc(x, Lx, Ly, bc="periodic"):
    if bc == "periodic":
        x[:, 0] = np.mod(x[:, 0], Lx)  # Wrap x-coordinate
        x[:, 1] = np.mod(x[:, 1], Ly)  # Wrap y-coordinate
        flips = np.zeros_like(x, dtype=bool)  # No velocity flips
        return x, flips
```

**Minimal image convention:**
For computing pairwise distances under periodic BC:
$$
\Delta x = x_j - x_i - L_x \cdot \text{round}\left(\frac{x_j - x_i}{L_x}\right)
$$

This ensures the shortest distance across periodic boundaries.

**Example:**
```
Particle at x=19.5 with velocity v_x=1.0, dt=1.0
→ x_new = 20.5 → wraps to x=0.5 (enters from left)
```

**Use cases:**
- **Standard for Vicsek models** (mimics infinite system)
- **Uniform statistical properties** (no edge effects)
- **ROM/density field training** (all experiments use periodic)

#### **b) Reflecting Boundaries**

**Description:**
- Domain has physical walls
- Particles bounce off boundaries
- Velocity component normal to wall is reversed
- Confines particles to finite region

**Mathematical formulation:**
For particle crossing boundary at $x=0$ or $x=L_x$:
$$
\begin{aligned}
x_i &\to -x_i \quad &\text{(if } x_i < 0\text{)} \\
x_i &\to 2L_x - x_i \quad &\text{(if } x_i > L_x\text{)} \\
v_{x,i} &\to -v_{x,i} \quad &\text{(flip velocity)}
\end{aligned}
$$

**Code implementation:**
```python
def apply_bc(x, Lx, Ly, bc="reflecting"):
    if bc == "reflecting":
        flips = np.zeros_like(x, dtype=bool)
        for dim, L in enumerate((Lx, Ly)):
            below = x[:, dim] < 0
            above = x[:, dim] > L
            if np.any(below):
                x[below, dim] = -x[below, dim]      # Mirror position
                flips[below, dim] = True             # Flag velocity flip
            if np.any(above):
                x[above, dim] = 2*L - x[above, dim]  # Mirror position
                flips[above, dim] = True
        return x, flips
```

**Example:**
```
Particle at x=19.5 with velocity v_x=1.0, dt=1.0
→ x_new = 20.5 → reflects to x=19.5, v_x → -1.0
```

**Use cases:**
- **Confined systems** (e.g., particles in a box)
- **Wall effects** (boundary accumulation)
- **Less common** in collective motion studies

---

## Implementation Details

### 3. Domain Module

**Location:** `src/rectsim/domain.py` (316 lines)

**Key functions:**

#### **`apply_bc(x, Lx, Ly, bc)`**

Apply boundary conditions to positions.

**Returns:**
- `x`: Adjusted positions (wrapped or reflected)
- `flips`: Boolean mask `(N, 2)` indicating which velocity components to flip

**Usage in simulation loop:**
```python
# After position update
x, flips = apply_bc(x, Lx, Ly, bc="periodic")
if bc == "reflecting":
    v[flips] *= -1  # Flip velocities at walls
```

#### **`pair_displacements(x, Lx, Ly, bc)`**

Compute pairwise separations with BC handling.

**Returns:**
- `dx, dy`: Displacement matrices `(N, N)`
- `rij`: Distance matrix `(N, N)`
- `unit_vectors`: Direction vectors `(N, N, 2)`

**Handles minimal image for periodic BC:**
```python
if bc == "periodic":
    dx -= Lx * np.round(dx / Lx)  # Shortest path across boundaries
    dy -= Ly * np.round(dy / Ly)
```

#### **`build_cells(x, Lx, Ly, rcut, bc)`**

Construct linked-cell list for efficient neighbor searches.

**Algorithm:**
- Divide domain into grid: $n_x \times n_y$ cells
- Cell size: $\Delta_{\text{cell}} \approx r_{\text{cut}}$
- Assign particles to cells based on position
- For periodic BC: wrap cell indices modulo $n_x, n_y$

**Complexity:**
- Build: $O(N)$
- Neighbor query: $O(N)$ average (vs. $O(N^2)$ naive)

---

## Boundary Conditions in Different Components

### 4. BC Propagation Through Pipeline

**Microsimulation:**
- `src/rectsim/vicsek_discrete.py`: Reads `config["sim"]["bc"]`
- Applies BC every timestep via `apply_bc()`
- Affects:
  - Position wrapping/reflection
  - Neighbor search (minimal image)
  - Force computation

**Density Field (KDE):**
- `src/rectsim/legacy_functions.py` → `kde_density_movie()`
- Gaussian filter mode depends on BC:
  ```python
  mode = "wrap" if bc == "periodic" else "nearest"
  gaussian_filter(density, sigma=bandwidth, mode=mode)
  ```

**Periodic (`mode="wrap"`):**
- Kernel wraps around edges
- No edge artifacts
- Continuous density across boundaries
- **Used for all ROM training**

**Reflecting (`mode="nearest"`):**
- Edge values repeated beyond boundary
- Higher density near walls (particle accumulation)
- Discontinuity at boundaries

**POD Compression:**
- Agnostic to BC (operates on density snapshots)
- But trained densities inherit BC properties

**MVAR/LSTM Forecasting:**
- Indirect BC effect through latent dynamics
- Periodic BC → smoother latent evolution → better ROM R²
- Reflecting BC → boundary effects in latent space

---

## Practical Examples

### 5. Configuration Examples

#### **Standard Periodic Setup (Production)**

```yaml
sim:
  N: 200
  Lx: 20.0
  Ly: 20.0
  bc: "periodic"
  T: 10.0
  dt: 0.1
  v0: 0.5
  R: 1.0
```

**Characteristics:**
- Square domain
- Moderate density: $\rho = 200/400 = 0.5$
- Interaction range: $R/L_x = 1/20 = 5\%$ of domain
- Bulk-like behavior

#### **Confined System (Reflecting)**

```yaml
sim:
  N: 100
  Lx: 10.0
  Ly: 10.0
  bc: "reflecting"
  T: 10.0
  dt: 0.1
  v0: 0.5
  R: 1.0
```

**Characteristics:**
- Smaller domain
- Density: $\rho = 100/100 = 1.0$
- Particles accumulate near walls
- Finite-size effects stronger

#### **Large Domain (Low Density)**

```yaml
sim:
  N: 100
  Lx: 30.0
  Ly: 30.0
  bc: "periodic"
  T: 10.0
  dt: 0.1
  v0: 0.5
  R: 1.0
```

**Characteristics:**
- Sparse: $\rho = 100/900 = 0.11$
- Fewer interactions
- Slower collective dynamics
- Closer to dilute limit

---

## Boundary Condition Effects

### 6. Impact on Dynamics

**Periodic BC:**

| Aspect | Effect |
|--------|--------|
| **Symmetry** | Full translational invariance |
| **Clustering** | Clusters can merge across boundaries |
| **Order parameters** | No edge effects, clean averaging |
| **Density field** | Uniform statistics |
| **ROM performance** | Smoother latent dynamics → higher R² |

**Reflecting BC:**

| Aspect | Effect |
|--------|--------|
| **Symmetry** | Broken at boundaries |
| **Clustering** | Particles accumulate at walls |
| **Order parameters** | Edge effects in polarization |
| **Density field** | Boundary peaks |
| **ROM performance** | Discontinuities complicate latent space |

### 7. Interaction Range Considerations

**Cutoff radius vs. domain size:**

For periodic BC with minimal image:
- If $R > L_x/2$: Particle interacts with its own periodic image
- **Avoid:** $R \geq L_x/2$ (unphysical self-interaction)
- **Safe:** $R < L_x/3$ (clear separation)

**Example:**
```
Lx = 20.0, R = 1.0 → R/Lx = 0.05 ✓ Safe
Lx = 20.0, R = 12.0 → R/Lx = 0.60 ✗ Self-interaction!
```

---

## Neighbor Search Optimization

### 8. Linked-Cell Algorithm

For efficient $O(N)$ neighbor finding:

**Cell construction:**
```python
ncellx = max(1, int(np.ceil(Lx / rcut)))
ncelly = max(1, int(np.ceil(Ly / rcut)))
cell_size_x = Lx / ncellx
cell_size_y = Ly / ncelly
```

**Particle assignment:**
```python
for idx, (xi, yi) in enumerate(x):
    ix = int(np.floor(xi / cell_size_x))
    iy = int(np.floor(yi / cell_size_y))
    cells[(ix, iy)].append(idx)
```

**Neighbor search:**
For each cell $(i_x, i_y)$, check:
- Self: $(i_x, i_y)$
- Adjacent: $(i_x \pm 1, i_y \pm 1)$ (9 cells total)

**Periodic BC handling:**
```python
for dx_cell in (-1, 0, 1):
    nx = ix + dx_cell
    if bc == "periodic":
        nx %= ncellx  # Wrap cell index
```

**Performance:**
- Typical: $n_x \times n_y \approx 20 \times 20 = 400$ cells
- $N = 200$ particles → ~0.5 particles/cell
- Check ~10 neighbors per particle instead of 200

---

## Validation and Testing

### 9. Boundary Condition Tests

**Location:** `tests/test_bc.py`

**Test 1: Periodic wrapping**
```python
x = np.array([[1.2, -0.1], [9.9, 4.8]])  # Out-of-bounds
wrapped, flips = apply_bc(x, Lx=10.0, Ly=5.0, bc="periodic")
assert np.all((0 <= wrapped) & (wrapped < [10.0, 5.0]))  # In bounds
assert not flips.any()  # No velocity flips
```

**Test 2: Minimal image distance**
```python
dx, dy, rij, _ = pair_displacements(wrapped, Lx, Ly, bc="periodic")
# Distance should use shortest path across boundary
assert rij[0,1] <= np.linalg.norm(wrapped[1] - wrapped[0])
```

**Test 3: Reflecting boundary**
```python
x = np.array([[-0.5, 6.0]])  # Out-of-bounds
wrapped, flips = apply_bc(x, Lx=5.0, Ly=5.0, bc="reflecting")
assert 0 <= wrapped[0,0] <= 5.0  # Reflected back in
assert flips[0,0] and flips[0,1]  # Velocity should flip
```

---

## Domain in Different Model Types

### 10. Model-Specific Considerations

**Discrete Vicsek:**
- Update: $\mathbf{x}_i(t+1) = \mathbf{x}_i(t) + v_0 \hat{\mathbf{v}}_i(t) \Delta t$
- BC applied after each position update
- Periodic standard (mimics infinite flock)

**Continuous D'Orsogna:**
- ODE: $\dot{\mathbf{x}}_i = \mathbf{v}_i$
- BC checked at integration steps
- Both periodic and reflecting used

**Hybrid (Vicsek + Forces):**
- Alignment: uses neighbor list (periodic wrapping)
- Forces: computed via `pair_displacements()` (minimal image)
- BC consistent across all interactions

---

## Configuration Validation

### 11. Domain Checks

**Location:** `src/rectsim/unified_config.py` → `validate_config()`

**Validation rules:**
```python
errors = []

# Positive dimensions
if domain['Lx'] <= 0 or domain['Ly'] <= 0:
    errors.append("Domain dimensions must be positive")

# Valid BC type
if domain['bc'] not in ['periodic', 'reflecting']:
    errors.append(f"Invalid boundary condition: {domain['bc']}")

# Interaction range sanity
if params['R'] >= domain['Lx'] / 2:
    warnings.append(f"R={R} is large compared to Lx={Lx}")
```

---

## Summary for Thesis

### 12. Key Points

**Domain specification:**
> All simulations occur in a rectangular domain $\Omega = [0, L_x] \times [0, L_y]$ with $L_x = L_y = 20$ (square) in standard configurations. Particle density $\rho = N/(L_x L_y)$ ranges from 0.25 to 1.0 particles per unit area.

**Boundary conditions:**
> We employ **periodic boundary conditions** exclusively for ROM training, where the domain wraps toroidally: $\mathbf{x} \to \mathbf{x} \mod \mathbf{L}$. This eliminates edge effects, ensures translational invariance, and provides uniform statistical properties across the domain. Pairwise interactions use the minimal image convention to compute shortest distances across periodic boundaries.

**Interaction range:**
> Alignment interactions occur within radius $R = 1.0$, giving $R/L_x = 0.05$, well below the $L_x/2$ limit to avoid periodic self-interaction. This ensures bulk-like collective behavior without finite-size artifacts.

**Computational implementation:**
> Efficient $O(N)$ neighbor finding is achieved via linked-cell lists with cell size $\approx R$, dividing the domain into $\sim 20 \times 20$ spatial cells. Periodic boundary wrapping is handled at both the position update (modulo arithmetic) and density field computation (Gaussian filter `mode="wrap"`) stages.

**Effect on ROM:**
> Periodic boundaries yield smooth, continuous density fields with no edge discontinuities. This directly benefits POD compression (cleaner modes) and MVAR/LSTM forecasting (smoother latent dynamics), contributing to high forecast R² > 0.95 in typical experiments.

---

## Files and References

### 13. Key Implementation Files

| File | Description | Lines | Domain Role |
|------|-------------|-------|-------------|
| `src/rectsim/domain.py` | BC implementation, neighbor search | 316 | Core domain utilities |
| `src/rectsim/unified_config.py` | Configuration schema with defaults | 344 | Domain specification |
| `src/rectsim/vicsek_discrete.py` | Discrete Vicsek simulator | ~500 | Applies BC at each step |
| `src/rectsim/legacy_functions.py` | KDE density with BC handling | ~1800 | BC-aware smoothing |
| `tests/test_bc.py` | Boundary condition tests | ~30 | Validation |

### 14. Configuration Examples

**All configs use:**
```yaml
sim:
  Lx: 20.0
  Ly: 20.0
  bc: "periodic"
```

Files: `configs/*.yaml` (50+ configuration files)

**Typical setups:**
- `configs/vicsek_morse_base.yaml`: Standard 20×20 periodic
- `configs/long_duration_d40.yaml`: Standard domain, long simulation
- `configs/alvarez_*.yaml`: Systematic studies, consistent domain

---

## References

**Vicsek model conventions:**
- Vicsek et al. (1995): Used periodic BC in original paper
- Grégoire & Chaté (2004): Standard practice for flocking studies

**Linked-cell algorithm:**
- Allen & Tildesley, *Computer Simulation of Liquids* (2017)
- Verlet lists and cell lists for molecular dynamics

**Our implementation:**
- Based on standard practices from computational physics
- Optimized for rectangular domains
- Unified interface across discrete/continuous models
