# Collective Motion Models

Complete reference for all supported models in the rectsim package: D'Orsogna social forces, Vicsek alignment, and hybrid models.

---

## Table of Contents

1. [Overview](#overview)
2. [D'Orsogna Model](#dorsogna-model)
3. [Vicsek Model](#vicsek-model)
4. [Hybrid Models](#hybrid-models)
5. [Model Comparison](#model-comparison)
6. [Implementation Details](#implementation-details)
7. [Configuration Examples](#configuration-examples)

---

## Overview

**Three model families:**

1. **D'Orsogna (Social Force)**
   - Continuous ODE evolution
   - Morse potential (repulsion + attraction)
   - Self-propulsion + damping
   - Optional Vicsek alignment
   - Implementation: `src/rectsim/dynamics.py`

2. **Vicsek (Discrete Alignment)**
   - Discrete time steps
   - Alignment with neighbors
   - Constant speed constraint
   - Optional noise
   - Implementation: `src/rectsim/vicsek_discrete.py`

3. **Hybrid (Vicsek + Morse)**
   - Combines alignment with forces
   - Two integration schemes available
   - Variable particle speeds
   - Rich emergent behavior

---

## D'Orsogna Model

### Mathematical Formulation

**Position evolution:**
```
dx/dt = v
```

**Velocity evolution:**
```
dv/dt = (α - β|v|²)v + F_morse + F_noise

where:
- (α - β|v|²)v: Self-propulsion + damping
- F_morse: Pairwise Morse potential
- F_noise: Gaussian noise
```

**Morse potential:**
```
F_morse(r) = (C_rep/l_rep) exp(-r/l_rep) - (C_att/l_att) exp(-r/l_att)
```

### Key Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| `alpha` | α | Self-propulsion strength | 1.0-10.0 |
| `beta` | β | Velocity damping | 0.1-2.0 |
| `C_rep` | C_r | Repulsion magnitude | 1.0-10.0 |
| `l_rep` | l_r | Repulsion length scale | 0.5-2.0 |
| `C_att` | C_a | Attraction magnitude | 0.5-5.0 |
| `l_att` | l_a | Attraction length scale | 2.0-10.0 |

### Optional Vicsek Alignment Add-On

The D'Orsogna model can include Vicsek-style alignment as a **post-integration step**:

**Code location:** `src/rectsim/dynamics.py`, lines 336-352

```python
for step in pbar:
    # Step 1: Integrate forces (Morse + self-propulsion + damping)
    new_state = integrator(state, param_cfg, dt, force_calc, {"Lx": Lx, "Ly": Ly, "bc": bc})

    # Step 2: Apply Vicsek alignment (if enabled)
    if align_enabled:
        new_state.v = _alignment_step(
            new_state.x, new_state.v,
            Lx, Ly, bc,
            align_radius, align_rate, align_Dtheta, dt, v0_mag,
            rng=rng,
        )

    state = new_state
```

**Alignment function** (`_alignment_step`, lines 223-269):
- Computes neighbor-averaged velocity direction
- Retains particle speed magnitudes
- Adds angular noise `D_theta`
- Applies rate parameter `kappa` (0.0 = no alignment, 1.0 = full alignment)

**Alignment parameters:**
```yaml
forces:
  alignment:
    enabled: true
    kappa: 0.5         # Alignment rate (0-1)
    r_align: 3.0       # Alignment radius
    D_theta: 0.1       # Angular noise
```

### Configuration Example

```yaml
model: social_force
sim:
  N: 100
  Lx: 20.0
  Ly: 20.0
  bc: periodic
params:
  dt: 0.01
  T_max: 50.0
  alpha: 5.0
  beta: 1.0
forces:
  repulsion:
    C_rep: 5.0
    l_rep: 1.0
  attraction:
    C_att: 1.0
    l_att: 3.0
  alignment:
    enabled: true
    kappa: 0.5
    r_align: 3.0
noise:
  kind: gaussian
  sigma_v: 0.1
```

---

## Vicsek Model

### Mathematical Formulation

**Discrete time step evolution:**

1. **Heading Update** (alignment with neighbors):
   ```
   θᵢ(t+Δt) = ⟨θⱼ(t)⟩_neighbors + ξᵢ
   
   where:
   - ⟨θⱼ⟩: Average heading of neighbors within radius R
   - ξᵢ ~ Uniform(-η/2, η/2): Angular noise
   ```

2. **Position Update** (constant speed):
   ```
   xᵢ(t+Δt) = xᵢ(t) + v₀ Δt [cos(θᵢ), sin(θᵢ)]
   ```

### Key Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| `v0` | v₀ | Constant particle speed | 0.1-1.0 |
| `R` | R | Alignment interaction radius | 1.0-5.0 |
| `eta` | η | Angular noise magnitude | 0.0-2π |
| `dt` | Δt | Time step | 0.01-0.1 |

### Implementation

**File:** `src/rectsim/vicsek_discrete.py`

**Key features:**
- Discrete explicit Euler integration
- Periodic or reflecting boundary conditions
- Efficient neighbor search using cell lists
- Optional density field computation

### Files Using Vicsek Module

1. **`src/rectsim/__init__.py`**: Model routing
   ```python
   if model == "vicsek_discrete":
       from .vicsek_discrete import simulate_vicsek
       return simulate_vicsek(config["vicsek"])
   ```

2. **`src/rectsim/cli.py`**: CLI interface
   ```python
   if model == "vicsek_discrete":
       result = simulate_vicsek(cfg["vicsek"])
   ```

3. **`src/rectsim/config.py`**: Config validation
   ```python
   if model == "vicsek_discrete":
       vicsek = config.get("vicsek")
       # Validate vicsek-specific parameters
   ```

4. **`tests/test_vicsek_discrete.py`**: Unit tests

### Configuration Example

```yaml
model: vicsek_discrete
vicsek:
  N: 200
  Lx: 25.0
  Ly: 25.0
  v0: 0.5
  R: 1.0
  eta: 0.1
  dt: 0.1
  T_max: 100.0
  bc: periodic
  rho_grid_nx: 64
  rho_grid_ny: 64
  compute_density: true
```

---

## Hybrid Models

**Two implementations** combining Vicsek alignment with D'Orsogna forces:

### 1. Discrete Explicit Euler (Recommended)

**File:** `src/rectsim/vicsek_discrete.py`

**Features:**
- Discrete time steps
- Explicit Euler integration
- Variable particle speeds (due to forces)
- Fast, simple, interpretable

**Evolution equations:**

1. **Heading Update** (Vicsek alignment):
   ```
   θᵢ(t+Δt) = θᵢ(t) + φᵢ(t)
   where φᵢ ~ N(0, η) is angular noise
   ```

2. **Self-Propulsion Velocity**:
   ```
   uᵢ = v₀ [cos(θᵢ), sin(θᵢ)]
   ```

3. **Morse Forces**:
   ```
   Fᵢ = ∑ⱼ F_morse(rᵢⱼ) r̂ᵢⱼ
   F_morse(r) = (Cᵣ/lᵣ)exp(-r/lᵣ) - (Cₐ/lₐ)exp(-r/lₐ)
   ```

4. **Position Update** (discrete Euler):
   ```
   xᵢ(t+Δt) = xᵢ(t) + Δt · (uᵢ + μₜ Fᵢ)
   ```

**Key parameters:**
- `v0`: Base self-propulsion speed
- `μₜ` (`mu_t`): Translational mobility (force coupling strength)
- `Cr, Ca`: Repulsion/attraction magnitudes
- `lr, la`: Repulsion/attraction length scales
- `R`: Alignment radius

### 2. Continuous RK4

**File:** `src/rectsim/dynamics.py`

**Features:**
- Continuous ODE evolution
- 4th-order Runge-Kutta integration
- Self-propulsion term: (α - β|v|²)v
- More accurate, smoother trajectories

**Evolution equations:**
```
dx/dt = v
dv/dt = (α - β|v|²)v + F_morse + F_alignment + F_noise
```

**When to use:**
- Need high-accuracy trajectories
- Smooth velocity fields
- Research-grade simulations

### Hybrid Model Configuration

```yaml
model: vicsek_discrete
vicsek:
  N: 100
  Lx: 20.0
  Ly: 20.0
  v0: 0.5           # Self-propulsion speed
  R: 2.0            # Alignment radius
  eta: 0.1          # Angular noise
  mu_t: 1.0         # Force coupling strength
  dt: 0.01
  T_max: 50.0
  bc: periodic
  
  # Morse forces
  C_rep: 5.0
  l_rep: 1.0
  C_att: 1.0
  l_att: 3.0
  
  # Density computation
  compute_density: true
  rho_grid_nx: 64
  rho_grid_ny: 64
```

---

## Model Comparison

| Feature | D'Orsogna | Vicsek | Hybrid |
|---------|-----------|--------|--------|
| **Integration** | Continuous ODE | Discrete Euler | Both options |
| **Forces** | Morse potential | None | Morse potential |
| **Alignment** | Optional add-on | Core mechanism | Core mechanism |
| **Speed** | Variable | Constant | Variable |
| **Complexity** | Medium | Low | High |
| **Use Case** | Social forces | Flocking | Combined behavior |

### When to Use Each Model

**D'Orsogna:**
- Studying social force dynamics
- Smooth, continuous trajectories
- Repulsion/attraction balance
- Optional alignment as perturbation

**Vicsek:**
- Classic flocking behavior
- Fast simulations
- Constant-speed agents
- Phase transitions in collective motion

**Hybrid:**
- Emergent complex patterns
- Combination of forces and alignment
- Rich parameter space exploration
- Research on multi-mechanism models

---

## Implementation Details

### Integration Schemes

**Explicit Euler** (Vicsek discrete):
```python
x_new = x + dt * v
v_new = v + dt * F(x, v)
```
- Simple, fast
- First-order accuracy
- Small timesteps required for stability

**Runge-Kutta 4th Order** (D'Orsogna continuous):
```python
k1 = F(x, v)
k2 = F(x + dt/2*k1, v + dt/2*k1)
k3 = F(x + dt/2*k2, v + dt/2*k2)
k4 = F(x + dt*k3, v + dt*k3)
x_new = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
```
- More accurate
- Fourth-order accuracy
- Larger timesteps allowed

### Boundary Conditions

**Periodic:**
```python
x = x % Lx  # Wrap around domain
```

**Reflecting:**
```python
if x < 0 or x > Lx:
    x = clip(x, 0, Lx)
    v_x = -v_x  # Reverse velocity component
```

### Neighbor Search

**Cell List Algorithm:**
1. Divide domain into grid cells of size R (interaction radius)
2. Assign particles to cells
3. For each particle, only check neighboring cells
4. **Complexity:** O(N) instead of O(N²)

---

## Configuration Examples

### Classic D'Orsogna (Cohesive Clusters)

```yaml
model: social_force
sim:
  N: 200
  Lx: 30.0
  Ly: 30.0
  bc: periodic
params:
  dt: 0.01
  T_max: 100.0
  alpha: 5.0
  beta: 1.0
forces:
  repulsion:
    C_rep: 10.0
    l_rep: 1.0
  attraction:
    C_att: 2.0
    l_att: 5.0
noise:
  kind: gaussian
  sigma_v: 0.1
```

### Classic Vicsek (Phase Transition)

```yaml
model: vicsek_discrete
vicsek:
  N: 400
  Lx: 25.0
  Ly: 25.0
  v0: 0.3
  R: 1.0
  eta: 0.5      # Vary this to see phase transition
  dt: 0.1
  T_max: 200.0
  bc: periodic
```

### Hybrid (Rich Dynamics)

```yaml
model: vicsek_discrete
vicsek:
  N: 150
  Lx: 25.0
  Ly: 25.0
  v0: 0.5
  R: 2.0
  eta: 0.2
  mu_t: 1.0
  C_rep: 5.0
  l_rep: 1.0
  C_att: 1.0
  l_att: 3.0
  dt: 0.01
  T_max: 100.0
  bc: periodic
  compute_density: true
  rho_grid_nx: 64
  rho_grid_ny: 64
```

---

## Additional Resources

- **Main README**: [README.md](../README.md) - Installation and quickstart
- **Configuration Guide**: [DEVELOPMENT.md](../DEVELOPMENT.md) - Config schema details
- **Ensemble Guide**: [ENSEMBLE.md](../ENSEMBLE.md) - Multiple simulations
- **ROM/MVAR Pipeline**: [ROM_MVAR.md](../ROM_MVAR.md) - Reduced-order modeling
