# D'Orsogna Alignment Mechanism

## Overview

The D'Orsogna model in `src/rectsim/dynamics.py` uses **Vicsek-style alignment** as an optional add-on to the Morse force dynamics. This is applied AFTER the force-based integration step.

## Code Location

The alignment is implemented in three functions in `src/rectsim/dynamics.py`:

### 1. Main Simulation Loop (lines 336-352)
```python
for step in pbar:
    if step % neighbor_rebuild == 1:
        force_calc.rebuild(state.x)

    # Step 1: Integrate forces (Morse potential + self-propulsion + damping)
    new_state = integrator(
        state,
        param_cfg,
        dt,
        force_calc,
        {"Lx": Lx, "Ly": Ly, "bc": bc},
    )

    # Step 2: Apply Vicsek alignment (if enabled)
    if align_enabled:
        new_state.v = _alignment_step(
            new_state.x,
            new_state.v,
            Lx,
            Ly,
            bc,
            align_radius,
            align_rate,
            align_Dtheta,
            dt,
            v0_mag,
            rng=rng,
        )

    state = new_state
```

### 2. Alignment Wrapper (lines 223-269)
```python
def _alignment_step(
    x: ArrayLike,
    v: ArrayLike,
    Lx: float,
    Ly: float,
    bc: str,
    radius: float,
    rate: float,
    Dtheta: float,
    dt: float,
    target_speed: float,
    rng: np.random.Generator | None = None,
) -> ArrayLike:
    """Apply Vicsek alignment to the velocity field while retaining speeds."""
    
    # Extract speeds and unit headings from velocity vectors
    speeds = np.linalg.norm(v, axis=1, keepdims=True)
    headings = v / speeds  # Unit vectors
    
    # Apply Vicsek alignment to headings only
    headings_new = vicsek_alignment_step(
        x,
        headings,
        Lx,
        Ly,
        bc,
        radius,
        rate,
        Dtheta,
        dt,
        cell_list=cell_list,
        rng=rng,
    )
    
    # Restore original speeds
    return headings_new * speeds
```

### 3. Core Vicsek Alignment (lines 178-220)
```python
def vicsek_alignment_step(
    x: ArrayLike,
    p: ArrayLike,  # Unit heading vectors
    Lx: float,
    Ly: float,
    bc: str,
    lV: float,      # Alignment radius
    mu_r: float,    # Alignment rate
    Dtheta: float,  # Rotational diffusion
    dt: float,
    cell_list: CellList | None = None,
    rng: np.random.Generator | None = None,
) -> ArrayLike:
    """Vicsek alignment update (AIM-1 Eq. 6) using linked-cell neighbours."""
    
    # Find neighbors within radius lV
    neighbours = neighbor_indices_from_celllist(x, local_cells, Lx, Ly, lV, bc)
    
    # For each particle
    for i in range(p.shape[0]):
        idx = neighbours[i]
        
        # Alignment drift: turn towards mean heading of neighbors
        if mu_r > 0.0 and idx.size:
            mean_vec = np.sum(p[idx], axis=0)
            mean_vec /= norm(mean_vec)  # Normalize
            drift_vec = mu_r * (mean_vec - p[i])
        
        # Rotational noise
        noise = sqrt(2 * Dtheta * dt) * random_normal()
        
        # Update heading (Euler-Maruyama)
        p_tmp = p[i] + drift_vec * dt + noise
        p_new[i] = p_tmp / norm(p_tmp)  # Re-normalize
    
    return p_new
```

## Mathematical Formulation

The alignment follows the stochastic differential equation (SDE):

```
dθᵢ/dt = μᵣ · sin(⟨θ⟩ⱼ - θᵢ) + √(2Dθ) · ξᵢ(t)
```

Where:
- `θᵢ`: Heading angle of particle i
- `⟨θ⟩ⱼ`: Mean heading of neighbors j within radius lV
- `μᵣ`: Alignment rate (how quickly particles turn to match neighbors)
- `Dθ`: Rotational diffusion coefficient (angular noise)
- `ξᵢ(t)`: White noise

This is implemented in **vector form** to avoid explicit angle calculations:
```
p̂ᵢ(t+dt) = normalize(p̂ᵢ(t) + μᵣ·(⟨p̂⟩ⱼ - p̂ᵢ)·dt + √(2Dθ·dt)·ηᵢ)
```

## Key Parameters

From your config `dorsogna_low_speed_aligned.yaml`:

```yaml
alignment:
  enabled: true
  radius: 2.0      # lV: neighborhood radius for alignment
  rate: 2.0        # μᵣ: alignment strength (how fast to turn)
  Dtheta: 0.05     # Dθ: rotational noise (angular randomness)
```

## Integration with D'Orsogna Dynamics

The full D'Orsogna model with alignment combines:

1. **Force Integration** (first):
   - Morse forces: `F = Cᵣ/lᵣ·exp(-r/lᵣ) - Cₐ/lₐ·exp(-r/lₐ)`
   - Self-propulsion: `α·v̂`
   - Friction: `-β·|v|·v`

2. **Alignment** (second, if enabled):
   - Adjusts heading direction only
   - Preserves speed magnitude from force integration

This creates a **hybrid model**:
- Forces control spacing and cohesion
- Alignment synchronizes velocity directions
- Together: cohesive flocks with regular spacing!

## Comparison to Pure Vicsek

**Pure Vicsek** (in `vicsek_discrete.py`):
- Fixed speed v₀
- Only alignment interactions
- Discrete time steps

**D'Orsogna + Alignment** (in `dynamics.py`):
- Variable speeds from force balance
- Alignment + spatial forces
- Continuous-time integration (RK4 or semi-implicit Euler)

## Reference

This implements the alignment rule from:
- **AIM-1 paper**: The alignment equation (Eq. 6) from the Active Inference Models paper
- Based on the classic Vicsek model (Vicsek et al., 1995)
- Extended to handle variable speeds in force-based models
# D'Orsogna vs Discrete Vicsek: Separate Models

## You're Correct! 

The **D'Orsogna model** (continuous, force-based) and **discrete Vicsek model** are **completely separate implementations** that don't work together.

## Two Different Simulators

### 1. D'Orsogna Model (Continuous)
**File**: `src/rectsim/dynamics.py`  
**Entry point**: `simulate(config)`  
**Runner script**: `scripts/run_dorsogna.py`

```python
# Continuous-time ODE integration
def simulate(config):
    # Initialize particles
    # For each time step:
    #   1. Compute Morse forces: F = f(positions)
    #   2. Integrate ODEs: dv/dt = α - β|v|²v + μₜF
    #   3. Update positions: dx/dt = v
    #   4. (Optional) Apply Vicsek alignment to headings
    return results
```

**Physics**:
- Continuous time (RK4 or semi-implicit Euler)
- Variable speeds from force balance
- Morse potential forces (repulsion + attraction)
- Self-propulsion + friction
- Optional Vicsek alignment on top

**Config type**: `model.type: dorsogna`

---

### 2. Discrete Vicsek Model
**File**: `src/rectsim/vicsek_discrete.py`  
**Entry point**: `simulate_vicsek(config)` or `simulate_backend(config, rng)`  
**Runner script**: `scripts/run_standardized.py`

```python
# Discrete-time update rule
def simulate_backend(config, rng):
    # Initialize particles with fixed speed v0
    # For each time step:
    #   1. Find neighbors within radius R
    #   2. Compute mean heading: θ̄ = angle(Σ v̂ⱼ)
    #   3. Update heading: θᵢ = θ̄ + noise
    #   4. Move: xᵢ += v₀ · (cos θᵢ, sin θᵢ) · dt
    #   (Force hook is placeholder only!)
    return results
```

**Physics**:
- Discrete time steps
- **Fixed constant speed** v₀
- Only alignment interactions
- Noise on heading angle
- **No forces** (placeholder exists but not implemented)

**Config type**: `model.type: vicsek`

---

## Router in `__init__.py`

The main entry point routes to the correct simulator:

```python
def simulate(config: Dict[str, Any]):
    from .dynamics import simulate as _simulate
    
    model = config.get("model", "social_force")
    
    # Special case: discrete Vicsek
    if model == "vicsek_discrete":
        from .vicsek_discrete import simulate_vicsek
        return simulate_vicsek(vicsek_cfg)
    
    # Default: continuous D'Orsogna/social force model
    return _simulate(config)
```

---

## Key Differences

| Feature | D'Orsogna (dynamics.py) | Discrete Vicsek (vicsek_discrete.py) |
|---------|------------------------|-----------------------------------|
| **Time integration** | Continuous ODE (RK4, Euler) | Discrete steps |
| **Speed** | Variable (force-dependent) | Fixed constant v₀ |
| **Forces** | ✅ Morse potential implemented | ❌ Placeholder only |
| **Alignment** | Optional add-on | Core mechanism |
| **Interactions** | Spatial forces + optional alignment | Alignment only |
| **Config** | `model.type: dorsogna` | `model.type: vicsek` |
| **Runner** | `run_dorsogna.py` | `run_standardized.py` |

---

## Can You Combine Them?

**Not directly!** They're fundamentally different update schemes:

### What You CAN Do:
1. ✅ **D'Orsogna + Vicsek alignment**: WORKS (what we just ran!)
   - Forces integrate continuously
   - Alignment applied as post-step adjustment
   - `alignment.enabled: true` in dorsogna config

2. ❌ **Discrete Vicsek + Morse forces**: DOESN'T WORK
   - Force hook exists but not implemented
   - Would need discrete force integration logic
   - Fixed speed conflicts with force-driven motion

### To Make Vicsek Use Forces:
You'd need to:
1. Implement force calculation in `vicsek_discrete.py`
2. Allow variable speeds (breaks classic Vicsek assumption)
3. Integrate forces into velocity update
4. Essentially... you'd be recreating `dynamics.py`!

---

## Which Should You Use?

### Use D'Orsogna (`dynamics.py`) when:
- You want spatial forces (repulsion, attraction)
- Variable speeds are important
- Realistic cohesion/spacing matters
- Optional: add alignment for coordinated motion

### Use Discrete Vicsek (`vicsek_discrete.py`) when:
- You want pure alignment dynamics
- Fixed speed v₀ is acceptable
- Classic Vicsek model behavior
- No spatial forces needed

---

## Your Recent Simulations

1. **`dorsogna_simple.yaml`**: Pure forces, no alignment
   - Low polarization (Φ ~ 0.1)
   - Clustering from forces
   - Random directions

2. **`dorsogna_low_speed_aligned.yaml`**: Forces + alignment
   - High polarization (Φ → 0.99)
   - Cohesive flocks with regular spacing
   - Coordinated migration

Both used `dynamics.py` (continuous D'Orsogna), just with different settings! 🎯
