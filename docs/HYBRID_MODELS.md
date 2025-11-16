# Hybrid Vicsek-D'Orsogna Models

This document describes the implementation of hybrid models that combine Vicsek alignment with D'Orsogna (Morse) forces.

## Overview

We now support **two implementations** of the Vicsek-D'Orsogna hybrid model:

1. **Discrete Explicit Euler** (`vicsek_discrete.py`)
   - Discrete time steps
   - Explicit Euler integration
   - Variable particle speeds (due to forces)
   - Fast, simple, interpretable

2. **Continuous RK4** (`dynamics.py`)
   - Continuous ODE evolution
   - 4th-order Runge-Kutta integration
   - Self-propulsion term: (α - β|v|²)v
   - More accurate, smoother trajectories

Both models support **periodic** and **reflecting** boundary conditions.

---

## Mathematical Formulation

### Discrete Model (Explicit Euler)

At each time step Δt:

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

Key parameters:
- `v0`: Base self-propulsion speed
- `μₜ`: Translational mobility (force coupling strength)
- `Cr, Ca`: Repulsion and attraction magnitudes
- `lr, la`: Repulsion and attraction length scales
- `R`: Alignment radius
- `η`: Noise strength

### Continuous Model (RK4)

System of ODEs:

```
dxᵢ/dt = vᵢ

dvᵢ/dt = (α - β|vᵢ|²)vᵢ + μₜ Fᵢ + alignment_term

where:
- α: self-propulsion magnitude
- β: friction coefficient
- Natural speed: v₀ = α/β
```

**Alignment term** (if enabled):
```
dp̂ᵢ/dt = μᵣ(⟨p̂ⱼ⟩ - p̂ᵢ) + √(2Dθ) ξ(t)

where:
- p̂ᵢ = vᵢ/|vᵢ| is the heading
- μᵣ: rotational mobility
- Dθ: rotational diffusion
- ⟨p̂ⱼ⟩: mean heading of neighbors within radius R
```

The continuous model integrates these ODEs using 4th-order Runge-Kutta.

---

## Implementation Details

### Discrete Model (`vicsek_discrete.py`)

Located in `simulate_backend()` function:

```python
# 1. Compute alignment heading
neighbors = nf.neighbors_of(x)
p_bar = compute_mean_heading(neighbors, p)

# 2. Add noise and rotate headings
phi = angle_noise(rng, noise_kind, eta, size=N)
p = rotate_headings(p_bar, phi)

# 3. Compute Morse forces (if enabled)
if forces_enabled:
    fx, fy = morse_force(x, Lx, Ly, bc, Cr, Ca, lr, la, rcut)
    F = np.column_stack([fx, fy])
else:
    F = zeros()

# 4. Update positions
x = x + dt * (v0 * p + mu_t * F)
x = apply_boundary_conditions(x, bc)
```

### Continuous Model (`dynamics.py`)

The RK4 integrator in `step_rk4()` evaluates the ODEs:

```python
def acceleration(state, params):
    # Self-propulsion
    acc = (alpha - beta * |v|²) * v
    
    # Morse forces
    fx, fy = morse_force(state.x, ...)
    acc += mu_t * F
    
    return acc

# RK4 substeps
k1 = f(t, y)
k2 = f(t + dt/2, y + k1·dt/2)
k3 = f(t + dt/2, y + k2·dt/2)
k4 = f(t + dt, y + k3·dt)
y_new = y + (k1 + 2k2 + 2k3 + k4)·dt/6

# Optional alignment applied after integration step
if alignment_enabled:
    v = apply_vicsek_alignment(x, v, ...)
```

---

## Boundary Conditions

### Periodic Boundaries

**Minimal Image Convention**: When computing distances, always use the shortest path:
```python
dx = x[j] - x[i]
dx = dx - Lx * round(dx / Lx)  # Wrap to [-Lx/2, Lx/2)
```

**Position Wrapping**:
```python
x = mod(x, Lx)  # Keep in [0, Lx)
```

### Reflecting Boundaries

**Wall Collisions**: When a particle hits a wall:
```python
if x < 0:
    x = -x          # Reflect position
    vx = -vx        # Flip velocity
if x > Lx:
    x = 2*Lx - x
    vx = -vx
```

**Heading Flip** (discrete model):
```python
if particle_hit_wall:
    theta = -theta  # Flip heading
```

---

## Configuration

### Discrete Vicsek-D'Orsogna

Example: `vicsek_dorsogna_discrete.yaml`

```yaml
model:
  type: vicsek_discrete
  speed: 0.5              # Base speed v0

sim:
  dt: 0.01                # Small time step for stability
  bc: periodic            # Or "reflecting"
  
params:
  R: 2.0                  # Alignment radius

noise:
  kind: gaussian
  eta: 0.3                # Noise strength

forces:
  enabled: true           # Enable Morse forces!
  params:
    Cr: 2.0               # Repulsion
    Ca: 1.0               # Attraction
    lr: 0.5               # Repulsion scale
    la: 1.5               # Attraction scale
    mu_t: 0.5             # Force coupling
    rcut_factor: 5.0      # Cutoff radius
```

### Continuous Vicsek-D'Orsogna

Example: `vicsek_dorsogna_continuous.yaml`

```yaml
model:
  type: dorsogna          # Use continuous backend

sim:
  integrator: rk4         # Or "euler_semiimplicit"
  dt: 0.01
  bc: periodic
  
params:
  alpha: 1.5              # Self-propulsion
  beta: 1.0               # Friction (natural speed = α/β)
  
  # Morse forces (always enabled in D'Orsogna)
  Cr: 2.0
  Ca: 1.0
  lr: 0.5
  la: 1.5
  
  # Optional alignment
  alignment:
    enabled: true         # Add Vicsek alignment!
    radius: 2.0
    rate: 1.0             # μᵣ
    Dtheta: 0.001         # Rotational diffusion
```

---

## Stability and Time Step Selection

### Discrete Model Constraints

1. **CFL Condition**: Particles shouldn't jump too far in one step
   ```
   (v₀ + μₜ·F_max)·Δt < 0.3·min(lᵣ, lₐ, R)
   ```

2. **Force Stability**: Keep translational mobility moderate
   ```
   μₜ ∈ [0.1, 1.0]
   ```

3. **Typical dt**: 0.001 - 0.01

### Continuous Model (RK4)

RK4 is more stable and allows larger time steps:
- Typical dt: 0.01 - 0.05
- Still respect CFL for accuracy

---

## Expected Behaviors

### Pure Vicsek (No Forces)
- Fixed speed v₀
- Polarization increases with low noise
- Possible phase transition (order-disorder)

### Pure D'Orsogna (No Alignment)
- Variable speeds around v₀ = α/β
- Mills, rings, or clumps (depending on parameters)
- Low polarization

### Hybrid Model (Both)
- Coordinated flocks with regular spacing
- High polarization + spatial structure
- Speeds vary but maintain group cohesion
- Rich dynamics: migrating bands, rotating mills, etc.

---

## Performance Tips

1. **Use cell lists**: Set `neighbor_rebuild` appropriately
   - Discrete: rebuild every 1-5 steps
   - Continuous: rebuild every 5-10 steps

2. **Force cutoff**: Set `rcut_factor` to 3-5× max(lr, la)
   - Larger = more accurate but slower
   - Smaller = faster but may miss interactions

3. **Save frequency**: Don't save every step
   - Discrete: `save_every: 10-50`
   - Continuous: `save_every: 50-100`

4. **Particle count**: Cell lists scale as O(N)
   - N < 1000: very fast
   - N = 1000-10000: moderate
   - N > 10000: consider optimization

---

## References

1. **Vicsek Model**: Vicsek et al., PRL 75, 1226 (1995)
2. **D'Orsogna Model**: D'Orsogna et al., PRL 96, 104302 (2006)
3. **Hybrid Dynamics**: Couzin et al., J. Theor. Biol. 218, 1 (2002)

---

## Troubleshooting

### Particles Explode
- Reduce `dt`
- Reduce `mu_t`
- Check force parameters (Cr, Ca, lr, la)

### No Alignment
- Increase `R` (alignment radius)
- Increase alignment rate
- Reduce noise

### No Clustering
- Adjust Morse parameters (try Cr > Ca, lr < la)
- Increase `mu_t`
- Check force cutoff

### Simulations Too Slow
- Reduce `N`
- Increase `save_every`
- Increase `neighbor_rebuild`
- Use Euler instead of RK4 (less accurate but faster)
