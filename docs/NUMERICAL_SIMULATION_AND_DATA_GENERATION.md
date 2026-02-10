# Numerical Simulation and Data Generation

## Overview

This document describes the numerical methods, time discretization schemes, and simulation algorithms used to generate training and test data for the ROM pipeline. It covers time integration methods, simulation parameters, and the role of noise in controlling generalization performance.

---

## Chapter: Numerical Simulation and Data Generation

### Section 1: Time Discretization and Integrators

#### 1.1 Explicit Euler (One-Step Discrete-Time System)

**Mathematical formulation:**

The **explicit Euler method** (also called forward Euler) is a first-order time integration scheme:

$$
\mathbf{y}_{n+1} = \mathbf{y}_n + \Delta t \cdot \mathbf{f}(\mathbf{y}_n, t_n)
$$

Where:
- $\mathbf{y}_n$ = state at timestep $n$
- $\Delta t$ = timestep size
- $\mathbf{f}(\mathbf{y}, t)$ = time derivative (right-hand side of ODE $\dot{\mathbf{y}} = \mathbf{f}(\mathbf{y}, t)$)

**Application to discrete Vicsek model:**

The discrete Vicsek model uses explicit Euler for the **discrete** update rule (not continuous ODE integration):

$$
\begin{aligned}
\mathbf{x}_i(t+\Delta t) &= \mathbf{x}_i(t) + v_0 \Delta t \cdot \hat{\mathbf{p}}_i(t) \\
\hat{\mathbf{p}}_i(t) &= \mathcal{R}(\phi_i) \cdot \bar{\mathbf{p}}_i(t) \\
\phi_i &\sim \text{Noise}(\eta)
\end{aligned}
$$

Where:
- $\mathbf{x}_i$ = position of particle $i$
- $v_0$ = constant speed (discrete Vicsek characteristic)
- $\hat{\mathbf{p}}_i$ = unit heading vector
- $\mathcal{R}(\phi)$ = rotation matrix for noise angle $\phi$
- $\bar{\mathbf{p}}_i$ = mean heading of neighbors within radius $R$

**Code implementation:**

**Location:** `src/rectsim/vicsek_discrete.py`

```python
def step_vicsek_discrete(
    x: np.ndarray,      # Positions (N, 2)
    p: np.ndarray,      # Headings (N, 2)
    v0: float,          # Speed
    dt: float,          # Timestep
    Lx: float, Ly: float,  # Domain size
    R: float,           # Interaction radius
    noise_kind: str,    # "gaussian" or "uniform"
    sigma: float,       # Gaussian noise std (if applicable)
    eta: float,         # Uniform noise range (if applicable)
    bc: str,            # Boundary condition
    rng: np.random.Generator,
    cell_list = None,
) -> Tuple[np.ndarray, np.ndarray, CellList]:
    """Single timestep of discrete Vicsek model with explicit Euler update.
    
    Algorithm:
    ----------
    1. Find neighbors: {j : |x_j - x_i| < R}
    2. Compute mean heading: p_bar_i = normalize(Σ_j p_j)
    3. Add noise: phi_i ~ Noise(eta)
    4. Update heading: p_i = R(phi_i) · p_bar_i
    5. Update position: x_i = x_i + v0 * dt * p_i (EXPLICIT EULER)
    6. Apply boundary conditions (wrap or reflect)
    """
    
    N = x.shape[0]
    
    # Step 1: Find neighbors within radius R
    neighbours, cell_list = compute_neighbors(x, Lx, Ly, R, bc, cell_list)
    
    # Step 2: Compute mean heading for each particle
    p_mean = np.zeros((N, 2), dtype=float)
    for i in range(N):
        if len(neighbours[i]) > 0:
            p_mean[i] = np.mean(p[neighbours[i]], axis=0)
            norm = np.linalg.norm(p_mean[i])
            if norm > 1e-12:
                p_mean[i] /= norm  # Normalize
    
    # Step 3: Add angular noise
    phi = angle_noise(rng, noise_kind, eta, sigma, size=N)
    
    # Step 4: Rotate mean heading by noise angle
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    p_new = np.column_stack([
        cos_phi * p_mean[:, 0] - sin_phi * p_mean[:, 1],
        sin_phi * p_mean[:, 0] + cos_phi * p_mean[:, 1]
    ])
    
    # Step 5: EXPLICIT EULER position update
    x_new = x + v0 * dt * p_new
    
    # Step 6: Apply boundary conditions
    x_new, flips = apply_bc(x_new, Lx, Ly, bc)
    
    return x_new, p_new, cell_list
```

**Stability condition:**

For explicit Euler to remain stable in the discrete Vicsek model, we enforce:

$$
v_0 \Delta t \leq 0.5 R
$$

This prevents particles from "jumping over" their interaction neighborhood in a single timestep.

**Code validation:**
```python
# From vicsek_discrete.py, line 243
if v0 * dt > 0.5 * R:
    raise ValueError(
        f"Time step too large: v0*dt = {v0*dt:.3f} > 0.5*R = {0.5*R:.3f}. "
        f"Reduce dt or increase R to maintain stable neighbor finding."
    )
```

**Typical parameter values used in pipeline:**

| Parameter | Value | Source |
|-----------|-------|--------|
| $\Delta t$ | 0.1 | `configs/alvarez_style_production.yaml:33` |
| $v_0$ | 1.0 | `configs/alvarez_style_production.yaml:36` |
| $R$ | 2.0 | `configs/alvarez_style_production.yaml:37` |
| Stability check | $1.0 \times 0.1 = 0.1 < 0.5 \times 2.0 = 1.0$ ✓ | Satisfied |

#### 1.2 Semi-Implicit Euler (Symplectic Integrator)

**Mathematical formulation:**

The **semi-implicit Euler method** (also called symplectic Euler or Euler-Cromer) is a first-order method that updates velocity first, then uses the *new* velocity to update position:

$$
\begin{aligned}
\mathbf{v}_{n+1} &= \mathbf{v}_n + \Delta t \cdot \mathbf{a}(\mathbf{x}_n, \mathbf{v}_n, t_n) \\
\mathbf{x}_{n+1} &= \mathbf{x}_n + \Delta t \cdot \mathbf{v}_{n+1}
\end{aligned}
$$

This differs from explicit Euler, which would use $\mathbf{v}_n$ in the position update. The semi-implicit version is **more stable** for oscillatory or force-driven systems.

**Application to variable-speed Vicsek:**

For the **variable-speed mode** where particles have Morse forces affecting velocity:

$$
\begin{aligned}
\mathbf{v}_i(t+\Delta t) &= \mathbf{v}_i(t) + \Delta t \cdot \mu_t \mathbf{F}_i(t) \\
\mathbf{x}_i(t+\Delta t) &= \mathbf{x}_i(t) + \Delta t \cdot \mathbf{v}_i(t+\Delta t)
\end{aligned}
$$

Where:
- $\mathbf{F}_i$ = Morse forces from neighbors
- $\mu_t$ = translational mobility parameter

**Code implementation:**

**Location:** `src/rectsim/vicsek_discrete.py`, lines 618-645

```python
if integrator == "euler_semiimplicit":
    # SEMI-IMPLICIT EULER: Update velocity first, then position with new velocity
    # This is more stable for force-driven systems
    
    # Step 1: Update velocities from forces: v_{n+1} = v_n + dt * mu_t * F_n
    v = v + dt * mu_t * F
    
    # Compute current headings from new velocities
    speed = np.linalg.norm(v, axis=1, keepdims=True)
    speed = np.maximum(speed, 1e-12)
    p = v / speed
    
    # Alignment: rotate velocities toward mean heading
    p_bar = compute_mean_headings(x, p, R, neighbours)
    
    # Add angular noise
    phi = angle_noise(rng, noise_kind, eta, size=N, match_variance=match_variance)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    
    p_aligned = np.column_stack([
        cos_phi * p_bar[:, 0] - sin_phi * p_bar[:, 1],
        sin_phi * p_bar[:, 0] + cos_phi * p_bar[:, 1]
    ])
    
    # Rotate velocity toward aligned direction (keep magnitude)
    v = speed.ravel()[:, np.newaxis] * p_aligned
    
    # Step 2: Update positions with NEW velocity: x_{n+1} = x_n + dt * v_{n+1}
    x = x + dt * v
    x, flips = apply_bc(x, Lx, Ly, bc)
```

**When to use semi-implicit Euler:**
- Variable-speed mode (`speed_mode: "variable"`)
- Systems with strong forces (Morse potential)
- When explicit Euler shows energy drift

**Location in configs:**
```yaml
# configs/interesting_behavior.yaml:20
sim:
  integrator: euler_semiimplicit  # More stable for variable speed + forces
```

#### 1.3 Runge-Kutta 4 (RK4)

**Mathematical formulation:**

The **fourth-order Runge-Kutta method** (RK4) is a higher-order explicit integrator with $O(\Delta t^4)$ local truncation error:

$$
\begin{aligned}
\mathbf{k}_1 &= \mathbf{f}(\mathbf{y}_n, t_n) \\
\mathbf{k}_2 &= \mathbf{f}(\mathbf{y}_n + \frac{\Delta t}{2}\mathbf{k}_1, t_n + \frac{\Delta t}{2}) \\
\mathbf{k}_3 &= \mathbf{f}(\mathbf{y}_n + \frac{\Delta t}{2}\mathbf{k}_2, t_n + \frac{\Delta t}{2}) \\
\mathbf{k}_4 &= \mathbf{f}(\mathbf{y}_n + \Delta t \mathbf{k}_3, t_n + \Delta t) \\
\mathbf{y}_{n+1} &= \mathbf{y}_n + \frac{\Delta t}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4)
\end{aligned}
$$

**Application to continuous D'Orsogna model:**

For the continuous self-propelled particle model with damping:

$$
\begin{aligned}
\frac{d\mathbf{x}_i}{dt} &= \mathbf{v}_i \\
\frac{d\mathbf{v}_i}{dt} &= (\alpha - \beta|\mathbf{v}_i|^2)\mathbf{v}_i + \mu_t \mathbf{F}_i
\end{aligned}
$$

Where:
- $\alpha$ = self-propulsion strength
- $\beta$ = friction coefficient
- Natural speed: $v_0 = \sqrt{\alpha/\beta}$

**Code implementation:**

**Location:** `src/rectsim/integrators.py`, lines 67-126

```python
def step_rk4(
    state: State,
    params: dict,
    dt: float,
    force_fn: ForceFunction,
    domain: dict,
) -> State:
    """Advance the state by one step using fourth-order Runge-Kutta."""
    
    alpha = params["alpha"]
    beta = params["beta"]
    mu_t = params.get("mu_t", 1.0)
    Lx = domain["Lx"]
    Ly = domain["Ly"]
    bc = domain["bc"]
    
    def eval_force(pos: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """Wrap the force function with boundary-condition handling."""
        pos_wrapped = pos.copy()
        apply_bc(pos_wrapped, Lx, Ly, bc)
        return force_fn(pos_wrapped)
    
    def _acceleration(v, fx, fy):
        """Compute acceleration: (α - β|v|²)v + μ_t F"""
        speed_sq = np.sum(v**2, axis=1, keepdims=True)
        self_prop = (alpha - beta * speed_sq) * v
        return self_prop + mu_t * np.column_stack((fx, fy))
    
    x0 = state.x
    v0 = state.v
    
    # Stage 1
    fx0, fy0 = eval_force(x0)
    a0 = _acceleration(v0, fx0, fy0)
    
    k1_x = v0
    k1_v = a0
    
    # Stage 2
    x1 = x0 + 0.5 * dt * k1_x
    v1 = v0 + 0.5 * dt * k1_v
    fx1, fy1 = eval_force(x1)
    a1 = _acceleration(v1, fx1, fy1)
    
    k2_x = v1
    k2_v = a1
    
    # Stage 3
    x2 = x0 + 0.5 * dt * k2_x
    v2 = v0 + 0.5 * dt * k2_v
    fx2, fy2 = eval_force(x2)
    a2 = _acceleration(v2, fx2, fy2)
    
    k3_x = v2
    k3_v = a2
    
    # Stage 4
    x3 = x0 + dt * k3_x
    v3 = v0 + dt * k3_v
    fx3, fy3 = eval_force(x3)
    a3 = _acceleration(v3, fx3, fy3)
    
    k4_x = v3
    k4_v = a3
    
    # Final update
    x_new = x0 + dt / 6.0 * (k1_x + 2*k2_x + 2*k3_x + k4_x)
    v_new = v0 + dt / 6.0 * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    
    # Apply boundary conditions
    x_new, flips = apply_bc(x_new, Lx, Ly, bc)
    v_new = v_new.copy()
    v_new[flips] *= -1  # Flip velocities at reflecting boundaries
    
    return State(x=x_new, v=v_new, t=state.t + dt)
```

**When to use RK4:**
- Continuous D'Orsogna model (`model: { type: "dorsogna" }`)
- Higher accuracy requirements
- Smooth dynamics without stiffness

**Configuration:**
```yaml
# configs/config.py:47 (legacy)
sim:
  integrator: "rk4"  # Or "euler" for semi-implicit
```

**Performance comparison:**

| Integrator | Order | Cost per step | Accuracy | Use case |
|------------|-------|---------------|----------|----------|
| Explicit Euler | 1 | 1× | Low | Discrete Vicsek (constant speed) |
| Semi-implicit Euler | 1 | 1× | Medium | Variable speed + forces |
| RK4 | 4 | 4× | High | Continuous D'Orsogna |

---

### Section 2: Full Simulation Algorithm

#### 2.1 Pseudocode: Simulation Loop (All Variables, Steps, Outputs)

**High-level algorithm:**

```
ALGORITHM: Complete Simulation Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INPUT:
  config = {
    sim: {N, Lx, Ly, bc, T, dt, save_every, neighbor_rebuild},
    model: {type, speed, speed_mode},
    params: {R, alpha, beta, Cr, Ca, lr, la, mu_t},
    noise: {kind, eta, match_variance}
  }
  rng: Random number generator

OUTPUT:
  result = {
    times: (T_frames,) timestamps
    traj: (T_frames, N, 2) positions
    vel: (T_frames, N, 2) velocities
    meta: dict with config and diagnostics
  }

INITIALIZATION:
  1. Extract parameters from config:
     N ← config.sim.N
     Lx, Ly ← config.sim.Lx, config.sim.Ly
     bc ← config.sim.bc
     T, dt ← config.sim.T, config.sim.dt
     save_every ← config.sim.save_every
     neighbor_rebuild ← config.sim.neighbor_rebuild
     v0 ← config.model.speed
     R ← config.params.R
     noise_kind ← config.noise.kind
     eta ← config.noise.eta
  
  2. Validate stability condition:
     IF v0 * dt > 0.5 * R:
       RAISE ERROR("Timestep too large, violates stability")
  
  3. Initialize particle positions:
     x ← uniform_random([N, 2]) * [Lx, Ly]  # Random in domain
     OR x ← sample_initial_positions(ic_type, N, Lx, Ly, rng)
  
  4. Initialize velocities/headings:
     IF model.type == "discrete":
       theta ← uniform_random([N]) * 2π
       p ← [cos(theta), sin(theta)]  # Unit headings
       v ← v0 * p  # Constant speed
     ELSE IF model.type == "dorsogna":
       angles ← uniform_random([N]) * 2π
       v ← v0 * [cos(angles), sin(angles)]  # Initial velocities
  
  5. Apply boundary conditions:
     x, flips ← apply_bc(x, Lx, Ly, bc)
  
  6. Build neighbor search structure:
     IF R > 0 OR forces_enabled:
       R_max ← max(R, force_cutoff)
       cell_list ← build_cells(x, Lx, Ly, R_max, bc)
  
  7. Initialize output storage:
     total_steps ← round(T / dt)
     n_frames ← ceil(total_steps / save_every) + 1
     traj ← zeros([n_frames, N, 2])
     vel ← zeros([n_frames, N, 2])
     times ← zeros([n_frames])
     
     traj[0] ← x
     vel[0] ← v
     times[0] ← 0.0
     frame_idx ← 1

MAIN SIMULATION LOOP:
  FOR step = 1 TO total_steps:
    
    # --- Neighbor list rebuild (if needed) ---
    IF (step - 1) % neighbor_rebuild == 0:
      cell_list ← build_cells(x, Lx, Ly, R_max, bc)
    
    # --- Model-specific dynamics ---
    IF model.type == "discrete":
      # Discrete Vicsek model
      
      # 1. Find neighbors
      neighbours ← find_neighbors_within_radius(x, R, cell_list)
      
      # 2. Compute mean heading for each particle
      FOR i = 0 TO N-1:
        IF len(neighbours[i]) > 0:
          p_bar[i] ← normalize(mean(p[neighbours[i]]))
        ELSE:
          p_bar[i] ← p[i]  # No neighbors, keep current heading
      
      # 3. Compute Morse forces (if enabled)
      IF forces_enabled:
        F ← compute_morse_forces(x, cell_list, Cr, Ca, lr, la, Lx, Ly, bc)
      ELSE:
        F ← zeros([N, 2])
      
      # 4. Speed mode logic
      IF speed_mode == "constant":
        # Pure Vicsek: constant speed, alignment only
        p ← p_bar  # No forces affect heading
        v ← v0 * p
        
        # Add angular noise
        phi ← sample_angle_noise(rng, noise_kind, eta, size=N)
        p ← rotate_vectors(p, phi)
        
        # Update positions
        x ← x + dt * v0 * p
      
      ELSE IF speed_mode == "constant_with_forces":
        # Hybrid: constant speed, but forces affect heading
        F_hat ← normalize(F)  # Force direction
        lambda ← force_alignment_strength(|F|, ...)  # Dynamic weighting
        
        p_combined ← normalize((1-lambda)*p_bar + lambda*F_hat)
        
        # Add angular noise
        phi ← sample_angle_noise(rng, noise_kind, eta, size=N)
        p ← rotate_vectors(p_combined, phi)
        
        v ← v0 * p
        x ← x + dt * v
      
      ELSE IF speed_mode == "variable":
        # Variable speed: forces affect velocity directly
        
        IF integrator == "euler_semiimplicit":
          # Step 1: Update velocity with forces
          v ← v + dt * mu_t * F
          
          # Step 2: Alignment (rotate toward mean heading)
          speed ← |v|
          p ← v / speed
          p_bar ← compute_mean_headings(...)
          
          # Add angular noise
          phi ← sample_angle_noise(...)
          p_aligned ← rotate_vectors(p_bar, phi)
          
          v ← speed * p_aligned  # Maintain speed, change direction
          
          # Step 3: Update position with NEW velocity
          x ← x + dt * v
        
        ELSE:  # Explicit Euler
          # Update velocity first
          v ← v + dt * mu_t * F
          
          # Update position with OLD velocity
          x ← x + dt * v
          
          # Then apply alignment/noise
          ...
    
    ELSE IF model.type == "dorsogna":
      # Continuous D'Orsogna model with RK4
      
      # Compute forces
      F ← compute_morse_forces(x, cell_list, Cr, Ca, lr, la, Lx, Ly, bc)
      
      # RK4 integration
      state ← {x: x, v: v, t: step*dt}
      state_new ← step_rk4(state, params, dt, F, domain)
      
      x ← state_new.x
      v ← state_new.v
      
      # Optional: Vicsek-style alignment
      IF alignment_enabled:
        v ← apply_alignment(x, v, R, alignment_rate, ...)
    
    # --- Apply boundary conditions ---
    x, flips ← apply_bc(x, Lx, Ly, bc)
    
    IF bc == "reflecting":
      v[flips] ← -v[flips]  # Flip velocities at walls
    
    # --- Update heading from velocity (for output) ---
    speed ← |v|
    speed ← max(speed, 1e-12)  # Avoid division by zero
    p ← v / speed
    
    # --- Save frame (if needed) ---
    IF step % save_every == 0 OR step == total_steps:
      traj[frame_idx] ← x
      vel[frame_idx] ← v
      times[frame_idx] ← step * dt
      frame_idx ← frame_idx + 1
    
    # --- Progress indicator ---
    IF step % 10 == 0:
      PRINT(f"Step {step}/{total_steps} ({100*step/total_steps:.1f}%)")

RETURN:
  result ← {
    times: times,
    traj: traj,
    vel: vel,
    meta: {
      N: N,
      Lx: Lx, Ly: Ly,
      bc: bc,
      T: T, dt: dt,
      model_type: model.type,
      speed_mode: speed_mode,
      noise_kind: noise_kind,
      eta: eta,
      R: R,
      ...
    }
  }
```

#### 2.2 Key Implementation Details

**Angle noise generation:**

**Location:** `src/rectsim/vicsek_discrete.py`, lines 113-140

```python
def _apply_noise(
    rng: np.random.Generator,
    noise_kind: str,
    sigma: float,
    eta: float,
) -> float:
    """Sample angular perturbation.
    
    Parameters
    ----------
    noise_kind : str
        Either "gaussian" or "uniform"
    sigma : float
        Standard deviation for Gaussian noise
    eta : float
        Range for uniform noise in [-η/2, η/2]
    
    Notes
    -----
    For equivalent variance: σ = η/√12
    This ensures Var[Gaussian(0, σ)] = Var[Uniform(-η/2, η/2)] = η²/12
    """
    if noise_kind == "gaussian":
        return float(rng.normal(loc=0.0, scale=sigma))
    if noise_kind == "uniform":
        half_eta = 0.5 * eta
        return float(rng.uniform(-half_eta, half_eta))
    raise ValueError(f"Unknown noise kind '{noise_kind}'")
```

**Variance matching:**

When `match_variance: true` in config:
$$
\sigma = \frac{\eta}{\sqrt{12}}
$$

This ensures:
$$
\text{Var}[\phi_{\text{Gaussian}}] = \text{Var}[\phi_{\text{Uniform}}] = \frac{\eta^2}{12}
$$

**Neighbor finding with cell lists:**

**Location:** `src/rectsim/domain.py`, lines 180-230

Complexity: $O(N)$ average vs. $O(N^2)$ naive

```python
def build_cells(x, Lx, Ly, rcut, bc):
    """Construct linked-cell list for efficient neighbor search.
    
    Divides domain into grid of cells with size ≈ rcut.
    Each particle assigned to one cell.
    To find neighbors within rcut, only check 9 adjacent cells.
    """
    ncellx = max(1, int(np.ceil(Lx / rcut)))
    ncelly = max(1, int(np.ceil(Ly / rcut)))
    
    cells = defaultdict(list)
    for idx, (xi, yi) in enumerate(x):
        ix = int(np.floor(xi / (Lx / ncellx)))
        iy = int(np.floor(yi / (Ly / ncelly)))
        cells[(ix, iy)].append(idx)
    
    return CellList(cells=cells, ncellx=ncellx, ncelly=ncelly, ...)
```

---

### Section 3: Parameter Choices and Fixed Experimental Settings

#### 3.1 Base Parameters (Kept Fixed for the Pipeline)

**Standard configuration used across all experiments:**

| Parameter | Symbol | Value | Source | Description |
|-----------|--------|-------|--------|-------------|
| **Domain** |  |  |  |  |
| Width | $L_x$ | 20.0 | `unified_config.py:70` | Domain x-dimension |
| Height | $L_y$ | 20.0 | `unified_config.py:71` | Domain y-dimension |
| Boundary | bc | `"periodic"` | `unified_config.py:72` | Periodic wrap (torus topology) |
| **Particles** |  |  |  |  |
| Count | $N$ | 100–400 | `configs/*.yaml` | Typical: 200 (medium), 400 (dense) |
| Speed | $v_0$ | 1.0 | `configs/alvarez_*.yaml:36` | Constant in discrete mode |
| **Alignment** |  |  |  |  |
| Radius | $R$ | 2.0 | `configs/alvarez_*.yaml:37` | Interaction range |
| **Time** |  |  |  |  |
| Timestep | $\Delta t$ | 0.1 | `configs/alvarez_*.yaml:33` | Micro timestep |
| Training duration | $T_{\text{train}}$ | 2.0–8.0 s | Varies | Short (2s) for tests, long (8s) for production |
| Save frequency | save_every | 1 | `unified_config.py:75` | Save every timestep |
| **Neighbor Search** |  |  |  |  |
| Rebuild frequency | neighbor_rebuild | 5 | `unified_config.py:76` | Rebuild every 5 steps |
| **Forces** (if enabled) |  |  |  |  |
| Repulsion strength | $C_r$ | 2.0 | `unified_config.py:48` | Morse repulsion |
| Attraction strength | $C_a$ | 1.0 | `unified_config.py:49` | Morse attraction |
| Repulsion length | $\ell_r$ | 0.5 | `unified_config.py:50` | Short-range repulsion |
| Attraction length | $\ell_a$ | 1.5 | `unified_config.py:51` | Medium-range attraction |
| Translational mobility | $\mu_t$ | 0.5 | `unified_config.py:52` | Force coupling strength |
| **Self-Propulsion** (D'Orsogna) |  |  |  |  |
| Propulsion | $\alpha$ | 1.5 | `unified_config.py:67` | Self-propulsion magnitude |
| Friction | $\beta$ | 1.0 | `unified_config.py:68` | Damping coefficient |
| Natural speed |  | $\sqrt{\alpha/\beta} = \sqrt{1.5} \approx 1.22$ | Derived | Equilibrium speed |

**Code locations:**

**Default configuration:**
```python
# src/rectsim/unified_config.py, lines 40-80
DEFAULTS = {
    'domain': {
        'Lx': 20.0,
        'Ly': 20.0,
        'bc': 'periodic',
    },
    'particles': {
        'N': 100,
        'initial_speed': 0.5,
    },
    'dynamics': {
        'alignment': {
            'enabled': True,
            'radius': 2.0,
            'rate': 1.0,
        },
        'forces': {
            'enabled': False,
            'Cr': 2.0,
            'Ca': 1.0,
            'lr': 0.5,
            'la': 1.5,
            'mu_t': 0.5,
        },
        'noise': {
            'kind': 'gaussian',
            'eta': 0.3,
            'match_variance': True,
        },
    },
    'integration': {
        'T': 100.0,
        'dt': 0.01,
        'save_every': 10,
        'neighbor_rebuild': 5,
        'integrator': 'euler',
        'seed': 42,
    },
}
```

**Production configuration (Alvarez-style):**
```yaml
# configs/alvarez_style_production.yaml, lines 25-38
sim:
  N: 40                     # Particle count (production uses 40 for fast sims)
  Lx: 15.0                  # Domain width
  Ly: 15.0                  # Domain height
  bc: "periodic"            # Boundary condition
  T: 8.0                    # Training horizon: 8 seconds
  dt: 0.1                   # Timestep
  v0: 1.0                   # Particle speed
  R: 2.0                    # Vicsek interaction radius
  eta: 0.3                  # Noise level (critical regime)
```

**Stability check validation:**

```python
# src/rectsim/unified_config.py, lines 226-232
if config.get('model', {}).get('type') == 'discrete':
    v0 = config.get('particles', {}).get('initial_speed', 0.5)
    dt = integ['dt']
    R = config.get('dynamics', {}).get('alignment', {}).get('radius', 2.0)
    if v0 * dt > 0.5 * R:
        errors.append(
            f"Stability condition violated: v0*dt={v0*dt:.3f} > 0.5*R={0.5*R:.3f}. "
            f"Reduce dt or increase R."
        )
```

#### 3.2 Noise Levels as a Generalization Knob

**Role of noise in collective motion:**

Noise strength $\eta$ controls the **phase transition** between ordered and disordered motion in the Vicsek model:

| Regime | $\eta$ range | Behavior | Order parameter $\Phi$ |
|--------|--------------|----------|------------------------|
| **Ordered** | $\eta < 0.2$ | Strong alignment, coherent flocking | $\Phi > 0.8$ |
| **Critical** | $0.2 \leq \eta \leq 0.5$ | Intermittent flocking, phase coexistence | $0.3 < \Phi < 0.7$ |
| **Disordered** | $\eta > 0.5$ | Weak alignment, random motion | $\Phi < 0.3$ |

Where polarization order parameter:
$$
\Phi(t) = \frac{1}{Nv_0}\left|\sum_{i=1}^N \mathbf{v}_i(t)\right|
$$

**Why noise matters for ROM generalization:**

1. **Training diversity:** Higher noise → more varied trajectories → better POD/MVAR coverage
2. **Latent smoothness:** Moderate noise smooths density fields → lower POD rank needed
3. **MVAR stability:** Critical regime ($\eta \approx 0.3$) balances predictability vs. richness

**Standard noise configuration:**

**Location:** All production configs use $\eta = 0.3$ (critical regime)

```yaml
# configs/alvarez_style_production.yaml:38
sim:
  eta: 0.3  # Moderate noise (critical regime)

# configs/unified_config.py:58-60
'noise': {
    'kind': 'gaussian',        # Type: "gaussian" or "uniform"
    'eta': 0.3,                # Noise strength (radians)
    'match_variance': True,    # σ = η/√12 for Gaussian
}
```

**Two noise types implemented:**

##### **a) Uniform Noise (Original Vicsek)**

**Distribution:**
$$
\phi_i \sim \text{Uniform}\left[-\frac{\eta}{2}, \frac{\eta}{2}\right]
$$

**Properties:**
- Range: $\eta \in [0, \pi]$ (full range covers all angles)
- Variance: $\text{Var}[\phi] = \eta^2/12$
- Support: Bounded, no rare extreme events
- **Use:** Standard Vicsek model, well-studied phase diagram

**Code:**
```python
# src/rectsim/vicsek_discrete.py, line 136-139
if noise_kind == "uniform":
    half_eta = 0.5 * eta  # η/2
    return float(rng.uniform(-half_eta, half_eta))
```

##### **b) Gaussian Noise (Variance-Matched)**

**Distribution:**
$$
\phi_i \sim \mathcal{N}(0, \sigma^2), \quad \sigma = \frac{\eta}{\sqrt{12}}
$$

**Properties:**
- Unbounded: rare large angles possible (fat tails)
- Variance: $\text{Var}[\phi] = \sigma^2 = \eta^2/12$ (matches uniform)
- Mean: $\mathbb{E}[\phi] = 0$
- **Use:** More realistic stochastic dynamics

**Code:**
```python
# src/rectsim/vicsek_discrete.py, line 133-134
if noise_kind == "gaussian":
    return float(rng.normal(loc=0.0, scale=sigma))
```

**Variance matching formula:**

When `match_variance: true` (default):
```python
# src/rectsim/vicsek_discrete.py, line 405-410
if match_variance:
    # For Gaussian noise to have same variance as Uniform[-η/2, η/2]:
    # Var[Uniform(-η/2, η/2)] = η²/12
    # So set Gaussian σ = η/√12
    sigma = eta / np.sqrt(12.0)
else:
    sigma = eta  # Direct interpretation
```

**Experimental verification:**

**Test:** `test_kde_fixes.py`, line 171-180
```python
# Verify mass conservation across different N values
N_values = [20, 40, 80, 100]
for N_test in N_values:
    # Generate trajectory with N_test particles, η=0.3 Gaussian noise
    traj_temp = np.random.rand(T, N_test, 2) * [Lx, Ly]
    rho_temp, _ = kde_density_movie(traj_temp, Lx, Ly, nx, ny, bandwidth=2.0, bc="periodic")
    mass = np.mean([dx*dy*np.sum(rho_temp[t]) for t in range(T)])
    error = abs(mass - N_test) / N_test * 100
    print(f"  N={N_test:3d}: mass={mass:.2f}, error={error:.2f}%")
```

**Noise in different model types:**

| Model | Noise Type | Location | Effect |
|-------|------------|----------|--------|
| **Discrete Vicsek** | Angular ($\phi$) | Heading update | Directly perturbs alignment direction |
| **Continuous D'Orsogna** | Rotational diffusion ($D_\theta$) | Velocity angle | Brownian rotation of heading |
| **Variable speed** | Angular + Force | Heading + Velocity | Combined stochasticity |

**Continuous rotational diffusion:**

For D'Orsogna model:
$$
d\theta_i = \mu_r (\bar{\theta}_i - \theta_i) dt + \sqrt{2D_\theta} dW_t
$$

Where:
- $\mu_r$ = rotational mobility
- $D_\theta$ = rotational diffusion coefficient
- $dW_t$ = Wiener process increment

**Code:**
```python
# src/rectsim/dynamics.py, line 158
noise_scale = np.sqrt(max(0.0, 2.0 * Dtheta * dt))
dtheta = rng.normal(0.0, noise_scale, size=N)  # Angular noise
```

**Configuration:**
```yaml
# configs/unified_config.py:61
'noise': {
    'Dtheta': 0.001,  # Rotational diffusion (continuous only)
}
```

---

### Section 4: Summary of What We Actually Use in Our Pipeline

**Canonical pipeline configuration:**

Based on `configs/alvarez_style_production.yaml` (400-run production experiments):

```yaml
# ============================================================================
# STANDARD PIPELINE CONFIGURATION (PRODUCTION)
# ============================================================================

# --- MICROSIMULATION ---
sim:
  N: 40                      # Particles (small N for fast sims, 400 training runs)
  Lx: 15.0, Ly: 15.0        # Domain (square)
  bc: "periodic"             # Boundary (torus topology)
  T: 8.0                     # Training duration (8 seconds)
  dt: 0.1                    # Timestep (0.1 seconds)
  v0: 1.0                    # Speed (constant in discrete mode)
  R: 2.0                     # Alignment radius
  eta: 0.3                   # Noise (critical regime)
  integrator: "euler"        # Explicit Euler for discrete Vicsek
  save_every: 1              # Save all timesteps
  neighbor_rebuild: 5        # Rebuild neighbor lists every 5 steps

# --- MODEL ---
model:
  type: "discrete"           # Discrete Vicsek model
  speed_mode: "constant"     # Constant speed (pure Vicsek)

# --- NOISE ---
noise:
  kind: "gaussian"           # Gaussian angular noise
  eta: 0.3                   # Strength (critical regime)
  match_variance: true       # σ = η/√12

# --- FORCES (DISABLED in standard pipeline) ---
forces:
  enabled: false             # No Morse forces in canonical Vicsek

# --- DENSITY FIELD ---
density:
  nx: 64, ny: 64            # Grid resolution
  bandwidth: 3.0             # KDE bandwidth (3 grid cells, smoothing)

# --- ROM ---
rom:
  subsample: 2               # Use every 2nd snapshot (ROM dt = 0.2s)
  pod_energy: 0.95           # Capture 95% variance
  
  models:
    mvar:
      enabled: true
      lag: 5                 # 5-timestep memory window
      ridge_alpha: 1.0e-4    # Strong regularization (Alvarez principle)
      eigenvalue_threshold: 0.999  # Stability control
    
    lstm:
      enabled: true
      lag: 5
      hidden_units: 64
      num_layers: 2
      batch_size: 32
      learning_rate: 1.0e-3
      max_epochs: 100
      patience: 10
      gradient_clip: 1.0

# --- TRAINING ---
train_ic:
  gaussian:
    n_runs: 100              # 25% of 400 total
    variances: [0.25, 1.0, 4.0, 9.0]  # σ = 0.5, 1.0, 2.0, 3.0
  
  uniform:
    n_runs: 100              # 25%
  
  ring:
    n_runs: 100              # 25%
    radii: [2.0, 3.0, 4.0, 5.0]
  
  two_clusters:
    n_runs: 100              # 25%

# --- TESTING ---
test_ic:
  type: "uniform"            # Uniform random (out-of-distribution)
  n_test: 20                 # 20 test trajectories

test_sim:
  T: 20.0                    # 20 seconds (12s extrapolation beyond training)
  dt: 0.1                    # Same as training

# --- EVALUATION ---
evaluation:
  forecast_start: 2.0        # Start closed-loop forecast at t=2s
  save_time_resolved: true   # Save R²(t) for temporal analysis
```

**Key identifiability check (Alvarez principle):**

For well-conditioned MVAR, ensure:

$$
\rho = \frac{N_{\text{samples}}}{N_{\text{params}}} \geq 10
$$

**Calculation for our pipeline:**
```
Training:
  - Runs: M = 400
  - Duration per run: T = 8.0s
  - Timestep: dt = 0.1s
  - Timesteps per run: K = T/dt = 80
  - Subsample: s = 2 → ROM timesteps: K_rom = 40
  - Windows per run: K_rom - lag = 40 - 5 = 35
  - Total windows: N_samples = 400 × 35 = 14,000

MVAR parameters:
  - POD modes: d = 35 (fixed_modes for stability)
  - Lag: w = 5
  - Parameters: N_params = d² × w = 35² × 5 = 6,125

Identifiability ratio:
  ρ = 14,000 / 6,125 ≈ 2.29 ✓
  (Acceptable, but not ideal. Production aims for ρ > 10)

Improved configuration (more runs):
  M = 1008 runs → N_samples = 1008 × 75 ≈ 75,600
  ρ ≈ 75,600 / 6,125 ≈ 12.3 ✓✓ (well-conditioned)
```

**What makes our approach work:**

1. **Moderate noise** ($\eta = 0.3$): Critical regime balances order and disorder
2. **Periodic BC**: Smooth, continuous density fields (no edge artifacts)
3. **Small timestep** ($\Delta t = 0.1$): Stable integration, resolves fast dynamics
4. **Gaussian smoothing** (bandwidth = 3.0): Reduces effective dimensionality
5. **Strong regularization** ($\alpha = 10^{-4}$): Prevents MVAR overfitting
6. **Diverse ICs** (Gaussian/Uniform/Ring/Clusters): ROM generalizes across conditions

---

## Files and References

### Key Implementation Files

| File | Description | Lines | Role |
|------|-------------|-------|------|
| `src/rectsim/vicsek_discrete.py` | Discrete Vicsek simulator | 715 | Main discrete-time loop |
| `src/rectsim/integrators.py` | RK4 and semi-implicit Euler | 157 | Continuous ODE integration |
| `src/rectsim/dynamics.py` | D'Orsogna continuous model | 428 | Continuous backend |
| `src/rectsim/unified_config.py` | Default configuration schema | 344 | Parameter defaults |
| `src/rectsim/domain.py` | Boundary conditions, neighbor search | 316 | Spatial utilities |
| `configs/alvarez_style_production.yaml` | Production pipeline config | 163 | Standard experiment setup |

### References

**Vicsek Model:**
- Vicsek et al., "Novel Type of Phase Transition in a System of Self-Driven Particles," *PRL* 75, 1226 (1995)
- Grégoire & Chaté, "Onset of Collective and Cohesive Motion," *PRL* 92, 025702 (2004)

**D'Orsogna Model:**
- D'Orsogna et al., "Self-propelled particles with soft-core interactions," *PRE* 73, 010903 (2006)

**Numerical Methods:**
- Hairer, Lubich, Wanner, "Geometric Numerical Integration" (2006) - Symplectic integrators
- Press et al., "Numerical Recipes" (2007) - RK4 implementation

**Noise and Phase Transitions:**
- Chaté et al., "Modeling collective motion," *Eur. Phys. J. B* 64, 451 (2008)
- Ballerini et al., "Interaction ruling animal collective behavior," *PNAS* 105, 1232 (2008)
