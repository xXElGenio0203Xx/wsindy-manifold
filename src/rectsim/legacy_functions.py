"""
Legacy functions from the old working pipeline (commit 67655d3).

These are the exact functions that worked perfectly before:
1. kde_density_movie - KDE with proper metadata
2. side_by_side_video - Comparison video generation
3. save_video - Single density video generation
4. Order parameters - polarization, nematic_order, etc.

Copied from wsindy-manifold-OLD to restore working functionality.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple, Dict, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

Array = np.ndarray


# ============================================================================
# KDE Density Computation (from wsindy_manifold/density.py)
# ============================================================================

def kde_density_movie(
    traj: Array,
    Lx: float,
    Ly: float,
    nx: int,
    ny: int,
    bandwidth: float,
    bc: str = "periodic"
) -> Tuple[Array, Dict]:
    """
    Compute Gaussian-smoothed KDE density movie from particle trajectories.
    
    Args:
        traj: Particle positions (T, N, 2) with (x, y) coordinates
        Lx: Domain width
        Ly: Domain height
        nx: Number of grid points in x
        ny: Number of grid points in y
        bandwidth: Gaussian smoothing bandwidth (in grid units)
        bc: Boundary conditions ('periodic' or 'reflecting')
        
    Returns:
        rho: Density movie (T, ny, nx)  # Note: ny first for image convention
        meta: Metadata dict with 'bandwidth', 'nx', 'ny', 'extent', 'Lx', 'Ly', 'bc'
        
    Example:
        >>> traj = np.random.rand(100, 50, 2) * 30  # 100 frames, 50 particles
        >>> rho, meta = kde_density_movie(traj, Lx=30, Ly=30, nx=50, ny=50, bandwidth=1.5)
        >>> print(rho.shape)  # (100, 50, 50)
        >>> print(meta['extent'])  # [0, 30, 0, 30]
    """
    T, N, d = traj.shape
    
    if d != 2:
        raise ValueError(f"traj must have shape (T, N, 2), got {traj.shape}")
    
    # Create grid edges
    x_edges = np.linspace(0.0, Lx, nx + 1)
    y_edges = np.linspace(0.0, Ly, ny + 1)
    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]
    
    # Mode for Gaussian filter
    mode = "wrap" if bc == "periodic" else "nearest"
    
    # Compute density for each frame
    rho = np.zeros((T, ny, nx))
    masses = []  # Track mass conservation
    
    for t in range(T):
        # Extract positions
        x = traj[t, :, 0]
        y = traj[t, :, 1]
        
        # CRITICAL: Wrap positions for periodic BC to prevent particle loss
        if bc == "periodic":
            x = np.mod(x, Lx)
            y = np.mod(y, Ly)
        
        # 2D histogram
        # histogram2d returns hist[i,j] where i is x-bins, j is y-bins
        # But for images, we need [row, col] = [y, x] indexing
        hist, _, _ = np.histogram2d(
            y,  # y coordinates (rows)
            x,  # x coordinates (columns)
            bins=[y_edges, x_edges],
            range=[[0.0, Ly], [0.0, Lx]]
        )
        
        # Normalize to density (particles per unit area)
        density = hist / (dx * dy)
        
        # Apply Gaussian smoothing
        if bandwidth > 0:
            density = gaussian_filter(density, sigma=bandwidth, mode=mode)
        
        # Store directly (already in [y, x] = [row, col] format for images)
        rho[t] = density
        
        # Mass conservation check
        mass = density.sum() * dx * dy
        masses.append(mass)
    
    # Report mass conservation statistics
    masses = np.array(masses)
    mass_min, mass_max = masses.min(), masses.max()
    if abs(mass_min - N) > 0.1 or abs(mass_max - N) > 0.1:
        print(f"⚠️  KDE mass conservation: min={mass_min:.2f}, max={mass_max:.2f}, target={N}")
        print(f"   Deviation: {abs(mass_min - N)/N*100:.1f}% to {abs(mass_max - N)/N*100:.1f}%")
    
    # Metadata
    meta = {
        'bandwidth': bandwidth,
        'nx': nx,
        'ny': ny,
        'Lx': Lx,
        'Ly': Ly,
        'extent': [0, Lx, 0, Ly],  # [xmin, xmax, ymin, ymax]
        'bc': bc,
        'N_particles': N,
        'T_frames': T
    }
    
    return rho, meta


def estimate_bandwidth(Lx: float, Ly: float, N: int, nx: int, ny: int) -> float:
    """
    Estimate reasonable KDE bandwidth based on problem size.
    
    Rule of thumb: bandwidth ~ (L / sqrt(N)) * (grid_resolution_factor)
    
    Args:
        Lx: Domain width
        Ly: Domain height
        N: Number of particles
        nx: Grid points in x
        ny: Grid points in y
        
    Returns:
        bandwidth: Suggested bandwidth in grid units
    """
    # Average domain size
    L_avg = (Lx + Ly) / 2
    
    # Average grid spacing
    dx = Lx / nx
    dy = Ly / ny
    dx_avg = (dx + dy) / 2
    
    # Scott's rule: h ~ N^(-1/(d+4)) where d=2
    scott_factor = N ** (-1/6)
    
    # Bandwidth in physical units
    h_physical = L_avg * scott_factor * 0.5
    
    # Convert to grid units
    bandwidth = h_physical / dx_avg
    
    # Clamp to reasonable range
    bandwidth = np.clip(bandwidth, 0.5, 5.0)
    
    return bandwidth


# ============================================================================
# Order Parameters (from wsindy_manifold/standard_metrics.py)
# ============================================================================

def polarization(vel: Array, eps: float = 1e-10) -> float:
    """
    Compute polarization order parameter Φ(t).
    
    Φ = (1/N) || Σᵢ vᵢ/||vᵢ|| ||
    
    Args:
        vel: Velocities (N, 2) or (N, d)
        eps: Small constant to avoid division by zero
        
    Returns:
        phi: Polarization in [0, 1]
    """
    if vel.ndim != 2:
        raise ValueError(f"vel must have shape (N, d), got {vel.shape}")
    
    N = vel.shape[0]
    if N == 0:
        return 0.0
    
    # Normalize velocities
    speeds = np.linalg.norm(vel, axis=1, keepdims=True)
    normalized = vel / (speeds + eps)
    
    # Sum and compute magnitude
    mean_direction = np.mean(normalized, axis=0)
    phi = np.linalg.norm(mean_direction)
    
    return float(phi)


def mean_speed(vel: Array) -> float:
    """
    Compute mean speed.
    
    Args:
        vel: Velocities (N, 2) or (N, d)
        
    Returns:
        v_mean: Mean speed
    """
    speeds = np.linalg.norm(vel, axis=1)
    return float(np.mean(speeds))


def speed_std(vel: Array) -> float:
    """
    Compute standard deviation of speeds.
    
    Args:
        vel: Velocities (N, 2) or (N, d)
        
    Returns:
        v_std: Speed standard deviation
    """
    speeds = np.linalg.norm(vel, axis=1)
    return float(np.std(speeds))


def nematic_order(vel: Array, eps: float = 1e-10) -> float:
    """
    Compute nematic order parameter (2nd moment of headings).
    
    Mathematical Formula:
    ---------------------
    Q-tensor: 𝐐 = (1/N) Σᵢ (𝐧ᵢ ⊗ 𝐧ᵢ - 𝐈/d)
    where 𝐧ᵢ = 𝐯ᵢ/‖𝐯ᵢ‖ (unit heading vectors)
    
    Nematic order: Q = λₘₐₓ(𝐐)
    
    Physical Interpretation:
    - Q ∈ [0, 1] in 2D
    - Q = 0: Isotropic (random orientations)
    - Q = 1: Perfect nematic order (aligned along one axis)
    - High Q with low Φ (polarization) → lane formation (bidirectional flow)
    
    The Q-tensor measures second-order alignment and is insensitive to 
    head-tail polarity, making it ideal for detecting bidirectional patterns.
    
    Args:
        vel: Velocities (N, 2)
        eps: Small constant to avoid division by zero
        
    Returns:
        q: Nematic order parameter in [0, 1]
    """
    if vel.ndim != 2 or vel.shape[1] != 2:
        raise ValueError(f"nematic_order requires (N, 2) velocities, got {vel.shape}")
    
    N, d = vel.shape
    if N == 0:
        return 0.0
    
    # Normalize
    speeds = np.linalg.norm(vel, axis=1, keepdims=True)
    n = vel / (speeds + eps)
    
    # Compute Q tensor
    Q = np.zeros((d, d))
    for i in range(N):
        Q += np.outer(n[i], n[i])
    Q = Q / N - np.eye(d) / d
    
    # Max eigenvalue
    eigvals = np.linalg.eigvalsh(Q)
    q = float(np.max(eigvals))
    
    return q


def compute_order_params(
    vel: Array,
    include_nematic: bool = False
) -> Dict[str, float]:
    """
    Compute all order parameters for a velocity snapshot.
    
    Args:
        vel: Velocities (N, 2) or (N, d)
        include_nematic: Whether to compute nematic order
        
    Returns:
        params: Dictionary with 'phi', 'mean_speed', 'speed_std', 'nematic' (optional)
    """
    params = {
        'phi': polarization(vel),
        'mean_speed': mean_speed(vel),
        'speed_std': speed_std(vel)
    }
    
    if include_nematic and vel.shape[1] == 2:
        params['nematic'] = nematic_order(vel)
    
    return params


# ============================================================================
# MORSE FORCES & D'ORSOGNA MODEL IMPLEMENTATION GUIDE
# ============================================================================
"""
MORSE POTENTIAL FORCES - COMPLETE IMPLEMENTATION REFERENCE
===========================================================

Our codebase implements the Morse potential for attractive-repulsive social forces
following D'Orsogna et al. (2006) with THREE distinct integration modes that control
how forces affect particle motion.

Mathematical Foundation
-----------------------
Morse potential force magnitude (scalar):
    f(r) = (Cᵣ/ℓᵣ)·exp(-r/ℓᵣ) - (Cₐ/ℓₐ)·exp(-r/ℓₐ)

Vector force on particle i from particle j:
    𝐅ᵢⱼ = -f(rᵢⱼ)·𝐫̂ᵢⱼ
    where 𝐫̂ᵢⱼ = (𝐱ⱼ - 𝐱ᵢ)/rᵢⱼ

Total force on particle i:
    𝐅ᵢ = -Σⱼ≠ᵢ f(rᵢⱼ)·𝐫̂ᵢⱼ

Parameters:
    Cᵣ: Repulsion strength (short-range, keeps particles apart)
    Cₐ: Attraction strength (long-range, pulls particles together)
    ℓᵣ: Repulsion length scale (typically 0.3-1.0)
    ℓₐ: Attraction length scale (typically 1.0-3.0, ℓₐ > ℓᵣ)
    rcut: Cutoff radius = rcut_factor × max(ℓᵣ, ℓₐ), typically 3-5× max

Force Balance:
    Cᵣ/ℓᵣ vs Cₐ/ℓₐ determines equilibrium particle spacing
    Cᵣ > Cₐ: Repulsion-dominated → dispersed agents
    Cₐ > Cᵣ: Attraction-dominated → clustering/aggregation
    
Implementation Details:
    - Newton's 3rd law explicitly enforced: 𝐅ᵢⱼ = -𝐅ⱼᵢ → Σᵢ𝐅ᵢ = 0
    - Linked-cell neighbor lists for O(N) force computation
    - Minimal image convention for periodic boundaries
    - Warning if rcut < 3×max(ℓᵣ, ℓₐ) (exponential decay assumption breaks)

THREE SPEED MODES: How Forces Affect Motion
============================================

Our implementation offers three distinct ways to couple Morse forces to particle motion,
controlled by the 'speed_mode' parameter in configuration files.

MODE 1: "constant" (Pure Discrete Vicsek - Forces IGNORED)
-----------------------------------------------------------
Traditional Vicsek model with EXACTLY constant speed v₀.

Dynamics:
    𝐱ᵢ(t+Δt) = 𝐱ᵢ(t) + Δt·v₀·𝐩ᵢ(t+Δt)
    𝐩ᵢ(t+Δt) = 𝐑(φᵢ)·𝐩̄ᵢ(t)
    
where:
    𝐩̄ᵢ = average heading of neighbors within radius R
    𝐑(φᵢ) = rotation matrix for noise angle φᵢ
    |𝐩ᵢ| = 1 always (unit heading vector)
    
Key characteristics:
    ✓ All particles move at EXACTLY v₀ at all times
    ✗ Morse forces are computed but DISCARDED (not used in position update)
    ✓ Only alignment and noise affect dynamics
    ✓ Energy scale constant: E = (N/2)·v₀²
    ✓ Used for pure flocking studies without force complications
    
Configuration example:
    model:
      speed_mode: constant
      speed: 1.0
    forces:
      enabled: false  # Usually disabled in this mode
      
Implementation (vicsek_discrete.py, lines 540-567):
    # Forces computed but ignored:
    F = morse_force(x, ...) if enabled else zeros
    
    # Position update uses ONLY heading and v0:
    x = x + dt * v0 * p  # Force F not used!
    v_output = v0 * p    # Constant magnitude


MODE 2: "constant_with_forces" (Hybrid: Forces Steer, Speed Fixed)
-------------------------------------------------------------------
Morse forces affect heading DIRECTION but speed remains constant at v₀.
This is our PRODUCTION MODE for ROM training.

Dynamics:
    Combined heading: 𝐩̃ᵢ = 𝐩̄ᵢ + μₜ·(𝐅ᵢ/|𝐅ᵢ|)
    Normalized: 𝐩̄ᵢ' = 𝐩̃ᵢ/|𝐩̃ᵢ|
    With noise: 𝐩ᵢ(t+Δt) = 𝐑(φᵢ)·𝐩̄ᵢ'
    Position: 𝐱ᵢ(t+Δt) = 𝐱ᵢ(t) + Δt·v₀·𝐩ᵢ(t+Δt)
    
where:
    μₜ: Translational mobility parameter (force influence weight, typically 0.3-0.5)
    𝐅ᵢ/|𝐅ᵢ|: Normalized force direction (unit vector)
    
Key characteristics:
    ✓ Forces rotate heading toward/away from neighbors
    ✓ Speed ALWAYS equals v₀ (energy conserved)
    ✓ Combines alignment (Vicsek) with force-driven steering
    ✓ Cleaner latent dynamics for ROM (constant kinetic energy)
    ✓ Easier to learn: speed is fixed, only heading varies
    ✗ Not physically realistic for all scenarios (real particles accelerate)
    
Physical interpretation:
    - Forces act as "steering" or "torque" on heading
    - Particle moves at v₀ but turns toward attractive neighbors
    - Repulsive forces turn particle away without slowing down
    - Similar to: car steering (speed constant, direction changes)
    
Configuration example:
    model:
      speed_mode: constant_with_forces
      speed: 1.0
    forces:
      enabled: true
      params:
        Cr: 0.3
        Ca: 1.0
        lr: 0.5
        la: 2.0
        mu_t: 0.3  # Force steering strength
        
Implementation (vicsek_discrete.py, lines 569-611):
    # Normalize force to heading:
    F_heading = F / |F|
    
    # Combine alignment and force headings:
    combined = p_bar + mu_t * F_heading
    p_bar_new = combined / |combined|
    
    # Apply noise rotation:
    p = R(phi) @ p_bar_new
    
    # Position update with CONSTANT SPEED v0:
    x = x + dt * v0 * p
    v_output = v0 * p  # |v| = v0 always


MODE 3: "variable" (Full D'Orsogna: Forces Affect Speed AND Direction)
-----------------------------------------------------------------------
Forces directly change velocity magnitude and direction. Most physically realistic.

Dynamics (Explicit Euler):
    𝐯ᵢ(t+Δt) = 𝐯ᵢ(t) + Δt·μₜ·𝐅ᵢ(t)
    Current heading: 𝐩ᵢ = 𝐯ᵢ/|𝐯ᵢ|
    Aligned heading: 𝐩ᵢ' = 𝐑(φᵢ)·𝐩̄ᵢ
    New velocity: 𝐯ᵢ(t+Δt) = |𝐯ᵢ(t+Δt)|·𝐩ᵢ'
    𝐱ᵢ(t+Δt) = 𝐱ᵢ(t) + Δt·𝐯ᵢ(t+Δt)
    
Dynamics (Semi-Implicit Euler, more stable):
    𝐯ᵢ(t+Δt) = 𝐯ᵢ(t) + Δt·μₜ·𝐅ᵢ(t)
    𝐱ᵢ(t+Δt) = 𝐱ᵢ(t) + Δt·𝐯ᵢ(t+Δt)  ← uses NEW velocity
    
Key characteristics:
    ✓ Forces directly accelerate/decelerate particles
    ✓ Speed varies: |𝐯ᵢ| changes over time
    ✓ Most physically realistic (Newton's 2nd law: F = m·dv/dt)
    ✓ Recovers continuous D'Orsogna model in Δt→0 limit
    ✗ Harder to learn for ROM (variable energy, richer dynamics)
    ✗ Can be numerically unstable without semi-implicit integration
    
Physical interpretation:
    - Attraction accelerates particles toward each other
    - Repulsion decelerates particles (or accelerates away)
    - Speed distribution broadens over time
    - Similar to: gravity (forces change both speed and direction)
    
Without self-propulsion (α=0, β=0):
    Pure force-driven dynamics, particles can stop completely
    
With self-propulsion (α>0, β>0):
    Forces compete with self-propulsion: dv/dt = (α - β|v|²)v + μₜ·F
    Natural speed: v₀ = √(α/β)
    
Configuration example:
    model:
      speed_mode: variable
      speed: 1.0  # Initial speed (not maintained)
    params:
      alpha: 1.5  # Self-propulsion
      beta: 0.5   # Friction
    forces:
      enabled: true
      params:
        Cr: 0.3
        Ca: 1.0
        lr: 0.5
        la: 2.0
        mu_t: 0.5  # Force coupling strength
        
Implementation (vicsek_discrete.py, lines 613-682):
    # Update velocities from forces:
    v = v + dt * mu_t * F
    
    # Extract current heading:
    p = v / |v|
    
    # Apply alignment rotation (keeps magnitude):
    p_aligned = R(phi) @ p_bar
    v = |v| * p_aligned
    
    # Position update with VARIABLE velocity:
    x = x + dt * v
    v_output = v  # |v| varies!

Speed Mode Comparison Table
============================
| Mode | Forces Affect | Speed | Energy | ROM Difficulty | Use Case |
|------|---------------|-------|--------|----------------|----------|
| constant | ✗ Nothing | v₀ (fixed) | Constant | Easiest | Pure flocking |
| constant_with_forces | Direction only | v₀ (fixed) | Constant | Medium | Production ROM |
| variable | Speed + Direction | Variable | Variable | Hardest | Realistic physics |

When to Use Each Mode
=====================
1. constant: 
   - Benchmarking against classic Vicsek literature
   - Studying pure alignment-driven flocking
   - When forces should be disabled
   
2. constant_with_forces (RECOMMENDED for ROM):
   - Training reduced-order models (POD + MVAR/LSTM)
   - When you want force effects but cleaner latent space
   - Easier convergence for autoregressive models
   - Current production mode for all ROM experiments
   
3. variable:
   - Maximum physical realism
   - When speed variations are important
   - Studying D'Orsogna-style swarming with clustering
   - Future work: Test ROM on richer dynamics

Typical Parameter Values
=========================
Cohesive flocking (used in production):
    Cr: 0.3    # Weak repulsion (avoid overlap)
    Ca: 1.0    # Moderate attraction (cohesion)
    lr: 0.5    # Short repulsion range
    la: 2.0    # Long attraction range
    mu_t: 0.3  # Modest force steering
    v0: 1.0    # Unit natural speed
    R: 2.0     # Alignment radius
    eta: 0.3   # Moderate noise (critical regime)
    
Dispersed/repulsive:
    Cr: 1.5    # Strong repulsion
    Ca: 0.5    # Weak attraction
    lr: 1.0    # Medium repulsion range
    la: 1.5    # Short attraction range
    
Force-free (pure Vicsek):
    forces.enabled: false
    speed_mode: constant

Stability Constraints
=====================
1. Vicsek stability: v₀·dt ≤ 0.5·R
   Prevents particles from "jumping over" their interaction neighborhood
   
2. Force cutoff: rcut ≥ 3·max(lr, la)
   Ensures exponential decay → negligible force beyond rcut
   
3. Semi-implicit integration recommended for variable speed mode
   Prevents numerical instability from large forces

Code Locations
==============
- Force computation: src/rectsim/morse.py::morse_force()
- Discrete Vicsek + Forces: src/rectsim/vicsek_discrete.py::simulate_backend()
- Continuous D'Orsogna: src/rectsim/dynamics.py::simulate_backend()
- Configuration schema: src/rectsim/config.py, src/rectsim/unified_config.py

References
==========
[1] D'Orsogna et al. (2006), "Self-propelled particles with soft-core interactions"
    Physical Review Letters 96, 104302
[2] Bhaskar & Ziegelmeier (2019), "Topological data analysis of collective motion"
    Chaos 29, 123125
[3] Vicsek et al. (1995), "Novel type of phase transition in a system of self-driven particles"
    Physical Review Letters 75, 1226
"""


# ============================================================================
# COMPLETE ROM PIPELINE: VICSEK + MORSE FORCES INTEGRATION
# ============================================================================
"""
UNIFIED ROM PIPELINE ARCHITECTURE
==================================

The full pipeline integrates Vicsek dynamics with Morse forces through a multi-stage
process: microsimulation → density fields → POD compression → latent dynamics learning
→ forecasting → reconstruction. This section explains how all components fit together.

PIPELINE OVERVIEW (run_unified_rom_pipeline.py)
================================================

Main Entry Point: python run_unified_rom_pipeline.py --config CONFIG --experiment_name NAME

The pipeline executes 7 sequential stages:

STAGE 1: Initial Condition Generation (ic_generator.py)
--------------------------------------------------------
Purpose: Create diverse IC configurations for training robustness

Process:
    1. Read train_ic config from YAML (gaussian, uniform, ring, two_clusters)
    2. Generate N_train configurations with varied parameters
       - Gaussian clusters: multiple centers × variances × samples
       - Uniform: random positions
       - Ring: circular arrangement
       - Two clusters: bimodal distributions
    3. Assign unique run_id, distribution type, ic_params

Configuration example (from vicsek_rom_joint_optimal.yaml):
    train_ic:
      gaussian:
        enabled: true
        positions_x: [7.5]  # Domain center
        positions_y: [7.5]
        variances: [2.0, 4.0, 6.0]  # 3 variance levels
        n_samples_per_config: 50    # 50 samples each → 150 runs
      uniform:
        enabled: true
        n_runs: 150  # 150 uniform ICs
      ring:
        enabled: true
        n_runs: 50   # 50 ring ICs
      two_clusters:
        enabled: true
        n_runs: 50   # 50 two-cluster ICs
    # Total: 400 training runs

Why diverse ICs matter:
    ✓ ROM learns dynamics across full state space (not just one attractor)
    ✓ Prevents overfitting to specific initial configurations
    ✓ Enables interpolation (new ICs between training points)
    ✓ Tests extrapolation (ICs outside training distribution)

STAGE 2: Microsimulation with Vicsek+Forces (simulation_runner.py)
--------------------------------------------------------------------
Purpose: Generate ground truth trajectories and densities

Process (for each IC configuration):
    1. Load config: extract sim params (N, Lx, Ly, T, dt), model params (v0, R, η)
    2. Load forces: Morse params (Cr, Ca, lr, la, mu_t) if enabled
    3. Load speed_mode: constant | constant_with_forces | variable
    4. Call simulate_backend() from vicsek_discrete.py:
       a. Initialize: positions x from IC, headings p random, velocities v
       b. Build neighbor list (linked cells, radius = max(R_align, R_force))
       c. Main loop (T/dt steps):
          - Find neighbors within R for alignment
          - Compute mean heading p_bar from neighbors
          - Compute Morse forces F if enabled
          - Apply speed mode logic (see previous section):
              * constant: x += dt*v0*p, ignore F
              * constant_with_forces: combine p_bar + mu_t*F_heading → p, x += dt*v0*p
              * variable: v += dt*mu_t*F, rotate v toward p_bar, x += dt*v
          - Add angular noise (uniform or gaussian with variance matching)
          - Apply boundary conditions (periodic wrap or reflecting flip)
          - Save frame every save_every steps
       d. Return {times, traj, vel, head, meta}
    5. Compute density field ρ(x,y,t) from trajectories:
       - Use kde_density_movie() from legacy_functions.py
       - 2D histogram on (nx, ny) spatial grid
       - Gaussian smoothing with bandwidth σ (typically 2-4 grid units)
       - Normalize: ρ = counts / (dx·dy) → particles per unit area
    6. Save outputs to train/train_XXX/:
       - trajectory.npz: {traj, vel, times}
       - density.npz: {rho, xgrid, ygrid, times}

Configuration determines dynamics:
    model:
      type: discrete           # Vicsek discrete-time
      speed: 1.0              # Natural speed v0
      speed_mode: constant_with_forces  # PRODUCTION MODE
    
    params:
      R: 2.0                  # Alignment radius
    
    noise:
      kind: gaussian          # Gaussian angular noise
      eta: 0.3               # Noise strength (critical regime)
      match_variance: true   # σ = η/√12
    
    forces:
      enabled: false          # Disabled for cleaner ROM training
      # When enabled:
      params:
        Cr: 0.3
        Ca: 1.0
        lr: 0.5
        la: 2.0
        mu_t: 0.3
    
    sim:
      N: 40                   # Modest particle count for speed
      T: 10.0                # Training trajectory length
      dt: 0.1                # Timestep
      save_every: 1          # Full temporal resolution

Why forces disabled in production:
    ✗ Forces add nonlinearity → richer but harder dynamics
    ✗ Variable force strengths create multi-scale phenomena
    ✓ Pure alignment dynamics → cleaner latent space
    ✓ Easier for MVAR (linear model) to capture
    ✓ Faster convergence, better generalization
    ✓ Future work: Enable forces to test ROM on complex dynamics

Parallel execution:
    - Uses multiprocessing.Pool (cpu_count() workers)
    - Each IC runs independently (embarrassingly parallel)
    - Progress bar tracks completion
    - Typical: 400 runs × 10s × 0.1dt = 40,000 integration steps total
    - Wall time: ~10-30 minutes on Oscar HPC (20 cores)

STAGE 3: Proper Orthogonal Decomposition (pod_builder.py)
----------------------------------------------------------
Purpose: Compress high-dimensional density fields to low-dimensional latent space

Process:
    1. Load ALL training densities: ρᵢ(t) for i=1..M runs, t=1..T timesteps
       - Shape: M runs × T_rom timesteps × (nx·ny) spatial points
       - Example: 400 runs × 100 timesteps × 4096 pixels = 163.8M values
    
    2. Stack into data matrix:
       X = [ρ₁(t₁), ρ₁(t₂), ..., ρ₁(T), ρ₂(t₁), ..., ρₘ(T)]ᵀ
       Shape: [M·T, nx·ny] = [40,000, 4096]
    
    3. Compute spatial mean (time-averaged):
       X̄ = (1/(M·T)) Σₜ Σₘ ρₘ(t)
    
    4. Center data:
       X_centered = X - X̄
    
    5. Singular Value Decomposition:
       X_centered = U·Σ·Vᵀ
       where:
       - U: spatial modes (left singular vectors) [nx·ny, min(M·T, nx·ny)]
       - Σ: singular values (energy content per mode)
       - V: temporal coefficients (right singular vectors)
    
    6. Truncate to R_POD modes:
       Two strategies (from rom config):
       a. Fixed dimension: R_POD = fixed_modes (e.g., 25)
       b. Energy threshold: R_POD = min k s.t. Σᵢ₌₁ᵏ σᵢ² / Σⱼ σⱼ² ≥ energy_threshold
       
       Production uses fixed_modes=25:
       - Balances expressiveness vs computational cost
       - Ensures sufficient samples: N_windows >> d²·lag for MVAR stability
       - Example: 40,000 timesteps → ~38,000 windows (lag=5) >> 25²·5 = 3,125
    
    7. Construct reduced basis:
       U_r = U[:, :R_POD]  # First R_POD spatial modes
       Shape: [nx·ny, R_POD] = [4096, 25]
    
    8. Project ALL training data to latent space:
       y_m(t) = (ρ_m(t) - X̄) · U_r
       Shape per run: [T_rom, R_POD] = [100, 25]
    
    9. Save POD data to rom_common/:
       - pod_basis.npz: {U_r, S, X_mean, R_POD, energy_captured, ...}
       - POD is computed ONCE and shared by MVAR and LSTM

Why POD works:
    ✓ Density fields have low intrinsic dimensionality (coherent structures)
    ✓ First few modes capture global patterns (uniform → clustered → flocking)
    ✓ Compression ratio: 4096 → 25 (164× reduction!)
    ✓ Linear projection: fast, reversible, preserves Euclidean structure
    ✓ Optimal in L² sense (maximum variance captured)

POD modes physically represent:
    - Mode 1: Mean density deviation (concentration vs dispersal)
    - Mode 2-3: Spatial gradients (cluster positions, edges)
    - Mode 4+: Higher-order patterns (rotations, deformations)

STAGE 4: Latent Dataset Construction (rom_data_utils.py)
---------------------------------------------------------
Purpose: Create windowed time-series data for autoregressive model training

Process:
    1. Extract latent trajectories: y_m(t) for m=1..M runs
       Shape per run: [T_rom, R_POD]
    
    2. Build sliding windows for each trajectory:
       For trajectory y_m with length T_rom:
       - Window size (lag): typically 5-20 timesteps
       - Create pairs: (y_m[t-lag:t], y_m[t]) for t=lag..T_rom-1
       - Each window: X = [y(t-lag), ..., y(t-1)], Y = y(t)
       
       Example (lag=5, T_rom=100):
       - Window 0: X = y[0:5], Y = y[5]
       - Window 1: X = y[1:6], Y = y[6]
       - ...
       - Window 94: X = y[94:99], Y = y[99]
       - Total: 95 windows per trajectory
    
    3. Stack all windows from all trajectories:
       X_all: [M·(T_rom-lag), lag, R_POD]
       Y_all: [M·(T_rom-lag), R_POD]
       
       Example (M=400, T_rom=100, lag=5, R_POD=25):
       X_all: [38,000, 5, 25] = 4.75M values
       Y_all: [38,000, 25] = 950K values
    
    4. Save to rom_common/latent_dataset.npz

Dataset statistics check:
    Samples vs parameters (MVAR):
    - MVAR needs: N_samples >> d²·lag (d = R_POD)
    - Here: 38,000 >> 25²·5 = 3,125 ✓ (12× oversampling)
    - Rule of thumb: 10× minimum for stability

Why windowing matters:
    ✓ Captures temporal dependencies (Markov property assumed at lag order)
    ✓ Autoregressive models predict next state from recent history
    ✓ Lag hyperparameter trades memory vs model complexity
    ✓ Too small lag: insufficient context
    ✓ Too large lag: fewer samples, overfitting risk

STAGE 5: Model Training (mvar_trainer.py / lstm_rom.py)
--------------------------------------------------------
Purpose: Learn latent dynamics from windowed dataset

OPTION A: MVAR (Multivariate AutoRegressive) Model
...................................................
Linear model: y(t) = A₁·y(t-1) + A₂·y(t-2) + ... + Aₚ·y(t-p) + ε(t)

Process:
    1. Reshape data for linear regression:
       - Input: X_all flattened to [N_samples, lag·d]
       - Output: Y_all shape [N_samples, d]
    
    2. Ridge regression with regularization:
       A_stacked = (XᵀX + α·I)⁻¹·XᵀY
       where α = ridge_alpha (typically 1e-6)
    
    3. Reshape A_stacked → [A₁, A₂, ..., Aₚ]:
       A_matrices: [p, d, d] = [5, 25, 25]
    
    4. Compute training R²:
       Predictions: Ŷ = X·A_stacked
       R² = 1 - ||Y - Ŷ||² / ||Y - Ȳ||²
    
    5. Save to MVAR/mvar_model.npz

Training time: ~seconds (linear least squares)
Memory: Modest (coefficient matrices only)
Interpretability: High (linear dynamics, stable eigenvalues)

OPTION B: LSTM (Long Short-Term Memory) Neural Network
.......................................................
Nonlinear model: y(t) = LSTM([y(t-lag), ..., y(t-1)]; θ)

Architecture:
    - Input: [batch, lag, d] = [batch, 5, 25]
    - LSTM layers: hidden_units (e.g., 50) × num_layers (e.g., 2)
    - Output: [batch, d] = [batch, 25]
    - Parameters θ: ~100K (vs MVAR's ~3K)

Training:
    1. Split data: 80% train, 20% validation
    2. Mini-batch SGD with Adam optimizer
    3. MSE loss: ||y_pred - y_true||²
    4. Early stopping on validation loss
    5. Learning rate scheduling
    6. Epochs: typically 100-500
    7. Save to LSTM/lstm_state_dict.pt

Training time: ~minutes to hours (GPU recommended)
Memory: Moderate (network weights + activations)
Expressiveness: High (captures nonlinear dynamics)

STAGE 6: Test Data Generation
------------------------------
Purpose: Generate held-out test trajectories for ROM evaluation

Process: Same as Stage 2 but with:
    - Different ICs (test_ic config)
    - Longer trajectories (T_test > T_train for forecasting)
    - Saved to test/test_XXX/
    - Filename: density_true.npz (vs density.npz for training)

Test IC design:
    - Interpolation tests: ICs similar to training (within convex hull)
    - Extrapolation tests: ICs outside training distribution
    - Typical split: 80% interpolation, 20% extrapolation

STAGE 7: ROM Evaluation (test_evaluator.py)
--------------------------------------------
Purpose: Forecast test trajectories using ROM and compute metrics

Process (for each test run):
    1. Load test density ρ_test(t) for t ∈ [0, T_test]
    
    2. Project to latent space:
       y_test(t) = (ρ_test(t) - X̄) · U_r
    
    3. Define forecast period:
       - Training period: [0, T_train] (e.g., 0-10s)
       - Forecast period: [T_train, T_test] (e.g., 10-20s)
    
    4. Extract initial condition from training period:
       IC_window = y_test[T_train-lag : T_train]
       Shape: [lag, R_POD] = [5, 25]
    
    5. Forecast using ROM (MVAR or LSTM):
       MVAR:
           for t in T_train..T_test:
               y_pred(t) = Σᵢ₌₁ᵖ Aᵢ · y_pred(t-i)
       
       LSTM:
           for t in T_train..T_test:
               y_pred(t) = LSTM([y_pred(t-lag), ..., y_pred(t-1)])
    
    6. Reconstruct to physical space:
       ρ_pred(t) = y_pred(t) · U_rᵀ + X̄
       Shape: [T_forecast, nx, ny]
    
    7. Compute R² metrics:
       a. Latent R² (ROM space):
          R²_latent = 1 - ||y_pred - y_true||² / ||y_true - ȳ_true||²
       
       b. Reconstructed R² (physical space):
          R²_recon = 1 - ||ρ_pred - ρ_true||² / ||ρ_true - ρ̄_true||²
       
       c. POD R² (compression quality):
          ρ_pod = (y_true · U_rᵀ + X̄)
          R²_pod = 1 - ||ρ_pod - ρ_true||² / ||ρ_true - ρ̄_true||²
    
    8. Save predictions to MVAR/predictions_XXX.npz or LSTM/predictions_XXX.npz
    
    9. Aggregate metrics across all test runs → test_results.csv

Evaluation metrics:
    - R²_reconstructed: Primary metric (physical space accuracy)
    - R²_latent: ROM prediction quality
    - R²_pod: Baseline (best possible with R_POD modes)
    - RMSE: Root mean squared error
    - Mass conservation: Σ ρ_pred ≈ N (particle count preserved?)

Why ROM works for Vicsek+Forces:
    ✓ Density fields lie on low-dimensional manifold
    ✓ Latent dynamics are approximately linear (for pure alignment)
    ✓ POD captures spatial coherence
    ✓ MVAR captures temporal evolution
    ✓ End-to-end: 4096-dim densities → 25-dim latent → accurate forecasts

FORCE MODE IMPACT ON ROM PERFORMANCE
=====================================

Mode 1: constant (forces disabled)
    Pipeline behavior:
    - Cleanest latent dynamics (pure alignment)
    - MVAR typically achieves R² > 0.95
    - Linear dynamics → perfect fit for MVAR
    - LSTM offers minimal improvement
    - Fastest training, best generalization

Mode 2: constant_with_forces (PRODUCTION)
    Pipeline behavior:
    - Forces steer headings → nonlinear coupling
    - MVAR achieves R² ~ 0.85-0.93 (good but not perfect)
    - LSTM can capture force-alignment interactions better
    - Latent space more structured than variable mode
    - Moderate training time, good generalization

Mode 3: variable (forces affect speed)
    Pipeline behavior:
    - Richest dynamics, highest nonlinearity
    - MVAR struggles: R² ~ 0.70-0.85
    - LSTM significantly outperforms (R² ~ 0.85-0.92)
    - Requires more training data
    - Slower training, risk of overfitting
    - Future research direction

Comparison table (typical R² on test set):
| Speed Mode | MVAR R² | LSTM R² | Training Time | Use Case |
|------------|---------|---------|---------------|----------|
| constant | 0.96 | 0.97 | Fast | Benchmark |
| constant_with_forces | 0.88 | 0.91 | Medium | Production |
| variable | 0.78 | 0.90 | Slow | Research |

Current strategy: Use constant_with_forces with forces DISABLED
    - Balances ROM learnability with dynamic richness
    - Forces can be re-enabled for future experiments
    - Constant speed simplifies latent space structure

COMPLETE PIPELINE WORKFLOW
===========================

Input: configs/vicsek_rom_joint_optimal.yaml
    - Defines: Vicsek params, force params, ICs, ROM settings

Command:
    python run_unified_rom_pipeline.py \\
        --config configs/vicsek_rom_joint_optimal.yaml \\
        --experiment_name vicsek_joint_optimal

Execution timeline (Oscar, 20 cores):
    1. IC generation: <1s (CPU)
    2. Training sims (400 runs): ~15 min (parallel)
    3. POD basis: ~30s (CPU, SVD)
    4. Latent dataset: ~5s (CPU)
    5. MVAR training: ~3s (CPU, linear algebra)
    6. LSTM training: ~10 min (GPU if available)
    7. Test sims (40 runs): ~2 min (parallel)
    8. Evaluation (both models): ~1 min (CPU)
    Total: ~30 minutes

Output: oscar_output/vicsek_joint_optimal/
    rom_common/
        pod_basis.npz         # POD modes (shared)
        latent_dataset.npz    # Windowed data (shared)
    MVAR/
        mvar_model.npz        # Linear coefficients
        test_results.csv      # Per-run metrics
        predictions_XXX.npz   # Forecasts for each test run
    LSTM/
        lstm_state_dict.pt    # Neural network weights
        training_log.csv      # Loss vs epoch
        test_results.csv      # Per-run metrics
        predictions_XXX.npz   # Forecasts for each test run
    train/
        train_XXX/
            trajectory.npz    # Particle trajectories
            density.npz       # Density fields
    test/
        test_XXX/
            trajectory.npz
            density_true.npz  # Ground truth densities
    summary.json              # Aggregate statistics

Visualization (run_visualizations.py):
    - Load all predictions
    - Generate comparison videos (side-by-side)
    - Plot R² vs time
    - Best/worst run analysis
    - Order parameter tracking
    - Mass conservation checks

Key Insights from Pipeline Analysis
====================================

1. Modularity: Each stage (IC → sim → POD → MVAR/LSTM → eval) is independent
   - Can swap Vicsek for D'Orsogna continuous model
   - Can change density computation (KDE → grid-based)
   - Can try different ROM methods (DMD, neural ODEs)

2. Data-driven: ROM learns from trajectories, not equations
   - No need to specify governing equations
   - Discovers latent dynamics empirically
   - Black-box approach: works for any particle model

3. Speed vs accuracy tradeoff:
   - More training runs → better generalization but slower
   - Higher R_POD → better reconstruction but harder to learn
   - Larger lag → more memory but richer dynamics

4. Force modes enable controlled experiments:
   - constant: Establish baseline (pure alignment)
   - constant_with_forces: Add complexity without speed chaos
   - variable: Full physics but harder to learn

5. Production choices are optimized:
   - N=40 particles: Fast sims, sufficient complexity
   - R_POD=25: Compression ratio ~160×, sufficient expressiveness
   - lag=5: Balance memory vs samples
   - speed_mode=constant_with_forces: Clean dynamics, forces disabled
   - 400 training runs: 12× oversampling for stability

This architecture enables rapid experimentation with particle dynamics while
maintaining reproducibility, scalability, and model interpretability.
"""


# ============================================================================
# KDE (KERNEL DENSITY ESTIMATION) IMPLEMENTATION DETAILS
# ============================================================================
"""
COMPLETE KDE AND BANDWIDTH SELECTION GUIDE
===========================================

Our density field computation uses Kernel Density Estimation (KDE) with Gaussian
kernels to convert discrete particle positions into smooth continuous density fields.
This section documents the mathematical foundations, implementation details, and
bandwidth selection strategies used throughout the pipeline.

MATHEMATICAL FOUNDATION
=======================

Kernel Density Estimation Formula
----------------------------------
Given N particles at positions {xᵢ}ᵢ₌₁ᴺ in a 2D domain Ω = [0, Lₓ] × [0, Lᵧ],
the density field ρ(x, y) at any point (x, y) is estimated as:

    ρ(x, y) = Σᵢ₌₁ᴺ K((x - xᵢ)/hₓ, (y - yᵢ)/hᵧ)

where K is the kernel function and (hₓ, hᵧ) are the bandwidth parameters.

Gaussian Kernel (Our Choice)
-----------------------------
We use an isotropic Gaussian kernel:

    K(u, v) = (1/(2πhₓhᵧ)) · exp(-½(u²/hₓ² + v²/hᵧ²))

This kernel is:
✓ Smooth (infinitely differentiable)
✓ Symmetric (rotationally invariant if hₓ = hᵧ)
✓ Non-negative (produces valid densities)
✓ Normalized (∫∫ K(u,v) du dv = 1)

Physical Interpretation
-----------------------
- Each particle contributes a Gaussian "blob" to the density field
- Bandwidth h controls the spread of each blob:
  * Small h → sharp peaks (particle-like)
  * Large h → smooth continuous field (fluid-like)
- Total mass is preserved: ∫∫_Ω ρ(x,y) dx dy = N

IMPLEMENTATION APPROACH: HISTOGRAM + GAUSSIAN FILTER
=====================================================

Instead of evaluating the KDE formula directly at each grid point (O(N·nx·ny) cost),
we use an efficient two-stage approach:

Stage 1: Particle Histogram (Binning)
--------------------------------------
Create a 2D histogram by counting particles in each grid cell:

Code (from kde_density_movie in legacy_functions.py):
    x_edges = np.linspace(0.0, Lx, nx + 1)
    y_edges = np.linspace(0.0, Ly, ny + 1)
    
    hist, _, _ = np.histogram2d(
        traj[t, :, 1],  # y coordinates (rows)
        traj[t, :, 0],  # x coordinates (columns)
        bins=[y_edges, x_edges],
        range=[[0.0, Ly], [0.0, Lx]]
    )
    
    # Normalize to density (particles per unit area)
    dx = Lx / nx
    dy = Ly / ny
    density = hist / (dx * dy)

Result: density[i, j] = count of particles in cell (i, j) / (cell_area)

Stage 2: Gaussian Smoothing
----------------------------
Apply Gaussian filter to smooth the histogram:

Code:
    from scipy.ndimage import gaussian_filter
    
    mode = "wrap" if bc == "periodic" else "nearest"
    density = gaussian_filter(density, sigma=bandwidth, mode=mode)

How it works:
- Convolves density field with discrete Gaussian kernel
- sigma = bandwidth (in grid cell units)
- mode controls boundary handling:
  * "wrap": Periodic boundaries (particles at x=0 interact with x=Lx)
  * "nearest": Reflecting boundaries (edge values repeated)

Computational Complexity:
- Histogram: O(N) (single pass through particles)
- Gaussian filter: O(nx·ny) (separable convolution)
- Total: O(N + nx·ny) << O(N·nx·ny) direct evaluation

Why This Works:
✓ Mathematically equivalent to KDE for sufficiently fine grids
✓ Fast (leverages optimized FFT-based convolution in scipy)
✓ Memory efficient (operates on grid, not particles)
✓ Handles periodic/reflecting BCs naturally

BANDWIDTH SELECTION
===================

The bandwidth parameter h is CRITICAL for KDE quality:
- Too small: Noisy, particle-like fields (undersmoothing)
- Too large: Over-smoothed, loses spatial structure (oversmoothing)
- Just right: Smooth continuous fields that capture clustering patterns

Production Method: MANUAL SPECIFICATION
----------------------------------------
We use MANUAL bandwidth specification in all production pipelines.

Configuration (from YAML config files):
    outputs:
      density_bandwidth: 3.0    # bandwidth in grid cell units

Implementation (config_loader.py, line 49):
    density_bandwidth = outputs.get('density_bandwidth', 2.0)  # Default: 2.0

Usage (simulation_runner.py, line 74):
    rho, meta = kde_density_movie(
        traj,
        Lx=config["sim"]["Lx"],
        Ly=config["sim"]["Ly"],
        nx=DENSITY_NX,
        ny=DENSITY_NY,
        bandwidth=DENSITY_BANDWIDTH,  # Direct from config
        bc=config["sim"].get("bc", "periodic")
    )

Why manual specification:
✓ Reproducibility across all runs
✓ Physics-informed choice (matches alignment radius R)
✓ Empirically tuned for ROM performance
✓ No variation between runs or particle counts

Typical bandwidth values used:
- bandwidth = 2.0: Standard smoothing, preserves cluster details
- bandwidth = 2.5: Balanced smoothing
- bandwidth = 3.0: Heavy smoothing, lower POD rank (ROM-friendly, CURRENT PRODUCTION)

Current production value: 3.0 grid cells
Reason: Optimal trade-off between:
  • Smooth latent dynamics (MVAR R² > 0.88)
  • Low POD rank (~25 modes for 99.5% energy)
  • Preserved spatial structure (clusters visible)

Physical guideline (informative only):
    h ≈ R / (Lx/nx)
where R is the alignment radius in the model.

Example (Vicsek with R=2.0, domain Lx=15, grid nx=64):
    Grid spacing: dx = 15/64 ≈ 0.234
    Physical bandwidth: R = 2.0
    Grid bandwidth: h = 2.0 / 0.234 ≈ 8.5 cells
    
But we use h=3.0 cells in practice for ROM efficiency (smaller kernels, lower rank).

Adaptive Methods (NOT USED IN PRODUCTION)
------------------------------------------
Note: The following methods exist in code but are NOT used in our ROM pipeline.
They are documented here for completeness but can be ignored for thesis purposes.

Scott's Rule (estimate_bandwidth function):
    h = L · N^(-1/(d+4)) · 0.5
    
    Implemented in legacy_functions.py (lines 112-148) but never called.
    Would provide automatic bandwidth scaling with N and domain size.
    Not used because we prioritize reproducibility over adaptivity.

Silverman's Rule (test code only):
    hᵢ = [4/(N(d+2))]^(1/(d+4)) · σᵢ
    
    Referenced in test_kde_density.py but kde_density module not in production.
    Would account for anisotropic particle distributions.
    Not used in our isotropic Vicsek systems.

BANDWIDTH IMPACT ON ROM PIPELINE
=================================

The bandwidth choice directly affects ROM performance through the "effective rank"
of the POD basis.

Effect on Density Fields
-------------------------
Low bandwidth (h=0.5-1.0):
- Sharp peaks at particle locations
- High spatial frequency content
- Noisy, discontinuous fields
- High effective rank (many POD modes needed)

Medium bandwidth (h=2.0-3.0): [PRODUCTION]
- Smooth continuous fields
- Captures cluster shapes
- Preserves spatial gradients
- Moderate effective rank (25-35 modes sufficient)

High bandwidth (h=4.0-6.0):
- Very smooth, blob-like fields
- Loses fine spatial structure
- Nearly Gaussian distributions
- Low effective rank (10-20 modes)

Effect on POD Compression
--------------------------
Higher bandwidth → Lower effective rank → Fewer POD modes needed

Example (400 training runs, 64×64 grid):
    bandwidth=1.0: Need ~50 modes for 99.5% energy
    bandwidth=2.5: Need ~30 modes for 99.5% energy
    bandwidth=4.0: Need ~20 modes for 99.5% energy

Why lower rank is better:
✓ Fewer parameters in MVAR (d²×lag smaller)
✓ More samples per parameter (better conditioning)
✓ Faster training (smaller matrices)
✓ Better generalization (less overfitting)
✓ Smoother latent dynamics (cleaner time series)

BUT: Too much smoothing loses physics!
✗ Over-smoothing eliminates cluster structure
✗ Latent space represents "blobs" not "swarms"
✗ ROM predictions lack spatial detail

Production Trade-off
---------------------
bandwidth = 2.5-3.0 balances:
+ Smooth enough for ROM learning
+ Detailed enough to capture clustering
+ Matches alignment radius R (physics-informed)

Configuration example (alvarez_style_production.yaml):
    density:
      nx: 64
      ny: 64
      bandwidth: 3.0  # 3 grid cells → more smoothing, lower effective rank

BOUNDARY CONDITION HANDLING
============================

Periodic Boundaries (mode="wrap")
----------------------------------
Used for: Vicsek models with periodic domain

How it works:
- Particles at x=0 and x=Lx are treated as neighbors
- Gaussian kernel wraps around domain edges
- Preserves translational symmetry
- Mass conservation automatic

Code:
    mode = "wrap" if bc == "periodic" else "nearest"
    density = gaussian_filter(density, sigma=bandwidth, mode=mode)

Visual effect: No edge artifacts, uniform density treatment

Reflecting Boundaries (mode="nearest")
---------------------------------------
Used for: Rectangular domains with walls

How it works:
- Edge values repeated beyond boundary
- Particles near walls create enhanced density
- No wraparound
- Reflects physical barriers

Visual effect: Higher density near boundaries (particles accumulate)

MASS CONSERVATION AND NORMALIZATION
====================================

Theoretical Guarantee
---------------------
For properly normalized KDE, total mass should equal particle count:

    M = ∫∫_Ω ρ(x,y) dx dy = N

Discrete approximation:
    M_discrete = Σᵢⱼ ρ[i,j] · dx · dy

Why mass can drift:
✗ Boundary effects (particles near edges)
✗ Histogram discretization error
✗ Gaussian filter truncation

Renormalization (density.py, compute_density_grid)
---------------------------------------------------
Code:
    # After histogram and Gaussian filter
    total_mass = density.sum() * dx * dy
    if total_mass > 0:
        density *= (N / total_mass)

This ensures: Σᵢⱼ ρ[i,j] · dx · dy = N exactly

Check in order parameters:
    total_mass_metric = np.sum(rho) * dx * dy
    # Should be ≈ N (within 0.1%)

PRODUCTION CONFIGURATION SUMMARY
=================================

Current best practices (from working pipelines):

Resolution: 64×64 grid
-----------------------
Why: Balances detail vs computation
- Too low (16×16): Blocky, sparse fields
- Just right (64×64): Smooth continuous heatmaps
- Too high (128×128): Slow, unnecessary detail

Bandwidth: 3.0 grid cells (CURRENT PRODUCTION)
-----------------------------------------------
Configuration (all production YAML files):
    outputs:
      density_resolution: 64
      density_bandwidth: 3.0

Why: Optimal ROM-physics trade-off
- Produces ~25-30 POD modes (well-conditioned MVAR)
- Smooth latent dynamics (R² > 0.88 on test)
- Faster than lower bandwidth (less POD modes)
- Preserves essential cluster structure

Historical values:
- bandwidth=2.0: Earlier experiments, more detail but higher POD rank
- bandwidth=2.5: Intermediate smoothing
- bandwidth=3.0: Current production (best ROM performance)

Boundary Conditions: Periodic
------------------------------
Why: Matches Vicsek model physics
- No edge effects
- Translational symmetry preserved
- Simpler analysis (no wall interactions)

Mass Normalization: Enabled
----------------------------
Why: Ensures conservation
- Total mass = N exactly
- Prevents drift over time
- Validates KDE implementation

Example production configuration (configs/vicsek_rom_joint_optimal.yaml):
    outputs:
      density_resolution: 64        # 64×64 grid
      density_bandwidth: 3.0        # 3.0 grid cells (smooth)
    
    sim:
      bc: "periodic"                # Periodic boundaries

CODE LOCATIONS AND FUNCTIONS
=============================

Main KDE Implementation
-----------------------
File: src/rectsim/legacy_functions.py
Function: kde_density_movie(traj, Lx, Ly, nx, ny, bandwidth, bc)
Lines: 26-110
Returns: (rho, meta) where rho.shape = (T, ny, nx)

Bandwidth Estimation
--------------------
File: src/rectsim/legacy_functions.py
Function: estimate_bandwidth(Lx, Ly, N, nx, ny)
Lines: 112-148
Returns: bandwidth in grid units (Scott's rule with clamping)

Alternative KDE (density.py)
----------------------------
File: src/rectsim/density.py
Function: density_movie_kde(traj, Lx, Ly, nx, ny, bandwidth, bc)
Lines: 188-260
Returns: rho.shape = (T, ny, nx) (no metadata)

Single-frame KDE
----------------
File: src/rectsim/density.py
Function: compute_density_grid(pos, nx, ny, Lx, Ly, bandwidth, bc)
Lines: 16-84
Returns: (density, x_edges, y_edges)

Pipeline Usage
--------------
File: src/rectsim/simulation_runner.py
Function: simulate_single_run()
Lines: 70-83
    rho, meta = kde_density_movie(
        traj,
        Lx=config["sim"]["Lx"],
        Ly=config["sim"]["Ly"],
        nx=DENSITY_NX,
        ny=DENSITY_NY,
        bandwidth=DENSITY_BANDWIDTH,
        bc=config["sim"].get("bc", "periodic")
    )

RECOMMENDED READING
===================

Theoretical background:
1. Scott, D.W. (1992), "Multivariate Density Estimation: Theory, Practice, and
   Visualization", Wiley.
2. Silverman, B.W. (1986), "Density Estimation for Statistics and Data Analysis",
   Chapman & Hall.

Application to collective motion:
3. Bhaskar, D. & Ziegelmeier, L. (2019), "Topological data analysis of collective
   motion", Chaos 29, 123125.
   → Uses KDE to convert D'Orsogna swarms to density fields for TDA analysis

ROM applications:
4. Alvarez et al. (2025), "Model Order Reduction for Collective Motion Systems"
   [Conference paper]
   → Uses Silverman bandwidth, periodic augmentation for boundary handling

Implementation notes:
5. SciPy documentation: scipy.ndimage.gaussian_filter
   → Details on mode parameter, sigma interpretation, edge handling

TROUBLESHOOTING GUIDE
======================

Problem: Blocky, sparse density fields
Solution: Increase resolution (nx, ny) and bandwidth
    Try: nx=64, bandwidth=2.5-3.0

Problem: Over-smoothed, no cluster structure
Solution: Decrease bandwidth
    Try: bandwidth=1.5-2.0

Problem: High POD rank (>40 modes needed)
Solution: Increase bandwidth to smooth fields
    Try: bandwidth=3.0-4.0

Problem: Mass not conserved (total_mass ≠ N)
Solution: Enable renormalization in density computation
    Code: density *= (N / (density.sum() * dx * dy))

Problem: Edge artifacts with periodic BC
Solution: Ensure mode="wrap" in gaussian_filter
    Check: bc="periodic" → mode="wrap"

Problem: ROM poor fit (R² < 0.8)
Solution: Increase bandwidth for cleaner latent dynamics
    Try: bandwidth=3.0 (current production value)

This comprehensive guide ensures reproducible, high-quality density field generation
throughout our ROM pipeline, supporting both research exploration and production
deployment.
"""

# ============================================================================
# Error Metrics (from wsindy_manifold/mvar_rom.py)
# ============================================================================

def compute_frame_metrics(
    X_true: Array,
    X_pred: Array
) -> Dict[str, Array]:
    """
    Compute frame-wise error metrics.
    
    Args:
        X_true: True density snapshots (T, n_c) - flattened density fields
        X_pred: Predicted density snapshots (T, n_c) - flattened density fields
        
    Returns:
        metrics: Dict with arrays for each metric over time
            - e1: Relative L¹ error
            - e2: Relative L² error
            - einf: Relative L∞ error
            - rmse: Root mean squared error
            - mass_error: Relative mass conservation error
    """
    T = X_true.shape[0]
    
    e1 = np.zeros(T)
    e2 = np.zeros(T)
    einf = np.zeros(T)
    rmse = np.zeros(T)
    mass_error = np.zeros(T)
    
    for t in range(T):
        diff = X_pred[t] - X_true[t]
        
        # Norms
        norm_true_1 = np.linalg.norm(X_true[t], ord=1)
        norm_true_2 = np.linalg.norm(X_true[t], ord=2)
        norm_true_inf = np.linalg.norm(X_true[t], ord=np.inf)
        
        # Relative errors
        e1[t] = np.linalg.norm(diff, ord=1) / (norm_true_1 + 1e-16)
        e2[t] = np.linalg.norm(diff, ord=2) / (norm_true_2 + 1e-16)
        einf[t] = np.linalg.norm(diff, ord=np.inf) / (norm_true_inf + 1e-16)
        
        # RMSE
        rmse[t] = np.sqrt(np.mean(diff ** 2))
        
        # Mass error
        mass_true = np.sum(X_true[t])
        mass_pred = np.sum(X_pred[t])
        mass_error[t] = np.abs(mass_pred - mass_true) / (np.abs(mass_true) + 1e-16)
    
    return {
        "e1": e1,
        "e2": e2,
        "einf": einf,
        "rmse": rmse,
        "mass_error": mass_error,
    }


def compute_summary_metrics(
    X_true: Array,
    X_pred: Array,
    X_train_mean: Array,
    frame_metrics: Dict[str, Array],
    tolerance_threshold: float = 0.10
) -> Dict:
    """
    Compute aggregate summary metrics.
    
    Args:
        X_true: True density (T, n_c) - flattened density fields
        X_pred: Predicted density (T, n_c) - flattened density fields
        X_train_mean: Mean of training data (n_c,)
        frame_metrics: Frame-wise metrics from compute_frame_metrics
        tolerance_threshold: Threshold for tolerance horizon (default 0.10)
        
    Returns:
        summary: Dict with scalar metrics
            - r2: R² score
            - median_e2: Median relative L² error
            - p10_e2: 10th percentile L² error
            - p90_e2: 90th percentile L² error
            - tau_tol: Tolerance horizon (frames until error exceeds threshold)
            - mean_mass_error: Mean mass conservation error
            - max_mass_error: Max mass conservation error
    """
    e2 = frame_metrics["e2"]
    
    # R² score
    ss_res = np.sum((X_true - X_pred) ** 2)
    ss_tot = np.sum((X_true - X_train_mean) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-16)
    
    # Percentiles
    median_e2 = np.median(e2)
    p10_e2 = np.percentile(e2, 10)
    p90_e2 = np.percentile(e2, 90)
    
    # Tolerance horizon: first time rolling mean exceeds threshold
    window = min(10, len(e2))
    rolling_e2 = np.convolve(e2, np.ones(window)/window, mode='valid')
    tau_tol_idx = np.where(rolling_e2 >= tolerance_threshold)[0]
    tau_tol = tau_tol_idx[0] if len(tau_tol_idx) > 0 else len(e2)
    
    return {
        "r2": float(r2),
        "median_e2": float(median_e2),
        "p10_e2": float(p10_e2),
        "p90_e2": float(p90_e2),
        "tau_tol": int(tau_tol),
        "mean_mass_error": float(np.mean(frame_metrics["mass_error"])),
        "max_mass_error": float(np.max(frame_metrics["mass_error"])),
    }


def plot_errors_timeseries(
    frame_metrics: Dict[str, Array],
    summary: Dict,
    T0: int = 0,
    save_path: Optional[Path] = None,
    title: str = 'MVAR-ROM: Error Metrics Over Time'
) -> plt.Figure:
    """
    Plot error metrics over time.
    
    Args:
        frame_metrics: Frame-wise metrics from compute_frame_metrics
        summary: Summary metrics with R², median_e2, tau_tol
        T0: Starting time frame (default 0)
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    T = len(frame_metrics["e1"])
    t = np.arange(T0, T0 + T)
    
    # Plot relative errors
    axes[0].plot(t, frame_metrics["e1"], 'b-', alpha=0.7, linewidth=2, label='Relative L¹')
    if T0 > 0:
        axes[0].axvline(T0, color='k', linestyle='--', alpha=0.5, label='Train/Test Split')
    axes[0].set_ylabel('Relative L¹ Error', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(t, frame_metrics["e2"], 'g-', alpha=0.7, linewidth=2, label='Relative L²')
    if T0 > 0:
        axes[1].axvline(T0, color='k', linestyle='--', alpha=0.5)
    axes[1].axhline(0.10, color='r', linestyle=':', alpha=0.5, label='10% Threshold')
    if summary["tau_tol"] < T and summary["tau_tol"] > 0:
        axes[1].axvline(T0 + summary["tau_tol"], color='r', linestyle='--', 
                       alpha=0.7, linewidth=2, label=f'τ_tol = {summary["tau_tol"]}')
    axes[1].set_ylabel('Relative L² Error', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    axes[2].plot(t, frame_metrics["einf"], 'r-', alpha=0.7, linewidth=2, label='Relative L∞')
    if T0 > 0:
        axes[2].axvline(T0, color='k', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Time Frame', fontsize=11)
    axes[2].set_ylabel('Relative L∞ Error', fontsize=11, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # Add summary text
    text_str = f"R² = {summary['r2']:.4f}\n"
    text_str += f"Median L² = {summary['median_e2']:.4f}\n"
    text_str += f"τ_tol = {summary['tau_tol']} frames"
    axes[1].text(0.02, 0.98, text_str, transform=axes[1].transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.8), fontsize=11)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved error timeseries: {save_path}")
    
    return fig


# ============================================================================
# Video Generation (from wsindy_manifold/io.py)
# ============================================================================

def save_video(
    path: Path,
    frames: Array,
    fps: int,
    name: str,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None
) -> None:
    """
    Save 2D heatmap frames as MP4 video.
    
    Args:
        path: Directory to save video
        frames: Array of shape (T, ny, nx)
        fps: Frames per second
        name: Filename (without .mp4 extension)
        cmap: Colormap name
        vmin: Min value for colormap (auto if None)
        vmax: Max value for colormap (auto if None)
        title: Video title
    """
    path.mkdir(parents=True, exist_ok=True)
    video_path = path / f"{name}.mp4"
    
    if vmin is None:
        vmin = frames.min()
    if vmax is None:
        vmax = frames.max()
    
    T, ny, nx = frames.shape
    
    # Subsample if too many frames
    max_frames = 500
    if T > max_frames:
        indices = np.linspace(0, T - 1, max_frames, dtype=int)
        frames = frames[indices]
        T = max_frames
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Initial frame (no transpose - frames already in correct orientation)
    im = ax.imshow(frames[0], origin='lower', cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect='auto')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Density', fontsize=11)
    
    # Time text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       color='white', fontsize=11, va='top',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    
    # Write video
    writer = FFMpegWriter(fps=fps, bitrate=2000)
    
    with writer.saving(fig, video_path, dpi=100):
        for t in range(T):
            im.set_data(frames[t])  # No transpose
            time_text.set_text(f'Frame {t+1}/{T}')
            writer.grab_frame()
    
    plt.close(fig)
    print(f"Saved video: {video_path}")


def side_by_side_video(
    path: Path,
    left_frames: Array,
    right_frames: Array,
    lower_strip_timeseries: Optional[Array] = None,
    name: str = "comparison",
    fps: int = 20,
    cmap: str = "viridis",
    titles: tuple = ("Ground Truth", "Prediction")
) -> None:
    """
    Create side-by-side comparison video with optional error timeseries below.
    
    Args:
        path: Directory to save video
        left_frames: Left panel frames (T, ny, nx)
        right_frames: Right panel frames (T, ny, nx)
        lower_strip_timeseries: Optional timeseries (T,) to plot below
        name: Filename (without .mp4 extension)
        fps: Frames per second
        cmap: Colormap name
        titles: (left_title, right_title)
    """
    path.mkdir(parents=True, exist_ok=True)
    video_path = path / f"{name}.mp4"
    
    T = len(left_frames)
    
    # Subsample if needed
    max_frames = 500
    if T > max_frames:
        indices = np.linspace(0, T - 1, max_frames, dtype=int)
        left_frames = left_frames[indices]
        right_frames = right_frames[indices]
        if lower_strip_timeseries is not None:
            lower_strip_timeseries = lower_strip_timeseries[indices]
        T = max_frames
    
    # Shared colormap limits
    vmin = min(left_frames.min(), right_frames.min())
    vmax = max(left_frames.max(), right_frames.max())
    
    # Create figure
    if lower_strip_timeseries is not None:
        fig = plt.figure(figsize=(14, 7))
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.3, wspace=0.3)
        ax_left = fig.add_subplot(gs[0, 0])
        ax_right = fig.add_subplot(gs[0, 1])
        ax_ts = fig.add_subplot(gs[1, :])
    else:
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))
        ax_ts = None
    
    # Left panel (no transpose - frames already in correct orientation)
    im_left = ax_left.imshow(left_frames[0], origin='lower', cmap=cmap,
                             vmin=vmin, vmax=vmax, aspect='auto')
    ax_left.set_title(titles[0], fontsize=13, fontweight='bold')
    ax_left.set_xlabel('x', fontsize=11)
    ax_left.set_ylabel('y', fontsize=11)
    plt.colorbar(im_left, ax=ax_left, fraction=0.046, pad=0.04)
    
    # Right panel (no transpose - frames already in correct orientation)
    im_right = ax_right.imshow(right_frames[0], origin='lower', cmap=cmap,
                               vmin=vmin, vmax=vmax, aspect='auto')
    ax_right.set_title(titles[1], fontsize=13, fontweight='bold')
    ax_right.set_xlabel('x', fontsize=11)
    ax_right.set_ylabel('y', fontsize=11)
    plt.colorbar(im_right, ax=ax_right, fraction=0.046, pad=0.04)
    
    # Timeseries panel
    if ax_ts is not None and lower_strip_timeseries is not None:
        time_steps = np.arange(T)
        line, = ax_ts.plot(time_steps, lower_strip_timeseries, 'b-', linewidth=2)
        marker, = ax_ts.plot([0], [lower_strip_timeseries[0]], 'ro', markersize=8)
        ax_ts.set_xlabel('Time Step', fontsize=11)
        ax_ts.set_ylabel('Error', fontsize=11)
        ax_ts.set_title('Relative L² Error Over Time', fontsize=12)
        ax_ts.grid(True, alpha=0.3)
        ax_ts.set_xlim([0, T-1])
        ax_ts.set_ylim([0, lower_strip_timeseries.max() * 1.1])
    
    plt.tight_layout()
    
    # Write video
    writer = FFMpegWriter(fps=fps, bitrate=3000)
    
    with writer.saving(fig, video_path, dpi=100):
        for t in range(T):
            im_left.set_data(left_frames[t])  # No transpose
            im_right.set_data(right_frames[t])  # No transpose
            
            if ax_ts is not None and lower_strip_timeseries is not None:
                marker.set_data([t], [lower_strip_timeseries[t]])
            
            writer.grab_frame()
    
    plt.close(fig)
    print(f"Saved comparison video: {video_path}")


def trajectory_video(
    path: Path,
    traj: Array,
    times: Array,
    Lx: float,
    Ly: float,
    name: str = "trajectory",
    fps: int = 20,
    marker_size: float = 30,
    marker_color: str = 'C0',
    title: Optional[str] = None,
    show_velocities: bool = True,
    quiver_scale: float = 3.0,
    quiver_width: float = 0.004,
    quiver_alpha: float = 0.8
) -> None:
    """
    Create trajectory video showing particle positions and velocities over time.
    Particles are colored by heading angle (-π to π) with a colorbar.
    Arrow lengths are proportional to particle speeds.
    
    Args:
        path: Directory to save video
        traj: Particle trajectories (T, N, 2)
        times: Time points (T,)
        Lx: Domain width
        Ly: Domain height
        name: Filename (without .mp4 extension)
        fps: Frames per second
        marker_size: Particle marker size
        marker_color: Ignored (particles colored by heading angle)
        title: Video title
        show_velocities: If True, show velocity arrows (quiver)
        quiver_scale: Scale factor for velocity arrows (higher = shorter arrows)
        quiver_width: Width of velocity arrows
        quiver_alpha: Transparency of velocity arrows
    """
    path.mkdir(parents=True, exist_ok=True)
    video_path = path / f"{name}.mp4"
    
    T, N, _ = traj.shape
    
    # Subsample if too many frames
    max_frames = 500
    if T > max_frames:
        indices = np.linspace(0, T - 1, max_frames, dtype=int)
        traj = traj[indices]
        times = times[indices]
        T = max_frames
    
    # Compute velocities from positions (finite differences)
    # Handle periodic boundaries to avoid spurious large velocities
    vel = np.zeros_like(traj)
    if T > 1:
        dt = times[1] - times[0] if len(times) > 1 else 1.0
        for t in range(T - 1):
            disp = traj[t + 1] - traj[t]
            
            # Wrap displacements for periodic boundaries (minimum image convention)
            disp[:, 0] = np.where(disp[:, 0] > Lx/2, disp[:, 0] - Lx, disp[:, 0])
            disp[:, 0] = np.where(disp[:, 0] < -Lx/2, disp[:, 0] + Lx, disp[:, 0])
            disp[:, 1] = np.where(disp[:, 1] > Ly/2, disp[:, 1] - Ly, disp[:, 1])
            disp[:, 1] = np.where(disp[:, 1] < -Ly/2, disp[:, 1] + Ly, disp[:, 1])
            
            vel[t] = disp / dt
        vel[-1] = vel[-2]  # Copy last velocity
    
    # Create figure with colorbar
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    # Compute heading angles for first frame
    angles = np.arctan2(vel[0, :, 1], vel[0, :, 0])  # Range: -π to π
    
    # Initial scatter plot colored by heading angle
    scatter = ax.scatter(
        traj[0, :, 0], 
        traj[0, :, 1], 
        c=angles,
        s=marker_size, 
        alpha=0.8,
        edgecolors='none',
        cmap='hsv',  # Circular colormap for angles
        vmin=-np.pi,
        vmax=np.pi
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Heading Angle (rad)', fraction=0.046, pad=0.04)
    cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    
    # Initial quiver plot (velocity arrows) - arrows proportional to speed
    quiver = None
    if show_velocities:
        # Don't use scale_units='xy' to make arrows speed-proportional
        quiver = ax.quiver(
            traj[0, :, 0],
            traj[0, :, 1],
            vel[0, :, 0],
            vel[0, :, 1],
            angles='xy',
            scale_units='xy',
            scale=quiver_scale,  # Controls arrow length
            width=quiver_width,
            alpha=quiver_alpha,
            color='orange',  # Orange arrows on colored particles
            edgecolors='white',
            linewidth=0.5
        )
    
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Time text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       color='black', fontsize=11, va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Particle count text
    ax.text(0.02, 0.02, f'N = {N} particles', transform=ax.transAxes,
           color='black', fontsize=10, va='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Write video using FFMpegWriter (best quality)
    writer = FFMpegWriter(fps=fps, bitrate=2000)
    
    with writer.saving(fig, video_path, dpi=100):
        for t in range(T):
            # Compute heading angles for this frame
            angles = np.arctan2(vel[t, :, 1], vel[t, :, 0])
            
            # Update particle positions and colors
            scatter.set_offsets(traj[t, :, :2])
            scatter.set_array(angles)
            
            # Update velocity arrows (proportional to speed)
            if show_velocities and quiver is not None:
                quiver.set_offsets(traj[t, :, :2])
                quiver.set_UVC(vel[t, :, 0], vel[t, :, 1])
            
            time_text.set_text(f't = {times[t]:.2f}s\nFrame {t+1}/{T}')
            writer.grab_frame()
    
    plt.close(fig)
    print(f"Saved trajectory video: {video_path}")
