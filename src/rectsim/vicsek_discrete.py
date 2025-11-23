"""Discrete-time Vicsek model backend with unified interface.

This module implements the discrete Vicsek self-propelled particle model, following:
Vicsek et al., "Novel Type of Phase Transition in a System of Self-Driven Particles"
Physical Review Letters 75, 1226 (1995)

This implementation uses the unified backend interface for consistency with other
model types (e.g., continuous/RK models)

The model consists of particles moving at constant speed and updating their headings
based on local alignment with neighbors. Key features:

- Pure discrete-time updates (no ODE integration)
- Fixed-speed particles with heading-only dynamics 
- Two types of noise:
    * Gaussian: Normally distributed angular perturbations with std σ
    * Uniform: Random angles drawn from [-η/2, η/2]
- Efficient neighbor search using cell lists
- Support for periodic and reflecting boundaries
- Optional neighbor rebuild frequency for performance

Example
-------
A basic simulation with 100 particles and Gaussian noise:

    cfg = {
        "seed": 42,
        "N": 100,           # number of particles
        "Lx": 10.0,        # domain size x
        "Ly": 10.0,        # domain size y 
        "bc": "periodic",  # boundary condition
        "T": 100.0,       # total time
        "dt": 1.0,        # time step
        "v0": 1.0,        # particle speed
        "R": 1.0,         # interaction radius
        "noise": {
            "kind": "gaussian",
            "sigma": 0.1    # noise strength
        },
        "save_every": 10,      # save interval
        "neighbor_rebuild": 1,  # neighbor list rebuild interval
    }
    result = simulate_vicsek(cfg)

The simulation returns trajectories, headings, and the order parameter evolution
that can be used to analyze the collective behavior.

References
----------
[1] Vicsek et al. PRE 75, 1226 (1995) - Original model
[2] Chaté et al. PRE 77, 046113 (2008) - Phase diagram analysis
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .domain import CellList, apply_bc, build_cells, neighbor_indices_from_celllist

ArrayLike = np.ndarray


def rotation(phi: float) -> np.ndarray:
    """Return a 2x2 rotation matrix for angle ``phi`` (radians)."""

    c = float(np.cos(phi))
    s = float(np.sin(phi))
    return np.array([[c, -s], [s, c]], dtype=float)


def headings_from_angles(theta: np.ndarray) -> np.ndarray:
    """Convert angles ``theta`` with shape ``(N,)`` to unit headings ``(N, 2)``."""

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    headings = np.column_stack((cos_theta, sin_theta))
    return headings.astype(float, copy=False)


def angles_from_headings(p: np.ndarray) -> np.ndarray:
    """Return polar angles (radians) for heading vectors with last axis 2."""

    return np.arctan2(p[..., 1], p[..., 0])


def compute_neighbors(
    x: np.ndarray,
    Lx: float,
    Ly: float,
    R: float,
    bc: str,
    cell_list: CellList | None = None,
) -> Tuple[List[np.ndarray], CellList | None]:
    """Return neighbour indices within radius ``R`` using linked-cell search."""

    if R <= 0.0:
        return [np.array([i], dtype=int) for i in range(x.shape[0])], None

    local_cells = cell_list if cell_list is not None else build_cells(x, Lx, Ly, R, bc)
    neighbours = neighbor_indices_from_celllist(x, local_cells, Lx, Ly, R, bc)
    enhanced: List[np.ndarray] = []
    for i in range(x.shape[0]):
        if neighbours[i].size:
            unique = np.union1d(neighbours[i], np.array([i], dtype=int))
            enhanced.append(unique.astype(int, copy=False))
        else:
            enhanced.append(np.array([i], dtype=int))
    return enhanced, local_cells


def _apply_noise(
    rng: np.random.Generator,
    noise_kind: str,
    sigma: float,
    eta: float,
) -> float:
    """Sample an angular perturbation according to the requested noise type.
    
    Parameters
    ----------
    rng : np.random.Generator
        Random number generator
    noise_kind : str
        Either "gaussian" or "uniform"
    sigma : float
        Standard deviation for Gaussian noise
    eta : float
        Range parameter for uniform noise in [0, π]. The actual noise is uniform in [-η/2, η/2].
        For Gaussian noise with equivalent variance, use sigma = eta/sqrt(12).
    """
    if noise_kind == "gaussian":
        # For equivalent variance to uniform[-η/2, η/2], use η/√12
        return float(rng.normal(loc=0.0, scale=sigma))
    if noise_kind == "uniform":
        # Original Vicsek model uses uniform noise in [-η/2, η/2] where η ∈ [0,π]
        half_eta = 0.5 * eta  # η/2
        return float(rng.uniform(-half_eta, half_eta))
    raise ValueError(f"Unknown noise kind '{noise_kind}'")


def step_vicsek_discrete(
    x: np.ndarray,
    p: np.ndarray,
    v0: float,
    dt: float,
    Lx: float,
    Ly: float,
    R: float,
    noise_kind: str,
    sigma: float,
    eta: float,
    bc: str,
    rng: np.random.Generator,
    cell_list: CellList | None = None,
) -> Tuple[np.ndarray, np.ndarray, CellList | None]:
    """Advance the discrete-time Vicsek dynamics by one step."""

    neighbours, local_cells = compute_neighbors(x, Lx, Ly, R, bc, cell_list=cell_list)

    p_new = np.empty_like(p)
    for i in range(p.shape[0]):
        idx = neighbours[i]
        if idx.size:  # Has neighbors (including self)
            mean_vec = np.sum(p[idx], axis=0)
            norm = float(np.linalg.norm(mean_vec))
            if norm > 1e-12:
                pbar = mean_vec / norm
            else:  # Opposing headings cancel out
                pbar = p[i]  # Keep current heading
        else:  # No neighbors (shouldn't happen with self-inclusion)
            pbar = p[i]  # Keep current heading
        phi = _apply_noise(rng, noise_kind, sigma, eta)
        p_new[i] = rotation(phi) @ pbar

    norms = np.linalg.norm(p_new, axis=1, keepdims=True)
    small = norms.squeeze(axis=1) < 1e-12
    if np.any(small):
        p_new[small] = p[small]
        norms = np.linalg.norm(p_new, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        p_new = np.divide(p_new, norms, out=p_new, where=norms > 0.0)

    x_new = x + v0 * dt * p_new
    x_new = x_new.astype(float, copy=False)
    x_new, flips = apply_bc(x_new, Lx, Ly, bc)
    if np.any(flips):
        p_new = p_new.copy()
        p_new[flips] *= -1.0

    return x_new, p_new, local_cells


def simulate_vicsek(cfg: dict) -> dict:
    """Simulate the discrete-time Vicsek model controlled by ``cfg``."""

    required_keys = [
        "seed",
        "N",
        "Lx",
        "Ly",
        "bc",
        "T",
        "dt",
        "v0",
        "R",
        "noise",
        "save_every",
        "neighbor_rebuild",
    ]
    for key in required_keys:
        if key not in cfg:
            raise KeyError(f"Vicsek config missing required key '{key}'")

    noise_cfg = cfg.get("noise", {})
    noise_kind = str(noise_cfg.get("kind", "gaussian")).lower()
    sigma = float(noise_cfg.get("sigma", 0.0))
    eta = float(noise_cfg.get("eta", 0.0))

    rng = np.random.default_rng(int(cfg.get("seed", 0)))

    N = int(cfg["N"])
    Lx = float(cfg["Lx"])
    Ly = float(cfg["Ly"])
    bc = str(cfg["bc"])
    T = float(cfg["T"])
    dt = float(cfg["dt"])
    v0 = float(cfg["v0"])
    R = float(cfg["R"])
    save_every = int(cfg.get("save_every", 1))
    neighbor_rebuild = int(cfg.get("neighbor_rebuild", 1))

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if T <= 0.0:
        raise ValueError("T must be positive")
    if save_every <= 0:
        raise ValueError("save_every must be positive")
    if neighbor_rebuild <= 0:
        raise ValueError("neighbor_rebuild must be positive")
    
    # Enforce v0*dt ≲ 0.5*R to prevent agents from jumping over their neighborhood
    if v0 * dt > 0.5 * R:
        raise ValueError(
            f"Time step too large: v0*dt = {v0*dt:.3f} > 0.5*R = {0.5*R:.3f}. "
            "Reduce dt to prevent agents from jumping over their metric neighborhood."
        )

    total_steps = int(np.round(T / dt))
    if total_steps <= 0:
        raise ValueError("Number of steps must be positive")

    x = rng.uniform(low=[0.0, 0.0], high=[Lx, Ly], size=(N, 2))
    theta0 = rng.uniform(0.0, 2.0 * np.pi, size=N)
    p = headings_from_angles(theta0)

    frames_x = [x.copy()]
    frames_p = [p.copy()]
    times = [0.0]
    psi_vals = [float(np.linalg.norm(np.mean(p, axis=0)))]

    cell_list: CellList | None = None
    if R > 0.0:
        cell_list = build_cells(x, Lx, Ly, R, bc)

    for step in range(1, total_steps + 1):
        if R > 0.0:
            if cell_list is None or (step - 1) % neighbor_rebuild == 0:
                cell_list = build_cells(x, Lx, Ly, R, bc)

        x, p, cell_list = step_vicsek_discrete(
            x,
            p,
            v0,
            dt,
            Lx,
            Ly,
            R,
            noise_kind,
            sigma,
            eta,
            bc,
            rng,
            cell_list=cell_list,
        )
        current_time = step * dt

        if step % save_every == 0 or step == total_steps:
            frames_x.append(x.copy())
            frames_p.append(p.copy())
            times.append(current_time)
            psi_vals.append(float(np.linalg.norm(np.mean(p, axis=0))))

    traj = np.stack(frames_x, axis=0)
    headings = np.stack(frames_p, axis=0)
    times_array = np.array(times, dtype=float)
    psi_array = np.array(psi_vals, dtype=float)

    result = {
        "traj": traj,
        "headings": headings,
        "vel": v0 * headings,
        "times": times_array,
        "psi": psi_array,
        "config": cfg,
        "v0": v0,
        "R": R,
        "noise": {
            "kind": noise_kind,
            "sigma": sigma,
            "eta": eta,
        },
    }
    return result


def simulate_backend(config: dict, rng: np.random.Generator) -> dict:
    """Unified backend interface for discrete Vicsek model.
    
    This function implements the standardized backend interface that produces
    consistent outputs across all model types (discrete VM, continuous/RK, etc.).
    
    Parameters
    ----------
    config : dict
        Configuration dictionary with keys:
        - sim: {N, Lx, Ly, bc, T, dt, save_every, neighbor_rebuild}
        - model: {type, speed, speed_mode}
          - speed_mode: "constant" (default), "constant_with_forces", or "variable"
            * constant: Traditional Vicsek, fixed speed v0, forces ignored
            * constant_with_forces: Forces affect heading only, speed stays v0
            * variable: Forces affect velocity, speed can change
        - noise: {kind, eta, match_variance}
        - forces: {enabled, type, params}
        - params: {R} (alignment radius)
    rng : np.random.Generator
        Random number generator instance.
        
    Returns
    -------
    result : dict
        Standardized result dictionary with keys:
        - times: (T,) array of time points
        - traj: (T, N, 2) array of positions
        - vel: (T, N, 2) array of velocities
        - head: (T, N, 2) array of unit headings  
        - meta: dict of configuration parameters
        
    Notes
    -----
    The discrete Vicsek model supports three speed modes:
    
    CONSTANT SPEED MODE (speed_mode="constant", default):
    1. Find neighbors within radius R
    2. Compute mean heading of neighbors
    3. Add angular noise
    4. Update positions: x += dt * v0 * p
    5. Forces are completely ignored, pure Vicsek alignment
    
    CONSTANT WITH FORCES MODE (speed_mode="constant_with_forces"):
    1. Find neighbors within radius R
    2. Compute mean heading of neighbors + force heading
    3. Add angular noise
    4. Update positions: x += dt * v0 * p
    5. Forces steer heading but speed stays constant at v0
    
    VARIABLE SPEED MODE (speed_mode="variable"):
    1. Find neighbors within radius R
    2. Update velocities from forces: v += dt * mu_t * F
    3. Compute mean heading of neighbors
    4. Rotate velocities toward aligned direction (preserving speed)
    5. Update positions: x += dt * v
    6. Forces directly affect velocity magnitude and direction
    """
    from .noise import angle_noise
    from .domain import NeighborFinder, apply_bc
    
    # Extract parameters
    sim = config["sim"]
    model = config.get("model", {})
    noise_cfg = config.get("noise", {})
    forces_cfg = config.get("forces", {"enabled": False})
    params = config.get("params", {})
    
    N = sim["N"]
    Lx = sim["Lx"]
    Ly = sim["Ly"]
    bc = sim["bc"]
    T = sim["T"]
    dt = sim["dt"]
    save_every = sim.get("save_every", 1)
    neighbor_rebuild = sim.get("neighbor_rebuild", 1)
    
    v0 = model.get("speed", 0.5)
    R = params.get("R", 1.0)
    speed_mode = model.get("speed_mode", "constant")  # "constant" or "variable"
    integrator = sim.get("integrator", "euler")  # "euler" or "euler_semiimplicit"
    
    noise_kind = noise_cfg.get("kind", "uniform")
    eta = noise_cfg.get("eta", 0.5)
    match_variance = noise_cfg.get("match_variance", True)
    
    # Determine maximum interaction radius for optimal cell list
    # This ensures a single cell list can serve both alignment and forces
    R_align = R
    R_max = R_align
    
    if forces_cfg.get("enabled", False):
        force_params = forces_cfg.get("params", {})
        lr = force_params.get("lr", 0.5)
        la = force_params.get("la", 1.5)
        rcut_factor = force_params.get("rcut_factor", 5.0)
        R_force = rcut_factor * max(lr, la)
        R_max = max(R_align, R_force)
    
    # Validation
    if v0 * dt > 0.5 * R:
        raise ValueError(
            f"Time step too large: v0*dt = {v0*dt:.3f} > 0.5*R = {0.5*R:.3f}. "
            "Reduce dt to prevent particles from jumping over their neighborhood."
        )
    
    steps = int(np.round(T / dt))
    if steps <= 0:
        raise ValueError("Number of steps must be positive")
    
    # Initialize positions
    initial_dist = config.get("initial_distribution", "uniform")
    if initial_dist == "uniform":
        x = rng.uniform(low=[0.0, 0.0], high=[Lx, Ly], size=(N, 2))
    else:
        # Use initial_conditions module for other distributions
        from .initial_conditions import initialize_positions
        x = initialize_positions(initial_dist, N, Lx, Ly, rng)
    
    # Initialize headings (velocities)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=N)
    p = np.column_stack([np.cos(theta), np.sin(theta)])  # unit headings
    
    # Initialize velocities based on speed mode
    if speed_mode == "variable":
        v = v0 * p.copy()  # Initial velocity at natural speed
    else:
        v = None  # Not used in constant-speed mode
    
    # Setup neighbor finder with maximum interaction radius
    # Cell size will be >= R_max, allowing efficient 3x3 search for both alignment and forces
    nf = NeighborFinder(Lx, Ly, R_max, bc)
    nf.rebuild(x)
    
    # Preallocate output arrays
    n_frames = steps // save_every + 1
    traj = np.empty((n_frames, N, 2))
    head = np.empty((n_frames, N, 2))
    vel = np.empty((n_frames, N, 2))
    times = np.empty(n_frames)
    
    # Save initial state
    frame_idx = 0
    traj[frame_idx] = x
    head[frame_idx] = p
    if speed_mode == "variable":
        vel[frame_idx] = v
    else:
        vel[frame_idx] = v0 * p
    times[frame_idx] = 0.0
    
    # Main simulation loop
    import sys
    print(f"  Progress: 0% (step 0/{steps})", flush=True)
    for step in range(1, steps + 1):
        # Progress indicator every 10%
        if step % max(1, steps // 10) == 0 or step == steps:
            pct = int(100 * step / steps)
            print(f"\r  Progress: {pct}% (step {step}/{steps})", end='', flush=True)
            sys.stdout.flush()  # Force flush
        
        # Rebuild neighbor list periodically
        if step % neighbor_rebuild == 0:
            nf.rebuild(x)
        
        # Get neighbors and compute mean heading
        neighbors = nf.neighbors_of(x)
        p_bar = np.empty_like(p)
        
        for i in range(N):
            idx = neighbors[i]
            if idx.size == 0:  # No neighbors (shouldn't happen with self-inclusion)
                p_bar[i] = p[i]
            else:
                # Filter neighbors within alignment radius R_align
                if R_align < R_max:
                    # Need to filter by distance
                    dists = np.linalg.norm(x[idx] - x[i], axis=1)
                    within_R = idx[dists <= R_align]
                    if within_R.size == 0:
                        p_bar[i] = p[i]
                    else:
                        s = p[within_R].sum(axis=0)
                        norm = np.linalg.norm(s)
                        if norm > 1e-12:
                            p_bar[i] = s / norm
                        else:
                            p_bar[i] = p[i]
                else:
                    # All neighbors from cell list are within R_align
                    s = p[idx].sum(axis=0)
                    norm = np.linalg.norm(s)
                    if norm > 1e-12:
                        p_bar[i] = s / norm
                    else:  # Opposing headings cancel
                        p_bar[i] = p[i]
        
        # Force hook: Add Morse forces if enabled
        if forces_cfg.get("enabled", False):
            from .morse import morse_force
            
            # Get force parameters
            force_params = forces_cfg.get("params", {})
            Cr = force_params.get("Cr", 2.0)
            Ca = force_params.get("Ca", 1.0)
            lr = force_params.get("lr", 0.5)
            la = force_params.get("la", 1.5)
            rcut_factor = force_params.get("rcut_factor", 5.0)
            rcut = rcut_factor * max(lr, la)
            mu_t = force_params.get("mu_t", 0.5)  # Translational mobility
            
            # Compute Morse forces using the shared cell list
            # Pass the cell list from NeighborFinder to avoid rebuilding
            fx, fy = morse_force(x, Lx, Ly, bc, Cr, Ca, lr, la, rcut, cell_list=nf._cell_list)
            F = np.column_stack([fx, fy])
        else:
            F = np.zeros_like(p)  # No forces
            mu_t = 0.0
        
        # ========== Update dynamics based on speed mode ==========
        
        if speed_mode == "constant":
            # CONSTANT SPEED MODE (traditional Vicsek)
            # Particles move at EXACTLY constant speed v0
            # Forces are IGNORED to maintain constant speed
            
            # Add angular noise and rotate headings
            phi = angle_noise(rng, noise_kind, eta, size=N, match_variance=match_variance)
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            
            # Rotation: p_new = R(phi) @ p_bar
            p = np.column_stack([
                cos_phi * p_bar[:, 0] - sin_phi * p_bar[:, 1],
                sin_phi * p_bar[:, 0] + cos_phi * p_bar[:, 1]
            ])
            
            # Update positions: x_{t+1} = x_t + dt * v0 * p
            # NOTE: Forces are ignored in constant speed mode to maintain |velocity| = v0
            x = x + dt * v0 * p
            x, flips = apply_bc(x, Lx, Ly, bc)
            
            # Flip headings on reflection
            if np.any(flips):
                p[flips] *= -1.0
            
            # Velocity for output: In constant speed mode, all particles move at exactly v0
            v_output = v0 * p  # Constant magnitude v0
            
        elif speed_mode == "constant_with_forces":
            # CONSTANT SPEED WITH FORCES MODE
            # Forces affect heading direction but speed stays constant at v0
            
            # Combine alignment heading with force-induced heading
            if forces_cfg.get("enabled", False) and np.any(F):
                # Normalize force to get force-induced heading
                F_norm = np.linalg.norm(F, axis=1, keepdims=True)
                F_heading = np.zeros_like(F)
                nonzero = F_norm[:, 0] > 1e-12
                F_heading[nonzero] = F[nonzero] / F_norm[nonzero]
                
                # Combine alignment and force headings (weighted average)
                # mu_t controls the influence of forces
                combined_heading = p_bar + mu_t * F_heading
                
                # Normalize to get unit heading
                combined_norm = np.linalg.norm(combined_heading, axis=1, keepdims=True)
                p_bar = np.zeros_like(combined_heading)
                nonzero = combined_norm[:, 0] > 1e-12
                p_bar[nonzero] = combined_heading[nonzero] / combined_norm[nonzero]
            
            # Add angular noise and rotate headings
            phi = angle_noise(rng, noise_kind, eta, size=N, match_variance=match_variance)
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            
            # Rotation: p_new = R(phi) @ p_bar
            p = np.column_stack([
                cos_phi * p_bar[:, 0] - sin_phi * p_bar[:, 1],
                sin_phi * p_bar[:, 0] + cos_phi * p_bar[:, 1]
            ])
            
            # Update positions: x_{t+1} = x_t + dt * v0 * p
            # Forces affect DIRECTION but not SPEED (speed = v0 always)
            x = x + dt * v0 * p
            x, flips = apply_bc(x, Lx, Ly, bc)
            
            # Flip headings on reflection
            if np.any(flips):
                p[flips] *= -1.0
            
            # Velocity for output: Constant speed v0 in heading direction
            v_output = v0 * p
            
        else:
            # VARIABLE SPEED MODE (like D'Orsogna but discrete)
            # Forces directly affect velocity, alignment affects heading
            
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
                
            else:
                # EXPLICIT EULER: Standard forward integration
                # Update velocities based on forces: dv/dt = mu_t * F
                v = v + dt * mu_t * F
                
                # Compute current headings from velocities
                speed = np.linalg.norm(v, axis=1, keepdims=True)
                speed = np.maximum(speed, 1e-12)  # Avoid division by zero
                p = v / speed
                
                # Alignment: rotate velocities toward mean heading
                phi = angle_noise(rng, noise_kind, eta, size=N, match_variance=match_variance)
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                
                # Rotation: p_bar_rotated = R(phi) @ p_bar
                p_aligned = np.column_stack([
                    cos_phi * p_bar[:, 0] - sin_phi * p_bar[:, 1],
                    sin_phi * p_bar[:, 0] + cos_phi * p_bar[:, 1]
                ])
                
                # Rotate velocity toward aligned direction (keep magnitude)
                v = speed.ravel()[:, np.newaxis] * p_aligned
                
                # Update positions: x_{t+1} = x_t + dt * v
                x = x + dt * v
                x, flips = apply_bc(x, Lx, Ly, bc)
            
            # Flip velocities on reflection
            if np.any(flips):
                v[flips] *= -1.0
            
            # Update headings from velocity
            speed = np.linalg.norm(v, axis=1, keepdims=True)
            speed = np.maximum(speed, 1e-12)
            p = v / speed
            
            v_output = v
        
        # Save frame if needed
        if step % save_every == 0:
            frame_idx = step // save_every
            traj[frame_idx] = x
            head[frame_idx] = p
            vel[frame_idx] = v_output
            times[frame_idx] = step * dt
    
    print()  # Newline after progress indicator
    
    # Return unified result format
    return {
        "times": times.astype(np.float32),
        "traj": traj.astype(np.float32),
        "vel": vel.astype(np.float32),
        "head": head.astype(np.float32),
        "meta": config,
    }


__all__ = [
    "rotation",
    "headings_from_angles",
    "angles_from_headings",
    "compute_neighbors",
    "step_vicsek_discrete",
    "simulate_vicsek",
    "simulate_backend",
]
