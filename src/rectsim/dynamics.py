"""Time evolution for the rectangular collective motion simulations.

This module contains the main simulator that integrates the particles'
positions and velocities. Responsibilities:
- Initialize particle positions and velocities from a random seed.
- Use a chosen integrator (RK4 or semi-implicit Euler) to step dynamics
    arising from social forces (Morse potential) and damping.
- Optionally apply Vicsek-style alignment and rotational noise to headings.
- Record frames (positions and velocities) at configured intervals and
    return a results dict consumed by plotting and I/O utilities.

The simulator separates numerical integration (``integrators``) from
side-effects (plotting & file I/O) to keep the numerics testable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm

from .domain import apply_bc, pair_displacements, neighbor_indices_from_celllist
from .integrators import State, step_euler_semiimplicit, step_rk4
from .morse import CellList, build_cells, morse_force

ArrayLike = np.ndarray


@dataclass
class ForceCalculator:
    """Cached Morse force evaluations with neighbor-list reuse."""

    Cr: float
    Ca: float
    lr: float
    la: float
    Lx: float
    Ly: float
    bc: str
    rcut: float
    cell_list: CellList
    total_time: float = 0.0
    calls: int = 0

    def rebuild(self, positions: ArrayLike) -> None:
        """Rebuild the neighbor cell list for the provided particle positions."""

        start = time.perf_counter()
        self.cell_list = build_cells(positions, self.Lx, self.Ly, self.rcut, self.bc)
        self.total_time += time.perf_counter() - start

    def __call__(self, positions: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """Evaluate Morse forces using cached neighbor information."""

        start = time.perf_counter()
        fx, fy = morse_force(
            positions,
            self.Lx,
            self.Ly,
            self.bc,
            self.Cr,
            self.Ca,
            self.lr,
            self.la,
            self.rcut,
            cell_list=self.cell_list,
        )
        self.total_time += time.perf_counter() - start
        self.calls += 1
        return fx, fy


def _neighbor_finder_internal(
    x: ArrayLike,
    Lx: float,
    Ly: float,
    bc: str,
    rcut: float,
) -> list[np.ndarray]:
    """Internal metric-ball neighbour search used by the alignment step."""

    _, _, rij, _ = pair_displacements(x, Lx, Ly, bc)
    idx_all = np.arange(x.shape[0])
    result: list[np.ndarray] = []
    for i in range(x.shape[0]):
        mask = (rij[i] <= rcut) & (idx_all != i)
        result.append(idx_all[mask])
    return result


def apply_alignment_step(
    x: ArrayLike,
    p: ArrayLike,
    Lx: float,
    Ly: float,
    bc: str,
    mu_r: float,
    lV: float,
    Dtheta: float,
    dt: float,
    neighbor_finder=None,
    rng=None,
) -> ArrayLike:
    """Vicsek-style alignment update producing unit headings.

    Parameters
    ----------
    x : ndarray, shape (N, 2)
        Particle positions.
    p : ndarray, shape (N, 2)
        Current unit heading vectors.
    Lx, Ly : float
        Domain lengths.
    bc : {"periodic", "reflecting"}
        Boundary condition name.
    mu_r : float
        Rotational mobility (alignment rate) ``μ_r``.
    lV : float
        Alignment radius ``l_V``.
    Dtheta : float
        Rotational diffusion coefficient ``D_θ``.
    dt : float
        Time step for Euler–Maruyama integration.
    neighbor_finder : callable, optional
        Function returning neighbour indices for each particle. Defaults to an
        :math:`O(N^2)` metric-ball search suitable for moderate ``N``.
    rng : numpy.random.Generator, optional
        Random generator used for stochastic forcing. If ``None`` a fresh
        generator is created.

    Returns
    -------
    ndarray, shape (N, 2)
        Updated, unit-normalised heading vectors.
    """

    # Backwards compatibility: if caller passed an RNG as the positional
    # argument (older signature), shift it into rng and clear neighbor_finder.
    if isinstance(neighbor_finder, np.random.Generator):
        rng = neighbor_finder
        neighbor_finder = None

    # Backwards compatibility: allow callers using older signature to pass
    # rng as the next positional argument.
    if isinstance(neighbor_finder, np.random.Generator):
        rng = neighbor_finder
        neighbor_finder = None

    if rng is None:
        rng = np.random.default_rng()
    if neighbor_finder is None:
        neighbor_finder = _neighbor_finder_internal

    neighbours = neighbor_finder(x, Lx, Ly, bc, lV)
    noise_scale = np.sqrt(max(0.0, 2.0 * Dtheta * dt))
    p_new = np.empty_like(p)
    for i in range(p.shape[0]):
        idx = neighbours[i]
        drift_vec = np.zeros(2, dtype=float)
        if mu_r > 0.0 and len(idx) > 0:
            mean_vec = np.sum(p[idx], axis=0)
            norm = float(np.linalg.norm(mean_vec))
            if norm > 1e-12:
                drift_vec = mu_r * (mean_vec / norm - p[i])
        noise = noise_scale * rng.normal(size=2)
        p_tmp = p[i] + drift_vec * dt + noise
        norm_tmp = float(np.linalg.norm(p_tmp))
        if norm_tmp < 1e-12:
            p_new[i] = p[i]
        else:
            p_new[i] = p_tmp / norm_tmp
    return p_new


def vicsek_alignment_step(
    x: ArrayLike,
    p: ArrayLike,
    Lx: float,
    Ly: float,
    bc: str,
    lV: float,
    mu_r: float,
    Dtheta: float,
    dt: float,
    cell_list: CellList | None = None,
    rng: np.random.Generator | None = None,
) -> ArrayLike:
    """Vicsek alignment update (AIM-1 Eq. 6) using linked-cell neighbours."""

    if rng is None:
        rng = np.random.default_rng()
    if lV <= 0.0 or (mu_r <= 0.0 and Dtheta <= 0.0):
        return p.copy()
    if bc not in {"periodic", "reflecting"}:
        raise ValueError("Unknown boundary condition for alignment")

    local_cells = cell_list or build_cells(x, Lx, Ly, lV, bc)
    neighbours = neighbor_indices_from_celllist(x, local_cells, Lx, Ly, lV, bc)

    noise_scale = np.sqrt(max(0.0, 2.0 * Dtheta * dt))
    p_new = np.empty_like(p)
    for i in range(p.shape[0]):
        idx = neighbours[i]
        drift_vec = np.zeros(2, dtype=float)
        if mu_r > 0.0 and idx.size:
            mean_vec = np.sum(p[idx], axis=0)
            norm = float(np.linalg.norm(mean_vec))
            if norm > 1e-12:
                drift_vec = mu_r * (mean_vec / norm - p[i])
        noise = noise_scale * rng.normal(size=2)
        p_tmp = p[i] + drift_vec * dt + noise
        norm_tmp = float(np.linalg.norm(p_tmp))
        if norm_tmp < 1e-12:
            p_new[i] = p[i]
        else:
            p_new[i] = p_tmp / norm_tmp
    return p_new


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

    if rate <= 0.0 and Dtheta <= 0.0:
        return v

    speeds = np.linalg.norm(v, axis=1, keepdims=True)
    headings = np.zeros_like(v)
    zero_mask = speeds.squeeze() < 1e-12
    if rng is None:
        rng = np.random.default_rng()

    if np.any(zero_mask):
        # Provide a deterministic orientation for zero-speed agents.
        rand_angles = rng.uniform(0.0, 2.0 * np.pi, size=np.count_nonzero(zero_mask))
        headings[zero_mask] = np.column_stack((np.cos(rand_angles), np.sin(rand_angles)))
        speeds[zero_mask] = target_speed
    headings[~zero_mask] = v[~zero_mask] / speeds[~zero_mask]

    cell_list = build_cells(x, Lx, Ly, radius, bc)
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

    return headings_new * speeds


def simulate_backend(config: Dict[str, Dict], rng: np.random.Generator) -> Dict[str, ArrayLike]:
    """Unified backend interface for continuous D'Orsogna model with RK4/Euler integration.
    
    This function implements the standardized backend interface that produces
    consistent outputs across all model types (discrete VM, continuous/RK, etc.).
    
    Parameters
    ----------
    config : dict
        Configuration dictionary with keys:
        - sim: {N, Lx, Ly, bc, T, dt, save_every, neighbor_rebuild, integrator}
        - params: {alpha, beta, Cr, Ca, lr, la, rcut_factor, alignment (optional)}
    rng : np.random.Generator
        Random number generator instance.
        
    Returns
    -------
    result : dict
        Standardized result dictionary with keys:
        - times: (T,) array of time points
        - traj: (T, N, 2) array of positions
        - vel: (T, N, 2) array of velocities
        - meta: dict of configuration parameters (includes force_evals, force_time)
        
    Notes
    -----
    The continuous D'Orsogna model uses ODE integration (RK4 or semi-implicit Euler):
    1. Compute Morse forces on all particles
    2. Integrate dx/dt = v, dv/dt = (α - β|v|²)v + F using chosen integrator
    3. Optionally apply Vicsek-style alignment to velocities
    4. Save frames at specified intervals
    """

    sim_cfg = config["sim"]
    param_cfg = config["params"]
    N = sim_cfg["N"]
    Lx = sim_cfg["Lx"]
    Ly = sim_cfg["Ly"]
    bc = sim_cfg["bc"]
    T = sim_cfg["T"]
    dt = sim_cfg["dt"]
    save_every = sim_cfg["save_every"]
    neighbor_rebuild = sim_cfg["neighbor_rebuild"]

    alpha = param_cfg["alpha"]
    beta = param_cfg["beta"]
    Cr = param_cfg["Cr"]
    Ca = param_cfg["Ca"]
    lr = param_cfg["lr"]
    la = param_cfg["la"]
    rcut = param_cfg["rcut_factor"] * max(lr, la)

    # Initialize positions
    ic_cfg = config.get("ic", {})
    ic_type = ic_cfg.get("type", "uniform")
    
    if ic_type == "uniform":
        # Fast path for uniform (most common case)
        x0 = rng.uniform(low=[0.0, 0.0], high=[Lx, Ly], size=(N, 2))
    else:
        # Use IC generation module for other distributions
        from .ic import sample_initial_positions
        x0 = sample_initial_positions(ic_type, N, Lx, Ly, rng)
    
    # Initialize velocities
    v0_mag = alpha / beta
    angles = rng.uniform(0.0, 2 * np.pi, size=N)
    v0 = v0_mag * np.column_stack((np.cos(angles), np.sin(angles)))

    state = State(x=x0, v=v0, t=0.0)
    apply_bc(state.x, Lx, Ly, bc)

    total_steps = int(np.round(T / dt))

    integrator = step_rk4 if sim_cfg["integrator"] == "rk4" else step_euler_semiimplicit

    frames_x = [state.x.copy()]
    frames_v = [state.v.copy()]
    frame_times = [state.t]

    cell_list = build_cells(state.x, Lx, Ly, rcut, bc)
    force_calc = ForceCalculator(Cr, Ca, lr, la, Lx, Ly, bc, rcut, cell_list)

    align_cfg = param_cfg.get("alignment", {})
    align_enabled = align_cfg.get("enabled", False)
    align_radius = align_cfg.get("radius", 1.5)
    align_rate = align_cfg.get("rate", 0.0)
    align_Dtheta = align_cfg.get("Dtheta", 0.0)

    pbar = tqdm(range(1, total_steps + 1), desc="Simulating", unit="step")
    sim_start = time.perf_counter()

    for step in pbar:
        if step % neighbor_rebuild == 1:
            force_calc.rebuild(state.x)

        new_state = integrator(
            state,
            param_cfg,
            dt,
            force_calc,
            {"Lx": Lx, "Ly": Ly, "bc": bc},
        )

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

        if step % save_every == 0 or step == total_steps:
            frames_x.append(state.x.copy())
            frames_v.append(state.v.copy())
            frame_times.append(state.t)

        elapsed = time.perf_counter() - sim_start
        pbar.set_postfix(
            {
                "t": f"{state.t:.2f}",
                "force": f"{force_calc.total_time:.2f}",
                "elapsed": f"{elapsed:.2f}",
            }
        )

    pbar.close()

    traj = np.stack(frames_x, axis=0).astype(np.float32)
    vel = np.stack(frames_v, axis=0).astype(np.float32)
    times = np.array(frame_times).astype(np.float32)

    # Standardized return format matching vicsek_discrete.simulate_backend
    # Store extra metadata in the meta dict
    meta = config.copy()
    meta['force_evals'] = force_calc.calls
    meta['force_time'] = force_calc.total_time
    meta['rcut'] = rcut

    result = {
        "times": times,
        "traj": traj,
        "vel": vel,
        "meta": meta,
    }

    return result


def simulate(config: Dict[str, Dict]) -> Dict[str, ArrayLike]:
    """Legacy wrapper for simulate_backend (for backwards compatibility).
    
    This function maintains the old API where the RNG is created from config["seed"].
    New code should use simulate_backend() directly with an explicit RNG.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary (must include config["seed"])
        
    Returns
    -------
    result : dict
        Legacy format with keys: traj, vel, times, params, sim, rcut, force_evals, force_time
    """
    rng = np.random.default_rng(config["seed"])
    result = simulate_backend(config, rng)
    
    # Convert to legacy format for backwards compatibility
    meta = result['meta']
    legacy_result = {
        "traj": result['traj'],
        "vel": result['vel'],
        "times": result['times'],
        "params": meta['params'],
        "sim": meta['sim'],
        "rcut": meta['rcut'],
        "force_evals": meta['force_evals'],
        "force_time": meta['force_time'],
    }
    
    return legacy_result


__all__ = ["simulate", "simulate_backend", "apply_alignment_step", "vicsek_alignment_step"]
