"""Time evolution for the rectangular collective motion simulations."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm

from .domain import apply_bc, pair_displacements
from .integrators import State, step_euler_semiimplicit, step_rk4
from .morse import CellList, build_cells, morse_force_pairs

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
        fx, fy = morse_force_pairs(
            positions,
            self.Cr,
            self.Ca,
            self.lr,
            self.la,
            self.Lx,
            self.Ly,
            self.bc,
            self.rcut,
            cell_list=self.cell_list,
        )
        self.total_time += time.perf_counter() - start
        self.calls += 1
        return fx, fy


def _alignment_step(
    x: ArrayLike,
    v: ArrayLike,
    Lx: float,
    Ly: float,
    bc: str,
    radius: float,
    rate: float,
    dt: float,
    target_speed: float,
) -> ArrayLike:
    """Relax velocities toward neighborhood averages for alignment interactions."""

    if rate <= 0:
        return v

    dx, dy, rij, _ = pair_displacements(x, Lx, Ly, bc)
    new_v = v.copy()
    for i in range(x.shape[0]):
        mask = rij[i] <= radius
        if not np.any(mask):
            continue
        mean_dir = np.sum(v[mask], axis=0)
        norm_mean = np.linalg.norm(mean_dir)
        if norm_mean < 1e-12:
            continue
        mean_dir /= norm_mean
        vi = v[i]
        speed = np.linalg.norm(vi)
        if speed < 1e-12:
            vi = target_speed * mean_dir
            speed = target_speed
        else:
            vi = vi / speed
        blend = (1 - rate * dt) * vi + rate * dt * mean_dir
        norm_blend = np.linalg.norm(blend)
        if norm_blend < 1e-12:
            blend = mean_dir
            norm_blend = 1.0
        speed_new = (1 - rate * dt) * speed + rate * dt * target_speed
        new_v[i] = speed_new * (blend / norm_blend)
    return new_v


def simulate(config: Dict[str, Dict]) -> Dict[str, ArrayLike]:
    """Run a simulation using the provided configuration."""

    rng = np.random.default_rng(config["seed"])

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

    x0 = rng.uniform(low=[0.0, 0.0], high=[Lx, Ly], size=(N, 2))
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
                align_cfg.get("radius", 1.5),
                align_cfg.get("rate", 0.1),
                dt,
                v0_mag,
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

    traj = np.stack(frames_x, axis=0)
    vel = np.stack(frames_v, axis=0)
    times = np.array(frame_times)

    result = {
        "traj": traj,
        "vel": vel,
        "times": times,
        "params": param_cfg,
        "sim": sim_cfg,
        "rcut": rcut,
        "force_evals": force_calc.calls,
        "force_time": force_calc.total_time,
    }

    return result


__all__ = ["simulate"]
