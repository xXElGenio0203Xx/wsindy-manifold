"""Time integration schemes for the D'Orsogna model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

from .domain import apply_bc

ArrayLike = np.ndarray
ForceFunction = Callable[[ArrayLike], Tuple[ArrayLike, ArrayLike]]


@dataclass
class State:
    """Container for simulation state."""

    x: ArrayLike
    v: ArrayLike
    t: float

    def copy(self) -> "State":
        return State(x=self.x.copy(), v=self.v.copy(), t=float(self.t))


def _acceleration(
    v: ArrayLike,
    alpha: float,
    beta: float,
    fx: ArrayLike,
    fy: ArrayLike,
) -> ArrayLike:
    speed_sq = np.sum(v**2, axis=1, keepdims=True)
    self_prop = (alpha - beta * speed_sq) * v
    return self_prop + np.column_stack((fx, fy))


def step_rk4(
    state: State,
    params: dict,
    dt: float,
    force_fn: ForceFunction,
    domain: dict,
) -> State:
    """Advance the state by one step using fourth-order Runge--Kutta."""

    alpha = params["alpha"]
    beta = params["beta"]
    Lx = domain["Lx"]
    Ly = domain["Ly"]
    bc = domain["bc"]

    def eval_force(pos: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        pos_wrapped = pos.copy()
        apply_bc(pos_wrapped, Lx, Ly, bc)
        return force_fn(pos_wrapped)

    x0 = state.x
    v0 = state.v

    fx0, fy0 = eval_force(x0)
    a0 = _acceleration(v0, alpha, beta, fx0, fy0)

    k1_x = v0
    k1_v = a0

    x1 = x0 + 0.5 * dt * k1_x
    v1 = v0 + 0.5 * dt * k1_v
    fx1, fy1 = eval_force(x1)
    a1 = _acceleration(v1, alpha, beta, fx1, fy1)

    k2_x = v1
    k2_v = a1

    x2 = x0 + 0.5 * dt * k2_x
    v2 = v0 + 0.5 * dt * k2_v
    fx2, fy2 = eval_force(x2)
    a2 = _acceleration(v2, alpha, beta, fx2, fy2)

    k3_x = v2
    k3_v = a2

    x3 = x0 + dt * k3_x
    v3 = v0 + dt * k3_v
    fx3, fy3 = eval_force(x3)
    a3 = _acceleration(v3, alpha, beta, fx3, fy3)

    k4_x = v3
    k4_v = a3

    x_new = x0 + dt / 6.0 * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
    v_new = v0 + dt / 6.0 * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    x_new, flips = apply_bc(x_new, Lx, Ly, bc)
    v_new = v_new.copy()
    v_new[flips] *= -1

    return State(x=x_new, v=v_new, t=state.t + dt)


def step_euler_semiimplicit(
    state: State,
    params: dict,
    dt: float,
    force_fn: ForceFunction,
    domain: dict,
    iterations: int = 3,
) -> State:
    """Advance the state using a semi-implicit Euler scheme."""

    alpha = params["alpha"]
    beta = params["beta"]
    Lx = domain["Lx"]
    Ly = domain["Ly"]
    bc = domain["bc"]

    fx, fy = force_fn(state.x)
    accel_force = np.column_stack((fx, fy))

    v_new = state.v.copy()
    for _ in range(iterations):
        speed_sq = np.sum(v_new**2, axis=1, keepdims=True)
        v_new = state.v + dt * (accel_force + (alpha - beta * speed_sq) * v_new)

    x_new = state.x + dt * v_new
    x_new, flips = apply_bc(x_new, Lx, Ly, bc)
    v_new[flips] *= -1

    return State(x=x_new, v=v_new, t=state.t + dt)


__all__ = ["State", "step_rk4", "step_euler_semiimplicit"]
