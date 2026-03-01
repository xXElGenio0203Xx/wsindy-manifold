"""
High-level forecast interface for WSINDy models.

Given a discovered :class:`WSINDyModel`, an initial condition ``U0``,
and a :class:`GridSpec`, this module chooses the best integrator
(ETDRK4 when feasible, RK4 otherwise) and returns the full trajectory.
"""

from __future__ import annotations

import numpy as np

from .grid import GridSpec
from .model import WSINDyModel
from .integrators import (
    _can_use_etdrk4,
    etdrk4_integrate,
    rk4_integrate,
    split_linear_nonlinear,
)


def wsindy_forecast(
    U0: np.ndarray,
    model: WSINDyModel,
    grid: GridSpec,
    n_steps: int,
    method: str = "auto",
    clip_negative: bool = False,
) -> np.ndarray:
    """Forecast a density field forward in time.

    Parameters
    ----------
    U0 : ndarray (nx, ny)
        Initial condition.
    model : WSINDyModel
        Discovered PDE model with active terms and coefficients.
    grid : GridSpec
        Must contain *dt* (integration time-step), *dx*, *dy*.
    n_steps : int
        Number of time steps to integrate.
    method : ``"auto"`` | ``"etdrk4"`` | ``"rk4"``
        ``"auto"`` selects ETDRK4 when linear-in-*u* terms exist,
        otherwise RK4.
    clip_negative : bool
        If True, enforce :math:`u \\ge 0` after every step (useful for
        density fields).

    Returns
    -------
    U_pred : ndarray (n_steps + 1, nx, ny)
        Trajectory including the initial condition at index 0.
    """
    if method == "auto":
        linear_terms, _ = split_linear_nonlinear(model)
        use_etdrk4 = _can_use_etdrk4(linear_terms)
    elif method == "etdrk4":
        use_etdrk4 = True
    elif method == "rk4":
        use_etdrk4 = False
    else:
        raise ValueError(
            f"Unknown method '{method}'; use 'auto', 'etdrk4', or 'rk4'"
        )

    if use_etdrk4:
        return etdrk4_integrate(
            U0, n_steps, grid.dt, grid, model, clip_negative,
        )
    else:
        return rk4_integrate(
            U0, n_steps, grid.dt, grid, model, clip_negative,
        )
