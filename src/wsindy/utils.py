"""
Low-level numerical utilities for WSINDy.

Provides finite-difference derivative approximations on uniform 1-D grids,
used to differentiate the compactly supported test functions on their
support lattice.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ── public API ───────────────────────────────────────────────────────────────

def finite_diff_1d(
    arr: NDArray[np.floating],
    step: float,
    order: int,
) -> NDArray[np.floating]:
    """Differentiate a uniformly sampled 1-D array via finite differences.

    Parameters
    ----------
    arr : 1-D ndarray
        Function values on a uniform grid with spacing *step*.
    step : float
        Grid spacing Δ.
    order : int
        Derivative order (0, 1, or 2).

    Returns
    -------
    deriv : 1-D ndarray, same length as *arr*
        Approximation of d^(order) f / ds^(order).

    Notes
    -----
    * Order 0 returns a copy of *arr*.
    * Interior points use second-order central differences.
    * Boundary points use second-order one-sided (forward/backward)
      stencils so the output length equals the input length.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D array, got shape {arr.shape}")
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")

    n = arr.shape[0]

    if order == 0:
        return arr.copy()

    if n < 3:
        raise ValueError(
            f"Need at least 3 points for order-{order} finite differences, "
            f"got {n}"
        )

    if order == 1:
        return _fd1(arr, step)
    if order == 2:
        return _fd2(arr, step)

    raise NotImplementedError(f"Derivatives of order {order} > 2 not supported")


# ── internal stencils ────────────────────────────────────────────────────────

def _fd1(f: NDArray[np.floating], h: float) -> NDArray[np.floating]:
    """First derivative, O(h²) everywhere."""
    n = f.shape[0]
    out = np.empty_like(f)

    # interior: central  (-1, 0, +1) / 2h
    out[1:-1] = (f[2:] - f[:-2]) / (2.0 * h)

    # left boundary: forward  (-3f0 + 4f1 - f2) / 2h
    out[0] = (-3.0 * f[0] + 4.0 * f[1] - f[2]) / (2.0 * h)

    # right boundary: backward  (3fn - 4fn-1 + fn-2) / 2h
    out[-1] = (3.0 * f[-1] - 4.0 * f[-2] + f[-3]) / (2.0 * h)

    return out


def _fd2(f: NDArray[np.floating], h: float) -> NDArray[np.floating]:
    """Second derivative, O(h²) everywhere."""
    n = f.shape[0]
    out = np.empty_like(f)

    # interior: central  (1, -2, 1) / h²
    h2 = h * h
    out[1:-1] = (f[2:] - 2.0 * f[1:-1] + f[:-2]) / h2

    # left boundary: forward  (2f0 - 5f1 + 4f2 - f3) / h²
    if n >= 4:
        out[0] = (2.0 * f[0] - 5.0 * f[1] + 4.0 * f[2] - f[3]) / h2
        out[-1] = (2.0 * f[-1] - 5.0 * f[-2] + 4.0 * f[-3] - f[-4]) / h2
    else:
        # fallback to central (already handled n>=3 check above)
        out[0] = (f[2] - 2.0 * f[1] + f[0]) / h2
        out[-1] = (f[-1] - 2.0 * f[-2] + f[-3]) / h2

    return out
