"""Gaussian kernel density estimation on a rectangular grid."""

from __future__ import annotations

import math
from typing import Callable, Dict, Tuple

import numpy as np

Array = np.ndarray

DistanceFn = Callable[[Array, Array, float, float, str], Tuple[Array, Array]]


def make_grid(Lx: float, Ly: float, nx: int, ny: int) -> Tuple[Array, float, float]:
    """Return grid cell centres and spacings for a rectangular domain.

    Parameters
    ----------
    Lx, Ly
        Domain lengths along ``x`` and ``y``.
    nx, ny
        Number of grid cells along each dimension.

    Returns
    -------
    Xc, dx, dy
        ``Xc`` has shape ``(nc, 2)`` with ``nc = nx * ny``. The second and
        third values are the uniform cell widths.
    """

    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive integers")

    dx = Lx / nx
    dy = Ly / ny
    x_centres = (np.arange(nx, dtype=float) + 0.5) * dx
    y_centres = (np.arange(ny, dtype=float) + 0.5) * dy
    xv, yv = np.meshgrid(x_centres, y_centres, indexing="xy")
    Xc = np.stack([xv.ravel(), yv.ravel()], axis=-1)
    return Xc, dx, dy


def _grid_shape_from_centres(Xc: Array, Lx: float, Ly: float) -> Tuple[int, int, float, float]:
    """Infer grid shape and spacing from cell centres."""

    if Xc.ndim != 2 or Xc.shape[1] != 2:
        raise ValueError("Xc must have shape (nc, 2)")

    x_unique = np.unique(np.round(Xc[:, 0], decimals=12))
    y_unique = np.unique(np.round(Xc[:, 1], decimals=12))
    nx = x_unique.size
    ny = y_unique.size
    if nx * ny != Xc.shape[0]:
        raise ValueError("Xc does not appear to describe a tensor-product grid")
    dx = Lx / nx
    dy = Ly / ny
    return nx, ny, dx, dy


def minimal_image_dxdy(delta: Array, length: float) -> Array:
    """Wrap coordinate differences onto ``[-L/2, L/2]`` using minimal images."""

    if length <= 0:
        return delta
    return delta - length * np.round(delta / length)


def pair_dxdy_to_grid(X: Array, Xc: Array, Lx: float, Ly: float, bc: str) -> Tuple[Array, Array]:
    """Return differences between grid cell centres and particle positions.

    Parameters
    ----------
    X
        Particle coordinates with shape ``(N, 2)``.
    Xc
        Grid cell centres with shape ``(nc, 2)``.
    Lx, Ly
        Domain sizes used for periodic wrapping.
    bc
        Boundary condition string. ``"periodic"`` applies minimal images,
        otherwise plain differences are used.

    Returns
    -------
    Δx, Δy : ndarray
        Arrays with shape ``(nc, N)`` giving the offsets from particles to grid
        cells along each coordinate axis.
    """

    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must have shape (N, 2)")
    if Xc.ndim != 2 or Xc.shape[1] != 2:
        raise ValueError("Xc must have shape (nc, 2)")

    dx = Xc[:, 0][:, None] - X[:, 0][None, :]
    dy = Xc[:, 1][:, None] - X[:, 1][None, :]

    if bc == "periodic":
        dx = minimal_image_dxdy(dx, Lx)
        dy = minimal_image_dxdy(dy, Ly)
    elif bc == "reflecting":
        dx = np.clip(dx, -Lx, Lx)
        dy = np.clip(dy, -Ly, Ly)
    else:
        raise ValueError(f"Unsupported boundary condition '{bc}'")

    return dx, dy


def kde_gaussian(
    X: Array,
    Xc: Array,
    hx: float,
    hy: float,
    Lx: float,
    Ly: float,
    bc: str = "periodic",
    tile: int = 1,
    distance_fn: DistanceFn | None = None,
) -> Array:
    """Evaluate an anisotropic Gaussian KDE on a rectangular grid.

    The kernel is ``Kh(u) = (1/(2π hx hy)) * exp(-0.5 * (((ux/hx)^2 + (uy/hy)^2)))``.
    The function returns values at the grid centres stored in ``Xc``.

    Parameters
    ----------
    X
        Point cloud of shape ``(N, 2)`` inside the domain.
    Xc
        Grid centres of shape ``(nc, 2)``. Typically produced by :func:`make_grid`.
    hx, hy
        Gaussian bandwidths. Must be strictly positive.
    Lx, Ly
        Domain lengths used for periodic wrapping.
    bc
        Either ``"periodic"`` (default) or ``"reflecting"``.
    tile
        Reserved for future tiling-based evaluation strategies. Currently
        ignored (kept for API compatibility.

    Returns
    -------
    rho : ndarray
        Array of shape ``(nc,)`` with probability density values that integrate
        to unity under the Riemann sum with spacings ``dx`` and ``dy``.
    """

    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must have shape (N, 2)")
    if Xc.ndim != 2 or Xc.shape[1] != 2:
        raise ValueError("Xc must have shape (nc, 2)")
    if hx <= 0 or hy <= 0:
        raise ValueError("Bandwidths hx and hy must be positive")

    nx, ny, dx, dy = _grid_shape_from_centres(Xc, Lx=Lx, Ly=Ly)

    N = X.shape[0]
    if N == 0:
        return np.full(Xc.shape[0], 1.0 / (dx * dy * nx * ny), dtype=float)

    if distance_fn is None:
        dx_grid, dy_grid = pair_dxdy_to_grid(X, Xc, Lx=Lx, Ly=Ly, bc=bc)
    else:
        dx_grid, dy_grid = distance_fn(X, Xc, Lx, Ly, bc)

    scaled = (dx_grid / hx) ** 2 + (dy_grid / hy) ** 2
    kernel = np.exp(-0.5 * scaled)
    prefactor = 1.0 / (2.0 * math.pi * hx * hy)
    rho = prefactor * kernel.sum(axis=1) / N

    # Enforce exact mass conservation under the Riemann sum.
    mass = float(np.sum(rho) * dx * dy)
    if mass <= 0:
        raise RuntimeError("Computed density has non-positive mass")
    rho /= mass
    return rho


def trajectories_to_density_movie(
    X_all: Array,
    Lx: float,
    Ly: float,
    nx: int,
    ny: int,
    hx: float,
    hy: float,
    bc: str = "periodic",
    distance_fn: DistanceFn | None = None,
) -> Tuple[Array, Dict[str, Array]]:
    """Convert trajectories into a stack of KDE density snapshots.

    Parameters
    ----------
    X_all
        Trajectory array of shape ``(T, N, 2)``.
    Lx, Ly, nx, ny, hx, hy
        Domain and KDE parameters.
    bc
        Boundary condition, typically ``"periodic"`` or ``"reflecting"``.

    Returns
    -------
    Rho, meta
        ``Rho`` has shape ``(T, nc)`` with ``nc = nx * ny``. ``meta`` contains
        the grid centres, spacings, original parameters, and a mask placeholder
        for future extensions.
    """

    if X_all.ndim != 3 or X_all.shape[2] != 2:
        raise ValueError("X_all must have shape (T, N, 2)")

    X_all = np.asarray(X_all, dtype=float)
    T, N, _ = X_all.shape
    Xc, dx, dy = make_grid(Lx=Lx, Ly=Ly, nx=nx, ny=ny)
    nc = Xc.shape[0]
    rho_frames = np.empty((T, nc), dtype=float)

    for t in range(T):
        rho_frames[t] = kde_gaussian(
            X=X_all[t],
            Xc=Xc,
            hx=hx,
            hy=hy,
            Lx=Lx,
            Ly=Ly,
            bc=bc,
            distance_fn=distance_fn,
        )

    meta: Dict[str, Array] = {
        "Xc": Xc,
        "dx": np.array(dx, dtype=float),
        "dy": np.array(dy, dtype=float),
        "nx": np.array(nx, dtype=int),
        "ny": np.array(ny, dtype=int),
        "Lx": np.array(Lx, dtype=float),
        "Ly": np.array(Ly, dtype=float),
        "hx": np.array(hx, dtype=float),
        "hy": np.array(hy, dtype=float),
        "bc": np.array(bc),
    }
    return rho_frames, meta


__all__ = [
    "make_grid",
    "kde_gaussian",
    "trajectories_to_density_movie",
    "minimal_image_dxdy",
    "pair_dxdy_to_grid",
]
