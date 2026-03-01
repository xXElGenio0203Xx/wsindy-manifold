"""
Separable compactly supported test functions for WSINDy.

The test function is

    ψ(x, y, t) = φ_x(x) · φ_y(y) · φ_t(t)

where each 1-D component is a polynomial bump

    φ_i(s) = (1 - s² / (ℓ_i Δ_i)²)^{p_i},   |s| ≤ ℓ_i Δ_i

with *ℓ_i* an integer half-width (in grid cells) and *p_i* the polynomial
degree controlling smoothness.

We also provide the required partial-derivative arrays (ψ_t, ψ_x, ψ_y,
ψ_xx, ψ_yy) computed analytically via the product rule + 1-D finite
differences on the support lattice.

All returned arrays have axis order **(t, x, y)** — time first — matching
the ``(T, nx, ny)`` convention assumed by the pipeline.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from .grid import GridSpec
from .utils import finite_diff_1d


# ── 1-D bump on discrete support ────────────────────────────────────────────

def make_1d_phi(
    grid_step: float,
    ell: int,
    p: int,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Sample a 1-D polynomial bump on its discrete support.

    Parameters
    ----------
    grid_step : float
        Uniform grid spacing Δ for this dimension.
    ell : int
        Half-width in *grid cells*.  The support is
        ``[-ell * Δ, +ell * Δ]``, sampled at ``2*ell + 1`` points.
    p : int
        Polynomial exponent (smoothness).  Must be ≥ 1.

    Returns
    -------
    coords : 1-D ndarray of shape ``(2*ell + 1,)``
        Sample coordinates ``s = -ell*Δ, ..., 0, ..., +ell*Δ``.
    phi : 1-D ndarray, same shape
        ``(1 - s²/(ell*Δ)²)^p`` evaluated at *coords*.
    """
    if ell < 1:
        raise ValueError(f"ell must be >= 1, got {ell}")
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")

    half_width = ell * grid_step
    coords = np.linspace(-half_width, half_width, 2 * ell + 1)
    # Normalised squared distance in [0, 1]
    u = (coords / half_width) ** 2
    phi = (1.0 - u) ** p

    return coords, phi


# ── full separable ψ bundle ─────────────────────────────────────────────────

#: Keys guaranteed to be present in the dict returned by
#: :func:`make_separable_psi`.
PSI_KEYS = frozenset({
    "psi",
    "psi_t",
    "psi_x",
    "psi_y",
    "psi_xx",
    "psi_yy",
    "support_coords",
    "grid",
    "ell",
    "p",
})


def make_separable_psi(
    grid: GridSpec,
    ellx: int,
    elly: int,
    ellt: int,
    px: int,
    py: int,
    pt: int,
) -> Dict[str, NDArray[np.floating] | dict]:
    """Build the separable test function and its required derivatives.

    Parameters
    ----------
    grid : GridSpec
        Uniform grid metadata (dt, dx, dy).
    ellx, elly, ellt : int
        Half-widths (in grid cells) for x, y, t.
    px, py, pt : int
        Polynomial exponents for x, y, t.

    Returns
    -------
    bundle : dict
        ``"psi"``
            3-D ndarray of shape ``(2*ellt+1, 2*ellx+1, 2*elly+1)``.
        ``"psi_t"``
            ∂_t ψ, same shape.
        ``"psi_x"``
            ∂_x ψ, same shape.
        ``"psi_y"``
            ∂_y ψ, same shape.
        ``"psi_xx"``
            ∂_xx ψ, same shape.
        ``"psi_yy"``
            ∂_yy ψ, same shape.
        ``"support_coords"``
            Dict with keys ``"t"``, ``"x"``, ``"y"`` — 1-D coordinate
            vectors on the support lattice.
        ``"grid"``
            The :class:`GridSpec` used.
        ``"ell"``
            Tuple ``(ellt, ellx, elly)``.
        ``"p"``
            Tuple ``(pt, px, py)``.
    """
    # 1-D components and their coordinates
    sx, phi_x = make_1d_phi(grid.dx, ellx, px)
    sy, phi_y = make_1d_phi(grid.dy, elly, py)
    st, phi_t = make_1d_phi(grid.dt, ellt, pt)

    # 1-D derivatives (on the support grid)
    dphi_x = finite_diff_1d(phi_x, grid.dx, order=1)
    dphi_y = finite_diff_1d(phi_y, grid.dy, order=1)
    dphi_t = finite_diff_1d(phi_t, grid.dt, order=1)

    d2phi_x = finite_diff_1d(phi_x, grid.dx, order=2)
    d2phi_y = finite_diff_1d(phi_y, grid.dy, order=2)

    # ── outer products (t, x, y) ────────────────────────────────────────
    # Shapes: phi_t → (nt,), phi_x → (nx,), phi_y → (ny,)
    # Broadcast via [:,None,None] * [None,:,None] * [None,None,:]
    T = phi_t[:, None, None]
    X = phi_x[None, :, None]
    Y = phi_y[None, None, :]

    dT = dphi_t[:, None, None]
    dX = dphi_x[None, :, None]
    dY = dphi_y[None, None, :]

    d2X = d2phi_x[None, :, None]
    d2Y = d2phi_y[None, None, :]

    psi = T * X * Y
    psi_t = dT * X * Y        # ∂_t ψ = φ'_t · φ_x · φ_y
    psi_x = T * dX * Y        # ∂_x ψ = φ_t · φ'_x · φ_y
    psi_y = T * X * dY        # ∂_y ψ = φ_t · φ_x · φ'_y
    psi_xx = T * d2X * Y      # ∂_xx ψ = φ_t · φ''_x · φ_y
    psi_yy = T * X * d2Y      # ∂_yy ψ = φ_t · φ_x · φ''_y

    return {
        "psi": psi,
        "psi_t": psi_t,
        "psi_x": psi_x,
        "psi_y": psi_y,
        "psi_xx": psi_xx,
        "psi_yy": psi_yy,
        "support_coords": {"t": st, "x": sx, "y": sy},
        "grid": grid,
        "ell": (ellt, ellx, elly),
        "p": (pt, px, py),
    }
