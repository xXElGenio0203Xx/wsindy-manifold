"""
Time integrators for discovered PDEs: RK4 and ETDRK4.

* **RK4** – classic fourth-order Runge–Kutta.  Always safe, but requires
  small *dt* for stiff diffusion.
* **ETDRK4** – Cox–Matthews / Kassam–Trefethen exponential integrator.
  Treats the linear part :math:`Lu` exactly in Fourier space and the
  nonlinear remainder :math:`N(u)` with a fourth-order explicit scheme.
  Coefficients are evaluated via the contour-integral trick so that the
  :math:`\\varphi`-functions are numerically stable near zero eigenvalues.
"""

from __future__ import annotations

import numpy as np

from .grid import GridSpec
from .model import WSINDyModel
from .operators import fft_wavenumbers
from .rhs import (
    apply_operator_pointwise,
    eval_feature_pointwise,
    parse_term,
    wsindy_rhs,
)


# ════════════════════════════════════════════════════════════════════
#  Linear / nonlinear splitting
# ════════════════════════════════════════════════════════════════════

def split_linear_nonlinear(
    model: WSINDyModel,
) -> tuple[dict[str, float], dict[str, float]]:
    """Split active model terms into *linear-in-u* and *nonlinear* groups.

    A term is **linear** if its feature is ``"u"`` (any spatial operator).
    Everything else (``"I:1"``, ``"I:u2"``, ``"lap:u3"``, …) goes into the
    nonlinear bucket.

    Returns
    -------
    linear_terms : dict  {col_name: weight}
    nonlinear_terms : dict  {col_name: weight}
    """
    linear: dict[str, float] = {}
    nonlinear: dict[str, float] = {}
    for i, name in enumerate(model.col_names):
        if not model.active[i]:
            continue
        _op, feat = parse_term(name)
        if feat == "u":
            linear[name] = float(model.w[i])
        else:
            nonlinear[name] = float(model.w[i])
    return linear, nonlinear


def _can_use_etdrk4(linear_terms: dict[str, float]) -> bool:
    """Return True when all linear terms have a known Fourier symbol."""
    if not linear_terms:
        return False
    allowed_ops = {"I", "dx", "dy", "dxx", "dyy", "lap"}
    for name in linear_terms:
        op, _feat = parse_term(name)
        if op not in allowed_ops:
            return False
    return True


# ════════════════════════════════════════════════════════════════════
#  Build L_hat (Fourier-space linear operator)
# ════════════════════════════════════════════════════════════════════

def build_L_hat(
    linear_terms: dict[str, float],
    nx: int,
    ny: int,
    grid: GridSpec,
) -> np.ndarray:
    r"""Return :math:`\hat L` of shape ``(nx, ny)`` — the Fourier multiplier
    for the linear part of the PDE.

    .. math::
        \hat L(k_x,k_y) = \sum_m w_m\,\sigma_m(k_x,k_y)

    Operator symbols:

    ======  ============================
    op      :math:`\sigma(k_x,k_y)`
    ======  ============================
    ``I``   1
    ``dx``  :math:`i k_x`
    ``dy``  :math:`i k_y`
    ``dxx`` :math:`-k_x^2`
    ``dyy`` :math:`-k_y^2`
    ``lap`` :math:`-(k_x^2+k_y^2)`
    ======  ============================
    """
    kx = fft_wavenumbers(nx, grid.dx)
    ky = fft_wavenumbers(ny, grid.dy)
    KX = kx[:, None] * np.ones((1, ny))
    KY = np.ones((nx, 1)) * ky[None, :]

    L_hat = np.zeros((nx, ny), dtype=np.complex128)
    for name, w in linear_terms.items():
        op, _feat = parse_term(name)
        if op == "I":
            L_hat += w
        elif op == "dx":
            L_hat += w * 1j * KX
        elif op == "dy":
            L_hat += w * 1j * KY
        elif op == "dxx":
            L_hat += w * (-(KX ** 2))
        elif op == "dyy":
            L_hat += w * (-(KY ** 2))
        elif op == "lap":
            L_hat += w * (-(KX ** 2 + KY ** 2))
    return L_hat


# ════════════════════════════════════════════════════════════════════
#  ETDRK4 coefficients via contour integral (Kassam–Trefethen 2005)
# ════════════════════════════════════════════════════════════════════

def _etdrk4_coefficients(
    L_hat: np.ndarray,
    dt: float,
    M: int = 64,
    r: float = 1.0,
) -> tuple[np.ndarray, ...]:
    r"""Pre-compute the six ETDRK4 coefficient arrays.

    The scheme reads

    .. math::
        \hat u_{n+1} = E\,\hat u_n
            + f_1\,\hat N_1
            + 2\,f_2\,(\hat N_2 + \hat N_3)
            + f_3\,\hat N_4

    Parameters
    ----------
    L_hat : complex array (nx, ny)
    dt : time step
    M : contour-integral quadrature points (64 is ample)
    r : contour radius

    Returns
    -------
    E, E2, Q, f1, f2, f3 : real arrays (nx, ny)
        E  = exp(z),  E2 = exp(z/2)  where z = dt·L̂
        Q  = dt·(exp(z/2)−1)/z   (used in intermediate stages)
        f1, f2, f3  = Cox–Matthews final-stage weights
    """
    z = dt * L_hat  # (nx, ny), complex

    # Contour points on a circle of radius r
    theta = np.arange(1, M + 1) * (2.0 * np.pi / M)
    circ = r * np.exp(1j * theta)  # (M,)

    E = np.exp(z)
    E2 = np.exp(z / 2.0)

    # ── Q = dt·(exp(z/2) − 1) / z  via contour average ─────────
    #    Rewrite as (dt/2)·φ₁(z/2)  with  φ₁(w) = (e^w − 1)/w
    zc_half = z[..., np.newaxis] / 2.0 + circ  # (..., M)
    Q = (dt / 2.0) * np.mean(
        (np.exp(zc_half) - 1.0) / zc_half, axis=-1,
    ).real

    # ── Final-stage weights ─────────────────────────────────────
    zc = z[..., np.newaxis] + circ  # (..., M)
    Ezc = np.exp(zc)
    zc2 = zc * zc
    zc3 = zc2 * zc

    f1 = dt * np.mean(
        (-4.0 - zc + Ezc * (4.0 - 3.0 * zc + zc2)) / zc3,
        axis=-1,
    ).real
    f2 = dt * np.mean(
        (2.0 + zc + Ezc * (-2.0 + zc)) / zc3,
        axis=-1,
    ).real
    f3 = dt * np.mean(
        (-4.0 - 3.0 * zc - zc2 + Ezc * (4.0 - zc)) / zc3,
        axis=-1,
    ).real

    return E, E2, Q, f1, f2, f3


# ════════════════════════════════════════════════════════════════════
#  RK4 integrator
# ════════════════════════════════════════════════════════════════════

def rk4_step(u: np.ndarray, dt: float, rhs_fn) -> np.ndarray:
    """One step of classic fourth-order Runge–Kutta."""
    k1 = rhs_fn(u)
    k2 = rhs_fn(u + 0.5 * dt * k1)
    k3 = rhs_fn(u + 0.5 * dt * k2)
    k4 = rhs_fn(u + dt * k3)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_integrate(
    u0: np.ndarray,
    n_steps: int,
    dt: float,
    grid: GridSpec,
    model: WSINDyModel,
    clip_negative: bool = False,
) -> np.ndarray:
    """Integrate with RK4.  Returns trajectory ``(n_steps+1, nx, ny)``."""
    traj = np.empty((n_steps + 1, *u0.shape), dtype=np.float64)
    traj[0] = u0.copy()

    def rhs_fn(u):
        return wsindy_rhs(u, model, grid)

    u = u0.copy()
    for k in range(n_steps):
        u = rk4_step(u, dt, rhs_fn)
        if clip_negative:
            u = np.clip(u, 0.0, None)
        traj[k + 1] = u
    return traj


# ════════════════════════════════════════════════════════════════════
#  ETDRK4 integrator
# ════════════════════════════════════════════════════════════════════

def _nonlinear_rhs_fn(
    u: np.ndarray,
    nonlinear_terms: dict[str, float],
    grid: GridSpec,
) -> np.ndarray:
    """Evaluate only the *nonlinear* part of the RHS in physical space."""
    rhs = np.zeros_like(u)
    for name, w in nonlinear_terms.items():
        op, feat = parse_term(name)
        f = eval_feature_pointwise(u, feat)
        rhs += w * apply_operator_pointwise(f, op, grid)
    return rhs


def etdrk4_integrate(
    u0: np.ndarray,
    n_steps: int,
    dt: float,
    grid: GridSpec,
    model: WSINDyModel,
    clip_negative: bool = False,
) -> np.ndarray:
    r"""Integrate with ETDRK4 (Cox–Matthews).

    Splits the PDE into :math:`u_t = Lu + N(u)`, constructs the Fourier
    multiplier :math:`\hat L`, and time-steps in spectral domain.

    Returns
    -------
    trajectory : ndarray (n_steps+1, nx, ny)
    """
    nx, ny = u0.shape
    linear_terms, nonlinear_terms = split_linear_nonlinear(model)
    L_hat = build_L_hat(linear_terms, nx, ny, grid)
    E, E2, Q, f1, f2, f3 = _etdrk4_coefficients(L_hat, dt)

    traj = np.empty((n_steps + 1, nx, ny), dtype=np.float64)
    traj[0] = u0.copy()

    u_hat = np.fft.fft2(u0)

    def Nhat(u_real: np.ndarray) -> np.ndarray:
        return np.fft.fft2(
            _nonlinear_rhs_fn(u_real, nonlinear_terms, grid)
        )

    for k in range(n_steps):
        u_real = np.fft.ifft2(u_hat).real

        N1 = Nhat(u_real)
        a_hat = E2 * u_hat + Q * N1

        N2 = Nhat(np.fft.ifft2(a_hat).real)
        b_hat = E2 * u_hat + Q * N2

        N3 = Nhat(np.fft.ifft2(b_hat).real)
        c_hat = E2 * a_hat + Q * (2.0 * N3 - N1)

        N4 = Nhat(np.fft.ifft2(c_hat).real)

        u_hat = (
            E * u_hat
            + N1 * f1
            + 2.0 * (N2 + N3) * f2
            + N4 * f3
        )

        u_new = np.fft.ifft2(u_hat).real
        if clip_negative:
            u_new = np.clip(u_new, 0.0, None)
            u_hat = np.fft.fft2(u_new)
        traj[k + 1] = u_new

    return traj
