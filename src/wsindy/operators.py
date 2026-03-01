"""
Spectral derivative utilities for 2-D periodic grids.

All functions assume the spatial field *u* has shape ``(nx, ny)`` and lives
on a doubly-periodic domain with uniform spacings *dx*, *dy*.
Derivatives are computed via multiplication in Fourier space; the result
is always real (`.real` is taken after the inverse FFT).
"""

from __future__ import annotations

import numpy as np


# ── wavenumber helper ───────────────────────────────────────────────

def fft_wavenumbers(n: int, d: float) -> np.ndarray:
    """Return angular wavenumbers *k* of shape ``(n,)`` compatible with
    ``np.fft.fftfreq``.

    .. math::
        k_j = 2\\pi \\, \\text{fftfreq}(n, d)_j

    Parameters
    ----------
    n : int
        Number of grid points.
    d : float
        Grid spacing.

    Returns
    -------
    k : ndarray (n,)
    """
    return 2.0 * np.pi * np.fft.fftfreq(n, d=d)


# ── first derivatives ──────────────────────────────────────────────

def grad_spectral(
    u: np.ndarray, dx: float, dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(u_x, u_y)`` via FFT differentiation on a periodic grid.

    Parameters
    ----------
    u : ndarray, shape (nx, ny)
    dx, dy : float – uniform grid spacings

    Returns
    -------
    ux, uy : ndarray, shape (nx, ny), real
    """
    nx, ny = u.shape
    kx = fft_wavenumbers(nx, dx)
    ky = fft_wavenumbers(ny, dy)
    u_hat = np.fft.fft2(u)
    ux = np.fft.ifft2(1j * kx[:, None] * u_hat).real
    uy = np.fft.ifft2(1j * ky[None, :] * u_hat).real
    return ux, uy


# ── second derivatives ─────────────────────────────────────────────

def dxx_spectral(u: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Return :math:`\\partial^2 u / \\partial x^2` via FFT."""
    nx, _ny = u.shape
    kx = fft_wavenumbers(nx, dx)
    u_hat = np.fft.fft2(u)
    return np.fft.ifft2(-(kx[:, None] ** 2) * u_hat).real


def dyy_spectral(u: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Return :math:`\\partial^2 u / \\partial y^2` via FFT."""
    _nx, ny = u.shape
    ky = fft_wavenumbers(ny, dy)
    u_hat = np.fft.fft2(u)
    return np.fft.ifft2(-(ky[None, :] ** 2) * u_hat).real


def laplacian_spectral(u: np.ndarray, dx: float, dy: float) -> np.ndarray:
    r"""Return :math:`\Delta u = u_{xx} + u_{yy}` via FFT.

    In Fourier space:

    .. math::
        \widehat{\Delta u} = -(k_x^2 + k_y^2)\,\hat u
    """
    nx, ny = u.shape
    kx = fft_wavenumbers(nx, dx)
    ky = fft_wavenumbers(ny, dy)
    u_hat = np.fft.fft2(u)
    ksq = kx[:, None] ** 2 + ky[None, :] ** 2
    return np.fft.ifft2(-ksq * u_hat).real
