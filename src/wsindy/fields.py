"""
Field computation for multi-field WSINDy PDE discovery.
=========================================================

Computes the derived field arrays needed for the thesis-grade library:

1. **Flux / polarization** ``p_x(x,t), p_y(x,t)`` from velocity-weighted KDE
2. **Morse potential** ``Φ(x,t) = W * ρ`` via FFT convolution
3. **Derived quantities** (gradients, |p|², divergences, etc.)

All fields are on the same (T, ny, nx) grid as ρ.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from .grid import GridSpec


# ═══════════════════════════════════════════════════════════════════
#  Flux / polarization fields via vector-weighted KDE
# ═══════════════════════════════════════════════════════════════════

def compute_flux_kde(
    traj: np.ndarray,
    vel: np.ndarray,
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    Lx: float,
    Ly: float,
    bandwidth: float = 5.0,
    bc: str = "periodic",
    subsample: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute polarization/flux fields ``p_x, p_y`` via velocity-weighted KDE.

    For each snapshot the flux is defined as::

        p(x, t) = Σ_i  K_h(x − x_i(t))  v_i(t)

    where ``K_h`` is the same Gaussian kernel used for density.

    Parameters
    ----------
    traj : ndarray, shape (T, N, 2)
        Particle positions at every saved frame.
    vel : ndarray, shape (T, N, 2)
        Particle velocities at every saved frame.
    xgrid, ygrid : ndarray, shape (nx,) and (ny,)
        Cell-centre coordinates of the density grid.
    Lx, Ly : float
        Domain extents.
    bandwidth : float
        Gaussian smoothing σ in **grid-cell units** (same as density KDE).
    bc : str
        ``"periodic"`` or ``"reflecting"``.
    subsample : int
        Temporal sub-sampling factor (match what's used for ρ).

    Returns
    -------
    px, py : ndarray, shape (T_sub, ny, nx)
    """
    traj = np.asarray(traj, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)

    T_full = traj.shape[0]
    N = traj.shape[1]
    nx = len(xgrid)
    ny = len(ygrid)
    dx_phys = Lx / nx
    dy_phys = Ly / ny

    x_edges = np.linspace(0.0, Lx, nx + 1)
    y_edges = np.linspace(0.0, Ly, ny + 1)
    mode = "wrap" if bc == "periodic" else "nearest"

    frames = range(0, T_full, subsample)
    T_sub = len(list(frames))

    px = np.zeros((T_sub, ny, nx), dtype=np.float64)
    py = np.zeros((T_sub, ny, nx), dtype=np.float64)

    for out_idx, t in enumerate(range(0, T_full, subsample)):
        pos_t = traj[t]  # (N, 2)
        vel_t = vel[t]   # (N, 2)

        # Weighted histograms: weight by v_x or v_y
        hx, _, _ = np.histogram2d(
            pos_t[:, 0], pos_t[:, 1],
            bins=[x_edges, y_edges],
            weights=vel_t[:, 0],
        )
        hy, _, _ = np.histogram2d(
            pos_t[:, 0], pos_t[:, 1],
            bins=[x_edges, y_edges],
            weights=vel_t[:, 1],
        )

        # Convert to flux density (per unit area)
        hx = hx.T / (dx_phys * dy_phys)
        hy = hy.T / (dx_phys * dy_phys)

        # Gaussian smoothing (same kernel as density KDE)
        if bandwidth > 0:
            hx = gaussian_filter(hx, sigma=bandwidth, mode=mode)
            hy = gaussian_filter(hy, sigma=bandwidth, mode=mode)

        px[out_idx] = hx
        py[out_idx] = hy

    return px, py


# ═══════════════════════════════════════════════════════════════════
#  Morse potential Φ = W * ρ via FFT
# ═══════════════════════════════════════════════════════════════════

def morse_kernel_grid(
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    Lx: float,
    Ly: float,
    Cr: float,
    Ca: float,
    lr: float,
    la: float,
    bc: str = "periodic",
) -> np.ndarray:
    """Build the Morse potential kernel W(r) on the spatial grid.

    The Morse potential is::

        W(r) = -C_r exp(-r/l_r) + C_a exp(-r/l_a)

    (repulsive at short range, attractive at long range).

    Returns
    -------
    W : ndarray, shape (ny, nx)
        Centred kernel (DC at [0,0] for FFT use).
    """
    nx = len(xgrid)
    ny = len(ygrid)

    # Build centred coordinate arrays
    # For periodic FFT convolution, distances wrap around half-domain
    cx = np.fft.fftfreq(nx, d=1.0) * Lx  # [-Lx/2, ..., Lx/2]
    cy = np.fft.fftfreq(ny, d=1.0) * Ly

    CX, CY = np.meshgrid(cx, cy)  # (ny, nx)
    R = np.sqrt(CX**2 + CY**2)
    R = np.maximum(R, 1e-10)  # avoid division by zero at origin

    W = -Cr * np.exp(-R / lr) + Ca * np.exp(-R / la)
    W[0, 0] = 0.0  # remove self-interaction at r=0

    return W


def compute_morse_potential(
    rho: np.ndarray,
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    Lx: float,
    Ly: float,
    Cr: float,
    Ca: float,
    lr: float,
    la: float,
    bc: str = "periodic",
) -> np.ndarray:
    """Compute Φ(x,t) = W * ρ for all time steps via FFT.

    Parameters
    ----------
    rho : ndarray, shape (T, ny, nx)
        Density field.
    xgrid, ygrid : 1-D arrays
        Cell-centre coordinates.
    Cr, Ca, lr, la : float
        Morse potential parameters.

    Returns
    -------
    Phi : ndarray, shape (T, ny, nx)
        Morse potential field.
    """
    W = morse_kernel_grid(xgrid, ygrid, Lx, Ly, Cr, Ca, lr, la, bc)
    dx = Lx / len(xgrid)
    dy = Ly / len(ygrid)

    W_hat = np.fft.fft2(W)

    T = rho.shape[0]
    Phi = np.zeros_like(rho)
    for t in range(T):
        rho_hat = np.fft.fft2(rho[t])
        # Convolution theorem: Φ = ifft(W_hat * ρ_hat) * dx * dy
        Phi[t] = np.real(np.fft.ifft2(W_hat * rho_hat)) * dx * dy

    return Phi


# ═══════════════════════════════════════════════════════════════════
#  Spatial derivative helpers — finite-difference (FD) backend
# ═══════════════════════════════════════════════════════════════════

def _dx(f: np.ndarray, dx: float) -> np.ndarray:
    """Central difference ∂f/∂x along last axis (periodic)."""
    return (np.roll(f, -1, axis=-1) - np.roll(f, 1, axis=-1)) / (2 * dx)


def _dy(f: np.ndarray, dy: float) -> np.ndarray:
    """Central difference ∂f/∂y along second-to-last axis (periodic)."""
    return (np.roll(f, -1, axis=-2) - np.roll(f, 1, axis=-2)) / (2 * dy)


def _dxx(f: np.ndarray, dx: float) -> np.ndarray:
    """Second derivative ∂²f/∂x² (periodic)."""
    return (np.roll(f, -1, axis=-1) - 2 * f + np.roll(f, 1, axis=-1)) / dx**2


def _dyy(f: np.ndarray, dy: float) -> np.ndarray:
    """Second derivative ∂²f/∂y² (periodic)."""
    return (np.roll(f, -1, axis=-2) - 2 * f + np.roll(f, 1, axis=-2)) / dy**2


def _lap(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Laplacian Δf = ∂²f/∂x² + ∂²f/∂y²."""
    return _dxx(f, dx) + _dyy(f, dy)


def _bilap(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Biharmonic Δ²f."""
    return _lap(_lap(f, dx, dy), dx, dy)


def _div_vec(fx: np.ndarray, fy: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Divergence ∇·(f_x, f_y)."""
    return _dx(fx, dx) + _dy(fy, dy)


# ═══════════════════════════════════════════════════════════════════
#  Spatial derivative helpers — spectral (FFT) backend
#  Machine-precision on periodic grids, unconditionally stable.
# ═══════════════════════════════════════════════════════════════════

def _spectral_dx(f: np.ndarray, Lx: float) -> np.ndarray:
    """Spectral ∂f/∂x on periodic domain [0, Lx)."""
    nx = f.shape[-1]
    dx = Lx / nx
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    # Zero the Nyquist mode for odd-order derivative accuracy
    if nx % 2 == 0:
        kx[nx // 2] = 0.0
    f_hat = np.fft.fft(f, axis=-1)
    return np.real(np.fft.ifft(1j * kx * f_hat, axis=-1))


def _spectral_dy(f: np.ndarray, Ly: float) -> np.ndarray:
    """Spectral ∂f/∂y on periodic domain [0, Ly)."""
    ny = f.shape[-2]
    dy = Ly / ny
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    if ny % 2 == 0:
        ky[ny // 2] = 0.0
    # Broadcast: make ky column vector so it multiplies along axis=-2
    ky = ky.reshape((-1,) + (1,) * (f.ndim - (f.ndim - 1)))  # (ny, 1, ...)
    shape = [1] * f.ndim
    shape[-2] = ny
    ky = ky.reshape(shape)
    f_hat = np.fft.fft(f, axis=-2)
    return np.real(np.fft.ifft(1j * ky * f_hat, axis=-2))


def _spectral_wavenumbers_2d(
    ny: int, nx: int, Lx: float, Ly: float,
) -> np.ndarray:
    """Return K² = kx² + ky²  shape (ny, nx)."""
    dx, dy = Lx / nx, Ly / ny
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    return KX**2 + KY**2


def _spectral_lap(f: np.ndarray, Lx: float, Ly: float) -> np.ndarray:
    """Spectral Laplacian Δf on periodic domain."""
    ny, nx = f.shape[-2], f.shape[-1]
    K2 = _spectral_wavenumbers_2d(ny, nx, Lx, Ly)
    f_hat = np.fft.fft2(f, axes=(-2, -1))
    return np.real(np.fft.ifft2(-K2 * f_hat, axes=(-2, -1)))


def _spectral_bilap(f: np.ndarray, Lx: float, Ly: float) -> np.ndarray:
    """Spectral biharmonic Δ²f on periodic domain."""
    ny, nx = f.shape[-2], f.shape[-1]
    K2 = _spectral_wavenumbers_2d(ny, nx, Lx, Ly)
    f_hat = np.fft.fft2(f, axes=(-2, -1))
    return np.real(np.fft.ifft2(K2**2 * f_hat, axes=(-2, -1)))


# ═══════════════════════════════════════════════════════════════════
#  Master field container
# ═══════════════════════════════════════════════════════════════════

class FieldData:
    """Container holding all precomputed fields on a shared grid.

    Attributes
    ----------
    rho : (T, ny, nx)
    px, py : (T, ny, nx) — flux / polarization
    Phi : (T, ny, nx) — Morse potential  (None if Morse disabled)
    grid : GridSpec
    Lx, Ly : domain size
    """

    def __init__(
        self,
        rho: np.ndarray,
        px: np.ndarray,
        py: np.ndarray,
        grid: GridSpec,
        Lx: float,
        Ly: float,
        Phi: Optional[np.ndarray] = None,
        use_spectral: bool = False,
    ):
        self.rho = rho
        self.px = px
        self.py = py
        self.Phi = Phi
        self.grid = grid
        self.Lx = Lx
        self.Ly = Ly
        self.use_spectral = use_spectral

    @property
    def dx(self) -> float:
        return self.grid.dx

    @property
    def dy(self) -> float:
        return self.grid.dy

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.rho.shape

    # ── Derivative dispatch (FD vs spectral) ─────────────────────

    def _ddx(self, f: np.ndarray) -> np.ndarray:
        """∂f/∂x — dispatches to spectral or FD backend."""
        if self.use_spectral:
            return _spectral_dx(f, self.Lx)
        return _dx(f, self.dx)

    def _ddy(self, f: np.ndarray) -> np.ndarray:
        """∂f/∂y — dispatches to spectral or FD backend."""
        if self.use_spectral:
            return _spectral_dy(f, self.Ly)
        return _dy(f, self.dy)

    def _laplacian(self, f: np.ndarray) -> np.ndarray:
        """Δf — dispatches to spectral or FD backend."""
        if self.use_spectral:
            return _spectral_lap(f, self.Lx, self.Ly)
        return _lap(f, self.dx, self.dy)

    def _bilaplacian(self, f: np.ndarray) -> np.ndarray:
        """Δ²f — dispatches to spectral or FD backend."""
        if self.use_spectral:
            return _spectral_bilap(f, self.Lx, self.Ly)
        return _bilap(f, self.dx, self.dy)

    def _div(self, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
        """∇·(fx, fy) — dispatches to spectral or FD backend."""
        return self._ddx(fx) + self._ddy(fy)

    # ── Cached derived quantities ────────────────────────────────

    def _cache(self, name: str, compute_fn):
        if not hasattr(self, f"_c_{name}"):
            setattr(self, f"_c_{name}", compute_fn())
        return getattr(self, f"_c_{name}")

    # Density derivatives
    def dx_rho(self):
        return self._cache("dx_rho", lambda: self._ddx(self.rho))

    def dy_rho(self):
        return self._cache("dy_rho", lambda: self._ddy(self.rho))

    def lap_rho(self):
        return self._cache("lap_rho", lambda: self._laplacian(self.rho))

    # Flux derivatives
    def div_p(self):
        """∇·p"""
        return self._cache("div_p", lambda: self._div(self.px, self.py))

    def lap_px(self):
        return self._cache("lap_px", lambda: self._laplacian(self.px))

    def lap_py(self):
        return self._cache("lap_py", lambda: self._laplacian(self.py))

    def bilap_px(self):
        return self._cache("bilap_px", lambda: self._bilaplacian(self.px))

    def bilap_py(self):
        return self._cache("bilap_py", lambda: self._bilaplacian(self.py))

    # Polarization magnitude squared
    def p_sq(self):
        """|p|²"""
        return self._cache("p_sq", lambda: self.px**2 + self.py**2)

    # Nonlinear density
    def rho2(self):
        return self._cache("rho2", lambda: self.rho**2)

    def rho3(self):
        return self._cache("rho3", lambda: self.rho**3)

    # Laplacians of nonlinear density
    def lap_rho2(self):
        return self._cache("lap_rho2", lambda: self._laplacian(self.rho2()))

    def lap_rho3(self):
        return self._cache("lap_rho3", lambda: self._laplacian(self.rho3()))

    def lap_p_sq(self):
        """Δ(|p|²)"""
        return self._cache("lap_p_sq", lambda: self._laplacian(self.p_sq()))

    # Density-weighted divergence of flux (used internally)
    def div_rho_grad_rho(self):
        """∇·(ρ ∇ρ)"""
        return self._cache("div_rho_grad_rho", lambda: self._div(
            self.rho * self.dx_rho(),
            self.rho * self.dy_rho()))

    # ── Morse-derived fields ─────────────────────────────────────

    def grad_Phi_x(self):
        if self.Phi is None:
            raise ValueError("Morse potential not computed")
        return self._cache("grad_Phi_x", lambda: self._ddx(self.Phi))

    def grad_Phi_y(self):
        if self.Phi is None:
            raise ValueError("Morse potential not computed")
        return self._cache("grad_Phi_y", lambda: self._ddy(self.Phi))

    def rho_grad_Phi_x(self):
        """ρ ∂_x Φ"""
        return self._cache("rho_grad_Phi_x", lambda: self.rho * self.grad_Phi_x())

    def rho_grad_Phi_y(self):
        """ρ ∂_y Φ"""
        return self._cache("rho_grad_Phi_y", lambda: self.rho * self.grad_Phi_y())

    def div_rho_gradPhi(self):
        """∇·(ρ ∇Φ)"""
        return self._cache("div_rho_gradPhi", lambda: self._div(
            self.rho_grad_Phi_x(),
            self.rho_grad_Phi_y()))

    # ── Self-advection (p·∇)p ────────────────────────────────────

    def p_dot_grad_px(self):
        """(p·∇)p_x"""
        return self._cache("p_dot_grad_px", lambda:
            self.px * self._ddx(self.px) + self.py * self._ddy(self.px))

    def p_dot_grad_py(self):
        """(p·∇)p_y"""
        return self._cache("p_dot_grad_py", lambda:
            self.px * self._ddx(self.py) + self.py * self._ddy(self.py))

    # ── Pressure gradients ────────────────────────────────────────

    def dx_rho2(self):
        """∂_x(ρ²)"""
        return self._cache("dx_rho2", lambda: self._ddx(self.rho2()))

    def dy_rho2(self):
        """∂_y(ρ²)"""
        return self._cache("dy_rho2", lambda: self._ddy(self.rho2()))

    # ── Coupling: ∇·(ρ ∇|p|²) ────────────────────────────────────

    def div_rho_grad_p_sq(self):
        """∇·(ρ ∇|p|²) — dangerous, may overfit"""
        p2 = self.p_sq()
        return self._cache("div_rho_grad_p_sq", lambda: self._div(
            self.rho * self._ddx(p2),
            self.rho * self._ddy(p2)))


def build_field_data(
    rho: np.ndarray,
    traj: np.ndarray,
    vel: np.ndarray,
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    Lx: float,
    Ly: float,
    dt: float,
    bandwidth: float = 5.0,
    bc: str = "periodic",
    subsample: int = 1,
    morse_params: Optional[Dict[str, float]] = None,
) -> FieldData:
    """One-shot construction of all fields from simulation data.

    Parameters
    ----------
    rho : (T, ny, nx) — density (already sub-sampled)
    traj : (T_full, N, 2) — particle positions (full temporal resolution)
    vel : (T_full, N, 2) — particle velocities (full temporal resolution)
    xgrid, ygrid : 1-D grid arrays
    Lx, Ly : domain extents
    dt : time step of ρ (after sub-sampling)
    bandwidth : KDE bandwidth in grid-cell units
    bc : boundary condition
    subsample : temporal sub-sample factor (applied to traj/vel → match ρ)
    morse_params : dict with keys ``Cr, Ca, lr, la`` or None

    Returns
    -------
    FieldData
    """
    # Compute flux fields from particle data
    px, py = compute_flux_kde(
        traj, vel, xgrid, ygrid, Lx, Ly,
        bandwidth=bandwidth, bc=bc, subsample=subsample,
    )

    # Trim to match ρ length (in case of off-by-one)
    T = min(rho.shape[0], px.shape[0])
    rho = rho[:T]
    px = px[:T]
    py = py[:T]

    grid = GridSpec(dt=dt, dx=Lx / len(xgrid), dy=Ly / len(ygrid))

    # Morse potential
    Phi = None
    if morse_params is not None:
        Phi = compute_morse_potential(
            rho, xgrid, ygrid, Lx, Ly,
            Cr=morse_params["Cr"],
            Ca=morse_params["Ca"],
            lr=morse_params["lr"],
            la=morse_params["la"],
            bc=bc,
        )

    return FieldData(rho, px, py, grid, Lx, Ly, Phi=Phi)


def build_field_data_rho_only(
    rho: np.ndarray,
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    Lx: float,
    Ly: float,
    dt: float,
    morse_params: Optional[Dict[str, float]] = None,
    bc: str = "periodic",
) -> FieldData:
    """Build FieldData when only ρ is available (no trajectory data).

    Flux fields ``px, py`` are set to zero, so flux-dependent terms
    will be zero and should not be selected by MSTLS.
    """
    T, ny, nx = rho.shape
    zeros = np.zeros_like(rho)
    grid = GridSpec(dt=dt, dx=Lx / nx, dy=Ly / ny)

    Phi = None
    if morse_params is not None:
        Phi = compute_morse_potential(
            rho, xgrid, ygrid, Lx, Ly,
            Cr=morse_params["Cr"],
            Ca=morse_params["Ca"],
            lr=morse_params["lr"],
            la=morse_params["la"],
            bc=bc,
        )

    return FieldData(rho, zeros, zeros, grid, Lx, Ly, Phi=Phi)
