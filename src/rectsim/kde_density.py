"""
KDE-based density estimation following Alvarez et al. (2025).

This module implements Algorithm C.1 from the paper "Next Generation Equation-Free
Multiscale Modelling of Crowd Dynamics via Machine Learning" with exact replication
of their Gaussian kernel, bandwidth handling (Silverman or manual), periodic
augmentation, obstacle masking, and two-step normalization.

Reference:
    Alvarez et al. (2025), Appendix C: Kernel Density Estimation
"""

import numpy as np
from typing import Optional, Tuple, Dict, Union


def silverman_bandwidth(positions: np.ndarray) -> Tuple[float, float]:
    """
    Compute Silverman's rule of thumb bandwidth for 2D KDE.
    
    From paper Eq. (C.2):
    h_i = [4 / (N * (d + 2))]^(1/(d + 4)) * σ_i,  where d = 2
    
    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        Particle positions at a single time snapshot
        
    Returns
    -------
    h_x, h_y : float
        Bandwidth in x and y directions
    """
    N = positions.shape[0]
    d = 2  # dimension
    
    # Sample standard deviations
    sigma_x = np.std(positions[:, 0], ddof=1)  # ddof=1 for sample std
    sigma_y = np.std(positions[:, 1], ddof=1)
    
    # Silverman's rule: h_i = [4/(N(d+2))]^(1/(d+4)) * σ_i
    multiplier = (4.0 / (N * (d + 2))) ** (1.0 / (d + 4))
    
    h_x = multiplier * sigma_x
    h_y = multiplier * sigma_y
    
    return h_x, h_y


def gaussian_kernel_2d(x: np.ndarray, y: np.ndarray, 
                       x_particle: float, y_particle: float,
                       h_x: float, h_y: float) -> np.ndarray:
    """
    Evaluate 2D Gaussian kernel K_H(x - x_i) on a grid.
    
    From paper Eq. (C.1):
    K_H(z) = [1 / (2π * det(H))^(1/2)] * exp(-1/2 * z^T H^(-1) z)
    where H = diag(h_x, h_y)
    
    Parameters
    ----------
    x, y : ndarray
        Grid coordinates (meshgrid format)
    x_particle, y_particle : float
        Particle position
    h_x, h_y : float
        Bandwidth in x and y directions
        
    Returns
    -------
    kernel_values : ndarray
        Kernel evaluated at each grid point
    """
    # Distance from particle
    dx = x - x_particle
    dy = y - y_particle
    
    # H = diag(h_x, h_y), so det(H) = h_x * h_y
    # H^(-1) = diag(1/h_x, 1/h_y)
    det_H = h_x * h_y
    
    # z^T H^(-1) z = (dx/h_x)^2 + (dy/h_y)^2
    exponent = -0.5 * ((dx / h_x)**2 + (dy / h_y)**2)
    
    # K_H(z) = [1 / (2π * det(H))^(1/2)] * exp(exponent)
    normalization = 1.0 / (2.0 * np.pi * np.sqrt(det_H))
    
    return normalization * np.exp(exponent)


def augment_positions_periodic(positions: np.ndarray, 
                               domain: Tuple[float, float, float, float],
                               extension_n: int = 5) -> np.ndarray:
    """
    Create augmented position set with mirrored particles near boundaries.
    
    From paper Algorithm C.1, steps 2-5:
    - Compute Δx = (b - a) / n with n = 5
    - For particles within Δx of left boundary, add ghost at (x - (b-a), y)
    - For particles within Δx of right boundary, add ghost at (x + (b-a), y)
    
    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        Original particle positions
    domain : tuple of (a, b, c, d)
        Domain boundaries [a,b] × [c,d]
    extension_n : int, default=5
        Parameter n for computing Δx = (b - a) / n
        
    Returns
    -------
    augmented_positions : ndarray, shape (N_all, 2)
        Original positions plus mirrored ghosts
    """
    a, b, c, d = domain
    Lx = b - a
    
    # Extension parameter: Δx = Lx / n
    delta_x = Lx / extension_n
    
    augmented = [positions]  # Start with original positions
    
    # Mirror particles near left boundary
    left_mask = positions[:, 0] <= (a + delta_x)
    if np.any(left_mask):
        left_ghosts = positions[left_mask].copy()
        left_ghosts[:, 0] = left_ghosts[:, 0] - Lx  # Translate left
        augmented.append(left_ghosts)
    
    # Mirror particles near right boundary
    right_mask = positions[:, 0] >= (b - delta_x)
    if np.any(right_mask):
        right_ghosts = positions[right_mask].copy()
        right_ghosts[:, 0] = right_ghosts[:, 0] + Lx  # Translate right
        augmented.append(right_ghosts)
    
    return np.vstack(augmented)


def create_obstacle_mask(domain: Tuple[float, float, float, float],
                        nx: int, ny: int,
                        obstacle_rect: Optional[Tuple[float, float, float, float]] = None) -> np.ndarray:
    """
    Create binary mask M(x,y) for obstacle region Γ.
    
    From paper Eq. (C.5):
    M(x,y) = 0 if (x,y) ∈ Γ, else 1
    
    Parameters
    ----------
    domain : tuple of (a, b, c, d)
        Domain boundaries
    nx, ny : int
        Grid resolution
    obstacle_rect : tuple of (a_o, b_o, c_o, d_o), optional
        Obstacle rectangle [a_o, b_o] × [c_o, d_o]
        
    Returns
    -------
    mask : ndarray, shape (nx, ny)
        Binary mask (1 = free space, 0 = obstacle)
    """
    a, b, c, d = domain
    
    # Create grid coordinates (cell centers)
    dx = (b - a) / nx
    dy = (d - c) / ny
    
    x_centers = np.linspace(a + dx/2, b - dx/2, nx)
    y_centers = np.linspace(c + dy/2, d - dy/2, ny)
    
    X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')
    
    # Initialize mask to all ones (free space)
    mask = np.ones((nx, ny), dtype=float)
    
    # Apply obstacle if provided
    if obstacle_rect is not None:
        a_o, b_o, c_o, d_o = obstacle_rect
        obstacle_mask = (X >= a_o) & (X <= b_o) & (Y >= c_o) & (Y <= d_o)
        mask[obstacle_mask] = 0.0
    
    return mask


def kde_density_snapshot(
    positions: np.ndarray,
    domain: Tuple[float, float, float, float],
    nx: int = 80,
    ny: int = 20,
    bandwidth_mode: str = "manual",
    manual_H: Tuple[float, float] = (3.0, 2.0),
    periodic_x: bool = True,
    periodic_extension_n: int = 5,
    obstacle_rect: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, float, Dict]:
    """
    Compute normalized KDE density field following Algorithm C.1 exactly.
    
    Implements the complete pipeline from Alvarez et al. (2025):
    1. Bandwidth selection (Silverman or manual)
    2. Periodic augmentation with ghost particles
    3. Extended domain KDE computation
    4. Obstacle masking
    5. Two-step normalization (by N_eff, then by integral)
    
    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        Particle positions at a single time snapshot, columns (x, y)
    domain : tuple of (a, b, c, d)
        Domain boundaries [a,b] × [c,d]
    nx, ny : int, default=80, 20
        Grid resolution (paper defaults)
    bandwidth_mode : {"silverman", "manual"}, default="manual"
        Bandwidth selection method
    manual_H : tuple of (h_x, h_y), default=(3.0, 2.0)
        Manual bandwidth (paper tuned values)
    periodic_x : bool, default=True
        Enable periodic boundary handling in x-direction
    periodic_extension_n : int, default=5
        Extension parameter n for Δx = (b-a)/n
    obstacle_rect : tuple of (a_o, b_o, c_o, d_o), optional
        Obstacle rectangle to mask out
        
    Returns
    -------
    rho : ndarray, shape (nx, ny)
        Normalized density field on cell centers
    H : ndarray, shape (2, 2)
        Bandwidth matrix (diagonal)
    S : float
        Total mass before final normalization
    meta : dict
        Metadata with {N, N_all, delta_x, delta_y, bandwidth_mode, etc.}
        
    Notes
    -----
    This function exactly replicates Algorithm C.1 from the paper:
    - Gaussian kernel with diagonal covariance H = diag(h_x, h_y)
    - Silverman bandwidth: h_i = [4/(N(d+2))]^(1/(d+4)) * σ_i, d=2
    - Periodic augmentation with ghost particles in extended domain
    - Obstacle masking: ρ ← M · ρ
    - Two-step normalization: divide by N_eff, then by ∫_Ω ρ dx dy
    
    References
    ----------
    Alvarez et al. (2025), Appendix C, Algorithm C.1
    """
    N = positions.shape[0]
    a, b, c, d = domain
    Lx = b - a
    Ly = d - c
    
    # ========== Step 1: Bandwidth selection ==========
    if bandwidth_mode == "silverman":
        h_x, h_y = silverman_bandwidth(positions)
    elif bandwidth_mode == "manual":
        h_x, h_y = manual_H
    else:
        raise ValueError(f"Unknown bandwidth_mode: {bandwidth_mode}")
    
    H = np.diag([h_x, h_y])
    
    # ========== Step 2-5: Periodic augmentation ==========
    if periodic_x:
        augmented_positions = augment_positions_periodic(
            positions, domain, periodic_extension_n
        )
        
        # Extended domain: Ω_ext = [a - Δx, b + Δx] × [c, d]
        delta_x_ext = Lx / periodic_extension_n
        a_ext = a - delta_x_ext
        b_ext = b + delta_x_ext
        c_ext = c
        d_ext = d
        
        # Extended grid resolution
        extension_cells = int(2 * nx / periodic_extension_n)
        nx_ext = nx + extension_cells
        ny_ext = ny
    else:
        augmented_positions = positions
        a_ext, b_ext, c_ext, d_ext = a, b, c, d
        nx_ext = nx
        ny_ext = ny
    
    N_all = augmented_positions.shape[0]
    
    # ========== Step 4: Create extended grid ==========
    dx = (b_ext - a_ext) / nx_ext
    dy = (d_ext - c_ext) / ny_ext
    
    # Cell centers
    x_centers_ext = np.linspace(a_ext + dx/2, b_ext - dx/2, nx_ext)
    y_centers_ext = np.linspace(c_ext + dy/2, d_ext - dy/2, ny_ext)
    
    X_ext, Y_ext = np.meshgrid(x_centers_ext, y_centers_ext, indexing='ij')
    
    # ========== Step 6-7: Initialize and compute KDE on extended domain ==========
    rho_ext = np.zeros((nx_ext, ny_ext), dtype=float)
    
    # Add contribution from each particle (including ghosts)
    for i in range(N_all):
        x_i, y_i = augmented_positions[i]
        kernel_contrib = gaussian_kernel_2d(X_ext, Y_ext, x_i, y_i, h_x, h_y)
        rho_ext += kernel_contrib
    
    # ========== Step 8: Normalize by N_eff (total particle count) ==========
    rho_ext = rho_ext / N_all
    
    # ========== Step 9: Restrict to original domain Ω ==========
    if periodic_x:
        # Find indices corresponding to original domain
        left_crop = int(extension_cells / 2)
        right_crop = left_crop + nx
        rho = rho_ext[left_crop:right_crop, :]
    else:
        rho = rho_ext
    
    # Verify dimensions
    assert rho.shape == (nx, ny), f"Shape mismatch: {rho.shape} != ({nx}, {ny})"
    
    # ========== Step 10: Apply obstacle mask ==========
    if obstacle_rect is not None:
        mask = create_obstacle_mask(domain, nx, ny, obstacle_rect)
        rho = mask * rho  # Eq. (C.6): ρ ← M · ρ
    
    # ========== Step 11: Compute total mass S(t) ==========
    # Cell dimensions in original domain
    dx_orig = Lx / nx
    dy_orig = Ly / ny
    
    S = np.sum(rho) * dx_orig * dy_orig
    
    # ========== Step 12: Final mass normalization ==========
    if S > 1e-12:  # Avoid division by zero
        rho = rho / S
    else:
        # No particles, return uniform zero field
        rho = np.zeros((nx, ny))
    
    # ========== Return results ==========
    meta = {
        'N': N,
        'N_all': N_all,
        'delta_x': dx_orig,
        'delta_y': dy_orig,
        'bandwidth_mode': bandwidth_mode,
        'h_x': h_x,
        'h_y': h_y,
        'periodic_x': periodic_x,
        'periodic_extension_n': periodic_extension_n if periodic_x else None,
        'obstacle_applied': obstacle_rect is not None,
        'mass_before_normalization': S,
    }
    
    return rho, H, S, meta


def kde_density_timeseries(
    trajectories: np.ndarray,
    times: np.ndarray,
    domain: Tuple[float, float, float, float],
    nx: int = 80,
    ny: int = 20,
    bandwidth_mode: str = "manual",
    manual_H: Tuple[float, float] = (3.0, 2.0),
    periodic_x: bool = True,
    periodic_extension_n: int = 5,
    obstacle_rect: Optional[Tuple[float, float, float, float]] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Compute KDE density field for entire trajectory time series.
    
    Parameters
    ----------
    trajectories : ndarray, shape (T, N, 2)
        Particle positions over time
    times : ndarray, shape (T,)
        Time points
    domain : tuple of (a, b, c, d)
        Domain boundaries
    nx, ny : int
        Grid resolution
    bandwidth_mode : {"silverman", "manual"}
        Bandwidth selection method
    manual_H : tuple of (h_x, h_y)
        Manual bandwidth
    periodic_x : bool
        Enable periodic boundary handling
    periodic_extension_n : int
        Extension parameter
    obstacle_rect : tuple, optional
        Obstacle rectangle
    verbose : bool
        Print progress
        
    Returns
    -------
    density_fields : ndarray, shape (T, nx, ny)
        Density at each time
    H_array : ndarray, shape (T, 2, 2)
        Bandwidth matrix at each time (varies if Silverman)
    S_array : ndarray, shape (T,)
        Total mass before normalization at each time
    meta_list : list of dict
        Metadata for each snapshot
    """
    T = trajectories.shape[0]
    
    density_fields = np.zeros((T, nx, ny), dtype=float)
    H_array = np.zeros((T, 2, 2), dtype=float)
    S_array = np.zeros(T, dtype=float)
    meta_list = []
    
    for t in range(T):
        if verbose and t % max(1, T // 10) == 0:
            print(f"Computing KDE: {t}/{T} ({100*t//T}%)")
        
        positions = trajectories[t]
        rho, H, S, meta = kde_density_snapshot(
            positions=positions,
            domain=domain,
            nx=nx,
            ny=ny,
            bandwidth_mode=bandwidth_mode,
            manual_H=manual_H,
            periodic_x=periodic_x,
            periodic_extension_n=periodic_extension_n,
            obstacle_rect=obstacle_rect,
        )
        
        density_fields[t] = rho
        H_array[t] = H
        S_array[t] = S
        meta_list.append(meta)
    
    if verbose:
        print(f"Computing KDE: {T}/{T} (100%)")
    
    return density_fields, H_array, S_array, meta_list


__all__ = [
    'kde_density_snapshot',
    'kde_density_timeseries',
    'silverman_bandwidth',
    'gaussian_kernel_2d',
    'augment_positions_periodic',
    'create_obstacle_mask',
]
