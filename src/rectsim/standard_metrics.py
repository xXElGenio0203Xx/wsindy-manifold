"""
Standardized order parameters and metrics for all Vicsek-type simulations.

This module provides unified metric computation functions that work across:
- Discrete Vicsek model
- Continuous (RK-integrated) Vicsek model
- Force-coupled (Morse/D'Orsogna) extensions

All functions accept (x, v) arrays and return scalar metrics.
"""

import numpy as np
from scipy.stats import gaussian_kde


def polarization(velocities):
    """
    Compute polarization order parameter Φ.
    
    Φ = ||⟨v̂ᵢ⟩|| where v̂ᵢ = vᵢ/|vᵢ|
    
    Measures global alignment of velocity directions.
    Range: [0, 1]
    - Φ = 0: completely disordered
    - Φ = 1: perfect alignment
    
    Parameters
    ----------
    velocities : ndarray, shape (N, 2)
        Velocity vectors for N particles
        
    Returns
    -------
    float
        Polarization order parameter in [0, 1]
        
    Notes
    -----
    For particles with zero velocity, we skip them in the average.
    If all particles have zero velocity, returns 0.0.
    """
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    nonzero = speeds.ravel() > 1e-10
    
    if not np.any(nonzero):
        return 0.0
    
    # Normalize velocities to unit vectors
    headings = np.zeros_like(velocities)
    headings[nonzero] = velocities[nonzero] / speeds[nonzero]
    
    # Average heading vector and its magnitude
    mean_heading = np.mean(headings[nonzero], axis=0)
    return float(np.linalg.norm(mean_heading))


def angular_momentum(positions, velocities, center=None):
    """
    Compute angular momentum order parameter L.
    
    L = |∑ᵢ rᵢ × vᵢ| / (N ⟨|r|⟩ ⟨|v|⟩)
    
    Measures collective rotation around the center of mass.
    
    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        Position vectors
    velocities : ndarray, shape (N, 2)
        Velocity vectors
    center : ndarray, shape (2,), optional
        Center point for angular momentum calculation.
        If None, uses center of mass.
        
    Returns
    -------
    float
        Normalized angular momentum
        
    Notes
    -----
    The cross product in 2D is: r × v = rx*vy - ry*vx (scalar)
    We sum all cross products and normalize by N, mean radius, and mean speed.
    """
    N = len(positions)
    
    if center is None:
        center = np.mean(positions, axis=0)
    
    # Relative positions from center
    r = positions - center
    
    # 2D cross product (z-component)
    cross = r[:, 0] * velocities[:, 1] - r[:, 1] * velocities[:, 0]
    total_L = np.abs(np.sum(cross))
    
    # Normalization
    mean_r = np.mean(np.linalg.norm(r, axis=1))
    mean_v = np.mean(np.linalg.norm(velocities, axis=1))
    
    if mean_r < 1e-10 or mean_v < 1e-10:
        return 0.0
    
    return float(total_L / (N * mean_r * mean_v))


def mean_speed(velocities):
    """
    Compute mean speed of all particles.
    
    ⟨|v|⟩ = (1/N) ∑ᵢ |vᵢ|
    
    Parameters
    ----------
    velocities : ndarray, shape (N, 2)
        Velocity vectors
        
    Returns
    -------
    float
        Mean speed
    """
    speeds = np.linalg.norm(velocities, axis=1)
    return float(np.mean(speeds))


def total_mass(positions, domain_bounds, resolution=50,
               bandwidth_mode="manual", manual_H=(3.0, 2.0),
               periodic_x=False, boundary_condition="periodic"):
    """
    Compute total mass of density field (should always be 1.0 for mass-normalized KDE).
    
    This order parameter verifies mass conservation over time. For properly
    mass-normalized density fields from KDE, this should remain constant at 1.0.
    
    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        Particle positions
    domain_bounds : tuple of (xmin, xmax, ymin, ymax)
        Domain boundaries for KDE evaluation
    resolution : int, optional
        Grid resolution for KDE evaluation (default: 50)
    bandwidth_mode : str, optional
        "silverman" or "manual" (default: "manual")
    manual_H : tuple, optional
        Manual bandwidth (h_x, h_y) when bandwidth_mode="manual" (default: (3.0, 2.0))
    periodic_x : bool, optional
        Apply periodic boundary handling in x-direction (default: False)
    boundary_condition : str, optional
        "periodic" or "reflecting" - used to determine if periodic_x should be True
        
    Returns
    -------
    float
        Total mass: ∫∫ ρ(x,y) dx dy (should be ≈ 1.0)
        
    Notes
    -----
    Uses paper-accurate KDE which enforces ∫ρ = 1 via two-step normalization.
    This metric serves as a validation check for mass conservation.
    
    References: Alvarez et al. (2025), Algorithm C.1
    """
    from .kde_density import kde_density_snapshot
    
    xmin, xmax, ymin, ymax = domain_bounds
    
    # Determine if we should use periodic x handling
    if boundary_condition == "periodic":
        periodic_x = True
    
    # Compute KDE using paper algorithm
    try:
        rho, H, S, meta = kde_density_snapshot(
            positions=positions,
            domain=(xmin, xmax, ymin, ymax),
            nx=resolution,
            ny=resolution,
            bandwidth_mode=bandwidth_mode,
            manual_H=manual_H,
            periodic_x=periodic_x,
            periodic_extension_n=5,
            obstacle_rect=None
        )
        
        # Compute total mass: integrate over grid
        # For uniform grid: ∫ρ ≈ sum(rho) * δx * δy
        delta_x = (xmax - xmin) / resolution
        delta_y = (ymax - ymin) / resolution
        total = float(np.sum(rho) * delta_x * delta_y)
        
        return total
    
    except (np.linalg.LinAlgError, ValueError) as e:
        # KDE can fail if all particles are at the same location
        # Return 0 to indicate failure (should trigger investigation)
        return 0.0


def density_variance(positions, domain_bounds, resolution=50, bandwidth=None,
                    bandwidth_mode="manual", manual_H=(3.0, 2.0), 
                    periodic_x=False, boundary_condition="periodic"):
    """
    Compute variance of density field using paper-accurate KDE (Alvarez et al., 2025).
    
    Measures spatial clustering/heterogeneity:
    - High variance: particles clustered in specific regions
    - Low variance: uniform spatial distribution
    
    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        Particle positions
    domain_bounds : tuple of (xmin, xmax, ymin, ymax)
        Domain boundaries for KDE evaluation
    resolution : int, optional
        Grid resolution for KDE evaluation (default: 50)
    bandwidth : float, optional
        Deprecated. Use bandwidth_mode and manual_H instead.
    bandwidth_mode : str, optional
        "silverman" or "manual" (default: "manual")
    manual_H : tuple, optional
        Manual bandwidth (h_x, h_y) when bandwidth_mode="manual" (default: (3.0, 2.0))
    periodic_x : bool, optional
        Apply periodic boundary handling in x-direction (default: False)
    boundary_condition : str, optional
        "periodic" or "reflecting" - used to determine if periodic_x should be True
        
    Returns
    -------
    float
        Variance of density field
        
    Notes
    -----
    Uses paper-accurate KDE with:
    - Proper periodic boundary handling via ghost particles
    - Silverman or manual bandwidth selection
    - Two-step normalization (ensures ∫ρ = 1)
    - Obstacle masking support
    
    References: Alvarez et al. (2025), Algorithm C.1
    """
    from .kde_density import kde_density_snapshot
    
    xmin, xmax, ymin, ymax = domain_bounds
    
    # Determine if we should use periodic x handling
    if boundary_condition == "periodic":
        periodic_x = True
    
    # Compute KDE using paper algorithm
    try:
        rho, H, S, meta = kde_density_snapshot(
            positions=positions,
            domain=(xmin, xmax, ymin, ymax),
            nx=resolution,
            ny=resolution,
            bandwidth_mode=bandwidth_mode,
            manual_H=manual_H,
            periodic_x=periodic_x,
            periodic_extension_n=5,
            obstacle_rect=None
        )
        
        # Return variance of density field
        return float(np.var(rho))
    
    except (np.linalg.LinAlgError, ValueError) as e:
        # KDE can fail if all particles are at the same location
        # or if there are too few particles
        return 0.0


def compute_all_metrics(positions, velocities, domain_bounds, 
                        resolution=50, bandwidth=None, boundary_condition="periodic",
                        bandwidth_mode="manual", manual_H=(3.0, 2.0)):
    """
    Compute all standard order parameters for a single frame.
    
    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        Particle positions
    velocities : ndarray, shape (N, 2)
        Particle velocities
    domain_bounds : tuple of (xmin, xmax, ymin, ymax)
        Domain boundaries
    resolution : int, optional
        Grid resolution for density variance (default: 50)
    bandwidth : float, optional
        Deprecated. Use bandwidth_mode and manual_H instead.
    boundary_condition : str, optional
        "periodic" or "reflecting" (for paper-accurate KDE)
    bandwidth_mode : str, optional
        "silverman" or "manual" (default: "manual")
    manual_H : tuple, optional
        Manual bandwidth (h_x, h_y) when bandwidth_mode="manual"
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'polarization': Φ(t)
        - 'angular_momentum': L(t)
        - 'mean_speed': ⟨|v|⟩(t)
        - 'density_variance': Var(ρ)(t)
        - 'total_mass': ∫ρ (should be ≈ 1.0)
    """
    return {
        'polarization': polarization(velocities),
        'angular_momentum': angular_momentum(positions, velocities),
        'mean_speed': mean_speed(velocities),
        'density_variance': density_variance(positions, domain_bounds, 
                                            resolution, bandwidth,
                                            bandwidth_mode=bandwidth_mode,
                                            manual_H=manual_H,
                                            boundary_condition=boundary_condition),
        'total_mass': total_mass(positions, domain_bounds,
                                resolution,
                                bandwidth_mode=bandwidth_mode,
                                manual_H=manual_H,
                                boundary_condition=boundary_condition)
    }


def compute_metrics_series(trajectory, velocities, domain_bounds,
                          resolution=50, bandwidth=None, verbose=False,
                          boundary_condition="periodic", bandwidth_mode="manual",
                          manual_H=(3.0, 2.0)):
    """
    Compute time series of all metrics for entire simulation.
    
    Parameters
    ----------
    trajectory : ndarray, shape (T, N, 2)
        Positions over time
    velocities : ndarray, shape (T, N, 2)
        Velocities over time
    domain_bounds : tuple of (xmin, xmax, ymin, ymax)
        Domain boundaries
    resolution : int, optional
        Grid resolution for density variance
    bandwidth : float, optional
        Deprecated. Use bandwidth_mode and manual_H instead.
    verbose : bool, optional
        If True, print progress
    boundary_condition : str, optional
        "periodic" or "reflecting" (for paper-accurate KDE)
    bandwidth_mode : str, optional
        "silverman" or "manual" (default: "manual")
    manual_H : tuple, optional
        Manual bandwidth (h_x, h_y) when bandwidth_mode="manual"
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'polarization': array of shape (T,)
        - 'angular_momentum': array of shape (T,)
        - 'mean_speed': array of shape (T,)
        - 'density_variance': array of shape (T,)
        - 'total_mass': array of shape (T,)
    """
    T = len(trajectory)
    
    metrics = {
        'polarization': np.zeros(T),
        'angular_momentum': np.zeros(T),
        'mean_speed': np.zeros(T),
        'density_variance': np.zeros(T),
        'total_mass': np.zeros(T)
    }
    
    for t in range(T):
        if verbose and t % max(1, T // 10) == 0:
            print(f"Computing metrics: {t}/{T} ({100*t//T}%)")
        
        frame_metrics = compute_all_metrics(
            trajectory[t], velocities[t], domain_bounds,
            resolution, bandwidth, boundary_condition=boundary_condition,
            bandwidth_mode=bandwidth_mode, manual_H=manual_H
        )
        
        for key in metrics:
            metrics[key][t] = frame_metrics[key]
    
    if verbose:
        print(f"Computing metrics: {T}/{T} (100%)")
    
    return metrics
