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


def nematic_order(velocities, eps=1e-10):
    """
    Compute nematic order parameter Q.
    
    Q-tensor: 𝐐 = (1/N) Σᵢ (𝐧ᵢ ⊗ 𝐧ᵢ - 𝐈/d)
    where 𝐧ᵢ = 𝐯ᵢ/‖𝐯ᵢ‖ (unit heading vectors)
    
    Nematic order: Q = λₘₐₓ(𝐐)
    
    Measures second-order alignment, insensitive to head-tail polarity.
    Critical for detecting bidirectional patterns:
    - High Q with low Φ → Lane formation (bidirectional flow)
    - High Q with high Φ → Polar flocking
    - Low Q → Disordered/isotropic
    
    Parameters
    ----------
    velocities : ndarray, shape (N, 2)
        Velocity vectors
    eps : float, optional
        Small constant to avoid division by zero (default: 1e-10)
        
    Returns
    -------
    float
        Nematic order parameter in [0, 1] (2D)
        
    Notes
    -----
    The Q-tensor is the second moment of the orientation distribution.
    Its largest eigenvalue quantifies alignment along the principal axis,
    regardless of polarity (+/- direction).
    
    References
    ----------
    - Vicsek & Zafeiris, Phys. Rep. 517, 71-140 (2012)
    - Chaté et al., Phys. Rev. E 77, 046113 (2008)
    """
    N, d = velocities.shape
    
    if N == 0:
        return 0.0
    
    if d != 2:
        raise ValueError(f"nematic_order requires 2D velocities, got shape {velocities.shape}")
    
    # Normalize to unit vectors
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    n = velocities / (speeds + eps)
    
    # Compute Q-tensor: (1/N) Σᵢ (nᵢ ⊗ nᵢ) - I/d
    Q = np.zeros((d, d))
    for i in range(N):
        Q += np.outer(n[i], n[i])
    Q = Q / N - np.eye(d) / d
    
    # Nematic order = largest eigenvalue
    eigvals = np.linalg.eigvalsh(Q)
    return float(np.max(eigvals))


def spatial_order(density_field):
    """
    Compute spatial order parameter from density field.
    
    Spatial order quantifies spatial heterogeneity via the standard deviation
    of the density field over all grid cells:
    
    S_spatial = std(ρ(x,y)) = sqrt(Var(ρ))
    
    where ρ is the discretized density field on a 2D grid.
    
    Interpretation:
    - High spatial order: particles clustered in specific regions (strong spatial structure)
    - Low spatial order: uniform spatial distribution (homogeneous)
    
    This metric is particularly useful for ROM evaluation, as it can be computed
    directly from predicted density fields without access to particle trajectories.
    
    Parameters
    ----------
    density_field : ndarray, shape (nx, ny)
        2D density field ρ(x,y) on discretized grid
        
    Returns
    -------
    float
        Standard deviation of density values across spatial grid
        
    Notes
    -----
    This is equivalent to computing:
    
    S = sqrt( (1/(nx*ny)) * Σᵢⱼ (ρᵢⱼ - ρ̄)² )
    
    where ρ̄ = (1/(nx*ny)) * Σᵢⱼ ρᵢⱼ is the spatial mean density.
    
    References
    ----------
    Used for density-based order parameter visualization in ROM validation.
    """
    if density_field.ndim != 2:
        raise ValueError(f"spatial_order requires 2D density field, got shape {density_field.shape}")
    
    return float(np.std(density_field))


def total_mass(positions, domain_bounds, resolution=50,
               bandwidth_mode="manual", manual_H=(3.0, 2.0),
               periodic_x=False, boundary_condition="periodic"):
    """
    Compute total mass of density field (should always be 1.0 for mass-normalized KDE).
    
    NOTE: Currently returns placeholder until kde_density module is implemented.
    
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
        Total mass: ∫∫ ρ(x,y) dx dy (currently placeholder: returns 1.0)
    """
    # TODO: Implement proper KDE mass conservation check
    # For now, return 1.0 placeholder to avoid import errors
    return 1.0


def density_variance(positions, domain_bounds, resolution=50, bandwidth=None,
                    bandwidth_mode="manual", manual_H=(3.0, 2.0), 
                    periodic_x=False, boundary_condition="periodic"):
    """
    Compute variance of density field using KDE.
    
    NOTE: Currently returns placeholder until kde_density module is implemented.
    
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
        Variance of density field (currently placeholder: returns 0.0)
    """
    # TODO: Implement proper KDE density variance
    # For now, return placeholder to avoid import errors
    return 0.0


def compute_all_metrics(positions, velocities, domain_bounds, 
                        resolution=50, bandwidth=None, boundary_condition="periodic",
                        bandwidth_mode="manual", manual_H=(3.0, 2.0),
                        density_field=None):
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
    density_field : ndarray, shape (nx, ny), optional
        Pre-computed density field for spatial_order calculation.
        If None, spatial_order will be None.
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'polarization': Φ(t)
        - 'angular_momentum': L(t)
        - 'mean_speed': ⟨|v|⟩(t)
        - 'nematic_order': Q(t)
        - 'density_variance': Var(ρ)(t)
        - 'total_mass': ∫ρ (should be ≈ 1.0)
        - 'spatial_order': std(ρ) if density_field provided, else None
    """
    metrics = {
        'polarization': polarization(velocities),
        'angular_momentum': angular_momentum(positions, velocities),
        'mean_speed': mean_speed(velocities),
        'nematic_order': nematic_order(velocities),
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
    
    # Add spatial order if density field provided
    if density_field is not None:
        metrics['spatial_order'] = spatial_order(density_field)
    else:
        metrics['spatial_order'] = None
    
    return metrics


def compute_metrics_series(trajectory, velocities, domain_bounds,
                          resolution=50, bandwidth=None, verbose=False,
                          boundary_condition="periodic", bandwidth_mode="manual",
                          manual_H=(3.0, 2.0), density_movie=None):
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
    density_movie : ndarray, shape (T, nx, ny), optional
        Pre-computed density field time series for spatial_order calculation.
        If None, spatial_order will be filled with NaN.
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'polarization': array of shape (T,)
        - 'angular_momentum': array of shape (T,)
        - 'mean_speed': array of shape (T,)
        - 'nematic_order': array of shape (T,)
        - 'density_variance': array of shape (T,)
        - 'total_mass': array of shape (T,)
        - 'spatial_order': array of shape (T,) if density_movie provided, else NaN
    """
    T = len(trajectory)
    
    metrics = {
        'polarization': np.zeros(T),
        'angular_momentum': np.zeros(T),
        'mean_speed': np.zeros(T),
        'nematic_order': np.zeros(T),
        'density_variance': np.zeros(T),
        'total_mass': np.zeros(T),
        'spatial_order': np.full(T, np.nan)
    }
    
    for t in range(T):
        if verbose and t % max(1, T // 10) == 0:
            print(f"Computing metrics: {t}/{T} ({100*t//T}%)")
        
        # Get density field for this timestep if available
        density_t = density_movie[t] if density_movie is not None else None
        
        frame_metrics = compute_all_metrics(
            trajectory[t], velocities[t], domain_bounds,
            resolution, bandwidth, boundary_condition=boundary_condition,
            bandwidth_mode=bandwidth_mode, manual_H=manual_H,
            density_field=density_t
        )
        
        for key in metrics:
            val = frame_metrics[key]
            metrics[key][t] = val if val is not None else np.nan
    
    if verbose:
        print(f"Computing metrics: {T}/{T} (100%)")
    
    return metrics
