"""Initial condition generators for particle positions and velocities.

This module provides functions to initialize particle configurations
with different spatial distributions and velocity patterns.
"""

import numpy as np
from typing import Tuple


def uniform_distribution(N: int, Lx: float, Ly: float, rng: np.random.Generator) -> np.ndarray:
    """Generate uniformly distributed particle positions.
    
    Parameters
    ----------
    N : int
        Number of particles
    Lx : float
        Domain width
    Ly : float
        Domain height
    rng : np.random.Generator
        Random number generator
        
    Returns
    -------
    positions : ndarray, shape (N, 2)
        Particle positions uniformly distributed in [0, Lx] × [0, Ly]
    """
    return rng.uniform(low=[0.0, 0.0], high=[Lx, Ly], size=(N, 2))


def gaussian_cluster(N: int, Lx: float, Ly: float, rng: np.random.Generator,
                     center: Tuple[float, float] = None, sigma: float = None) -> np.ndarray:
    """Generate particle positions in a Gaussian cluster.
    
    Parameters
    ----------
    N : int
        Number of particles
    Lx : float
        Domain width
    Ly : float
        Domain height
    rng : np.random.Generator
        Random number generator
    center : tuple of (float, float), optional
        Cluster center (cx, cy). Defaults to domain center (Lx/2, Ly/2)
    sigma : float, optional
        Standard deviation of cluster. Defaults to min(Lx, Ly) / 6
        
    Returns
    -------
    positions : ndarray, shape (N, 2)
        Particle positions sampled from 2D Gaussian, wrapped to domain
    """
    if center is None:
        center = (Lx / 2, Ly / 2)
    if sigma is None:
        sigma = min(Lx, Ly) / 6
    
    # Sample from 2D Gaussian
    positions = rng.normal(loc=center, scale=sigma, size=(N, 2))
    
    # Wrap to domain (periodic-like wrapping)
    positions[:, 0] = positions[:, 0] % Lx
    positions[:, 1] = positions[:, 1] % Ly
    
    return positions


def two_clusters(N: int, Lx: float, Ly: float, rng: np.random.Generator,
                 separation: float = None, sigma: float = None) -> np.ndarray:
    """Generate particle positions in two separated Gaussian clusters.
    
    Parameters
    ----------
    N : int
        Number of particles (will be split approximately equally)
    Lx : float
        Domain width
    Ly : float
        Domain height
    rng : np.random.Generator
        Random number generator
    separation : float, optional
        Distance between cluster centers. Defaults to Lx / 3
    sigma : float, optional
        Standard deviation of each cluster. Defaults to min(Lx, Ly) / 8
        
    Returns
    -------
    positions : ndarray, shape (N, 2)
        Particle positions in two clusters
    """
    if separation is None:
        separation = Lx / 3
    if sigma is None:
        sigma = min(Lx, Ly) / 8
    
    # Split particles between two clusters
    N1 = N // 2
    N2 = N - N1
    
    # Cluster centers (horizontally separated)
    center1 = (Lx / 2 - separation / 2, Ly / 2)
    center2 = (Lx / 2 + separation / 2, Ly / 2)
    
    # Generate positions for each cluster
    cluster1 = rng.normal(loc=center1, scale=sigma, size=(N1, 2))
    cluster2 = rng.normal(loc=center2, scale=sigma, size=(N2, 2))
    
    # Combine and wrap to domain
    positions = np.vstack([cluster1, cluster2])
    positions[:, 0] = positions[:, 0] % Lx
    positions[:, 1] = positions[:, 1] % Ly
    
    return positions


def ring_distribution(N: int, Lx: float, Ly: float, rng: np.random.Generator,
                      radius: float = None, width: float = None) -> np.ndarray:
    """Generate particle positions on a ring/annulus.
    
    Parameters
    ----------
    N : int
        Number of particles
    Lx : float
        Domain width
    Ly : float
        Domain height
    rng : np.random.Generator
        Random number generator
    radius : float, optional
        Mean radius of ring. Defaults to min(Lx, Ly) / 4
    width : float, optional
        Width of ring (radial spread). Defaults to radius / 10
        
    Returns
    -------
    positions : ndarray, shape (N, 2)
        Particle positions on ring centered in domain
    """
    if radius is None:
        radius = min(Lx, Ly) / 4
    if width is None:
        width = radius / 10
    
    # Random angles
    theta = rng.uniform(0, 2 * np.pi, size=N)
    
    # Radii with small variation
    r = rng.normal(loc=radius, scale=width, size=N)
    r = np.abs(r)  # Ensure positive
    
    # Convert to Cartesian, centered in domain
    cx, cy = Lx / 2, Ly / 2
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    
    positions = np.column_stack([x, y])
    
    # Wrap to domain
    positions[:, 0] = positions[:, 0] % Lx
    positions[:, 1] = positions[:, 1] % Ly
    
    return positions


def initialize_positions(distribution: str, N: int, Lx: float, Ly: float,
                        rng: np.random.Generator, **kwargs) -> np.ndarray:
    """Generate initial particle positions based on distribution type.
    
    Parameters
    ----------
    distribution : str
        Distribution type: 'uniform', 'gaussian_cluster', 'two_clusters', 'ring'
    N : int
        Number of particles
    Lx : float
        Domain width
    Ly : float
        Domain height
    rng : np.random.Generator
        Random number generator
    **kwargs : dict
        Additional parameters for specific distributions
        
    Returns
    -------
    positions : ndarray, shape (N, 2)
        Initial particle positions
        
    Raises
    ------
    ValueError
        If distribution type is not recognized
    """
    distributions = {
        'uniform': uniform_distribution,
        'gaussian_cluster': gaussian_cluster,
        'gaussian': gaussian_cluster,  # Alias
        'two_clusters': two_clusters,
        'bimodal': two_clusters,  # Alias
        'ring': ring_distribution,
        'annulus': ring_distribution,  # Alias
    }
    
    if distribution not in distributions:
        valid = ', '.join(distributions.keys())
        raise ValueError(
            f"Unknown distribution '{distribution}'. "
            f"Valid options: {valid}"
        )
    
    return distributions[distribution](N, Lx, Ly, rng, **kwargs)


def initialize_velocities(N: int, v0: float, rng: np.random.Generator,
                         distribution: str = 'random') -> np.ndarray:
    """Generate initial particle velocities.
    
    Parameters
    ----------
    N : int
        Number of particles
    v0 : float
        Speed magnitude
    rng : np.random.Generator
        Random number generator
    distribution : str, optional
        Velocity distribution type:
        - 'random': Random directions with magnitude v0
        - 'aligned': All particles moving in same direction
        - 'counter_rotating': Two groups rotating opposite directions
        
    Returns
    -------
    velocities : ndarray, shape (N, 2)
        Initial particle velocities
    """
    if distribution == 'random':
        # Random angles
        theta = rng.uniform(0, 2 * np.pi, size=N)
        velocities = v0 * np.column_stack([np.cos(theta), np.sin(theta)])
        
    elif distribution == 'aligned':
        # All particles moving right
        velocities = np.tile([v0, 0.0], (N, 1))
        
    elif distribution == 'counter_rotating':
        # First half rotating CCW, second half CW
        N1 = N // 2
        N2 = N - N1
        theta1 = rng.uniform(0, 2 * np.pi, size=N1)
        theta2 = rng.uniform(0, 2 * np.pi, size=N2)
        
        v1 = v0 * np.column_stack([np.cos(theta1), np.sin(theta1)])
        v2 = v0 * np.column_stack([np.cos(theta2), np.sin(theta2)])
        velocities = np.vstack([v1, v2])
        
    else:
        raise ValueError(
            f"Unknown velocity distribution '{distribution}'. "
            f"Valid options: 'random', 'aligned', 'counter_rotating'"
        )
    
    return velocities


__all__ = [
    'uniform_distribution',
    'gaussian_cluster',
    'two_clusters',
    'ring_distribution',
    'initialize_positions',
    'initialize_velocities',
]
