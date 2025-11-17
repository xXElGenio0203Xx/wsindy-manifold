"""Initial condition generation for ensemble simulations.

This module provides utilities to generate varied initial particle
configurations for large ensemble runs. It supports multiple IC types
inspired by the Alvarez et al. autoregressive ROM paper:
- uniform: particles uniformly distributed in the domain
- gaussian: particles in a Gaussian blob
- ring: particles arranged in a ring structure
- cluster: particles in multiple Gaussian clusters

These ICs provide diversity in the ensemble dataset for POD + MVAR training.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

ICType = Literal["uniform", "gaussian", "ring", "cluster"]


def sample_initial_positions(
    ic_type: str,
    N: int,
    Lx: float,
    Ly: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate initial particle positions according to the specified distribution.

    Parameters
    ----------
    ic_type : str
        Type of initial condition. One of:
        - "uniform": uniform distribution in [0,Lx]×[0,Ly]
        - "gaussian": single Gaussian blob centered randomly
        - "ring": particles arranged around a ring
        - "cluster": multiple Gaussian clusters
    N : int
        Number of particles.
    Lx : float
        Domain width.
    Ly : float
        Domain height.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    positions : np.ndarray, shape (N, 2)
        Initial positions in [0,Lx]×[0,Ly].

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> pos = sample_initial_positions("uniform", 100, 20.0, 20.0, rng)
    >>> pos.shape
    (100, 2)
    >>> np.all((pos >= 0) & (pos <= 20))
    True
    """
    if ic_type == "uniform":
        return _uniform_ic(N, Lx, Ly, rng)
    elif ic_type == "gaussian":
        return _gaussian_blob_ic(N, Lx, Ly, rng)
    elif ic_type == "ring":
        return _ring_ic(N, Lx, Ly, rng)
    elif ic_type == "cluster":
        return _cluster_ic(N, Lx, Ly, rng)
    else:
        raise ValueError(
            f"Unknown ic_type '{ic_type}'. "
            f"Supported types: uniform, gaussian, ring, cluster"
        )


def _uniform_ic(N: int, Lx: float, Ly: float, rng: np.random.Generator) -> np.ndarray:
    """Uniform distribution in [0,Lx]×[0,Ly]."""
    return rng.uniform(low=[0.0, 0.0], high=[Lx, Ly], size=(N, 2))


def _gaussian_blob_ic(
    N: int,
    Lx: float,
    Ly: float,
    rng: np.random.Generator,
    sigma_factor: float = 0.1,
) -> np.ndarray:
    """Single Gaussian blob centered at a random location.

    Parameters
    ----------
    N : int
        Number of particles.
    Lx, Ly : float
        Domain dimensions.
    rng : np.random.Generator
        Random number generator.
    sigma_factor : float, optional
        Standard deviation as a fraction of min(Lx, Ly). Default is 0.1.

    Returns
    -------
    positions : np.ndarray, shape (N, 2)
        Particle positions clipped to [0,Lx]×[0,Ly].
    """
    # Choose a random center, avoiding edges
    margin = 0.2
    center_x = rng.uniform(margin * Lx, (1 - margin) * Lx)
    center_y = rng.uniform(margin * Ly, (1 - margin) * Ly)
    center = np.array([center_x, center_y])

    # Standard deviation based on domain size
    sigma = sigma_factor * min(Lx, Ly)

    # Sample from 2D Gaussian
    positions = rng.normal(loc=center, scale=sigma, size=(N, 2))

    # Clip to domain (simpler than wrapping for non-periodic domains)
    positions[:, 0] = np.clip(positions[:, 0], 0.0, Lx)
    positions[:, 1] = np.clip(positions[:, 1], 0.0, Ly)

    return positions


def _ring_ic(
    N: int,
    Lx: float,
    Ly: float,
    rng: np.random.Generator,
    radius_factor: float = 0.3,
    radial_noise: float = 0.05,
) -> np.ndarray:
    """Particles arranged around a ring with small radial noise.

    Parameters
    ----------
    N : int
        Number of particles.
    Lx, Ly : float
        Domain dimensions.
    rng : np.random.Generator
        Random number generator.
    radius_factor : float, optional
        Ring radius as a fraction of min(Lx, Ly). Default is 0.3.
    radial_noise : float, optional
        Standard deviation of radial noise as a fraction of the radius. Default is 0.05.

    Returns
    -------
    positions : np.ndarray, shape (N, 2)
        Particle positions.
    """
    # Center of the domain
    center = np.array([Lx / 2, Ly / 2])

    # Ring radius
    radius = radius_factor * min(Lx, Ly)

    # Angular positions uniformly distributed
    angles = rng.uniform(0.0, 2 * np.pi, size=N)

    # Add radial noise
    radii = radius + rng.normal(0.0, radial_noise * radius, size=N)

    # Convert to Cartesian coordinates
    positions = np.empty((N, 2))
    positions[:, 0] = center[0] + radii * np.cos(angles)
    positions[:, 1] = center[1] + radii * np.sin(angles)

    # Wrap or clip to domain
    positions[:, 0] = np.clip(positions[:, 0], 0.0, Lx)
    positions[:, 1] = np.clip(positions[:, 1], 0.0, Ly)

    return positions


def _cluster_ic(
    N: int,
    Lx: float,
    Ly: float,
    rng: np.random.Generator,
    n_clusters: int | None = None,
    sigma_factor: float = 0.08,
) -> np.ndarray:
    """Multiple Gaussian clusters distributed in the domain.

    Parameters
    ----------
    N : int
        Number of particles.
    Lx, Ly : float
        Domain dimensions.
    rng : np.random.Generator
        Random number generator.
    n_clusters : int or None, optional
        Number of clusters. If None, randomly choose 2-4 clusters.
    sigma_factor : float, optional
        Standard deviation of each cluster as a fraction of min(Lx, Ly). Default is 0.08.

    Returns
    -------
    positions : np.ndarray, shape (N, 2)
        Particle positions.
    """
    # Choose number of clusters
    if n_clusters is None:
        n_clusters = rng.integers(2, 5)  # 2, 3, or 4 clusters

    # Generate random cluster centers, avoiding edges
    margin = 0.15
    centers = np.empty((n_clusters, 2))
    centers[:, 0] = rng.uniform(margin * Lx, (1 - margin) * Lx, size=n_clusters)
    centers[:, 1] = rng.uniform(margin * Ly, (1 - margin) * Ly, size=n_clusters)

    # Assign particles to clusters (roughly equal proportions)
    # Use Dirichlet distribution for varied proportions
    proportions = rng.dirichlet(np.ones(n_clusters))
    cluster_sizes = np.round(proportions * N).astype(int)

    # Adjust to ensure total is exactly N
    diff = N - cluster_sizes.sum()
    if diff > 0:
        # Add remaining particles to random clusters
        idx = rng.choice(n_clusters, size=diff, replace=True)
        for i in idx:
            cluster_sizes[i] += 1
    elif diff < 0:
        # Remove particles from largest clusters
        for _ in range(-diff):
            largest = np.argmax(cluster_sizes)
            cluster_sizes[largest] -= 1

    # Generate particle positions for each cluster
    sigma = sigma_factor * min(Lx, Ly)
    positions = []

    for k in range(n_clusters):
        if cluster_sizes[k] == 0:
            continue
        cluster_pos = rng.normal(
            loc=centers[k],
            scale=sigma,
            size=(cluster_sizes[k], 2),
        )
        positions.append(cluster_pos)

    positions = np.vstack(positions)

    # Clip to domain
    positions[:, 0] = np.clip(positions[:, 0], 0.0, Lx)
    positions[:, 1] = np.clip(positions[:, 1], 0.0, Ly)

    # Shuffle to randomize cluster assignment order
    rng.shuffle(positions)

    return positions
