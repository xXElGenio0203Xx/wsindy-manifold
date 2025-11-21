"""
Density estimation with KDE for particle trajectories.

Provides standardized KDE interface that returns density movie + metadata.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple, Dict

Array = np.ndarray


def kde_density_movie(
    traj: Array,
    Lx: float,
    Ly: float,
    nx: int,
    ny: int,
    bandwidth: float,
    bc: str = "periodic"
) -> Tuple[Array, Dict]:
    """
    Compute Gaussian-smoothed KDE density movie from particle trajectories.
    
    Args:
        traj: Particle positions (T, N, 2) with (x, y) coordinates
        Lx: Domain width
        Ly: Domain height
        nx: Number of grid points in x
        ny: Number of grid points in y
        bandwidth: Gaussian smoothing bandwidth (in grid units)
        bc: Boundary conditions ('periodic' or 'reflecting')
        
    Returns:
        rho: Density movie (T, ny, nx)  # Note: ny first for image convention
        meta: Metadata dict with 'bandwidth', 'nx', 'ny', 'extent', 'Lx', 'Ly', 'bc'
        
    Example:
        >>> traj = np.random.rand(100, 50, 2) * 30  # 100 frames, 50 particles
        >>> rho, meta = kde_density_movie(traj, Lx=30, Ly=30, nx=50, ny=50, bandwidth=1.5)
        >>> print(rho.shape)  # (100, 50, 50)
        >>> print(meta['extent'])  # [0, 30, 0, 30]
    """
    T, N, d = traj.shape
    
    if d != 2:
        raise ValueError(f"traj must have shape (T, N, 2), got {traj.shape}")
    
    # Create grid edges
    x_edges = np.linspace(0.0, Lx, nx + 1)
    y_edges = np.linspace(0.0, Ly, ny + 1)
    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]
    
    # Mode for Gaussian filter
    mode = "wrap" if bc == "periodic" else "nearest"
    
    # Compute density for each frame
    rho = np.zeros((T, ny, nx))
    
    for t in range(T):
        # 2D histogram
        hist, _, _ = np.histogram2d(
            traj[t, :, 0],  # x coordinates
            traj[t, :, 1],  # y coordinates
            bins=[x_edges, y_edges],
            range=[[0.0, Lx], [0.0, Ly]]
        )
        
        # Normalize to density (particles per unit area)
        density = hist / (dx * dy)
        
        # Apply Gaussian smoothing
        if bandwidth > 0:
            density = gaussian_filter(density, sigma=bandwidth, mode=mode)
        
        # Store (transposed for image convention: y as first axis)
        rho[t] = density.T
    
    # Metadata
    meta = {
        'bandwidth': bandwidth,
        'nx': nx,
        'ny': ny,
        'Lx': Lx,
        'Ly': Ly,
        'extent': [0, Lx, 0, Ly],  # [xmin, xmax, ymin, ymax]
        'bc': bc,
        'N_particles': N,
        'T_frames': T
    }
    
    return rho, meta


def grid_from_meta(meta: Dict) -> Tuple[Array, Array]:
    """
    Create meshgrid from metadata.
    
    Args:
        meta: Metadata dict from kde_density_movie
        
    Returns:
        xx: X coordinates (nx, ny)
        yy: Y coordinates (nx, ny)
    """
    nx, ny = meta['nx'], meta['ny']
    Lx, Ly = meta['Lx'], meta['Ly']
    
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    return xx, yy


def check_density_compatibility(meta1: Dict, meta2: Dict) -> bool:
    """
    Check if two density movies are compatible (same grid).
    
    Args:
        meta1: Metadata from first density movie
        meta2: Metadata from second density movie
        
    Returns:
        compatible: True if grids match
    """
    required_keys = ['nx', 'ny', 'extent']
    
    for key in required_keys:
        if meta1.get(key) != meta2.get(key):
            return False
    
    return True


def estimate_bandwidth(Lx: float, Ly: float, N: int, nx: int, ny: int) -> float:
    """
    Estimate reasonable KDE bandwidth based on problem size.
    
    Rule of thumb: bandwidth ~ (L / sqrt(N)) * (grid_resolution_factor)
    
    Args:
        Lx: Domain width
        Ly: Domain height
        N: Number of particles
        nx: Grid points in x
        ny: Grid points in y
        
    Returns:
        bandwidth: Suggested bandwidth in grid units
    """
    # Average domain size
    L_avg = (Lx + Ly) / 2
    
    # Average grid spacing
    dx = Lx / nx
    dy = Ly / ny
    dx_avg = (dx + dy) / 2
    
    # Scott's rule: h ~ N^(-1/(d+4)) where d=2
    scott_factor = N ** (-1/6)
    
    # Bandwidth in physical units
    h_physical = L_avg * scott_factor * 0.5
    
    # Convert to grid units
    bandwidth = h_physical / dx_avg
    
    # Clamp to reasonable range
    bandwidth = np.clip(bandwidth, 0.5, 5.0)
    
    return bandwidth
