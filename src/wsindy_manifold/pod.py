"""
POD (Proper Orthogonal Decomposition) utilities.

Provides dimensionality reduction via SVD-based POD:
- fit_pod: Extract basis from snapshots
- restrict: Project to latent space
- lift: Reconstruct from latent coordinates
"""
import numpy as np
from typing import Tuple, Optional

Array = np.ndarray


def fit_pod(
    X: Array,
    energy: float = 0.99,
    n_modes: Optional[int] = None
) -> Tuple[Array, Array, int, Array]:
    """
    Fit POD basis from density snapshots using energy threshold or fixed mode count.
    
    Uses economy SVD when T << n_c for efficiency.
    
    Args:
        X: Snapshot matrix (T, n_c) where n_c = nx * ny
        energy: Cumulative energy threshold (0 < energy <= 1), ignored if n_modes is set
        n_modes: Optional fixed number of modes to retain (overrides energy threshold)
        
    Returns:
        Ud: POD basis (n_c, d)
        xbar: Mean snapshot (n_c,)
        d: Number of modes retained
        energy_curve: Cumulative energy array
        
    Example:
        >>> X = np.random.randn(100, 2500)  # 100 snapshots, 50x50 grid
        >>> Ud, xbar, d, curve = fit_pod(X, energy=0.99)
        >>> print(f"Retained {d} modes for 99% energy")
        >>> # Or with fixed modes:
        >>> Ud, xbar, d, curve = fit_pod(X, n_modes=50)
    """
    if X.ndim != 2:
        raise ValueError(f"X must have shape (T, n_c), got {X.shape}")
    
    T, n_c = X.shape
    
    # Compute mean
    xbar = np.mean(X, axis=0)
    
    # Center data
    X_centered = X - xbar
    
    # SVD strategy based on problem size
    if T < n_c:
        # Economy SVD via temporal covariance (for T << n_c)
        # C = (1/T) X_centered @ X_centered.T
        C = (X_centered @ X_centered.T) / T
        eigvals, eigvecs = np.linalg.eigh(C)
        
        # Sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Clip negative eigenvalues (numerical artifacts)
        eigvals = np.maximum(eigvals, 0.0)
        
        # Compute spatial modes: Φ = X_centered.T @ Ψ
        Ud = X_centered.T @ eigvecs  # (n_c, T)
        
        # Normalize
        norms = np.linalg.norm(Ud, axis=0)
        Ud = Ud / (norms + 1e-16)
        
        # Singular values
        sigma = np.sqrt(eigvals * T)
    else:
        # Standard SVD (for T >= n_c)
        U, sigma, Vt = np.linalg.svd(X_centered, full_matrices=False)
        # U has shape (T, min(T, n_c)), but we want spatial modes (n_c, ...)
        # The spatial modes are the columns of X_centered @ V.T / sigma
        # Or equivalently: X_centered.T @ U / sigma, but U is already orthonormal
        # Actually, for X = U @ diag(sigma) @ Vt, the spatial modes are Vt.T
        Ud = Vt.T  # (min(T, n_c), min(T, n_c)) transposed -> (n_c, min(T, n_c))
    
    # Compute energy
    energy_vals = sigma ** 2
    total_energy = np.sum(energy_vals)
    cumulative_energy = np.cumsum(energy_vals) / (total_energy + 1e-16)
    
    # Find d that captures desired energy or use fixed n_modes
    if n_modes is not None:
        d = min(n_modes, len(cumulative_energy))
        print(f"POD: Retained {d}/{len(cumulative_energy)} modes (fixed, {cumulative_energy[d-1]*100:.2f}% energy)")
    else:
        d = np.searchsorted(cumulative_energy, energy) + 1
        d = min(d, len(cumulative_energy))
        print(f"POD: Retained {d}/{len(cumulative_energy)} modes ({cumulative_energy[d-1]*100:.2f}% energy)")
    
    return Ud[:, :d], xbar, d, cumulative_energy


def restrict(X: Array, Ud: Array, xbar: Array) -> Array:
    """
    Restrict (project) full-state snapshots to POD latent space.
    
    y(t) = Ud^T (x(t) - x̄)
    
    Args:
        X: Snapshots (T, n_c)
        Ud: POD basis (n_c, d)
        xbar: Mean snapshot (n_c,)
        
    Returns:
        Y: Latent coordinates (T, d)
        
    Example:
        >>> X_train = np.random.randn(100, 2500)
        >>> Ud, xbar, d, _ = fit_pod(X_train)
        >>> Y_train = restrict(X_train, Ud, xbar)
        >>> print(Y_train.shape)  # (100, d)
    """
    if X.ndim != 2:
        raise ValueError(f"X must have shape (T, n_c), got {X.shape}")
    
    X_centered = X - xbar
    Y = X_centered @ Ud
    
    return Y


def lift(Y: Array, Ud: Array, xbar: Array, preserve_mass: bool = True) -> Array:
    """
    Lift (reconstruct) latent coordinates to full state space.
    
    x̂(t) = Ud y(t) + x̄
    
    Optionally enforces mass conservation by rescaling each frame to match
    the mean mass of xbar.
    
    Args:
        Y: Latent coordinates (T, d)
        Ud: POD basis (n_c, d)
        xbar: Mean snapshot (n_c,)
        preserve_mass: If True, rescale each frame to preserve total mass
        
    Returns:
        X_reconstructed: Reconstructed snapshots (T, n_c)
        
    Example:
        >>> Y_pred = model.forecast(Y_seed, steps=50)
        >>> X_pred = lift(Y_pred, Ud, xbar, preserve_mass=True)
        >>> X_pred_2d = X_pred.reshape(-1, nx, ny)
    """
    if Y.ndim != 2:
        raise ValueError(f"Y must have shape (T, d), got {Y.shape}")
    
    X_reconstructed = Y @ Ud.T + xbar
    
    # Enforce mass conservation if requested
    if preserve_mass:
        target_mass = np.sum(xbar)
        for t in range(X_reconstructed.shape[0]):
            current_mass = np.sum(X_reconstructed[t])
            if current_mass > 1e-16:  # Avoid division by zero
                X_reconstructed[t] *= (target_mass / current_mass)
    
    return X_reconstructed


def pod_compression_ratio(n_c: int, d: int) -> float:
    """
    Compute compression ratio achieved by POD.
    
    Args:
        n_c: Original dimensionality (nx * ny)
        d: Latent dimensionality
        
    Returns:
        ratio: Compression ratio
    """
    return n_c / d


def pod_reconstruction_error(X: Array, Ud: Array, xbar: Array) -> float:
    """
    Compute POD reconstruction error (relative L2).
    
    Args:
        X: Original snapshots (T, n_c)
        Ud: POD basis (n_c, d)
        xbar: Mean snapshot (n_c,)
        
    Returns:
        error: Relative L2 reconstruction error
    """
    Y = restrict(X, Ud, xbar)
    X_recon = lift(Y, Ud, xbar)
    
    diff = X - X_recon
    error = np.linalg.norm(diff, 'fro') / (np.linalg.norm(X, 'fro') + 1e-16)
    
    return float(error)
