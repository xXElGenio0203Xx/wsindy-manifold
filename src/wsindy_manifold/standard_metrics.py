"""
Standard metrics for collective dynamics and model evaluation.

Provides:
- Order parameters (polarization, mean speed, speed std, nematic order)
- Relative error metrics (L1, L2, L∞, RMSE, mass error)
- Tolerance horizon computation
"""
import numpy as np
from typing import Dict, Optional

Array = np.ndarray


# ============================================================================
# Order Parameters (for simulations)
# ============================================================================

def polarization(vel: Array, eps: float = 1e-10) -> float:
    """
    Compute polarization order parameter Φ(t).
    
    Φ = (1/N) || Σᵢ vᵢ/||vᵢ|| ||
    
    Args:
        vel: Velocities (N, 2) or (N, d)
        eps: Small constant to avoid division by zero
        
    Returns:
        phi: Polarization in [0, 1]
    """
    if vel.ndim != 2:
        raise ValueError(f"vel must have shape (N, d), got {vel.shape}")
    
    N = vel.shape[0]
    if N == 0:
        return 0.0
    
    # Normalize velocities
    speeds = np.linalg.norm(vel, axis=1, keepdims=True)
    normalized = vel / (speeds + eps)
    
    # Sum and compute magnitude
    mean_direction = np.mean(normalized, axis=0)
    phi = np.linalg.norm(mean_direction)
    
    return float(phi)


def mean_speed(vel: Array) -> float:
    """
    Compute mean speed.
    
    Args:
        vel: Velocities (N, 2) or (N, d)
        
    Returns:
        v_mean: Mean speed
    """
    speeds = np.linalg.norm(vel, axis=1)
    return float(np.mean(speeds))


def speed_std(vel: Array) -> float:
    """
    Compute standard deviation of speeds.
    
    Args:
        vel: Velocities (N, 2) or (N, d)
        
    Returns:
        v_std: Speed standard deviation
    """
    speeds = np.linalg.norm(vel, axis=1)
    return float(np.std(speeds))


def nematic_order(vel: Array, eps: float = 1e-10) -> float:
    """
    Compute nematic order parameter (2nd moment of headings).
    
    Q = max eigenvalue of (1/N) Σᵢ (nᵢ ⊗ nᵢ - I/d)
    where nᵢ = vᵢ/||vᵢ||
    
    Args:
        vel: Velocities (N, 2)
        eps: Small constant
        
    Returns:
        q: Nematic order in [0, 1]
    """
    if vel.ndim != 2 or vel.shape[1] != 2:
        raise ValueError(f"nematic_order requires (N, 2) velocities, got {vel.shape}")
    
    N, d = vel.shape
    if N == 0:
        return 0.0
    
    # Normalize
    speeds = np.linalg.norm(vel, axis=1, keepdims=True)
    n = vel / (speeds + eps)
    
    # Compute Q tensor
    Q = np.zeros((d, d))
    for i in range(N):
        Q += np.outer(n[i], n[i])
    Q = Q / N - np.eye(d) / d
    
    # Max eigenvalue
    eigvals = np.linalg.eigvalsh(Q)
    q = float(np.max(eigvals))
    
    return q


def compute_order_params(
    vel: Array,
    include_nematic: bool = False
) -> Dict[str, float]:
    """
    Compute all order parameters for a velocity snapshot.
    
    Args:
        vel: Velocities (N, 2) or (N, d)
        include_nematic: Whether to compute nematic order
        
    Returns:
        params: Dictionary with 'phi', 'mean_speed', 'speed_std', 'nematic' (optional)
    """
    params = {
        'phi': polarization(vel),
        'mean_speed': mean_speed(vel),
        'speed_std': speed_std(vel)
    }
    
    if include_nematic and vel.shape[1] == 2:
        params['nematic'] = nematic_order(vel)
    
    return params


# ============================================================================
# Relative Error Metrics (for model evaluation)
# ============================================================================

def rel_errors(X_pred: Array, X_true: Array) -> Dict[str, Array]:
    """
    Compute relative error metrics over time.
    
    Args:
        X_pred: Predicted states (T, n) or (T, nx, ny)
        X_true: True states (T, n) or (T, nx, ny)
        
    Returns:
        metrics: Dictionary with 'e1', 'e2', 'einf', 'rmse', 'mass_error' arrays (T,)
    """
    # Flatten spatial dimensions if needed
    if X_pred.ndim == 3:
        T, nx, ny = X_pred.shape
        X_pred = X_pred.reshape(T, nx * ny)
        X_true = X_true.reshape(T, nx * ny)
    
    T, n = X_pred.shape
    
    # Compute errors
    diff = X_pred - X_true
    
    # L1 relative error
    e1 = np.linalg.norm(diff, ord=1, axis=1) / (np.linalg.norm(X_true, ord=1, axis=1) + 1e-16)
    
    # L2 relative error
    e2 = np.linalg.norm(diff, ord=2, axis=1) / (np.linalg.norm(X_true, ord=2, axis=1) + 1e-16)
    
    # L∞ relative error
    einf = np.max(np.abs(diff), axis=1) / (np.max(np.abs(X_true), axis=1) + 1e-16)
    
    # RMSE
    rmse = np.sqrt(np.mean(diff ** 2, axis=1))
    
    # Mass error (for density fields)
    mass_pred = np.sum(X_pred, axis=1)
    mass_true = np.sum(X_true, axis=1)
    mass_error = np.abs(mass_pred - mass_true) / (mass_true + 1e-16)
    
    return {
        'e1': e1,
        'e2': e2,
        'einf': einf,
        'rmse': rmse,
        'mass_error': mass_error
    }


def tolerance_horizon(
    e2: Array,
    threshold: float = 0.10,
    window: int = 5
) -> Optional[int]:
    """
    Compute tolerance horizon τ_tol.
    
    First time where rolling mean of L² error exceeds threshold.
    
    Args:
        e2: Relative L² error timeseries (T,)
        threshold: Tolerance threshold (default 10%)
        window: Rolling window size for smoothing
        
    Returns:
        tau_tol: Frame index where error exceeds threshold, or None if never
    """
    if len(e2) < window:
        return None
    
    # Compute rolling mean
    rolling_mean = np.convolve(e2, np.ones(window) / window, mode='valid')
    
    # Find first crossing
    crossings = np.where(rolling_mean > threshold)[0]
    
    if len(crossings) == 0:
        return None
    
    return int(crossings[0])


def r2_score(X_pred: Array, X_true: Array) -> float:
    """
    Compute R² score.
    
    R² = 1 - Σ||x̂ - x||² / Σ||x - x̄||²
    
    Args:
        X_pred: Predicted states (T, n)
        X_true: True states (T, n)
        
    Returns:
        r2: R² score
    """
    # Flatten if needed
    if X_pred.ndim == 3:
        X_pred = X_pred.reshape(X_pred.shape[0], -1)
        X_true = X_true.reshape(X_true.shape[0], -1)
    
    # Residual sum of squares
    ss_res = np.sum((X_pred - X_true) ** 2)
    
    # Total sum of squares
    x_mean = np.mean(X_true, axis=0)
    ss_tot = np.sum((X_true - x_mean) ** 2)
    
    r2 = 1 - ss_res / (ss_tot + 1e-16)
    
    return float(r2)


def compute_summary_metrics(
    X_pred: Array,
    X_true: Array,
    threshold: float = 0.10
) -> Dict[str, float]:
    """
    Compute summary metrics for model evaluation.
    
    Args:
        X_pred: Predicted states (T, n) or (T, nx, ny)
        X_true: True states (T, n) or (T, nx, ny)
        threshold: Tolerance threshold for horizon
        
    Returns:
        summary: Dictionary with r2, median_e2, p10_e2, p90_e2, tau_tol, etc.
    """
    # Compute frame-wise errors
    errors = rel_errors(X_pred, X_true)
    
    # Aggregate metrics
    summary = {
        'r2': r2_score(X_pred, X_true),
        'median_e1': float(np.median(errors['e1'])),
        'median_e2': float(np.median(errors['e2'])),
        'median_einf': float(np.median(errors['einf'])),
        'p10_e2': float(np.percentile(errors['e2'], 10)),
        'p90_e2': float(np.percentile(errors['e2'], 90)),
        'mean_rmse': float(np.mean(errors['rmse'])),
        'mean_mass_error': float(np.mean(errors['mass_error'])),
        'max_mass_error': float(np.max(errors['mass_error'])),
    }
    
    # Tolerance horizon
    tau = tolerance_horizon(errors['e2'], threshold=threshold)
    summary['tau_tol'] = int(tau) if tau is not None else None
    
    return summary


# ============================================================================
# Mass Conservation Check
# ============================================================================

def check_mass_conservation(
    densities: Array,
    rtol: float = 5e-3,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Check mass conservation in density evolution.
    
    Args:
        densities: Density snapshots (T, nx, ny) or (T, n)
        rtol: Relative tolerance
        verbose: Print diagnostics
        
    Returns:
        stats: Dictionary with initial_mass, final_mass, drift, max_drift
    """
    # Compute total mass at each timestep
    if densities.ndim == 3:
        mass = np.sum(densities, axis=(1, 2))
    else:
        mass = np.sum(densities, axis=1)
    
    initial_mass = mass[0]
    final_mass = mass[-1]
    
    # Relative drift
    drift = np.abs(mass - initial_mass) / (initial_mass + 1e-16)
    max_drift = float(np.max(drift))
    final_drift = float(drift[-1])
    
    stats = {
        'initial_mass': float(initial_mass),
        'final_mass': float(final_mass),
        'final_drift': final_drift,
        'max_drift': max_drift,
        'within_tolerance': max_drift < rtol
    }
    
    if verbose:
        print(f"Mass conservation check:")
        print(f"  Initial mass: {initial_mass:.6f}")
        print(f"  Final mass:   {final_mass:.6f}")
        print(f"  Max drift:    {max_drift*100:.3f}%")
        
        if stats['within_tolerance']:
            print(f"  ✓ Within tolerance ({rtol*100:.2f}%)")
        else:
            print(f"  ✗ Exceeds tolerance ({rtol*100:.2f}%)")
    
    return stats
