"""
Operator library columns for weak-form regression (placeholders).
"""
from __future__ import annotations
import numpy as np

def diffusion_column(grad_rho: np.ndarray, grad_phi: np.ndarray) -> float:
    """Weak integral contribution for diffusion term: ∫ ∇ρ · ∇φ dμ."""
    return float(np.tensordot(grad_rho, grad_phi, axes=grad_rho.ndim-1))

def transport_column(rho: np.ndarray, U: np.ndarray, grad_phi: np.ndarray) -> float:
    """Weak integral for transport term: ∫ ρ U · ∇φ dμ."""
    return float(np.tensordot(rho * U, grad_phi, axes=U.ndim-1))
