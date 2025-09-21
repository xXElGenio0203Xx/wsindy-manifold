"""
Assemble the weak-form linear system A θ ≈ b across time slabs.
"""
from __future__ import annotations
import numpy as np

def assemble_system(rho_t: np.ndarray, grad_rho_t: np.ndarray, grad_phi: np.ndarray):
    """Placeholder to show shapes and stacking strategy.
    rho_t:    (K × G)  density frames on grid
    grad_rho_t: (K × G × d) gradients
    grad_phi: (M × G × d) gradients of test functions
    """
    K, G = rho_t.shape[0], rho_t.shape[1]
    M = grad_phi.shape[0]
    # b via backward difference:
    b = (rho_t[1:].sum(axis=1) - rho_t[:-1].sum(axis=1))  # crude placeholder
    b = np.repeat(b[:, None], M, axis=1).reshape(-1)      # stack per test function
    # A with two columns as example
    A = np.random.randn((K-1) * M, 2)  # replace with proper weak integrals
    return A, b
