"""
Local charts, exp/log maps, and metric/volume utilities.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class ManifoldChart:
    """Minimal chart interface: forward/backward maps and metric tensor.
    This is a placeholder; replace with concrete implementations for S^2, T^2, etc.
    """
    def to_coords(self, x_world) -> np.ndarray:
        raise NotImplementedError

    def from_coords(self, u: np.ndarray):
        raise NotImplementedError

    def metric(self, u: np.ndarray) -> np.ndarray:
        """Return g(u) ∈ R^{d×d}."""
        raise NotImplementedError

def exp_map(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Chart-based approximation of exp_u(v). Placeholder."""
    return u + v

def log_map(u: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Chart-based approximation of log_u(x). Placeholder."""
    return x - u
