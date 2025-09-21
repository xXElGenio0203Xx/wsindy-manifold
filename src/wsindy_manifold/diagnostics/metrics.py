"""
Error metrics between density fields.
"""
from __future__ import annotations
import numpy as np

def l2_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.sqrt(a.size + 1e-12))
