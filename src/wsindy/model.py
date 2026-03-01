"""
WSINDy model container.

Stores the fitted sparse model: coefficients, active mask, preconditioning
information, and fit diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
from numpy.typing import NDArray


@dataclass
class WSINDyModel:
    """Fitted WSINDy sparse regression model.

    Attributes
    ----------
    col_names : list[str]
        Human-readable term labels, e.g. ``["I:u", "dxx:u", "lap:u"]``.
    w : ndarray, shape ``(M,)``
        Unscaled (physical) coefficients.  Inactive terms have ``w[i]=0``.
    active : ndarray, shape ``(M,)``, dtype ``bool``
        Which library terms survived thresholding.
    best_lambda : float
        Regularisation parameter that achieved the lowest normalised loss.
    col_scale : ndarray, shape ``(M,)``
        Column norms used during preconditioning.
        ``w_unscaled = w_scaled / col_scale``.
    diagnostics : dict
        Per-lambda history, final residual, R², etc.
    """

    col_names: List[str]
    w: NDArray[np.floating]
    active: NDArray[np.bool_]
    best_lambda: float
    col_scale: NDArray[np.floating]
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # ── convenience ──────────────────────────────────────────────────────

    @property
    def n_active(self) -> int:
        """Number of active (non-zero) terms."""
        return int(np.sum(self.active))

    @property
    def active_terms(self) -> List[str]:
        """Names of active terms."""
        return [n for n, a in zip(self.col_names, self.active) if a]

    @property
    def active_coeffs(self) -> NDArray[np.floating]:
        """Coefficients of active terms only."""
        return self.w[self.active]

    def summary(self) -> str:
        """One-line-per-term human-readable summary."""
        lines = [
            f"WSINDyModel  (λ*={self.best_lambda:.4e}, "
            f"{self.n_active}/{len(self.w)} active)",
        ]
        # Sort active terms by |coeff| descending
        order = np.argsort(-np.abs(self.w))
        for i in order:
            if not self.active[i]:
                continue
            lines.append(f"  {self.col_names[i]:>12s}  {self.w[i]:+.6e}")
        diag = self.diagnostics
        if "r2" in diag:
            lines.append(f"  R²(weak) = {diag['r2']:.6f}")
        if "normalised_loss" in diag:
            lines.append(f"  L(w)     = {diag['normalised_loss']:.6e}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return self.summary()
