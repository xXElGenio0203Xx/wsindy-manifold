"""
Bootstrap uncertainty quantification for WSINDy coefficients.

Resamples the query-point rows of the weak system ``b = G w``, refits
MSTLS for each resample, and collects coefficient distributions.

This implements the UQ strategy from Messenger & Bortz (2021):
bootstrap over the spatial/temporal query points that define the
weak-form linear system.

Public API
----------
* :func:`bootstrap_wsindy` — run B bootstrap replicates and return
  a :class:`BootstrapResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .fit import wsindy_fit_regression
from .model import WSINDyModel


@dataclass
class BootstrapResult:
    """Bootstrap coefficient distributions.

    Attributes
    ----------
    coeff_samples : ndarray, shape ``(B, M)``
        Coefficient vector for each replicate (zeros for inactive terms).
    active_counts : ndarray, shape ``(M,)``
        How many replicates included each term.
    col_names : list[str]
        Library term names (length ``M``).
    base_model : WSINDyModel
        Model fitted on the full dataset.
    B : int
        Number of bootstrap replicates.
    """

    coeff_samples: NDArray[np.floating]
    active_counts: NDArray[np.intp]
    col_names: List[str]
    base_model: WSINDyModel
    B: int

    # ── derived statistics ─────────────────────────────────────────

    @property
    def coeff_mean(self) -> NDArray[np.floating]:
        """Mean coefficient across replicates."""
        return np.mean(self.coeff_samples, axis=0)

    @property
    def coeff_std(self) -> NDArray[np.floating]:
        """Standard deviation of coefficients across replicates."""
        return np.std(self.coeff_samples, axis=0)

    @property
    def inclusion_probability(self) -> NDArray[np.floating]:
        """Fraction of replicates in which each term was active."""
        return self.active_counts.astype(np.float64) / self.B

    def confidence_interval(
        self, alpha: float = 0.05,
    ) -> NDArray[np.floating]:
        """Per-term ``[lo, hi]`` confidence interval.

        Parameters
        ----------
        alpha : float
            Two-tailed significance level (default 5 %).

        Returns
        -------
        CI : ndarray, shape ``(M, 2)``
        """
        lo = np.percentile(self.coeff_samples, 100 * alpha / 2, axis=0)
        hi = np.percentile(self.coeff_samples, 100 * (1 - alpha / 2), axis=0)
        return np.column_stack([lo, hi])

    def summary(self, alpha: float = 0.05) -> str:
        """Human-readable table of coefficient statistics."""
        ci = self.confidence_interval(alpha)
        lines = [
            f"Bootstrap UQ ({self.B} replicates, "
            f"{100 * (1 - alpha):.0f}% CI)",
            f"{'Term':>14s}  {'Mean':>12s}  {'Std':>12s}  "
            f"{'CI_lo':>12s}  {'CI_hi':>12s}  {'P(active)':>9s}",
        ]
        order = np.argsort(-self.inclusion_probability)
        for i in order:
            if self.inclusion_probability[i] < 1e-10:
                continue
            lines.append(
                f"{self.col_names[i]:>14s}  "
                f"{self.coeff_mean[i]:+12.4e}  "
                f"{self.coeff_std[i]:12.4e}  "
                f"{ci[i, 0]:+12.4e}  "
                f"{ci[i, 1]:+12.4e}  "
                f"{self.inclusion_probability[i]:9.3f}"
            )
        return "\n".join(lines)


def bootstrap_wsindy(
    b: NDArray[np.floating],
    G: NDArray[np.floating],
    col_names: List[str],
    B: int = 100,
    *,
    lambdas: Optional[NDArray[np.floating]] = None,
    max_iter: int = 25,
    tol: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    replace: bool = True,
    subsample_frac: float = 1.0,
) -> BootstrapResult:
    """Bootstrap uncertainty quantification for WSINDy.

    Parameters
    ----------
    b : ndarray, shape ``(K,)``
        Weak-form LHS at query points.
    G : ndarray, shape ``(K, M)``
        Weak-form RHS matrix.
    col_names : list[str]
        Library term labels (length M).
    B : int
        Number of bootstrap replicates.
    lambdas : ndarray or None
        Lambda grid for MSTLS.
    max_iter : int
        Max MSTLS iterations.
    tol : float
        MSTLS convergence tolerance.
    rng : numpy Generator or None
        Random number generator (for reproducibility).
    replace : bool
        Sample with replacement (True = standard bootstrap).
    subsample_frac : float
        Fraction of rows to sample (default 1.0 = same size as original).

    Returns
    -------
    BootstrapResult
    """
    b = np.asarray(b, dtype=np.float64).ravel()
    G = np.asarray(G, dtype=np.float64)
    K, M = G.shape
    if rng is None:
        rng = np.random.default_rng()

    n_sample = max(1, int(K * subsample_frac))

    # Fit base model on full dataset
    base_model = wsindy_fit_regression(
        b, G, col_names, lambdas=lambdas, max_iter=max_iter, tol=tol,
    )

    coeff_samples = np.zeros((B, M), dtype=np.float64)
    active_counts = np.zeros(M, dtype=np.intp)

    for rep in range(B):
        idx = rng.choice(K, size=n_sample, replace=replace)
        try:
            m = wsindy_fit_regression(
                b[idx], G[idx, :], col_names,
                lambdas=lambdas, max_iter=max_iter, tol=tol,
            )
            coeff_samples[rep, :] = m.w
            active_counts += m.active.astype(np.intp)
        except Exception:
            # Degenerate resample → leave zeros
            pass

    return BootstrapResult(
        coeff_samples=coeff_samples,
        active_counts=active_counts,
        col_names=list(col_names),
        base_model=base_model,
        B=B,
    )
