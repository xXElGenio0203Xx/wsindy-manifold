"""
Stability selection for WSINDy library terms.

Implements the stability selection framework (Meinshausen & Bühlmann, 2010)
adapted to WSINDy: across multiple bootstrap / sub-sample replicates,
track how frequently each library term is included in the active set.

This provides a complementary view to :mod:`uncertainty`:

* **Uncertainty** → coefficient distributions (mean, std, CI).
* **Stability** → selection frequency tables across ℓ scales **and/or**
  bootstrap replicates, identifying robust vs. fragile terms.

Public API
----------
* :func:`stability_selection` — run stability sweep and return a
  :class:`StabilityResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .grid import GridSpec
from .test_functions import make_separable_psi
from .system import build_weak_system, default_t_margin, make_query_indices
from .fit import wsindy_fit_regression
from .model import WSINDyModel


@dataclass
class StabilityResult:
    """Stability selection results.

    Attributes
    ----------
    freq : ndarray, shape ``(M,)``
        Selection frequency for each library term (fraction of trials
        in which it was active).
    freq_matrix : ndarray, shape ``(n_configs, M)``
        Per-configuration binary inclusion matrix (row = trial,
        column = term).
    col_names : list[str]
        Library term names (length ``M``).
    config_labels : list[str]
        Human-readable label for each row of ``freq_matrix``.
    threshold : float
        Frequency threshold used to flag robust terms.
    """

    freq: NDArray[np.floating]
    freq_matrix: NDArray[np.floating]
    col_names: List[str]
    config_labels: List[str]
    threshold: float

    @property
    def robust_terms(self) -> List[str]:
        """Terms whose selection frequency exceeds the threshold."""
        return [
            n for n, f in zip(self.col_names, self.freq)
            if f >= self.threshold
        ]

    @property
    def fragile_terms(self) -> List[str]:
        """Active terms that are NOT robust (0 < freq < threshold)."""
        return [
            n for n, f in zip(self.col_names, self.freq)
            if 0 < f < self.threshold
        ]

    def summary(self) -> str:
        """Human-readable stability table."""
        lines = [
            f"Stability selection ({len(self.config_labels)} trials, "
            f"threshold={self.threshold:.2f})",
            f"{'Term':>14s}  {'Freq':>6s}  {'Status':>8s}",
        ]
        order = np.argsort(-self.freq)
        for i in order:
            if self.freq[i] < 1e-10:
                continue
            status = "robust" if self.freq[i] >= self.threshold else "fragile"
            lines.append(
                f"{self.col_names[i]:>14s}  "
                f"{self.freq[i]:6.3f}  "
                f"{status:>8s}"
            )
        n_never = int(np.sum(self.freq < 1e-10))
        if n_never > 0:
            lines.append(f"  ({n_never} terms never selected)")
        return "\n".join(lines)


def stability_selection(
    U: NDArray[np.floating],
    grid: GridSpec,
    library_terms: List[Tuple[str, str]],
    *,
    ell_grid: Optional[Sequence[Tuple[int, int, int]]] = None,
    p: Tuple[int, int, int] = (2, 2, 2),
    stride: Tuple[int, int, int] = (1, 1, 1),
    n_bootstrap: int = 0,
    subsample_frac: float = 0.5,
    lambdas: Optional[NDArray[np.floating]] = None,
    max_iter: int = 25,
    tol: float = 0.0,
    threshold: float = 0.6,
    rng: Optional[np.random.Generator] = None,
) -> StabilityResult:
    """Run stability selection across ℓ scales and/or bootstrap resamples.

    Parameters
    ----------
    U : ndarray, shape ``(T, nx, ny)``
        Spatiotemporal field data.
    grid : GridSpec
    library_terms : list of ``(op, feature)`` pairs
    ell_grid : sequence of ``(ellt, ellx, elly)`` tuples, optional
        Test-function scales to sweep.  If ``None``, a single default
        scale is used (ℓ = 3 in each dimension).
    p : (pt, px, py)
        Polynomial orders for test functions.
    stride : (stride_t, stride_x, stride_y)
        Query-point strides.
    n_bootstrap : int
        Number of additional bootstrap replicates **per ℓ**.  If 0,
        only the ℓ sweep contributes (one fit per ℓ value).
    subsample_frac : float
        Fraction of rows to subsample for bootstrap replicates.
    lambdas : ndarray or None
    max_iter : int
    tol : float
    threshold : float
        Minimum selection frequency for "robust" classification.
    rng : numpy Generator or None

    Returns
    -------
    StabilityResult
    """
    U = np.asarray(U, dtype=np.float64)
    T, nx, ny = U.shape
    M = len(library_terms)

    if rng is None:
        rng = np.random.default_rng()

    if ell_grid is None:
        ell_grid = [(3, 3, 3)]

    inclusion_rows: List[NDArray[np.bool_]] = []
    labels: List[str] = []

    for ell in ell_grid:
        ellt, ellx, elly = ell
        pt, px, py = p
        psi_bundle = make_separable_psi(grid, ellt, ellx, elly, pt, px, py)
        t_margin = default_t_margin(psi_bundle)
        query_idx = make_query_indices(
            T, nx, ny, stride[0], stride[1], stride[2], t_margin,
        )

        b, G, col_names = build_weak_system(
            U, grid, psi_bundle, library_terms, query_idx,
        )

        # --- Full-data fit for this ℓ ---
        model = wsindy_fit_regression(
            b, G, col_names, lambdas=lambdas, max_iter=max_iter, tol=tol,
        )
        inclusion_rows.append(model.active.copy())
        labels.append(f"ell=({ellt},{ellx},{elly})")

        # --- Bootstrap replicates ---
        K = b.shape[0]
        n_sub = max(1, int(K * subsample_frac))
        for rep in range(n_bootstrap):
            idx = rng.choice(K, size=n_sub, replace=False)
            try:
                m = wsindy_fit_regression(
                    b[idx], G[idx, :], col_names,
                    lambdas=lambdas, max_iter=max_iter, tol=tol,
                )
                inclusion_rows.append(m.active.copy())
            except Exception:
                inclusion_rows.append(np.zeros(M, dtype=bool))
            labels.append(f"ell=({ellt},{ellx},{elly})/boot{rep}")

    # Aggregate
    freq_matrix = np.array(inclusion_rows, dtype=np.float64)
    freq = np.mean(freq_matrix, axis=0)

    return StabilityResult(
        freq=freq,
        freq_matrix=freq_matrix,
        col_names=list(col_names) if 'col_names' in dir() else [f"{o}:{f}" for o, f in library_terms],
        config_labels=labels,
        threshold=threshold,
    )
