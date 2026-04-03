"""
Automated model selection for WSINDy across test-function scales.

WSINDy's accuracy depends on the test-function half-widths
:math:`\\ell = (\\ell_t, \\ell_x, \\ell_y)`.  Too narrow → noisy
convolutions; too wide → smears out spatial structure.

This module provides :func:`wsindy_model_selection`, which:

1. Sweeps a grid of :math:`\\ell` values.
2. Fits a WSINDy model (Parts 2–3) for each.
3. Scores models by a composite criterion (fit quality + sparsity
   + optional stability).
4. Returns a diagnostics table, the Pareto frontier, and the
   recommended best model.

The recommended **composite score** penalises both poor fit and
over-complexity::

    score  =  normalised_loss  +  α · (n_active / M)
           +  β · condition_penalty

Lower is better.  The Pareto frontier is computed over the
(normalised_loss, n_active) plane.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .grid import GridSpec
from .test_functions import make_separable_psi
from .system import (
    build_weak_system,
    default_t_margin,
    fft_convolve3d_same,
    make_query_indices,
    nondimensionalize_field,
    rescale_coefficients,
)
from .fit import wsindy_fit_regression
from .model import WSINDyModel


# ═══════════════════════════════════════════════════════════════════
#  Result containers
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TrialResult:
    """Diagnostics for one (ℓ, stride, p) configuration."""

    ell: Tuple[int, int, int]
    p: Tuple[int, int, int]
    stride: Tuple[int, int, int]
    model: WSINDyModel
    n_query: int
    normalised_loss: float
    r2_weak: float
    n_active: int
    best_lambda: float
    condition_number: float
    composite_score: float
    elapsed_s: float

    def row_dict(self) -> Dict[str, Any]:
        """Flat dict for tabular display / DataFrame conversion."""
        return {
            "ellt": self.ell[0],
            "ellx": self.ell[1],
            "elly": self.ell[2],
            "pt": self.p[0],
            "px": self.p[1],
            "py": self.p[2],
            "stride_t": self.stride[0],
            "stride_x": self.stride[1],
            "stride_y": self.stride[2],
            "n_query": self.n_query,
            "n_active": self.n_active,
            "normalised_loss": self.normalised_loss,
            "r2_weak": self.r2_weak,
            "best_lambda": self.best_lambda,
            "cond": self.condition_number,
            "score": self.composite_score,
            "elapsed_s": self.elapsed_s,
            "active_terms": self.model.active_terms,
            "active_coeffs": list(self.model.active_coeffs),
        }


@dataclass
class SelectionResult:
    """Output of :func:`wsindy_model_selection`."""

    trials: List[TrialResult]
    best: TrialResult
    pareto: List[TrialResult]

    # ── convenience ──────────────────────────────────────────────────
    @property
    def best_model(self) -> WSINDyModel:
        return self.best.model

    def table(self) -> List[Dict[str, Any]]:
        """List-of-dicts for all trials (e.g. for ``pd.DataFrame``)."""
        return [t.row_dict() for t in self.trials]

    def summary(self, top_k: int = 5) -> str:
        """Human-readable ranking."""
        lines = [
            f"WSINDy model selection: {len(self.trials)} trials, "
            f"{len(self.pareto)} Pareto-optimal",
            "",
        ]
        ranked = sorted(self.trials, key=lambda t: t.composite_score)
        for i, t in enumerate(ranked[:top_k]):
            tag = " ★" if t is self.best else ""
            lines.append(
                f"  #{i + 1}  ℓ=({t.ell[0]},{t.ell[1]},{t.ell[2]})  "
                f"active={t.n_active}  loss={t.normalised_loss:.4e}  "
                f"score={t.composite_score:.4e}  "
                f"λ*={t.best_lambda:.2e}{tag}"
            )
        if len(ranked) > top_k:
            lines.append(f"  ... ({len(ranked) - top_k} more)")
        lines.append("")
        lines.append(f"  Best: ℓ=({self.best.ell[0]},"
                     f"{self.best.ell[1]},{self.best.ell[2]})  "
                     f"active={self.best.n_active}")
        lines.append(self.best.model.summary())
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  Pareto frontier
# ═══════════════════════════════════════════════════════════════════

def _pareto_frontier(
    trials: List[TrialResult],
) -> List[TrialResult]:
    """2-D Pareto frontier over (normalised_loss, n_active).

    A trial dominates another if it is no worse on both objectives
    and strictly better on at least one.
    """
    if not trials:
        return []

    # Sort by loss ascending, break ties by sparsity
    by_loss = sorted(trials, key=lambda t: (t.normalised_loss, t.n_active))
    frontier: List[TrialResult] = []
    best_nact = float("inf")

    for t in by_loss:
        if t.n_active < best_nact:
            frontier.append(t)
            best_nact = t.n_active

    return frontier


# ═══════════════════════════════════════════════════════════════════
#  Spectral validation of test-function smoothing
# ═══════════════════════════════════════════════════════════════════

def spectral_fidelity(
    U: NDArray[np.floating],
    psi_bundle: dict,
    grid: GridSpec,
    *,
    energy_fraction: float = 0.95,
) -> float:
    """Measure how much of U's spectral energy is preserved after ψ-convolution.

    Computes ``ψ * U`` via FFT convolution and compares the spatial
    power spectra.  A return value near 1.0 means the test function
    is *not* over-smoothing the data; values significantly below 1.0
    indicate that the test-function support is too wide and is
    destroying high-frequency signal content.

    The check is:
        r = ‖F[ψ*U]‖₂ / ‖F[U] · F[ψ]_max‖₂

    normalised so that a flat test function (no smoothing) gives r ≈ 1.

    Parameters
    ----------
    U : ndarray (T, nx, ny)
    psi_bundle : dict from ``make_separable_psi``
    grid : GridSpec
    energy_fraction : float
        Fraction of total spectral energy to consider (top modes).

    Returns
    -------
    fidelity : float in [0, 1]
        1.0 = perfect preservation, < 0.9 suggests over-smoothing.
    """
    psi_kernel = np.asarray(psi_bundle["psi"], dtype=np.float64)
    periodic = (grid.periodic_time, grid.periodic_space, grid.periodic_space)

    # Convolve
    smoothed = fft_convolve3d_same(U, psi_kernel, periodic)

    # Compare spatial power spectra (averaged over time)
    T_len = U.shape[0]
    numer = 0.0
    denom = 0.0
    for t in range(T_len):
        P_orig = np.abs(np.fft.fft2(U[t])) ** 2
        P_smooth = np.abs(np.fft.fft2(smoothed[t])) ** 2

        # Sort modes by original power (descending) and keep top fraction
        flat_orig = P_orig.ravel()
        flat_smooth = P_smooth.ravel()
        order = np.argsort(flat_orig)[::-1]
        cumulative = np.cumsum(flat_orig[order])
        total = cumulative[-1]
        if total < 1e-30:
            continue
        cutoff = np.searchsorted(cumulative, energy_fraction * total) + 1
        idx = order[:cutoff]

        numer += np.sum(flat_smooth[idx])
        denom += np.sum(flat_orig[idx])

    if denom < 1e-30:
        return 1.0
    return float(numer / denom)


# ═══════════════════════════════════════════════════════════════════
#  Composite score
# ═══════════════════════════════════════════════════════════════════

def _composite_score(
    normalised_loss: float,
    n_active: int,
    n_library: int,
    condition_number: float,
    alpha: float,
    beta: float,
    cond_threshold: float,
) -> float:
    r"""Compute composite model-selection score (lower is better).

    .. math::
        S = L_{\rm norm} + \alpha \frac{n_{\rm active}}{M}
          + \beta \max\!\bigl(0,\, \log_{10}(\kappa) - \log_{10}(\kappa_0)\bigr)

    Parameters
    ----------
    normalised_loss : float
        ‖b − Gw‖₂ / ‖b_LS‖₂  (from MSTLS).
    n_active : int
    n_library : int  (M)
    condition_number : float  (κ of active sub-matrix)
    alpha : float  (complexity weight, default 0.1)
    beta : float   (ill-conditioning penalty, default 0.01)
    cond_threshold : float  (κ₀, penalty kicks in above this)
    """
    complexity = alpha * n_active / max(n_library, 1)
    log_cond = np.log10(max(condition_number, 1.0))
    log_thresh = np.log10(max(cond_threshold, 1.0))
    cond_pen = beta * max(0.0, log_cond - log_thresh)
    return normalised_loss + complexity + cond_pen


# ═══════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════

def wsindy_model_selection(
    U: NDArray[np.floating],
    grid: GridSpec,
    library_terms: List[Tuple[str, str]],
    ell_grid: Sequence[Tuple[int, int, int]],
    *,
    p: Tuple[int, int, int] = (3, 5, 5),
    lambdas: Optional[NDArray[np.floating]] = None,
    stride: Tuple[int, int, int] = (2, 2, 2),
    stride_grid: Optional[Sequence[Tuple[int, int, int]]] = None,
    alpha: float = 0.1,
    beta: float = 0.01,
    cond_threshold: float = 1e8,
    max_iter: int = 25,
    verbose: bool = True,
) -> SelectionResult:
    r"""Sweep test-function scale :math:`\ell` and return ranked models.

    Parameters
    ----------
    U : ndarray (T, nx, ny)
        Spatiotemporal data.
    grid : GridSpec
    library_terms : list of (op, feature) tuples
        E.g. ``[("I","u"), ("lap","u"), ("I","u2")]``.
    ell_grid : sequence of (ellt, ellx, elly) tuples
        Test-function half-widths to sweep.
    p : (pt, px, py)
        Polynomial exponents for the bump function (fixed across sweep).
    lambdas : ndarray or None
        Threshold candidates (shared across all trials).
    stride : (st, sx, sy)
        Default query stride (used if *stride_grid* is None).
    stride_grid : sequence of stride tuples, or None
        If provided, also sweep strides (outer product with *ell_grid*).
    alpha : float
        Complexity weight in composite score (default 0.1).
    beta : float
        Conditioning penalty weight (default 0.01).
    cond_threshold : float
        Condition number above which penalty kicks in (default 1e8).
    max_iter : int
        Max MSTLS iterations per λ.
    verbose : bool
        Print progress.

    Returns
    -------
    SelectionResult
        ``.best`` — best trial, ``.pareto`` — Pareto frontier,
        ``.trials`` — all trials with full diagnostics.
    """
    T_data, _nx, _ny = U.shape

    # ── Nondimensionalize field data ────────────────────────────────
    U_nd, U_c = nondimensionalize_field(U)
    if verbose:
        print(f"  Nondimensionalization: U_c = {U_c:.4e}")

    if lambdas is None:
        lambdas = np.logspace(-4, 1, 40)

    strides = list(stride_grid) if stride_grid is not None else [stride]

    # ── build (ℓ, stride) configurations ────────────────────────────
    configs: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = []
    for ell in ell_grid:
        for s in strides:
            configs.append((tuple(ell), tuple(s)))  # type: ignore[arg-type]

    n_library = len(library_terms)
    trials: List[TrialResult] = []

    for idx, (ell, st) in enumerate(configs):
        ellt, ellx, elly = ell
        st_t, st_x, st_y = st

        t0 = time.perf_counter()

        # ── 1. Build ψ bundle ─────────────────────────────────────
        try:
            psi_bundle = make_separable_psi(
                grid,
                ellt=ellt, ellx=ellx, elly=elly,
                pt=p[0], px=p[1], py=p[2],
            )
        except Exception as exc:
            if verbose:
                print(f"  [{idx + 1}/{len(configs)}] "
                      f"ℓ=({ellt},{ellx},{elly}) SKIP (ψ build: {exc})")
            continue

        # ── 1b. Spectral fidelity check ──────────────────────────
        fidelity = spectral_fidelity(U_nd, psi_bundle, grid)
        if fidelity < 0.5:
            if verbose:
                print(f"  [{idx + 1}/{len(configs)}] "
                      f"ℓ=({ellt},{ellx},{elly}) SKIP "
                      f"(spectral fidelity={fidelity:.3f} < 0.5)")
            continue

        # ── 2. Query indices ──────────────────────────────────────
        t_margin = default_t_margin(psi_bundle)
        if 2 * t_margin >= T_data:
            if verbose:
                print(f"  [{idx + 1}/{len(configs)}] "
                      f"ℓ=({ellt},{ellx},{elly}) SKIP (t_margin too large)")
            continue

        query_idx = make_query_indices(
            T_data, _nx, _ny,
            stride_t=st_t, stride_x=st_x, stride_y=st_y,
            t_margin=t_margin,
        )
        n_query = query_idx.shape[0]
        if n_query < n_library + 1:
            if verbose:
                print(f"  [{idx + 1}/{len(configs)}] "
                      f"ℓ=({ellt},{ellx},{elly}) SKIP (too few query pts)")
            continue

        # ── 3. Build weak system (on nondimensionalized field) ────
        b, G, col_names = build_weak_system(
            U_nd, grid, psi_bundle, library_terms, query_idx,
        )

        # ── 4. Fit ────────────────────────────────────────────────
        model = wsindy_fit_regression(
            b, G, col_names, lambdas=lambdas, max_iter=max_iter,
        )

        # ── 4b. Rescale coefficients to physical units ────────────
        model.w = rescale_coefficients(model.w, col_names, U_c)

        elapsed = time.perf_counter() - t0

        # ── 5. Extract diagnostics ────────────────────────────────
        diag = model.diagnostics
        nloss = diag.get("normalised_loss", float("inf"))
        r2w = diag.get("r2", 0.0)

        # Condition number of active sub-matrix
        active_cols = G[:, model.active]
        if active_cols.shape[1] > 0:
            cond = float(np.linalg.cond(active_cols))
        else:
            cond = float("inf")

        score = _composite_score(
            nloss, model.n_active, n_library, cond,
            alpha, beta, cond_threshold,
        )

        trial = TrialResult(
            ell=ell,
            p=p,
            stride=st,
            model=model,
            n_query=n_query,
            normalised_loss=nloss,
            r2_weak=r2w,
            n_active=model.n_active,
            best_lambda=model.best_lambda,
            condition_number=cond,
            composite_score=score,
            elapsed_s=elapsed,
        )
        trials.append(trial)

        if verbose:
            print(
                f"  [{idx + 1}/{len(configs)}] "
                f"ℓ=({ellt},{ellx},{elly}) s=({st_t},{st_x},{st_y})  "
                f"active={model.n_active}  loss={nloss:.4e}  "
                f"score={score:.4e}  fidelity={fidelity:.3f}  "
                f"({elapsed:.2f}s)"
            )

    # ── Rank ──────────────────────────────────────────────────────
    if not trials:
        raise RuntimeError(
            "No valid trial completed. Check ell_grid values "
            "and data dimensions."
        )

    ranked = sorted(trials, key=lambda t: t.composite_score)
    best = ranked[0]
    pareto = _pareto_frontier(trials)

    return SelectionResult(trials=trials, best=best, pareto=pareto)


# ═══════════════════════════════════════════════════════════════════
#  Convenience: default ℓ grid
# ═══════════════════════════════════════════════════════════════════

def default_ell_grid(
    T: int,
    nx: int,
    ny: int,
    *,
    n_points: int = 5,
) -> List[Tuple[int, int, int]]:
    """Generate a reasonable default sweep grid for :math:`\\ell`.

    Returns ``n_points`` logarithmically-spaced configurations where
    each :math:`\\ell_i` ranges from ``_MIN_ELL`` to roughly 1/4 of the
    axis length, enforcing a minimum of 7 grid points per test function
    (the literature requires ≥ 6).

    Parameters
    ----------
    T, nx, ny : int
        Data dimensions.
    n_points : int
        Number of grid points.

    Returns
    -------
    list of (ellt, ellx, elly) tuples with integer half-widths.
    """
    _MIN_ELL = 3  # ell=3 ⇒ 2*3+1 = 7 gridpoints ≥ 6 (paper minimum)
    max_t = max(T // 4, _MIN_ELL)
    max_x = max(nx // 4, _MIN_ELL)
    max_y = max(ny // 4, _MIN_ELL)

    ells_t = np.unique(
        np.round(np.logspace(np.log10(_MIN_ELL), np.log10(max_t), n_points))
    ).astype(int)
    ells_x = np.unique(
        np.round(np.logspace(np.log10(_MIN_ELL), np.log10(max_x), n_points))
    ).astype(int)
    ells_y = np.unique(
        np.round(np.logspace(np.log10(_MIN_ELL), np.log10(max_y), n_points))
    ).astype(int)

    # Pair them up (matching indices, not full outer product —
    # that would be n³ which is excessive).
    n = min(len(ells_t), len(ells_x), len(ells_y))
    grid = []
    for i in range(n):
        it = min(i, len(ells_t) - 1)
        ix = min(i, len(ells_x) - 1)
        iy = min(i, len(ells_y) - 1)
        grid.append((int(ells_t[it]), int(ells_x[ix]), int(ells_y[iy])))

    # Deduplicate
    return list(dict.fromkeys(grid))
