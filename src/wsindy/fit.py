"""
High-level fitting interface for WSINDy sparse regression.

Orchestrates column preconditioning → MSTLS → coefficient un-scaling
and packages the result into a :class:`WSINDyModel`.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from .metrics import r2_score, wsindy_fit_metrics
from .model import WSINDyModel
from .regression import mstls, precondition_columns


def wsindy_fit_regression(
    b: NDArray[np.floating],
    G: NDArray[np.floating],
    col_names: List[str],
    lambdas: Optional[NDArray[np.floating]] = None,
    max_iter: int = 25,
    tol: float = 0.0,
) -> WSINDyModel:
    """Fit a sparse WSINDy model via preconditioned MSTLS.

    Parameters
    ----------
    b : ndarray, shape ``(K,)``
        Weak-form LHS vector  (ψ_t * U  at query points).
    G : ndarray, shape ``(K, M)``
        Weak-form RHS matrix.
    col_names : list of *M* strings
        Human-readable library term labels.
    lambdas : ndarray or None
        Candidate λ̂ values.  If ``None`` a default log-spaced grid
        ``[1e-4, 1e0]`` with 30 points is used.
    max_iter : int
        Max thresholding iterations per λ (default 25).
    tol : float
        Absolute coefficient floor (default 0 — rely only on
        dominant-balance thresholding).

    Returns
    -------
    model : WSINDyModel
        Fitted sparse model with unscaled physical coefficients.
    """
    b = np.asarray(b, dtype=np.float64).ravel()
    G = np.asarray(G, dtype=np.float64)
    K, M = G.shape
    if b.shape[0] != K:
        raise ValueError(
            f"Shape mismatch: b has {b.shape[0]} rows, G has {K} rows."
        )
    if len(col_names) != M:
        raise ValueError(
            f"col_names length {len(col_names)} != G column count {M}."
        )

    # ── 1. Precondition columns to unit norm ────────────────────────────
    Gs, scale_info = precondition_columns(G)
    col_scale: NDArray[np.floating] = scale_info["col_scale"]

    # ── 2. Default lambda grid ──────────────────────────────────────────
    if lambdas is None:
        lambdas = np.logspace(-4, 0, 30)

    # ── 3. Run MSTLS on normalised system ───────────────────────────────
    result = mstls(Gs, b, lambdas, max_iter=max_iter, tol=tol)

    w_scaled: NDArray[np.floating] = result["w"]
    active: NDArray[np.bool_] = result["active"]
    best_lambda: float = result["best_lambda"]

    # ── 4. Un-scale coefficients to physical units ──────────────────────
    #   Gs = G / col_scale   →   b ≈ Gs @ w_scaled = G @ (w_scaled / col_scale)
    #   so   w_physical = w_scaled / col_scale
    w_unscaled = np.zeros(M, dtype=np.float64)
    w_unscaled[active] = w_scaled[active] / col_scale[active]

    # ── 5. Diagnostics on the *original* (unscaled) system ──────────────
    fit_met = wsindy_fit_metrics(b, G, w_unscaled)

    diagnostics = {
        "r2": fit_met["r2"],
        "residual_norm": fit_met["residual_norm"],
        "relative_l2": fit_met["relative_l2"],
        "normalised_loss": result["history"][
            next(
                i for i, h in enumerate(result["history"])
                if np.isclose(h["lambda"], best_lambda)
            )
        ]["normalised_loss"],
        "n_active": int(np.sum(active)),
        "best_lambda": best_lambda,
        "lambda_history": [
            {
                "lambda": h["lambda"],
                "n_active": h["n_active"],
                "normalised_loss": h["normalised_loss"],
                "cond": h["cond"],
                "converged": h["converged"],
            }
            for h in result["history"]
        ],
    }

    return WSINDyModel(
        col_names=col_names,
        w=w_unscaled,
        active=active,
        best_lambda=best_lambda,
        col_scale=col_scale,
        diagnostics=diagnostics,
    )
