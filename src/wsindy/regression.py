"""
Sparse regression for WSINDy: preconditioning + MSTLS.

Implements the Modified Sequential Thresholded Least Squares algorithm
with dominant-balance thresholding following Minor et al. (2025).

Key steps
---------
1. **Column normalisation** — each column of G is scaled to unit ℓ₂ norm
   so that the thresholding criterion treats all terms on equal footing
   regardless of their original magnitudes.
2. **MSTLS** — for each candidate λ̂ on a log-spaced grid, iteratively
   solve the LS problem on active columns and prune terms whose
   *contribution* ``|w_i| · ‖G_i‖₂ / ‖b‖₂`` falls outside
   ``[λ̂, 1/λ̂]``.
3. **Model selection** — pick the λ̂ that minimises the normalised loss
   ``‖b − Gw‖₂ / ‖b_ls‖₂``; break ties by preferring fewer active terms.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ═══════════════════════════════════════════════════════════════════════════
# A)  Column scaling / preconditioning
# ═══════════════════════════════════════════════════════════════════════════

def precondition_columns(
    G: NDArray[np.floating],
    eps: float = 1e-12,
) -> Tuple[NDArray[np.floating], Dict[str, NDArray[np.floating]]]:
    """Scale each column of *G* to unit ℓ₂ norm.

    Parameters
    ----------
    G : ndarray, shape ``(K, M)``
    eps : float
        Floor for column norms to avoid division by zero.

    Returns
    -------
    Gs : ndarray, shape ``(K, M)``
        Column-normalised matrix.
    info : dict
        ``"col_scale"`` — 1-D array of length *M* with the original
        column norms.  To recover physical coefficients:
        ``w_unscaled = w_scaled / col_scale``.
    """
    G = np.asarray(G, dtype=np.float64)
    col_norms = np.linalg.norm(G, axis=0)
    col_scale = np.maximum(col_norms, eps)
    Gs = G / col_scale[np.newaxis, :]
    return Gs, {"col_scale": col_scale}


# ═══════════════════════════════════════════════════════════════════════════
# B)  Stable least-squares solver
# ═══════════════════════════════════════════════════════════════════════════

def solve_ls(
    G: NDArray[np.floating],
    b: NDArray[np.floating],
    rcond: float = 1e-12,
) -> NDArray[np.floating]:
    """SVD-based least squares: minimise ‖Gw − b‖₂.

    Parameters
    ----------
    G : ndarray, shape ``(K, M)``
    b : ndarray, shape ``(K,)``
    rcond : float
        Cut-off ratio for small singular values (passed to
        ``np.linalg.lstsq``).

    Returns
    -------
    w : ndarray, shape ``(M,)``
    """
    w, _, _, _ = np.linalg.lstsq(G, b, rcond=rcond)
    return w


# ═══════════════════════════════════════════════════════════════════════════
# C)  MSTLS — Modified Sequential Thresholded Least Squares
# ═══════════════════════════════════════════════════════════════════════════

def _dominant_balance_mask(
    w: NDArray[np.floating],
    G: NDArray[np.floating],
    b_norm: float,
    lambda_hat: float,
) -> NDArray[np.bool_]:
    """Apply the dominant-balance thresholding criterion.

    A term *i* is **kept** if its relative contribution satisfies

        λ̂  ≤  ‖w_i · G_i‖₂ / ‖b‖₂  ≤  1/λ̂

    Parameters
    ----------
    w : (M,) coefficient vector (full, with zeros for inactive terms).
    G : (K, M) matrix (may be the preconditioned version).
    b_norm : ‖b‖₂.
    lambda_hat : current threshold parameter.

    Returns
    -------
    keep : (M,) boolean mask — True for terms to retain.
    """
    M = w.shape[0]
    # col_contrib[i] = ||w[i] * G[:,i]||_2 / ||b||_2
    #                = |w[i]| * ||G[:,i]||_2 / ||b||_2
    col_contrib = np.zeros(M, dtype=np.float64)
    for i in range(M):
        col_contrib[i] = np.abs(w[i]) * np.linalg.norm(G[:, i]) / b_norm

    inv_lambda = 1.0 / lambda_hat
    keep = (col_contrib >= lambda_hat) & (col_contrib <= inv_lambda)
    return keep


def mstls(
    G: NDArray[np.floating],
    b: NDArray[np.floating],
    lambdas: NDArray[np.floating],
    max_iter: int = 25,
    tol: float = 0.0,
) -> Dict[str, Any]:
    """Run MSTLS over a grid of λ̂ candidates.

    Parameters
    ----------
    G : ndarray, shape ``(K, M)``
        Design matrix (typically already preconditioned).
    b : ndarray, shape ``(K,)``
        Target vector.
    lambdas : 1-D ndarray
        Candidate threshold values (e.g. ``np.logspace(-4, 0, 25)``).
    max_iter : int
        Maximum thresholding iterations per λ.
    tol : float
        Optional absolute-value floor: drop terms with ``|w_i| < tol``
        in addition to dominant-balance thresholding.

    Returns
    -------
    result : dict
        ``"w"``          — best sparse weight vector (M,)
        ``"active"``     — boolean mask (M,)
        ``"best_lambda"``— selected λ̂
        ``"history"``    — list of per-λ diagnostic dicts
    """
    G = np.asarray(G, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).ravel()
    K, M = G.shape
    b_norm = float(np.linalg.norm(b))
    if b_norm < 1e-30:
        raise ValueError(
            f"‖b‖₂ = {b_norm:.2e} is effectively zero; "
            "cannot run MSTLS (no signal in weak-form LHS)."
        )

    # ── baseline full LS ────────────────────────────────────────────────
    w_ls = solve_ls(G, b)
    b_ls_norm = float(np.linalg.norm(G @ w_ls))
    if b_ls_norm < 1e-30:
        b_ls_norm = b_norm  # fallback

    # ── sweep λ̂ ─────────────────────────────────────────────────────────
    history: List[Dict[str, Any]] = []
    best_loss = np.inf
    best_w = w_ls.copy()
    best_active = np.ones(M, dtype=bool)
    best_lam = float(lambdas[0])

    for lam in lambdas:
        active = np.ones(M, dtype=bool)
        w_full = np.zeros(M, dtype=np.float64)

        converged = False
        for iteration in range(max_iter):
            n_act = int(np.sum(active))
            if n_act == 0:
                break

            # Solve LS on active columns
            w_act = solve_ls(G[:, active], b)
            w_full[:] = 0.0
            w_full[active] = w_act

            # Optional absolute-value floor
            if tol > 0:
                w_full[np.abs(w_full) < tol] = 0.0

            # Dominant-balance thresholding
            new_active = _dominant_balance_mask(w_full, G, b_norm, lam)
            # Only keep terms that were already active AND pass the test
            new_active = active & new_active

            if np.array_equal(new_active, active):
                converged = True
                break
            active = new_active

        # Final fit on converged active set
        n_act = int(np.sum(active))
        w_full[:] = 0.0
        if n_act > 0:
            w_full[active] = solve_ls(G[:, active], b)

        # Normalised loss
        res_norm = float(np.linalg.norm(b - G @ w_full))
        norm_loss = res_norm / b_ls_norm

        # Condition number of active submatrix
        if n_act > 0:
            cond = float(np.linalg.cond(G[:, active]))
        else:
            cond = float("inf")

        rec = {
            "lambda": float(lam),
            "n_active": n_act,
            "residual_norm": res_norm,
            "normalised_loss": norm_loss,
            "cond": cond,
            "converged": converged,
            "w": w_full.copy(),
            "active": active.copy(),
        }
        history.append(rec)

        # Model selection: minimum normalised loss; tie-break by sparsity
        if n_act > 0 and (
            norm_loss < best_loss
            or (np.isclose(norm_loss, best_loss) and n_act < int(np.sum(best_active)))
        ):
            best_loss = norm_loss
            best_w = w_full.copy()
            best_active = active.copy()
            best_lam = float(lam)

    return {
        "w": best_w,
        "active": best_active,
        "best_lambda": best_lam,
        "history": history,
    }
