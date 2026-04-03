"""
Post-regression diagnostics for WSINDy.

Task 4.3 — Residual distribution analysis, OLS comparison, condition reporting.
Task 8.1 — Dominant balance / dimensionless group reporting.

All functions operate on the already-fitted weak system ``(b, G, model)``
and return data structures or generate Matplotlib figures.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .model import WSINDyModel


# ═══════════════════════════════════════════════════════════════════════════
#  Task 4.3 — Residual distribution diagnostics
# ═══════════════════════════════════════════════════════════════════════════

def residual_analysis(
    b: NDArray[np.floating],
    G: NDArray[np.floating],
    model: WSINDyModel,
    *,
    plot: bool = False,
    ax=None,
) -> Dict[str, Any]:
    r"""Analyse the residual distribution :math:`r = b - G w^*`.

    The weak-form residuals should follow a peaked, heavy-tailed
    distribution (Bessel-type) if the test function is well-chosen.
    Gaussian residuals indicate under-smoothing or model misspecification.

    Parameters
    ----------
    b : ndarray (K,)
    G : ndarray (K, M)
    model : WSINDyModel
    plot : bool
        If True, produce a histogram + Q-Q diagnostic figure.
    ax : matplotlib Axes, optional
        If provided and *plot* is True, draw into this axes pair
        (expects len(ax) >= 2).

    Returns
    -------
    dict with keys:
        ``residuals``  — (K,) array
        ``mean``, ``std``, ``skewness``, ``kurtosis``
        ``shapiro_p``  — p-value from Shapiro-Wilk normality test
                         (small p ⇒ non-Gaussian, expected for Bessel)
        ``max_abs``    — max |r_i|
    """
    r = b - G @ model.w
    N = len(r)
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1)) if N > 1 else 0.0
    skew = float(np.mean(((r - mu) / max(sigma, 1e-30)) ** 3)) if sigma > 1e-30 else 0.0
    kurt = float(np.mean(((r - mu) / max(sigma, 1e-30)) ** 4) - 3.0) if sigma > 1e-30 else 0.0

    # Shapiro-Wilk on a subsample (max 5000 for speed)
    try:
        from scipy.stats import shapiro
        sub = r[:: max(1, N // 5000)]
        _, shapiro_p = shapiro(sub)
    except Exception:
        shapiro_p = float("nan")

    result: Dict[str, Any] = {
        "residuals": r,
        "mean": mu,
        "std": sigma,
        "skewness": skew,
        "kurtosis": kurt,
        "shapiro_p": float(shapiro_p),
        "max_abs": float(np.max(np.abs(r))),
    }

    if plot:
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax0, ax1 = ax[0], ax[1]

        # Histogram
        ax0.hist(r, bins=60, density=True, alpha=0.7, edgecolor="k", linewidth=0.3)
        ax0.set_xlabel("Residual $r = b - Gw^*$")
        ax0.set_ylabel("Density")
        ax0.set_title(f"Residual distribution (kurtosis={kurt:.2f})")
        ax0.axvline(0, color="r", ls="--", lw=0.8)

        # Q-Q plot against normal
        from scipy.stats import probplot
        probplot(r, dist="norm", plot=ax1)
        ax1.set_title("Normal Q-Q plot")
        ax1.get_lines()[0].set_markersize(2)

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Task 4.3 — OLS comparison
# ═══════════════════════════════════════════════════════════════════════════

def ols_comparison(
    b: NDArray[np.floating],
    G: NDArray[np.floating],
    model: WSINDyModel,
) -> Dict[str, Any]:
    """Compare MSTLS sparse fit with unconstrained OLS on the same system.

    This serves as a sanity check: for correctly identified active terms,
    the OLS and MSTLS coefficients should agree closely.  Large
    discrepancies indicate either over-regularisation or model
    misspecification.

    Parameters
    ----------
    b : ndarray (K,)
    G : ndarray (K, M)
    model : WSINDyModel  (fitted via MSTLS)

    Returns
    -------
    dict with keys:
        ``w_ols``      — (M,) full OLS weights
        ``w_mstls``    — (M,) copy of model.w
        ``r2_ols``     — R² of OLS fit
        ``r2_mstls``   — R² of MSTLS fit
        ``max_rel_diff`` — max relative difference on active terms
        ``per_term``   — list of dicts with per-active-term comparison
    """
    w_ols_full, _, _, _ = np.linalg.lstsq(G, b, rcond=1e-12)
    r2_ols = 1.0 - float(np.sum((b - G @ w_ols_full) ** 2)) / float(
        np.sum((b - np.mean(b)) ** 2)
    )
    r2_mstls = 1.0 - float(np.sum((b - G @ model.w) ** 2)) / float(
        np.sum((b - np.mean(b)) ** 2)
    )

    per_term = []
    for i, name in enumerate(model.col_names):
        if model.active[i]:
            w_s = model.w[i]
            w_o = w_ols_full[i]
            denom = max(abs(w_s), abs(w_o), 1e-30)
            per_term.append({
                "term": name,
                "w_mstls": float(w_s),
                "w_ols": float(w_o),
                "rel_diff": float(abs(w_s - w_o) / denom),
            })

    max_rel = max((d["rel_diff"] for d in per_term), default=0.0)

    return {
        "w_ols": w_ols_full,
        "w_mstls": model.w.copy(),
        "r2_ols": r2_ols,
        "r2_mstls": r2_mstls,
        "max_rel_diff": max_rel,
        "per_term": per_term,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Task 8.1 — Dominant balance / dimensionless group reporting
# ═══════════════════════════════════════════════════════════════════════════

def dominant_balance_report(
    b: NDArray[np.floating],
    G: NDArray[np.floating],
    model: WSINDyModel,
) -> Dict[str, Any]:
    r"""Compute dimensionless importance ratios for each active term.

    For each active term *i* the dimensionless group is

    .. math::
        \Pi_i = \frac{\|w_i \, G_i\|_2}{\|b\|_2}

    This quantifies the relative contribution of term *i* to the
    LHS dynamics.  Ratios sum ≈ 1 for a well-conditioned fit.
    Large ratios indicate dominant physics; small ratios indicate
    marginal terms that may be spurious.

    Parameters
    ----------
    b : ndarray (K,)
    G : ndarray (K, M)
    model : WSINDyModel

    Returns
    -------
    dict with keys:
        ``groups``   — list of ``{"term", "Pi", "w", "||G_i||"}`` dicts
                        sorted by Π descending
        ``Pi_sum``   — sum of all Π_i (should be ≈ 1)
        ``b_norm``   — ‖b‖₂ used as reference
    """
    b_norm = float(np.linalg.norm(b))
    if b_norm < 1e-30:
        return {"groups": [], "Pi_sum": 0.0, "b_norm": b_norm}

    groups = []
    for i, name in enumerate(model.col_names):
        if not model.active[i]:
            continue
        Gi_norm = float(np.linalg.norm(G[:, i]))
        contrib = abs(model.w[i]) * Gi_norm
        Pi = contrib / b_norm
        groups.append({
            "term": name,
            "Pi": Pi,
            "w": float(model.w[i]),
            "G_i_norm": Gi_norm,
        })

    groups.sort(key=lambda d: d["Pi"], reverse=True)
    Pi_sum = sum(d["Pi"] for d in groups)

    return {
        "groups": groups,
        "Pi_sum": Pi_sum,
        "b_norm": b_norm,
    }


def print_dominant_balance(report: Dict[str, Any]) -> str:
    """Format a dominant-balance report as a human-readable table."""
    lines = [
        "Dominant Balance Report",
        "=" * 55,
        f"  {'Term':>20s}    {'Π_i':>10s}    {'w_i':>12s}",
        "  " + "-" * 50,
    ]
    for g in report["groups"]:
        lines.append(
            f"  {g['term']:>20s}    {g['Pi']:10.4e}    {g['w']:+12.4e}"
        )
    lines.append("  " + "-" * 50)
    lines.append(f"  {'Σ Π_i':>20s}    {report['Pi_sum']:10.4e}")
    lines.append(f"  {'‖b‖₂':>20s}    {report['b_norm']:10.4e}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  Task 6.3 — Model hierarchy comparison (AIC-based)
# ═══════════════════════════════════════════════════════════════════════════

def model_aic(
    b: NDArray[np.floating],
    G: NDArray[np.floating],
    model: WSINDyModel,
) -> float:
    r"""Compute the Akaike Information Criterion for a fitted model.

    .. math::
        \mathrm{AIC} = K \ln\!\bigl(\mathrm{RSS}/K\bigr) + 2 k

    where *K* = number of observations, *k* = number of active terms,
    and RSS = :math:`\|b - G w\|_2^2`.
    """
    K = len(b)
    k = model.n_active
    rss = float(np.sum((b - G @ model.w) ** 2))
    if rss <= 0:
        rss = 1e-30
    return K * np.log(rss / K) + 2 * k


def compare_model_hierarchy(
    b: NDArray[np.floating],
    G: NDArray[np.floating],
    col_names: List[str],
    hierarchy: Dict[str, List[str]],
    lambdas: Optional[NDArray[np.floating]] = None,
    max_iter: int = 25,
) -> Dict[str, Any]:
    r"""Fit and compare a hierarchy of nested models via ΔAIC.

    Parameters
    ----------
    b : ndarray (K,)
    G : ndarray (K, M)
    col_names : list of str
        Full column names matching G.
    hierarchy : dict
        ``{"model_name": ["term1", "term2", ...], ...}``
        Each entry specifies the allowed library terms for that model.
        Terms not in the list are forced to zero.
        Use ``"full"`` as a special key to include all terms.
    lambdas, max_iter : regression parameters.

    Returns
    -------
    dict with:
        ``models``   — dict of ``{name: WSINDyModel}``
        ``aic``      — dict of ``{name: float}``
        ``delta_aic``— dict of ``{name: float}`` (ΔAIC relative to best)
        ``ranking``  — list of (name, AIC) sorted ascending
    """
    from .fit import wsindy_fit_regression

    if lambdas is None:
        lambdas = np.logspace(-4, 1, 40)

    col_idx = {name: i for i, name in enumerate(col_names)}
    models_out: Dict[str, WSINDyModel] = {}
    aic_out: Dict[str, float] = {}

    for model_name, allowed_terms in hierarchy.items():
        if model_name == "full" or allowed_terms is None:
            # Use all columns
            G_sub = G
            names_sub = col_names
        else:
            indices = [col_idx[t] for t in allowed_terms if t in col_idx]
            if not indices:
                continue
            G_sub = G[:, indices]
            names_sub = [col_names[i] for i in indices]

        m = wsindy_fit_regression(b, G_sub, names_sub, lambdas=lambdas, max_iter=max_iter)

        # Embed back into full-size coefficient vector for AIC computation
        w_full = np.zeros(len(col_names))
        active_full = np.zeros(len(col_names), dtype=bool)
        if model_name == "full" or allowed_terms is None:
            w_full = m.w
            active_full = m.active
        else:
            for j, idx in enumerate(indices):
                w_full[idx] = m.w[j]
                active_full[idx] = m.active[j]

        m_full = WSINDyModel(
            col_names=col_names,
            w=w_full,
            active=active_full,
            best_lambda=m.best_lambda,
            col_scale=np.ones(len(col_names)),
            diagnostics=m.diagnostics,
        )
        models_out[model_name] = m_full
        aic_out[model_name] = model_aic(b, G, m_full)

    best_aic = min(aic_out.values()) if aic_out else 0.0
    delta_aic = {name: aic - best_aic for name, aic in aic_out.items()}
    ranking = sorted(aic_out.items(), key=lambda x: x[1])

    return {
        "models": models_out,
        "aic": aic_out,
        "delta_aic": delta_aic,
        "ranking": ranking,
    }
