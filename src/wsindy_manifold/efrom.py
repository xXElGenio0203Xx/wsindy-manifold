"""Reduced-order density modeling utilities (KDE → POD → VAR → forecast)."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

Array = np.ndarray


def pod_fit(
    rho: Array,
    rank: int,
    keep_mass_mode: bool = True,
) -> Tuple[Array, Array, Array, Array]:
    """Fit a POD basis via SVD on density snapshots."""

    T, ny, nx = rho.shape
    M = ny * nx
    X = rho.reshape(T, M)
    mean_frame = X.mean(axis=0) if keep_mass_mode else np.zeros(M)
    Xc = X - mean_frame
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    r = min(rank, U.shape[1]) if rank is not None else U.shape[1]
    return U[:, :r], s[:r], Vt[:r], mean_frame.reshape(ny, nx)


def project_latent(rho: Array, Vt: Array, mean_frame: Array) -> Array:
    """Project density snapshots to latent coordinates."""

    T = rho.shape[0]
    basis = Vt.T  # shape (M, r)
    X = rho.reshape(T, -1)
    mean_flat = mean_frame.reshape(-1)
    Xc = X - mean_flat
    return Xc @ basis


def lift(
    Z: Array,
    Vt: Array,
    mean_frame_flat: Array,
    ny: int,
    nx: int,
    keep_mass_mode: bool = True,
    mass_target: Optional[float] = None,
    cell_area: float = 1.0,
) -> Array:
    """Lift latent coordinates back to density fields with optional mass fix."""

    basis = Vt
    Xc = Z @ basis
    X = Xc + mean_frame_flat.reshape(1, -1)
    rho = X.reshape(Z.shape[0], ny, nx)
    if keep_mass_mode and mass_target is not None and mass_target > 0:
        masses = rho.reshape(Z.shape[0], -1).sum(axis=1) * cell_area
        for i in range(rho.shape[0]):
            if masses[i] <= 0:
                continue
            rho[i] *= mass_target / masses[i]
        drift = np.abs(masses - mass_target)
        if np.max(drift) > 1e-3 * max(1.0, mass_target):
            import warnings

            warnings.warn(
                f"Mass drift detected during lift (max drift={np.max(drift):.3e})",
                RuntimeWarning,
                stacklevel=2,
            )
    return rho


def mvar_fit(Z: Array, order: int) -> Tuple[Array, Sequence[Array]]:
    """Fit a VAR model with intercept using least squares."""

    if order <= 0:
        raise ValueError("order must be positive")
    T, r = Z.shape
    if T <= order:
        raise ValueError("Not enough samples to fit VAR")
    Y = Z[order:]
    rows = []
    for t in range(order, T):
        row = [1.0]
        for lag in range(1, order + 1):
            row.extend(Z[t - lag])
        rows.append(row)
    X = np.asarray(rows)
    coef, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    c = coef[0]
    A = coef[1:].reshape(order, r, r)
    return c, A


def mvar_forecast(
    Z_init: Array,
    intercept: Array,
    A_list: Sequence[Array],
    steps: int,
) -> Array:
    """Roll out the VAR model for ``steps`` future samples."""

    order = len(A_list)
    history = list(Z_init[-order:])
    preds = []
    for _ in range(steps):
        y = intercept.copy()
        for lag in range(order):
            y += A_list[lag] @ history[-(lag + 1)]
        preds.append(y)
        history.append(y)
        history = history[-order:]
    return np.asarray(preds)


def efrom_train_and_forecast(
    rho_train: Array,
    rho_test: Array,
    rank: int,
    order: int,
    horizon: int,
    keep_mass_mode: bool = True,
    cell_area: float = 1.0,
) -> Tuple[Array, dict]:
    """Train POD+VAR on training data and forecast test horizon."""

    ny, nx = rho_train.shape[1:]
    U, s, Vt, mean_frame = pod_fit(rho_train, rank, keep_mass_mode=keep_mass_mode)
    basis = Vt
    Z_train = project_latent(rho_train, basis, mean_frame)
    intercept, A_list = mvar_fit(Z_train, order)
    rho_train_flat = rho_train.reshape(rho_train.shape[0], -1)
    mass_target = float(np.mean(rho_train_flat.sum(axis=1) * cell_area))
    history = Z_train[-order:]
    Z_pred = mvar_forecast(history, intercept, A_list, horizon)
    rho_pred = lift(
        Z_pred,
        basis,
        mean_frame.reshape(-1),
        ny,
        nx,
        keep_mass_mode=keep_mass_mode,
        mass_target=mass_target,
        cell_area=cell_area,
    )
    mass_series = rho_pred.reshape(rho_pred.shape[0], -1).sum(axis=1) * cell_area
    info = {
        "U": U,
        "s": s,
        "Vt": Vt,
        "mean_frame": mean_frame,
        "intercept": intercept,
        "A_list": A_list,
        "mass_series": mass_series,
    }
    return rho_pred, info


__all__ = [
    "pod_fit",
    "project_latent",
    "lift",
    "mvar_fit",
    "mvar_forecast",
    "efrom_train_and_forecast",
]
