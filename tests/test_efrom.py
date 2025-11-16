import numpy as np

from wsindy_manifold.efrom import (
    efrom_train_and_forecast,
    lift,
    mvar_fit,
    mvar_forecast,
    pod_fit,
    project_latent,
)


def _synthetic_density(T=60, ny=6, nx=5, rank=2, seed=0):
    rng = np.random.default_rng(seed)
    mean = rng.uniform(0.2, 0.3, size=(ny, nx))
    basis = rng.normal(scale=0.05, size=(rank, ny * nx))
    A = np.array([[0.9, -0.05], [0.02, 0.95]])
    Z = np.zeros((T, rank))
    Z[0] = rng.normal(scale=0.1, size=rank)
    for t in range(1, T):
        Z[t] = A @ Z[t - 1]
    rho = mean.reshape(1, -1) + Z @ basis
    rho = np.clip(rho, 1e-6, None)
    rho = rho.reshape(T, ny, nx)
    mass = rho.reshape(T, -1).sum(axis=1)
    rho = rho / mass[:, None, None] * 10.0
    return rho


def test_pod_projection_and_lift_roundtrip():
    rho = _synthetic_density(T=20)
    U, s, Vt, mean_frame = pod_fit(rho, rank=3)
    Z = project_latent(rho, Vt, mean_frame)
    rho_rec = lift(Z, Vt, mean_frame.reshape(-1), rho.shape[1], rho.shape[2])
    assert np.allclose(rho, rho_rec, atol=1e-8)


def test_mvar_fit_forecast_identity():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(50, 3))
    intercept, A = mvar_fit(Z, order=1)
    Z_pred = mvar_forecast(Z[-1:], intercept, A, steps=1)
    assert Z_pred.shape == (1, 3)


def test_efrom_train_and_forecast_accuracy():
    rho = _synthetic_density(T=80)
    split = 60
    rho_train = rho[:split]
    rho_test = rho[split:]
    horizon = min(10, rho_test.shape[0])
    rho_pred, _ = efrom_train_and_forecast(
        rho_train,
        rho_test,
        rank=3,
        order=2,
        horizon=horizon,
        cell_area=1.0,
        keep_mass_mode=True,
    )
    err = np.linalg.norm(rho_pred - rho_test[:horizon]) / np.linalg.norm(rho_test[:horizon])
    assert err < 0.1
