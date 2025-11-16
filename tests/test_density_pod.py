import numpy as np

from rectsim.density import density_movie_kde
from wsindy_manifold.efrom import lift, pod_fit, project_latent


def test_kde_mass_conservation():
    rng = np.random.default_rng(0)
    traj = rng.uniform([0.0, 0.0], [2.0, 1.0], size=(5, 50, 2))
    rho = density_movie_kde(traj, Lx=2.0, Ly=1.0, nx=32, ny=16, bandwidth=0.5, bc="periodic")
    dx = 2.0 / 32
    dy = 1.0 / 16
    masses = rho.reshape(rho.shape[0], -1).sum(axis=1) * dx * dy
    np.testing.assert_allclose(masses, traj.shape[1], rtol=1e-6)


def test_pod_lift_identity():
    rng = np.random.default_rng(1)
    rho = rng.normal(size=(8, 10, 12))
    U, s, Vt, mean_frame = pod_fit(rho, rank=rho.shape[1] * rho.shape[2])
    Z = project_latent(rho, Vt, mean_frame)
    rho_rec = lift(Z, Vt, mean_frame.reshape(-1), rho.shape[1], rho.shape[2])
    np.testing.assert_allclose(rho, rho_rec, atol=1e-10)
