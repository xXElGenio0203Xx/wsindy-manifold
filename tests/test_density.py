import numpy as np

from rectsim.density import density_movie_kde


def test_density_movie_kde_mass_conservation_reflecting():
    traj = np.array([
        [[0.1, 0.2], [0.4, 0.5]],
        [[0.2, 0.3], [0.5, 0.6]],
    ])
    rho = density_movie_kde(traj, Lx=1.0, Ly=1.0, nx=10, ny=12, bandwidth=0.5, bc="reflecting")
    dx = 1.0 / 10
    dy = 1.0 / 12
    mass = rho[0].sum() * dx * dy
    assert np.isclose(mass, traj.shape[1])


def test_density_movie_kde_periodic_wrap():
    traj = np.array([
        [[0.95, 0.5], [0.05, 0.5]],
    ])
    rho = density_movie_kde(traj, Lx=1.0, Ly=1.0, nx=32, ny=32, bandwidth=0.2, bc="periodic")
    dx = dy = 1.0 / 32
    mass = rho[0].sum() * dx * dy
    assert np.isclose(mass, traj.shape[1])
