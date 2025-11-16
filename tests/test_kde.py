import numpy as np

from wsindy_manifold.latent.kde import (
    make_grid,
    kde_gaussian,
    trajectories_to_density_movie,
    minimal_image_dxdy,
    pair_dxdy_to_grid,
)


def test_kde_mass_conservation():
    Lx, Ly = 1.0, 1.0
    nx, ny = 16, 12
    Xc, dx, dy = make_grid(Lx, Ly, nx, ny)
    points = np.array([[0.25, 0.75], [0.75, 0.25], [0.5, 0.5]])
    rho = kde_gaussian(points, Xc, hx=0.1, hy=0.1, Lx=Lx, Ly=Ly)
    mass = rho.sum() * dx * dy
    assert np.isclose(mass, 1.0, atol=1e-8)


def test_density_movie_shape_and_mass():
    Lx, Ly = 2.0, 1.5
    nx, ny = 8, 6
    T, N = 5, 10
    rng = np.random.default_rng(0)
    traj = rng.uniform([0.0, 0.0], [Lx, Ly], size=(T, N, 2))
    Rho, meta = trajectories_to_density_movie(
        X_all=traj,
        Lx=Lx,
        Ly=Ly,
        nx=nx,
        ny=ny,
        hx=0.2,
        hy=0.15,
        bc="periodic",
    )
    assert Rho.shape == (T, nx * ny)
    dx = float(meta["dx"])
    dy = float(meta["dy"])
    masses = Rho.sum(axis=1) * dx * dy
    assert np.allclose(masses, 1.0, atol=1e-8)


def test_minimal_image_and_custom_distance():
    L = 2.0
    delta = np.array([3.0, -3.0, 1.0])
    wrapped = minimal_image_dxdy(delta, L)
    assert np.allclose(wrapped, np.array([-1.0, 1.0, 1.0]))

    X = np.array([[0.1, 0.1], [1.9, 1.8]])
    Xc, _, _ = make_grid(L, L, 4, 4)
    dx_grid, dy_grid = pair_dxdy_to_grid(X, Xc, L, L, bc="periodic")
    assert dx_grid.shape == (Xc.shape[0], X.shape[0])

    def zero_distance(*_args, **_kwargs):
        return np.zeros_like(dx_grid), np.zeros_like(dy_grid)

    rho = kde_gaussian(
        X,
        Xc,
        hx=0.2,
        hy=0.2,
        Lx=L,
        Ly=L,
        bc="periodic",
        distance_fn=zero_distance,
    )
    assert np.isclose(rho.sum() * (L / 4) * (L / 4), 1.0)
