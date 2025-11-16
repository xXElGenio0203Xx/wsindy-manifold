import numpy as np

from rectsim.domain import apply_bc, pair_displacements


def test_periodic_wrap_and_minimal_image():
    x = np.array([[1.2, -0.1], [9.9, 4.8]])
    wrapped, flips = apply_bc(x.copy(), Lx=10.0, Ly=5.0, bc="periodic")
    assert np.all((0 <= wrapped) & (wrapped[:, 0] < 10.0) & (wrapped[:, 1] < 5.0))
    assert not flips.any()

    dx, dy, rij, _ = pair_displacements(wrapped, Lx=10.0, Ly=5.0, bc="periodic")
    direct = wrapped[1] - wrapped[0]
    dist_wrap = np.hypot(dx[0, 1], dy[0, 1])
    assert dist_wrap <= np.linalg.norm(direct)


def test_reflecting_boundary_flips_velocity_flag():
    x = np.array([[1.0, 1.0], [-0.5, 6.0]])
    wrapped, flips = apply_bc(x.copy(), Lx=5.0, Ly=5.0, bc="reflecting")
    assert 0.0 <= wrapped[1, 0] <= 5.0
    assert 0.0 <= wrapped[1, 1] <= 5.0
    assert flips[1, 0] and flips[1, 1]
