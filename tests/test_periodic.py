import numpy as np

from rectsim.domain import pair_displacements


def test_minimal_image_periodic():
    x = np.array([[19.5, 10.0], [0.5, 10.0]])
    dx, dy, rij, _ = pair_displacements(x, Lx=20.0, Ly=20.0, bc="periodic")
    np.testing.assert_allclose(rij[0, 1], 1.0)
    np.testing.assert_allclose(dx[0, 1], 1.0)


def test_translation_invariance():
    x = np.array([[2.0, 3.0], [5.0, 7.0]])
    dx1, dy1, rij1, _ = pair_displacements(x, Lx=20.0, Ly=20.0, bc="periodic")
    x_shifted = x + np.array([20.0, 0.0])
    dx2, dy2, rij2, _ = pair_displacements(x_shifted, Lx=20.0, Ly=20.0, bc="periodic")
    np.testing.assert_allclose(rij1, rij2)
    np.testing.assert_allclose(dx1, dx2)
    np.testing.assert_allclose(dy1, dy2)
