import numpy as np

from rectsim.morse import morse_force_pairs


def test_two_particle_symmetry():
    x = np.array([[0.0, 0.0], [1.0, 0.0]])
    fx, fy = morse_force_pairs(x, Cr=2.0, Ca=1.0, lr=0.5, la=1.0, Lx=5.0, Ly=5.0, bc="reflecting", rcut=5.0)
    np.testing.assert_allclose(fx[0], -fx[1])
    np.testing.assert_allclose(fy[0], -fy[1])


def test_force_regimes():
    close = np.array([[0.0, 0.0], [0.2, 0.0]])
    far = np.array([[0.0, 0.0], [5.0, 0.0]])

    fx_close, _ = morse_force_pairs(close, Cr=3.0, Ca=1.0, lr=0.5, la=1.0, Lx=10.0, Ly=10.0, bc="reflecting", rcut=6.0)
    fx_far, _ = morse_force_pairs(far, Cr=3.0, Ca=1.0, lr=0.5, la=1.0, Lx=10.0, Ly=10.0, bc="reflecting", rcut=6.0)

    assert fx_close[0] < 0  # repulsive pushes left
    assert fx_far[0] > 0  # attractive pulls right
