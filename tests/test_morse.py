import numpy as np

from rectsim.morse import morse_force, morse_force_pairs


def test_two_particle_symmetry():
    x = np.array([[0.0, 0.0], [1.0, 0.0]])
    fx, fy = morse_force_pairs(x, Cr=2.0, Ca=1.0, lr=0.5, la=1.0, Lx=5.0, Ly=5.0, bc="reflecting", rcut=5.0)
    np.testing.assert_allclose(fx[0], -fx[1], atol=1e-10)
    np.testing.assert_allclose(fy[0], -fy[1], atol=1e-10)


def test_force_regimes():
    close = np.array([[0.0, 0.0], [0.2, 0.0]])
    far = np.array([[0.0, 0.0], [5.0, 0.0]])

    fx_close, _ = morse_force_pairs(close, Cr=3.0, Ca=1.0, lr=0.5, la=1.0, Lx=10.0, Ly=10.0, bc="reflecting", rcut=6.0)
    fx_far, _ = morse_force_pairs(far, Cr=3.0, Ca=1.0, lr=0.5, la=1.0, Lx=10.0, Ly=10.0, bc="reflecting", rcut=6.0)

    assert fx_close[0] < 0  # repulsive pushes left
    assert fx_far[0] > 0  # attractive pulls right
    assert abs(fx_close[0]) > abs(fx_far[0])  # magnitude decreases with distance


def test_morse_force_matches_pairs():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 10.0, size=(16, 2))
    params = dict(Cr=3.0, Ca=1.5, lr=0.8, la=1.2, Lx=10.0, Ly=10.0, bc="periodic", rcut=5.0)
    fx1, fy1 = morse_force_pairs(x, **params)
    fx2, fy2 = morse_force(x, params["Lx"], params["Ly"], params["bc"], params["Cr"], params["Ca"], params["lr"], params["la"], params["rcut"])
    np.testing.assert_allclose(fx1, fx2)
    np.testing.assert_allclose(fy1, fy2)


def test_momentum_conservation():
    rng = np.random.default_rng(1)
    x = rng.uniform(0.0, 20.0, size=(32, 2))
    fx, fy = morse_force(x, 20.0, 20.0, "periodic", 2.5, 1.8, 0.9, 1.3, 5.0)
    assert abs(fx.sum()) < 1e-12
    assert abs(fy.sum()) < 1e-12
