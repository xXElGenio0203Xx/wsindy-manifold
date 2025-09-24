import numpy as np

from rectsim.metrics import angular_momentum, polarization, abs_angular_momentum


def test_polarization_unity():
    v = np.tile(np.array([[1.0, 0.0]]), (10, 1))
    assert polarization(v) == 1.0


def test_angular_momentum_zero():
    x = np.column_stack((np.linspace(0, 1, 5), np.zeros(5)))
    v = np.tile(np.array([[1.0, 0.0]]), (5, 1))
    assert angular_momentum(x, v) == 0.0


def test_ring_positive_angular_momentum():
    theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    x = np.column_stack((np.cos(theta), np.sin(theta)))
    v = np.column_stack((-np.sin(theta), np.cos(theta)))
    assert angular_momentum(x, v) > 0
    assert abs_angular_momentum(x, v) > 0
