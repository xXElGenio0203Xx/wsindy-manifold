import numpy as np

from rectsim.dynamics import _alignment_step


def test_alignment_step_basic():
    # Three particles where two have the same direction; the third should tilt toward that mean.
    x = np.array([[0.0, 0.0], [0.4, 0.0], [0.8, 0.0]])
    # first two point right, third points up
    v = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    Lx = Ly = 10.0
    bc = "reflecting"
    radius = 1.0
    rate = 1.0
    dt = 0.1
    target_speed = 0.5

    new_v = _alignment_step(x, v, Lx, Ly, bc, radius, rate, dt, target_speed)
    # The third particle should gain a positive x-component (tilt toward the mean rightward direction).
    assert new_v[2][0] > 0
