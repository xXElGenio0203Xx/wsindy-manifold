import numpy as np

from rectsim.morse import _morse_pair_force


def test_morse_force_signs():
    # Choose parameters where lr < la so repulsion is shorter-ranged than attraction
    Cr = 3.0
    Ca = 1.0
    lr = 0.5
    la = 1.0

    # Short distance -> expect net repulsion (f > 0)
    r_short = 0.1
    assert _morse_pair_force(r_short, Cr, Ca, lr, la) > 0

    # Long distance -> expect net attraction (f < 0)
    r_long = 3.0
    assert _morse_pair_force(r_long, Cr, Ca, lr, la) < 0
