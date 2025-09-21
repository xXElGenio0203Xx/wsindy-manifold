import numpy as np
from wsindy_manifold.wsindy.mstls import mstls

def test_mstls_runs():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((200, 10))
    true = np.zeros(10); true[[1,4,7]] = [1.0, -0.5, 0.8]
    b = A @ true + 0.05*rng.standard_normal(200)
    S, theta = mstls(A, b, lam=1e-5, tau_schedule=(0.2, 0.1, 0.05))
    assert theta.shape == (10,)
    assert len(S) > 0
