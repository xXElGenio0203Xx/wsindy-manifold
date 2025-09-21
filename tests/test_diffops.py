import numpy as np
from wsindy_manifold.geometry.diffops import grad, laplace_beltrami

def test_diffops_shapes():
    f = np.sin(np.linspace(0, 2*np.pi, 128))
    g = grad(f, None)
    Lf = laplace_beltrami(f, None)
    assert g.shape == f.shape
    assert Lf.shape == f.shape
