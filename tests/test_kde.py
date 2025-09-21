import numpy as np
from wsindy_manifold.density.geodesic_kde import geodesic_kde

def test_kde_shapes():
    pts = np.random.randn(100, 2)
    grid = np.random.randn(50, 2)
    vals = geodesic_kde(pts, grid, 0.3)
    assert vals.shape == (50,)
    assert np.all(np.isfinite(vals))
