import numpy as np

from wsindy_manifold.latent.metrics import frame_metrics, series_metrics


def test_frame_metrics_basic():
    dx = dy = 0.5
    r_true = np.array([1.0, 1.0, 1.0, 1.0])
    r_pred = np.array([0.9, 1.1, 1.0, 1.0])
    metrics = frame_metrics(r_true, r_pred, dx, dy, grid_shape=(2, 2))
    assert np.isclose(metrics["mass_true"], 1.0)
    assert np.isclose(metrics["mass_pred"], 1.0)
    assert metrics["rmse"] > 0
    assert 0.0 <= metrics["r2"] <= 1.0


def test_series_metrics_tolerance():
    dx = dy = 1.0
    r_true = np.tile(np.array([[0.5, 0.5]]), (5, 1))
    r_pred = r_true.copy()
    r_pred[3:] += 0.2
    times = np.linspace(0.0, 4.0, 5)
    metrics = series_metrics(r_true, r_pred, times, dx, dy, grid_shape=(1, 2))
    assert metrics["t_tol"] == times[3]
    aggregate = metrics["aggregate"]
    assert aggregate["mae_mean"] >= 0.0
