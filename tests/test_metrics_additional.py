import numpy as np

from rectsim.metrics import mean_relative_error, r2, rmse, tolerance_horizon


def test_rmse_zero_for_identical_arrays():
    arr = np.array([1.0, 2.0, 3.0])
    assert rmse(arr, arr) == 0.0


def test_r2_perfect_prediction():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert r2(y_true, y_pred) == 1.0


def test_mean_relative_error_eps():
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([0.0, 1.5])
    rel = mean_relative_error(y_true, y_pred)
    assert rel > 0


def test_tolerance_horizon_basic():
    rel = np.array([0.05, 0.09, 0.12, 0.2])
    idx = tolerance_horizon(rel, tol=0.1)
    assert idx == 2
