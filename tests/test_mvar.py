import numpy as np

from wsindy_manifold.latent.mvar import fit_mvar, forecast_step, rollout


def _simulate_var2(A0, A, T):
    w = A.shape[0]
    d = A.shape[1]
    Y = np.zeros((T, d))
    rng = np.random.default_rng(1)
    Y[:w] = rng.normal(scale=0.1, size=(w, d))
    for t in range(w, T):
        state = A0.copy()
        for j in range(1, w + 1):
            state += A[j - 1] @ Y[t - j]
        Y[t] = state
    return Y


def test_fit_mvar_recovers_coefficients():
    A0 = np.array([0.2, -0.1])
    A = np.array(
        [
            [[0.6, 0.05], [-0.02, 0.55]],
            [[0.1, 0.0], [0.0, 0.08]],
        ]
    )
    T = 200
    Y = _simulate_var2(A0, A, T)

    model = fit_mvar(Y, w=2)
    assert np.allclose(model["A0"], A0, atol=1e-6)
    assert np.allclose(model["A"], A, atol=1e-6)


def test_forecast_and_rollout_match_ground_truth():
    A0 = np.array([0.1, -0.05])
    A = np.array(
        [
            [[0.7, 0.1], [-0.05, 0.6]],
            [[0.05, 0.02], [0.0, 0.1]],
        ]
    )
    T = 120
    Y = _simulate_var2(A0, A, T)
    model = fit_mvar(Y, w=2)

    idx = 50
    history = Y[idx - model["w"] : idx]
    pred = forecast_step(history, model)
    assert np.allclose(pred, Y[idx], atol=1e-6)

    steps = 5
    rollout_result = rollout(history, steps=steps, model=model)
    assert np.allclose(rollout_result, Y[idx : idx + steps], atol=1e-6)
