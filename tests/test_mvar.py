"""Tests for the MVAR module (rectsim.mvar).

Covers:
1. Synthetic VAR data generation and recovery
2. Load density movies from disk
3. Build global snapshot matrix
4. Compute POD basis
5. Project/reconstruct with POD
6. MVAR model fitting and forecasting
7. Evaluation metrics
"""

import json
from pathlib import Path

import numpy as np
import pytest

from rectsim.mvar import (
    MVARModel,
    build_global_snapshot_matrix,
    compute_pod,
    evaluate_mvar_on_runs,
    fit_mvar_from_runs,
    load_density_movies,
    mvar_forecast,
    plot_pod_energy,
    project_to_pod,
    reconstruct_from_pod,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def synthetic_var_data():
    """Generate synthetic VAR(2) data for testing.

    Returns a dict with Y, A0, A, and ground truth parameters.
    """
    rng = np.random.default_rng(42)

    r = 5  # latent dimension
    order = 2
    T = 200

    # Generate stable AR coefficients
    A0 = rng.standard_normal(r) * 0.1

    # A[0]: lag 1, A[1]: lag 2
    A = np.zeros((order, r, r))
    A[0] = rng.standard_normal((r, r)) * 0.3
    A[1] = rng.standard_normal((r, r)) * 0.2

    # Generate time series
    Y = np.zeros((T, r))
    Y[0] = rng.standard_normal(r) * 0.5
    Y[1] = rng.standard_normal(r) * 0.5

    for t in range(order, T):
        Y[t] = A0 + A[0] @ Y[t - 1] + A[1] @ Y[t - 2]
        Y[t] += rng.standard_normal(r) * 0.05  # Small noise

    return {
        "Y": Y,
        "A0": A0,
        "A": A,
        "order": order,
        "r": r,
    }


@pytest.fixture
def mock_density_runs(tmp_path):
    """Create mock density.npz files for testing."""
    rng = np.random.default_rng(123)

    ny, nx = 32, 32
    runs = []

    for i in range(3):
        run_dir = tmp_path / f"run_{i:03d}"
        run_dir.mkdir()

        T = 100 + i * 20  # Varying lengths
        rho = rng.random((T, ny, nx))
        times = np.arange(T) * 0.1

        np.savez(run_dir / "density.npz", rho=rho, times=times)

        # Optional run.json metadata
        meta = {"run_id": i, "N": 100}
        with open(run_dir / "run.json", "w") as f:
            json.dump(meta, f)

        runs.append(run_dir)

    return runs, (ny, nx)


@pytest.fixture
def simple_pod_data():
    """Simple data for POD testing."""
    rng = np.random.default_rng(999)

    # Generate low-rank data: X = U @ S @ Vt
    T, d = 100, 50
    r_true = 5

    U_true = rng.standard_normal((T, r_true))
    S_true = np.array([10.0, 5.0, 3.0, 2.0, 1.0])
    Vt_true = rng.standard_normal((r_true, d))

    # Orthonormalize
    U_true, _ = np.linalg.qr(U_true)
    Vt_true, _ = np.linalg.qr(Vt_true.T)
    Vt_true = Vt_true.T

    X = U_true @ np.diag(S_true) @ Vt_true

    return X, r_true


# ============================================================================
# Test: Synthetic VAR recovery
# ============================================================================


class TestSyntheticVAR:
    """Test MVAR fitting on synthetic VAR data."""

    def test_synthetic_var_recovery(self, synthetic_var_data):
        """Fit MVAR to synthetic VAR and check coefficient recovery."""
        Y_true = synthetic_var_data["Y"]
        A0_true = synthetic_var_data["A0"]
        A_true = synthetic_var_data["A"]
        order = synthetic_var_data["order"]

        # Prepare as latent_dict format
        latent_dict = {
            "run1": {"Y": Y_true, "times": np.arange(Y_true.shape[0])},
        }

        # Fit MVAR
        model, info = fit_mvar_from_runs(
            latent_dict,
            order=order,
            ridge=1e-8,
            train_frac=0.9,
        )

        # Check shapes
        assert model.A0.shape == A0_true.shape
        assert model.A.shape == A_true.shape

        # Check coefficient values (should be close for noiseless data)
        # Allow some error due to ridge and noise
        np.testing.assert_allclose(model.A0, A0_true, atol=0.5)
        np.testing.assert_allclose(model.A, A_true, atol=0.5)

    def test_mvar_forecast_stability(self, synthetic_var_data):
        """Check that MVAR forecast doesn't explode."""
        Y_true = synthetic_var_data["Y"]
        order = synthetic_var_data["order"]

        latent_dict = {
            "run1": {"Y": Y_true, "times": np.arange(Y_true.shape[0])},
        }

        model, _ = fit_mvar_from_runs(
            latent_dict, order=order, ridge=1e-6, train_frac=0.8
        )

        # Forecast 100 steps
        Y_init = Y_true[-order:]
        Y_pred = mvar_forecast(model, Y_init, steps=100)

        # Check shape
        assert Y_pred.shape == (100, Y_true.shape[1])

        # Check forecast doesn't explode (within reasonable bounds)
        assert np.all(np.abs(Y_pred) < 100.0)


# ============================================================================
# Test: Load density movies
# ============================================================================


class TestLoadDensityMovies:
    """Test loading density movies from disk."""

    def test_load_density_basic(self, mock_density_runs):
        """Load density movies from mock directories."""
        run_dirs, (ny, nx) = mock_density_runs

        density_dict = load_density_movies(run_dirs)

        assert len(density_dict) == 3
        assert "run_000" in density_dict
        assert "run_001" in density_dict
        assert "run_002" in density_dict

        # Check data shapes
        for run_name, data in density_dict.items():
            assert data["rho"].ndim == 3
            assert data["rho"].shape[1:] == (ny, nx)
            assert data["times"].shape[0] == data["rho"].shape[0]

    def test_load_with_metadata(self, mock_density_runs):
        """Check that metadata is loaded from run.json."""
        run_dirs, _ = mock_density_runs

        density_dict = load_density_movies(run_dirs)

        # Check metadata present
        assert "meta" in density_dict["run_000"]
        assert density_dict["run_000"]["meta"]["run_id"] == 0

    def test_load_missing_file(self, tmp_path):
        """Gracefully handle missing density.npz."""
        run_dir = tmp_path / "empty_run"
        run_dir.mkdir()

        density_dict = load_density_movies([run_dir])

        # Should be empty (warning printed)
        assert len(density_dict) == 0


# ============================================================================
# Test: Build global snapshot matrix
# ============================================================================


class TestSnapshotMatrix:
    """Test global snapshot matrix construction."""

    def test_build_snapshot_basic(self, mock_density_runs):
        """Build snapshot matrix from mock runs."""
        run_dirs, (ny, nx) = mock_density_runs
        density_dict = load_density_movies(run_dirs)

        X, run_slices, mean = build_global_snapshot_matrix(
            density_dict, subtract_mean=True
        )

        # Check shape
        d = ny * nx
        T_total = sum(v["rho"].shape[0] for v in density_dict.values())
        assert X.shape == (T_total, d)

        # Check slices
        assert len(run_slices) == 3
        for run_name, sl in run_slices.items():
            assert isinstance(sl, slice)

        # Check mean shape
        assert mean.shape == (d,)

    def test_snapshot_no_centering(self, mock_density_runs):
        """Test without mean subtraction."""
        run_dirs, _ = mock_density_runs
        density_dict = load_density_movies(run_dirs)

        X, _, mean = build_global_snapshot_matrix(
            density_dict, subtract_mean=False
        )

        # Mean should be zero array
        assert np.allclose(mean, 0.0)

    def test_snapshot_empty_dict(self):
        """Raise error on empty density_dict."""
        with pytest.raises(ValueError, match="empty"):
            build_global_snapshot_matrix({}, subtract_mean=True)


# ============================================================================
# Test: Compute POD
# ============================================================================


class TestComputePOD:
    """Test POD basis computation."""

    def test_pod_basic(self, simple_pod_data):
        """Compute POD on synthetic low-rank data."""
        X, r_true = simple_pod_data

        pod_basis = compute_pod(X, r=None, energy_threshold=0.99)

        # Check keys
        assert "Phi" in pod_basis
        assert "S" in pod_basis
        assert "U" in pod_basis
        assert "r" in pod_basis
        assert "energy" in pod_basis

        # Check shapes
        T, d = X.shape
        r = pod_basis["r"]
        assert pod_basis["Phi"].shape == (d, r)
        assert pod_basis["U"].shape == (T, r)

        # Check r is reasonable
        assert r <= min(T, d)

    def test_pod_orthonormality(self, simple_pod_data):
        """Check that POD modes are orthonormal."""
        X, _ = simple_pod_data

        pod_basis = compute_pod(X, r=10, energy_threshold=0.99)

        Phi = pod_basis["Phi"]

        # Phi.T @ Phi should be identity
        PhiT_Phi = Phi.T @ Phi
        np.testing.assert_allclose(PhiT_Phi, np.eye(10), atol=1e-10)

    def test_pod_energy_threshold(self, simple_pod_data):
        """Check that energy threshold is respected."""
        X, _ = simple_pod_data

        energy_thresh = 0.95
        pod_basis = compute_pod(X, r=None, energy_threshold=energy_thresh)

        r = pod_basis["r"]
        energy_captured = pod_basis["energy"][r - 1]

        # Energy should be >= threshold
        assert energy_captured >= energy_thresh

    def test_pod_fixed_r(self, simple_pod_data):
        """Test with fixed r parameter."""
        X, _ = simple_pod_data

        r_fixed = 8
        pod_basis = compute_pod(X, r=r_fixed, energy_threshold=0.99)

        assert pod_basis["r"] == r_fixed


# ============================================================================
# Test: Project and reconstruct
# ============================================================================


class TestProjectReconstruct:
    """Test POD projection and reconstruction."""

    def test_project_basic(self, mock_density_runs):
        """Project density fields to latent space."""
        run_dirs, (ny, nx) = mock_density_runs
        density_dict = load_density_movies(run_dirs)

        X, _, mean = build_global_snapshot_matrix(density_dict)
        pod_basis = compute_pod(X, r=10)

        latent_dict = project_to_pod(density_dict, pod_basis["Phi"], mean)

        # Check keys
        assert len(latent_dict) == len(density_dict)

        # Check shapes
        for run_name, data in latent_dict.items():
            T_r = density_dict[run_name]["rho"].shape[0]
            assert data["Y"].shape == (T_r, 10)

    def test_roundtrip(self, mock_density_runs):
        """Check projection -> reconstruction roundtrip."""
        run_dirs, (ny, nx) = mock_density_runs
        density_dict = load_density_movies(run_dirs)

        X, _, mean = build_global_snapshot_matrix(density_dict)
        pod_basis = compute_pod(X, r=20)  # High r for accurate reconstruction

        latent_dict = project_to_pod(density_dict, pod_basis["Phi"], mean)

        # Pick one run and reconstruct
        run_name = "run_000"
        Y = latent_dict[run_name]["Y"]
        rho_orig = density_dict[run_name]["rho"]

        rho_rec = reconstruct_from_pod(Y, pod_basis["Phi"], mean, ny, nx)

        # Should match closely (not exact due to truncation)
        assert rho_rec.shape == rho_orig.shape
        np.testing.assert_allclose(rho_rec, rho_orig, atol=0.5)


# ============================================================================
# Test: MVAR fitting
# ============================================================================


class TestMVARFitting:
    """Test MVAR model fitting."""

    def test_fit_basic(self, mock_density_runs):
        """Fit MVAR on mock latent data."""
        run_dirs, (ny, nx) = mock_density_runs
        density_dict = load_density_movies(run_dirs)

        X, _, mean = build_global_snapshot_matrix(density_dict)
        pod_basis = compute_pod(X, r=10)
        latent_dict = project_to_pod(density_dict, pod_basis["Phi"], mean)

        # Fit MVAR
        model, info = fit_mvar_from_runs(
            latent_dict, order=4, ridge=1e-6, train_frac=0.8
        )

        # Check model attributes
        assert model.order == 4
        assert model.latent_dim == 10
        assert model.A0.shape == (10,)
        assert model.A.shape == (4, 10, 10)
        assert model.ridge == 1e-6

        # Check info
        assert "total_samples" in info
        assert "num_runs" in info

    def test_fit_save_load(self, tmp_path, mock_density_runs):
        """Test saving and loading MVAR model."""
        run_dirs, (ny, nx) = mock_density_runs
        density_dict = load_density_movies(run_dirs)

        X, _, mean = build_global_snapshot_matrix(density_dict)
        pod_basis = compute_pod(X, r=10)
        latent_dict = project_to_pod(density_dict, pod_basis["Phi"], mean)

        model, _ = fit_mvar_from_runs(latent_dict, order=3, ridge=1e-5)

        # Save
        model_path = tmp_path / "mvar_model.npz"
        model.save(model_path)

        # Load
        model_loaded = MVARModel.load(model_path)

        # Check equality
        assert model_loaded.order == model.order
        np.testing.assert_array_equal(model_loaded.A0, model.A0)
        np.testing.assert_array_equal(model_loaded.A, model.A)


# ============================================================================
# Test: MVAR forecasting
# ============================================================================


class TestMVARForecasting:
    """Test MVAR multi-step forecasting."""

    def test_forecast_shape(self, synthetic_var_data):
        """Check forecast output shape."""
        Y_true = synthetic_var_data["Y"]
        order = synthetic_var_data["order"]

        latent_dict = {"run1": {"Y": Y_true}}
        model, _ = fit_mvar_from_runs(latent_dict, order=order)

        Y_init = Y_true[:order]
        steps = 50
        Y_pred = mvar_forecast(model, Y_init, steps)

        assert Y_pred.shape == (steps, Y_true.shape[1])

    def test_forecast_consistency(self, synthetic_var_data):
        """Check that forecast is deterministic."""
        Y_true = synthetic_var_data["Y"]
        order = synthetic_var_data["order"]

        latent_dict = {"run1": {"Y": Y_true}}
        model, _ = fit_mvar_from_runs(latent_dict, order=order)

        Y_init = Y_true[:order]

        Y_pred1 = mvar_forecast(model, Y_init, 30)
        Y_pred2 = mvar_forecast(model, Y_init, 30)

        np.testing.assert_array_equal(Y_pred1, Y_pred2)


# ============================================================================
# Test: Evaluation
# ============================================================================


class TestEvaluation:
    """Test MVAR evaluation metrics."""

    def test_evaluate_basic(self, mock_density_runs):
        """Run full evaluation pipeline."""
        run_dirs, (ny, nx) = mock_density_runs
        density_dict = load_density_movies(run_dirs)

        X, _, mean = build_global_snapshot_matrix(density_dict)
        pod_basis = compute_pod(X, r=10)
        latent_dict = project_to_pod(density_dict, pod_basis["Phi"], mean)

        model, _ = fit_mvar_from_runs(latent_dict, order=4, train_frac=0.7)

        results = evaluate_mvar_on_runs(
            model,
            latent_dict,
            density_dict,
            pod_basis,
            mean,
            ny,
            nx,
            train_frac=0.7,
        )

        # Check keys
        assert "per_run" in results
        assert "aggregate" in results

        # Check per-run results
        for run_name, run_results in results["per_run"].items():
            assert "R2" in run_results
            assert "rmse_time_series" in run_results
            assert isinstance(run_results["R2"], float)

        # Check aggregate
        if results["aggregate"]:
            assert "mean_R2" in results["aggregate"]

    def test_evaluate_R2_bounds(self, mock_density_runs):
        """Check that R² values are reasonable."""
        run_dirs, (ny, nx) = mock_density_runs
        density_dict = load_density_movies(run_dirs)

        X, _, mean = build_global_snapshot_matrix(density_dict)
        pod_basis = compute_pod(X, r=15)
        latent_dict = project_to_pod(density_dict, pod_basis["Phi"], mean)

        model, _ = fit_mvar_from_runs(latent_dict, order=4, train_frac=0.8)

        results = evaluate_mvar_on_runs(
            model, latent_dict, density_dict, pod_basis, mean, ny, nx
        )

        # R² should be in reasonable range (can be negative for bad fits)
        for run_name, run_results in results["per_run"].items():
            R2 = run_results["R2"]
            assert -10.0 <= R2 <= 1.0  # Allow some slack for bad fits


# ============================================================================
# Test: Plotting
# ============================================================================


class TestPlotting:
    """Test POD energy plotting."""

    def test_plot_pod_energy(self, tmp_path, simple_pod_data):
        """Generate POD energy plot."""
        X, _ = simple_pod_data
        pod_basis = compute_pod(X, r=10)

        out_path = tmp_path / "pod_energy.png"
        plot_pod_energy(pod_basis["S"], out_path, r_mark=10, energy_threshold=0.99)

        # Check file created
        assert out_path.exists()
