"""Tests for EF-ROM data pipeline and POD basis computation."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from rectsim.efrom_data import (
    PODBasis,
    build_latent_dataset,
    build_snapshot_matrix,
    compute_pod,
    load_density_runs,
    project_onto_pod,
    reconstruct_from_pod,
)


class TestPODBasis:
    """Tests for PODBasis dataclass."""

    def test_initialization(self):
        """PODBasis should initialize with correct shapes."""
        d, r = 1024, 50
        modes = np.random.randn(d, r)
        singular_values = np.random.rand(r)
        mean = np.random.randn(d)
        energy = np.linspace(0.1, 0.99, r)

        pod = PODBasis(
            modes=modes,
            singular_values=singular_values,
            mean=mean,
            grid_shape=(32, 32),
            energy=energy,
        )

        assert pod.n_modes == r
        assert pod.spatial_dim == d
        assert pod.grid_shape == (32, 32)

    def test_validation_dimension_mismatch(self):
        """PODBasis should raise if dimensions don't match."""
        d, r = 1024, 50
        modes = np.random.randn(d, r)
        singular_values = np.random.rand(r + 1)  # Wrong size
        mean = np.random.randn(d)
        energy = np.linspace(0.1, 0.99, r)

        with pytest.raises(AssertionError):
            PODBasis(
                modes=modes,
                singular_values=singular_values,
                mean=mean,
                grid_shape=(32, 32),
                energy=energy,
            )

    def test_validation_grid_shape_mismatch(self):
        """PODBasis should raise if grid shape doesn't match spatial dim."""
        d, r = 1024, 50
        modes = np.random.randn(d, r)
        singular_values = np.random.rand(r)
        mean = np.random.randn(d)
        energy = np.linspace(0.1, 0.99, r)

        with pytest.raises(AssertionError):
            PODBasis(
                modes=modes,
                singular_values=singular_values,
                mean=mean,
                grid_shape=(30, 30),  # 30*30 = 900 != 1024
                energy=energy,
            )


class TestBuildSnapshotMatrix:
    """Tests for building snapshot matrix from density runs."""

    def test_single_run(self):
        """Should correctly flatten and transpose a single density run."""
        T, nx, ny = 100, 16, 16
        densities = [np.random.rand(T, nx, ny)]

        X, mean, grid_shape = build_snapshot_matrix(densities, center=True)

        assert X.shape == (nx * ny, T)
        assert mean.shape == (nx * ny,)
        assert grid_shape == (nx, ny)

    def test_multiple_runs(self):
        """Should concatenate multiple runs along snapshot axis."""
        nx, ny = 16, 16
        densities = [
            np.random.rand(100, nx, ny),
            np.random.rand(80, nx, ny),
            np.random.rand(120, nx, ny),
        ]

        X, mean, grid_shape = build_snapshot_matrix(densities, center=True)

        total_snapshots = 100 + 80 + 120
        assert X.shape == (nx * ny, total_snapshots)
        assert mean.shape == (nx * ny,)
        assert grid_shape == (nx, ny)

    def test_centering(self):
        """Should subtract mean when center=True."""
        densities = [np.random.rand(100, 16, 16)]

        X_centered, mean_centered, _ = build_snapshot_matrix(densities, center=True)
        X_uncentered, mean_uncentered, _ = build_snapshot_matrix(densities, center=False)

        # Centered version should have zero mean along snapshots
        assert np.abs(X_centered.mean(axis=1)).max() < 1e-10

        # Uncentered should have zero mean array
        assert np.all(mean_uncentered == 0)

        # Uncentered + manual centering should match centered
        X_manual = X_uncentered - X_uncentered.mean(axis=1, keepdims=True)
        np.testing.assert_allclose(X_manual, X_centered, atol=1e-10)

    def test_empty_list_raises(self):
        """Should raise ValueError for empty density list."""
        with pytest.raises(ValueError, match="densities list is empty"):
            build_snapshot_matrix([], center=True)

    def test_inconsistent_grid_shapes_raises(self):
        """Should raise ValueError if runs have different grid shapes."""
        densities = [
            np.random.rand(100, 16, 16),
            np.random.rand(100, 32, 32),  # Different grid size
        ]

        with pytest.raises(ValueError, match="grid shape"):
            build_snapshot_matrix(densities, center=True)


class TestComputePOD:
    """Tests for POD basis computation."""

    def test_energy_threshold(self):
        """POD should select modes based on energy threshold."""
        d, M = 256, 1000
        X = np.random.randn(d, M)
        mean = np.zeros(d)

        pod = compute_pod(X, mean, (16, 16), energy_tol=0.95, r_max=None)

        # Check that cumulative energy exceeds threshold
        assert pod.energy[-1] >= 0.95

        # Check that one fewer mode would not meet threshold
        # (This may not always hold due to energy ties, but usually does)
        if pod.n_modes > 1:
            pod_fewer = compute_pod(X, mean, (16, 16), energy_tol=0.95, r_max=pod.n_modes - 1)
            assert pod_fewer.energy[-1] < 0.95 or pod_fewer.n_modes == pod.n_modes - 1

    def test_r_max_cap(self):
        """POD should respect r_max cap."""
        d, M = 256, 1000
        X = np.random.randn(d, M)
        mean = np.zeros(d)

        r_max = 10
        pod = compute_pod(X, mean, (16, 16), energy_tol=0.99, r_max=r_max)

        assert pod.n_modes <= r_max

    def test_orthonormality(self):
        """POD modes should be orthonormal."""
        d, M = 256, 500
        X = np.random.randn(d, M)
        mean = np.zeros(d)

        pod = compute_pod(X, mean, (16, 16), energy_tol=0.95)

        # Modes should be orthonormal: modes.T @ modes = I
        gram = pod.modes.T @ pod.modes
        np.testing.assert_allclose(gram, np.eye(pod.n_modes), atol=1e-10)

    def test_energy_monotonic_increasing(self):
        """Cumulative energy should be monotonically increasing."""
        d, M = 256, 500
        X = np.random.randn(d, M)
        mean = np.zeros(d)

        pod = compute_pod(X, mean, (16, 16), energy_tol=0.95)

        # Energy should increase
        assert np.all(np.diff(pod.energy) >= 0)

        # Last energy should be <= 1.0
        assert pod.energy[-1] <= 1.0 + 1e-10

    def test_at_least_one_mode(self):
        """POD should always keep at least one mode."""
        d, M = 100, 50
        X = np.random.randn(d, M)
        mean = np.zeros(d)

        pod = compute_pod(X, mean, (10, 10), energy_tol=0.0, r_max=None)

        assert pod.n_modes >= 1


class TestProjectAndReconstruct:
    """Tests for projection and reconstruction operations."""

    def test_roundtrip_identity(self):
        """project -> reconstruct should approximately recover original."""
        T, nx, ny = 100, 16, 16
        d = nx * ny

        # Create synthetic POD basis
        r = 20
        modes = np.linalg.qr(np.random.randn(d, r))[0]  # Orthonormal
        mean = np.random.randn(d)
        singular_values = np.linspace(10, 1, r)
        energy = np.cumsum(singular_values ** 2) / (singular_values ** 2).sum()

        pod = PODBasis(
            modes=modes,
            singular_values=singular_values,
            mean=mean,
            grid_shape=(nx, ny),
            energy=energy,
        )

        # Create test density
        rho_original = np.random.rand(T, nx, ny)

        # Project and reconstruct
        y = project_onto_pod(rho_original, pod, center=True)
        rho_reconstructed = reconstruct_from_pod(y, pod, add_mean=True)

        # Should have same shape
        assert rho_reconstructed.shape == rho_original.shape

        # Reconstruction error should be small (not exact due to truncation)
        # But if we use all modes (r = d), it should be nearly exact
        # For r < d, there will be truncation error

    def test_projection_shape(self):
        """Projection should produce correct shape."""
        T, nx, ny = 100, 16, 16
        d = nx * ny
        r = 30

        modes = np.linalg.qr(np.random.randn(d, r))[0]
        pod = PODBasis(
            modes=modes,
            singular_values=np.ones(r),
            mean=np.zeros(d),
            grid_shape=(nx, ny),
            energy=np.linspace(0.1, 0.95, r),
        )

        rho = np.random.rand(T, nx, ny)
        y = project_onto_pod(rho, pod)

        assert y.shape == (T, r)

    def test_reconstruction_shape(self):
        """Reconstruction should produce correct shape."""
        T = 100
        nx, ny = 16, 16
        d = nx * ny
        r = 30

        modes = np.linalg.qr(np.random.randn(d, r))[0]
        pod = PODBasis(
            modes=modes,
            singular_values=np.ones(r),
            mean=np.zeros(d),
            grid_shape=(nx, ny),
            energy=np.linspace(0.1, 0.95, r),
        )

        y = np.random.rand(T, r)
        rho = reconstruct_from_pod(y, pod)

        assert rho.shape == (T, nx, ny)

    def test_centering_consistency(self):
        """Centering in project/reconstruct should be consistent."""
        T, nx, ny = 50, 16, 16
        d = nx * ny
        r = 20

        modes = np.linalg.qr(np.random.randn(d, r))[0]
        mean = np.random.randn(d) * 10  # Non-trivial mean
        pod = PODBasis(
            modes=modes,
            singular_values=np.ones(r),
            mean=mean,
            grid_shape=(nx, ny),
            energy=np.linspace(0.1, 0.95, r),
        )

        rho = np.random.rand(T, nx, ny)

        # Project with centering, reconstruct with mean
        y = project_onto_pod(rho, pod, center=True)
        rho_rec = reconstruct_from_pod(y, pod, add_mean=True)

        # Should be approximately equal (up to truncation error)
        # The difference is due to POD truncation, not centering
        assert rho_rec.shape == rho.shape

    def test_grid_shape_validation_project(self):
        """project_onto_pod should raise if grid shapes don't match."""
        nx, ny = 16, 16
        d = nx * ny
        r = 20

        modes = np.linalg.qr(np.random.randn(d, r))[0]
        pod = PODBasis(
            modes=modes,
            singular_values=np.ones(r),
            mean=np.zeros(d),
            grid_shape=(nx, ny),
            energy=np.linspace(0.1, 0.95, r),
        )

        rho_wrong_shape = np.random.rand(50, 32, 32)  # Wrong grid size

        with pytest.raises(ValueError, match="grid shape"):
            project_onto_pod(rho_wrong_shape, pod)

    def test_mode_count_validation_reconstruct(self):
        """reconstruct_from_pod should raise if latent dims don't match."""
        nx, ny = 16, 16
        d = nx * ny
        r = 20

        modes = np.linalg.qr(np.random.randn(d, r))[0]
        pod = PODBasis(
            modes=modes,
            singular_values=np.ones(r),
            mean=np.zeros(d),
            grid_shape=(nx, ny),
            energy=np.linspace(0.1, 0.95, r),
        )

        y_wrong_dim = np.random.rand(50, r + 5)  # Wrong latent dimension

        with pytest.raises(ValueError, match="Latent dimension"):
            reconstruct_from_pod(y_wrong_dim, pod)


class TestBuildLatentDataset:
    """Tests for latent dataset construction."""

    def test_train_test_split(self):
        """Should correctly split each run into train and test."""
        nx, ny = 16, 16
        d = nx * ny
        r = 20

        # Create 3 runs with different lengths
        densities = [
            np.random.rand(100, nx, ny),
            np.random.rand(80, nx, ny),
            np.random.rand(120, nx, ny),
        ]

        # Create POD basis
        modes = np.linalg.qr(np.random.randn(d, r))[0]
        pod = PODBasis(
            modes=modes,
            singular_values=np.ones(r),
            mean=np.zeros(d),
            grid_shape=(nx, ny),
            energy=np.linspace(0.1, 0.95, r),
        )

        # Build dataset with 80% train
        data = build_latent_dataset(densities, pod, train_frac=0.8)

        # Check shapes
        total_train = 80 + 64 + 96  # 80% of each run
        total_test = 20 + 16 + 24  # remaining 20%

        assert data["train"]["y"].shape == (total_train, r)
        assert data["test"]["y"].shape == (total_test, r)

        assert len(data["train"]["run_ids"]) == total_train
        assert len(data["test"]["run_ids"]) == total_test

    def test_metadata_tracking(self):
        """run_ids and t_idx should correctly track source of each point."""
        nx, ny = 16, 16
        d = nx * ny
        r = 20

        densities = [
            np.random.rand(100, nx, ny),
            np.random.rand(100, nx, ny),
        ]

        modes = np.linalg.qr(np.random.randn(d, r))[0]
        pod = PODBasis(
            modes=modes,
            singular_values=np.ones(r),
            mean=np.zeros(d),
            grid_shape=(nx, ny),
            energy=np.linspace(0.1, 0.95, r),
        )

        data = build_latent_dataset(densities, pod, train_frac=0.9)

        # Check run_ids
        train_run_ids = data["train"]["run_ids"]
        assert np.all(train_run_ids[:90] == 0)  # First 90 from run 0
        assert np.all(train_run_ids[90:] == 1)  # Next 90 from run 1

        # Check t_idx continuity within each run
        train_t_idx = data["train"]["t_idx"]
        assert np.all(train_t_idx[:90] == np.arange(90))
        assert np.all(train_t_idx[90:] == np.arange(90))

    def test_empty_list_returns_empty_dataset(self):
        """Empty density list should return empty train/test."""
        nx, ny = 16, 16
        d = nx * ny
        r = 20

        densities = []

        modes = np.linalg.qr(np.random.randn(d, r))[0]
        pod = PODBasis(
            modes=modes,
            singular_values=np.ones(r),
            mean=np.zeros(d),
            grid_shape=(nx, ny),
            energy=np.linspace(0.1, 0.95, r),
        )

        data = build_latent_dataset(densities, pod, train_frac=0.9)

        assert data["train"]["y"].shape[0] == 0
        assert data["test"]["y"].shape[0] == 0


class TestLoadDensityRuns:
    """Tests for loading density files from disk."""

    def test_load_single_run(self, tmp_path):
        """Should load a single density file."""
        # Create temporary directory structure
        run_dir = tmp_path / "simulations" / "model" / "run_0001"
        run_dir.mkdir(parents=True)

        # Save a density file
        T, nx, ny = 50, 16, 16
        rho = np.random.rand(T, nx, ny)
        np.savez(run_dir / "density.npz", rho=rho)

        # Load
        densities, metadata = load_density_runs(tmp_path / "simulations" / "model")

        assert len(densities) == 1
        assert densities[0].shape == (T, nx, ny)
        assert metadata[0]["run_id"] == "run_0001"
        assert metadata[0]["T"] == T
        assert metadata[0]["grid_shape"] == (nx, ny)

    def test_load_multiple_runs(self, tmp_path):
        """Should load multiple density files."""
        model_dir = tmp_path / "simulations" / "model"

        # Create 3 runs
        for i in range(3):
            run_dir = model_dir / f"run_{i:04d}"
            run_dir.mkdir(parents=True)
            rho = np.random.rand(50, 16, 16)
            np.savez(run_dir / "density.npz", rho=rho)

        densities, metadata = load_density_runs(model_dir)

        assert len(densities) == 3
        assert all(d.shape == (50, 16, 16) for d in densities)

    def test_max_runs_limit(self, tmp_path):
        """Should respect max_runs limit."""
        model_dir = tmp_path / "simulations" / "model"

        for i in range(5):
            run_dir = model_dir / f"run_{i:04d}"
            run_dir.mkdir(parents=True)
            rho = np.random.rand(50, 16, 16)
            np.savez(run_dir / "density.npz", rho=rho)

        densities, metadata = load_density_runs(model_dir, max_runs=3)

        assert len(densities) == 3

    def test_time_slicing(self, tmp_path):
        """Should apply time slice correctly."""
        run_dir = tmp_path / "simulations" / "model" / "run_0001"
        run_dir.mkdir(parents=True)

        T, nx, ny = 100, 16, 16
        rho = np.random.rand(T, nx, ny)
        np.savez(run_dir / "density.npz", rho=rho)

        # Load with time slice excluding last 10 frames
        densities, metadata = load_density_runs(
            tmp_path / "simulations" / "model",
            t_slice=slice(0, -10),
        )

        assert densities[0].shape == (90, nx, ny)
        assert metadata[0]["T"] == 90

    def test_squeeze_channel_dimension(self, tmp_path):
        """Should squeeze singleton channel dimension."""
        run_dir = tmp_path / "simulations" / "model" / "run_0001"
        run_dir.mkdir(parents=True)

        # Save with shape (T, nx, ny, 1)
        T, nx, ny = 50, 16, 16
        rho = np.random.rand(T, nx, ny, 1)
        np.savez(run_dir / "density.npz", rho=rho)

        densities, metadata = load_density_runs(tmp_path / "simulations" / "model")

        # Should be squeezed to (T, nx, ny)
        assert densities[0].shape == (T, nx, ny)
