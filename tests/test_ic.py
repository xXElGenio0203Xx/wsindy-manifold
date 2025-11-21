"""Tests for initial condition generation module."""

import numpy as np
import pytest

from rectsim.ic import sample_initial_positions


class TestUniformIC:
    """Tests for uniform initial condition generation."""

    def test_uniform_within_bounds(self):
        """Uniform IC should produce positions within [0,Lx]×[0,Ly]."""
        rng = np.random.default_rng(42)
        N, Lx, Ly = 100, 20.0, 15.0
        pos = sample_initial_positions("uniform", N, Lx, Ly, rng)
        
        assert pos.shape == (N, 2)
        assert np.all(pos[:, 0] >= 0.0) and np.all(pos[:, 0] <= Lx)
        assert np.all(pos[:, 1] >= 0.0) and np.all(pos[:, 1] <= Ly)

    def test_uniform_reproducible(self):
        """Same seed should produce same positions."""
        N, Lx, Ly = 50, 10.0, 10.0
        
        rng1 = np.random.default_rng(123)
        pos1 = sample_initial_positions("uniform", N, Lx, Ly, rng1)
        
        rng2 = np.random.default_rng(123)
        pos2 = sample_initial_positions("uniform", N, Lx, Ly, rng2)
        
        np.testing.assert_array_equal(pos1, pos2)


class TestGaussianIC:
    """Tests for Gaussian blob initial condition generation."""

    def test_gaussian_within_bounds(self):
        """Gaussian IC should clip positions to domain bounds."""
        rng = np.random.default_rng(42)
        N, Lx, Ly = 100, 20.0, 15.0
        pos = sample_initial_positions("gaussian", N, Lx, Ly, rng)
        
        assert pos.shape == (N, 2)
        assert np.all(pos[:, 0] >= 0.0) and np.all(pos[:, 0] <= Lx)
        assert np.all(pos[:, 1] >= 0.0) and np.all(pos[:, 1] <= Ly)

    def test_gaussian_clustered(self):
        """Gaussian IC should create a concentrated distribution."""
        rng = np.random.default_rng(42)
        N, Lx, Ly = 200, 20.0, 20.0
        pos = sample_initial_positions("gaussian", N, Lx, Ly, rng)
        
        # Measure concentration: standard deviation should be small
        std_x = np.std(pos[:, 0])
        std_y = np.std(pos[:, 1])
        
        # Should be much smaller than uniform distribution (std ≈ Lx/sqrt(12))
        assert std_x < Lx / 4
        assert std_y < Ly / 4

    def test_gaussian_reproducible(self):
        """Same seed should produce same positions."""
        N, Lx, Ly = 50, 10.0, 10.0
        
        rng1 = np.random.default_rng(123)
        pos1 = sample_initial_positions("gaussian", N, Lx, Ly, rng1)
        
        rng2 = np.random.default_rng(123)
        pos2 = sample_initial_positions("gaussian", N, Lx, Ly, rng2)
        
        np.testing.assert_array_equal(pos1, pos2)


class TestRingIC:
    """Tests for ring initial condition generation."""

    def test_ring_within_bounds(self):
        """Ring IC should produce positions within domain."""
        rng = np.random.default_rng(42)
        N, Lx, Ly = 100, 20.0, 15.0
        pos = sample_initial_positions("ring", N, Lx, Ly, rng)
        
        assert pos.shape == (N, 2)
        assert np.all(pos[:, 0] >= 0.0) and np.all(pos[:, 0] <= Lx)
        assert np.all(pos[:, 1] >= 0.0) and np.all(pos[:, 1] <= Ly)

    def test_ring_centered(self):
        """Ring IC should be centered around domain center."""
        rng = np.random.default_rng(42)
        N, Lx, Ly = 200, 20.0, 20.0
        pos = sample_initial_positions("ring", N, Lx, Ly, rng)
        
        # Mean position should be near center
        mean_x = np.mean(pos[:, 0])
        mean_y = np.mean(pos[:, 1])
        
        assert np.abs(mean_x - Lx/2) < 1.0
        assert np.abs(mean_y - Ly/2) < 1.0

    def test_ring_shape(self):
        """Ring IC should have particles at similar distances from center."""
        rng = np.random.default_rng(42)
        N, Lx, Ly = 200, 20.0, 20.0
        pos = sample_initial_positions("ring", N, Lx, Ly, rng)
        
        center = np.array([Lx/2, Ly/2])
        distances = np.linalg.norm(pos - center, axis=1)
        
        # Most particles should be at similar radii (before clipping)
        # Standard deviation of distances should be small
        std_dist = np.std(distances)
        assert std_dist < 1.0  # Small radial spread

    def test_ring_reproducible(self):
        """Same seed should produce same positions."""
        N, Lx, Ly = 50, 10.0, 10.0
        
        rng1 = np.random.default_rng(123)
        pos1 = sample_initial_positions("ring", N, Lx, Ly, rng1)
        
        rng2 = np.random.default_rng(123)
        pos2 = sample_initial_positions("ring", N, Lx, Ly, rng2)
        
        np.testing.assert_array_equal(pos1, pos2)


class TestClusterIC:
    """Tests for multi-cluster initial condition generation."""

    def test_cluster_within_bounds(self):
        """Cluster IC should produce positions within domain."""
        rng = np.random.default_rng(42)
        N, Lx, Ly = 100, 20.0, 15.0
        pos = sample_initial_positions("cluster", N, Lx, Ly, rng)
        
        assert pos.shape == (N, 2)
        assert np.all(pos[:, 0] >= 0.0) and np.all(pos[:, 0] <= Lx)
        assert np.all(pos[:, 1] >= 0.0) and np.all(pos[:, 1] <= Ly)

    def test_cluster_multimodal(self):
        """Cluster IC should create multiple concentrated regions."""
        rng = np.random.default_rng(42)
        N, Lx, Ly = 400, 20.0, 20.0
        pos = sample_initial_positions("cluster", N, Lx, Ly, rng)
        
        # Use K-means or simple heuristic to detect multiple modes
        # For simplicity, check that distribution is more spread than single Gaussian
        # but more structured than uniform
        std_x = np.std(pos[:, 0])
        std_y = np.std(pos[:, 1])
        
        # Should be between gaussian (very concentrated) and uniform (very spread)
        assert std_x > Lx / 7  # More spread than single Gaussian (relaxed threshold)
        assert std_x < Lx / 3  # Less spread than uniform
        assert std_y > Ly / 7
        assert std_y < Ly / 3

    def test_cluster_reproducible(self):
        """Same seed should produce same positions."""
        N, Lx, Ly = 50, 10.0, 10.0
        
        rng1 = np.random.default_rng(123)
        pos1 = sample_initial_positions("cluster", N, Lx, Ly, rng1)
        
        rng2 = np.random.default_rng(123)
        pos2 = sample_initial_positions("cluster", N, Lx, Ly, rng2)
        
        np.testing.assert_array_equal(pos1, pos2)

    def test_cluster_total_count(self):
        """Cluster IC should produce exactly N particles."""
        rng = np.random.default_rng(42)
        N, Lx, Ly = 157, 20.0, 15.0  # Use non-round number
        pos = sample_initial_positions("cluster", N, Lx, Ly, rng)
        
        assert pos.shape[0] == N


class TestICTypeValidation:
    """Tests for IC type validation and error handling."""

    def test_invalid_ic_type_raises(self):
        """Invalid IC type should raise ValueError."""
        rng = np.random.default_rng(42)
        
        with pytest.raises(ValueError, match="Unknown ic_type"):
            sample_initial_positions("invalid_type", 100, 20.0, 20.0, rng)

    def test_all_valid_types(self):
        """All documented IC types should work."""
        rng = np.random.default_rng(42)
        N, Lx, Ly = 50, 10.0, 10.0
        
        for ic_type in ["uniform", "gaussian", "ring", "cluster"]:
            pos = sample_initial_positions(ic_type, N, Lx, Ly, rng)
            assert pos.shape == (N, 2)
