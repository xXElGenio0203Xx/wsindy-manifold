"""Tests for the new unified backend interface and noise module."""

import numpy as np
import pytest

from rectsim.domain import NeighborFinder
from rectsim.noise import angle_noise, noise_variance
from rectsim.vicsek_discrete import simulate_backend


class TestNoiseModule:
    """Tests for the centralized noise module."""
    
    def test_uniform_noise_range(self):
        """Test uniform noise stays within expected bounds."""
        rng = np.random.default_rng(42)
        eta = 1.0
        noise = angle_noise(rng, "uniform", eta, size=10000)
        
        assert noise.min() >= -eta / 2
        assert noise.max() <= eta / 2
        
    def test_uniform_noise_variance(self):
        """Test uniform noise has correct theoretical variance."""
        rng = np.random.default_rng(42)
        eta = 1.0
        noise = angle_noise(rng, "uniform", eta, size=100000)
        
        expected_var = noise_variance("uniform", eta)
        actual_var = np.var(noise)
        
        # Allow 1% tolerance due to finite sampling
        assert abs(actual_var - expected_var) / expected_var < 0.01
        
    def test_gaussian_noise_variance_matching(self):
        """Test Gaussian noise matches uniform variance when requested."""
        rng = np.random.default_rng(42)
        eta = 1.0
        
        uniform_noise = angle_noise(rng, "uniform", eta, size=100000)
        gaussian_noise = angle_noise(rng, "gaussian", eta, size=100000, match_variance=True)
        
        uniform_var = np.var(uniform_noise)
        gaussian_var = np.var(gaussian_noise)
        
        # Variances should match within 5%
        assert abs(gaussian_var - uniform_var) / uniform_var < 0.05
        
    def test_gaussian_noise_without_matching(self):
        """Test Gaussian noise uses eta as sigma when match_variance=False."""
        rng = np.random.default_rng(42)
        eta = 1.0
        
        noise = angle_noise(rng, "gaussian", eta, size=100000, match_variance=False)
        actual_var = np.var(noise)
        expected_var = eta**2
        
        # Allow 1% tolerance
        assert abs(actual_var - expected_var) / expected_var < 0.01
        
    def test_noise_kind_validation(self):
        """Test that invalid noise kind raises error."""
        rng = np.random.default_rng(42)
        
        with pytest.raises(ValueError, match="Unknown noise kind"):
            angle_noise(rng, "invalid", 1.0, size=10)


class TestNeighborFinder:
    """Tests for the unified NeighborFinder API."""
    
    def test_neighbor_finder_includes_self(self):
        """Test that each particle includes itself in its neighbor list."""
        nf = NeighborFinder(Lx=10.0, Ly=10.0, rcut=1.0, bc="periodic")
        x = np.random.uniform(0, 10, size=(20, 2))
        nf.rebuild(x)
        neighbors = nf.neighbors_of(x)
        
        for i in range(len(neighbors)):
            assert i in neighbors[i], f"Particle {i} not in its own neighbor list"
            
    def test_neighbor_finder_periodic_boundaries(self):
        """Test neighbor finding across periodic boundaries."""
        nf = NeighborFinder(Lx=10.0, Ly=10.0, rcut=1.0, bc="periodic")
        
        # Place particles at opposite corners (should be neighbors with periodic BC)
        x = np.array([[0.2, 0.2], [9.8, 9.8]])
        nf.rebuild(x)
        neighbors = nf.neighbors_of(x)
        
        # Both particles should see each other
        assert 1 in neighbors[0]
        assert 0 in neighbors[1]
        
    def test_neighbor_finder_rebuild_required(self):
        """Test that neighbors_of() raises error before rebuild()."""
        nf = NeighborFinder(Lx=10.0, Ly=10.0, rcut=1.0, bc="periodic")
        x = np.random.uniform(0, 10, size=(20, 2))
        
        with pytest.raises(RuntimeError, match="Must call rebuild"):
            nf.neighbors_of(x)
            
    def test_neighbor_finder_small_cutoff(self):
        """Test neighbor finding with small cutoff radius."""
        nf = NeighborFinder(Lx=10.0, Ly=10.0, rcut=0.1, bc="periodic")
        
        # Widely spaced particles
        x = np.array([[0.0, 0.0], [5.0, 5.0], [9.0, 9.0]])
        nf.rebuild(x)
        neighbors = nf.neighbors_of(x)
        
        # Each particle should only see itself
        for i in range(len(neighbors)):
            assert neighbors[i].tolist() == [i]


class TestUnifiedBackend:
    """Tests for the unified simulate_backend interface."""
    
    def test_backend_returns_correct_keys(self):
        """Test that backend returns all required keys."""
        config = {
            "sim": {"N": 20, "Lx": 10.0, "Ly": 10.0, "bc": "periodic",
                   "T": 5.0, "dt": 0.1, "save_every": 5, "neighbor_rebuild": 2},
            "model": {"type": "vicsek_discrete", "speed": 0.5},
            "noise": {"kind": "uniform", "eta": 0.5, "match_variance": True},
            "forces": {"enabled": False},
            "params": {"R": 1.0}
        }
        
        rng = np.random.default_rng(42)
        result = simulate_backend(config, rng)
        
        required_keys = {"times", "traj", "vel", "head", "meta"}
        assert set(result.keys()) == required_keys
        
    def test_backend_output_shapes(self):
        """Test that backend outputs have correct shapes."""
        N = 30
        T = 10.0
        dt = 0.1
        save_every = 5
        
        config = {
            "sim": {"N": N, "Lx": 10.0, "Ly": 10.0, "bc": "periodic",
                   "T": T, "dt": dt, "save_every": save_every, "neighbor_rebuild": 2},
            "model": {"speed": 0.5},
            "noise": {"kind": "uniform", "eta": 0.5},
            "forces": {"enabled": False},
            "params": {"R": 1.0}
        }
        
        rng = np.random.default_rng(42)
        result = simulate_backend(config, rng)
        
        expected_frames = int(T / dt) // save_every + 1
        
        assert result["times"].shape == (expected_frames,)
        assert result["traj"].shape == (expected_frames, N, 2)
        assert result["vel"].shape == (expected_frames, N, 2)
        assert result["head"].shape == (expected_frames, N, 2)
        
    def test_backend_unit_headings(self):
        """Test that all headings are unit vectors."""
        config = {
            "sim": {"N": 50, "Lx": 10.0, "Ly": 10.0, "bc": "periodic",
                   "T": 5.0, "dt": 0.1, "save_every": 5, "neighbor_rebuild": 2},
            "model": {"speed": 0.5},
            "noise": {"kind": "uniform", "eta": 0.5},
            "forces": {"enabled": False},
            "params": {"R": 1.0}
        }
        
        rng = np.random.default_rng(42)
        result = simulate_backend(config, rng)
        
        norms = np.linalg.norm(result["head"], axis=2)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-6)
        
    def test_backend_velocity_speed_relationship(self):
        """Test that velocity magnitude equals configured speed."""
        v0 = 0.75
        config = {
            "sim": {"N": 50, "Lx": 10.0, "Ly": 10.0, "bc": "periodic",
                   "T": 5.0, "dt": 0.1, "save_every": 5, "neighbor_rebuild": 2},
            "model": {"speed": v0},
            "noise": {"kind": "uniform", "eta": 0.5},
            "forces": {"enabled": False},
            "params": {"R": 1.0}
        }
        
        rng = np.random.default_rng(42)
        result = simulate_backend(config, rng)
        
        speeds = np.linalg.norm(result["vel"], axis=2)
        np.testing.assert_allclose(speeds, v0, rtol=1e-6)
        
    def test_backend_timestep_validation(self):
        """Test that backend validates v0*dt < 0.5*R constraint."""
        config = {
            "sim": {"N": 20, "Lx": 10.0, "Ly": 10.0, "bc": "periodic",
                   "T": 5.0, "dt": 2.0, "save_every": 1, "neighbor_rebuild": 1},
            "model": {"speed": 1.0},
            "noise": {"kind": "uniform", "eta": 0.5},
            "forces": {"enabled": False},
            "params": {"R": 1.0}
        }
        
        rng = np.random.default_rng(42)
        
        with pytest.raises(ValueError, match="Time step too large"):
            simulate_backend(config, rng)
            
    def test_backend_with_gaussian_noise(self):
        """Test backend works with Gaussian noise."""
        config = {
            "sim": {"N": 50, "Lx": 10.0, "Ly": 10.0, "bc": "periodic",
                   "T": 10.0, "dt": 0.1, "save_every": 10, "neighbor_rebuild": 5},
            "model": {"speed": 0.5},
            "noise": {"kind": "gaussian", "eta": 0.5, "match_variance": True},
            "forces": {"enabled": False},
            "params": {"R": 1.0}
        }
        
        rng = np.random.default_rng(42)
        result = simulate_backend(config, rng)
        
        # Should complete without errors
        assert result["times"].shape[0] > 0
        
    def test_backend_reflecting_boundaries(self):
        """Test backend with reflecting boundary conditions."""
        config = {
            "sim": {"N": 30, "Lx": 10.0, "Ly": 10.0, "bc": "reflecting",
                   "T": 5.0, "dt": 0.1, "save_every": 5, "neighbor_rebuild": 2},
            "model": {"speed": 0.5},
            "noise": {"kind": "uniform", "eta": 0.5},
            "forces": {"enabled": False},
            "params": {"R": 1.0}
        }
        
        rng = np.random.default_rng(42)
        result = simulate_backend(config, rng)
        
        # All positions should be within domain
        assert np.all(result["traj"] >= 0.0)
        assert np.all(result["traj"][:, :, 0] <= config["sim"]["Lx"])
        assert np.all(result["traj"][:, :, 1] <= config["sim"]["Ly"])
        
    def test_backend_deterministic_with_seed(self):
        """Test that backend gives reproducible results with same seed."""
        config = {
            "sim": {"N": 30, "Lx": 10.0, "Ly": 10.0, "bc": "periodic",
                   "T": 5.0, "dt": 0.1, "save_every": 5, "neighbor_rebuild": 2},
            "model": {"speed": 0.5},
            "noise": {"kind": "uniform", "eta": 0.5},
            "forces": {"enabled": False},
            "params": {"R": 1.0}
        }
        
        rng1 = np.random.default_rng(42)
        result1 = simulate_backend(config, rng1)
        
        rng2 = np.random.default_rng(42)
        result2 = simulate_backend(config, rng2)
        
        np.testing.assert_array_equal(result1["traj"], result2["traj"])
        np.testing.assert_array_equal(result1["head"], result2["head"])
