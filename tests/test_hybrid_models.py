"""Tests for hybrid Vicsek-D'Orsogna models (discrete and continuous)."""

import numpy as np
import pytest
from rectsim.morse import morse_force, minimal_image_displacement
from rectsim.vicsek_discrete import simulate_backend
from rectsim.dynamics import simulate


class TestMinimalImageDisplacement:
    """Test periodic minimal image convention."""
    
    def test_within_half_box(self):
        """Displacements within [-L/2, L/2) should be unchanged."""
        L = 10.0
        assert minimal_image_displacement(2.0, L) == pytest.approx(2.0)
        assert minimal_image_displacement(-3.0, L) == pytest.approx(-3.0)
        assert minimal_image_displacement(4.9, L) == pytest.approx(4.9)
        
    def test_wrap_positive(self):
        """Large positive displacements should wrap."""
        L = 10.0
        # 8.0 is closer via -2.0
        assert minimal_image_displacement(8.0, L) == pytest.approx(-2.0)
        # 11.0 wraps to 1.0
        assert minimal_image_displacement(11.0, L) == pytest.approx(1.0)
        
    def test_wrap_negative(self):
        """Large negative displacements should wrap."""
        L = 10.0
        # -8.0 is closer via 2.0
        assert minimal_image_displacement(-8.0, L) == pytest.approx(2.0)
        # -11.0 wraps to -1.0
        assert minimal_image_displacement(-11.0, L) == pytest.approx(-1.0)


class TestMorseForceSymmetry:
    """Test Morse force symmetry and conservation."""
    
    def test_force_symmetry_periodic(self):
        """Newton's third law: forces sum to zero with periodic BC."""
        N = 10
        Lx = Ly = 10.0
        x = np.random.rand(N, 2) * np.array([Lx, Ly])
        
        fx, fy = morse_force(
            x, Lx, Ly, "periodic",
            Cr=2.0, Ca=1.0, lr=0.5, la=1.5, rcut=7.5
        )
        
        # Total force should be nearly zero (Newton's 3rd law)
        assert abs(fx.sum()) < 1e-10
        assert abs(fy.sum()) < 1e-10
        
    def test_force_symmetry_reflecting(self):
        """Newton's third law with reflecting boundaries."""
        N = 10
        Lx = Ly = 10.0
        x = np.random.rand(N, 2) * np.array([Lx, Ly])
        
        fx, fy = morse_force(
            x, Lx, Ly, "reflecting",
            Cr=2.0, Ca=1.0, lr=0.5, la=1.5, rcut=7.5
        )
        
        # Total force should be nearly zero
        assert abs(fx.sum()) < 1e-10
        assert abs(fy.sum()) < 1e-10


class TestDiscreteVicsekDorsogna:
    """Test discrete Vicsek-D'Orsogna hybrid model."""
    
    def test_forces_disabled(self):
        """With forces disabled, should behave like pure Vicsek."""
        rng = np.random.default_rng(42)
        
        config = {
            "sim": {
                "N": 20,
                "Lx": 10.0,
                "Ly": 10.0,
                "bc": "periodic",
                "T": 1.0,
                "dt": 0.1,
                "save_every": 5,
                "neighbor_rebuild": 1,
            },
            "model": {"speed": 0.5},
            "params": {"R": 1.5},
            "noise": {"kind": "gaussian", "eta": 0.3},
            "forces": {"enabled": False},
        }
        
        result = simulate_backend(config, rng)
        
        assert "traj" in result
        assert "vel" in result
        assert result["traj"].shape[1] == 20  # N particles
        assert result["traj"].shape[2] == 2   # 2D
        
    def test_forces_enabled(self):
        """With forces enabled, speeds should vary."""
        rng = np.random.default_rng(42)
        
        config = {
            "sim": {
                "N": 20,
                "Lx": 10.0,
                "Ly": 10.0,
                "bc": "periodic",
                "T": 2.0,
                "dt": 0.01,
                "save_every": 50,
                "neighbor_rebuild": 5,
            },
            "model": {"speed": 0.5},
            "params": {"R": 1.5},
            "noise": {"kind": "gaussian", "eta": 0.2},
            "forces": {
                "enabled": True,
                "params": {
                    "Cr": 2.0,
                    "Ca": 1.0,
                    "lr": 0.5,
                    "la": 1.5,
                    "mu_t": 0.5,
                    "rcut_factor": 5.0,
                }
            },
        }
        
        result = simulate_backend(config, rng)
        
        # Check speeds vary (not constant like pure Vicsek)
        speeds = np.linalg.norm(result["vel"], axis=2)
        assert speeds.shape == result["traj"].shape[:2]
        
        # Speeds should vary across particles at final time
        final_speeds = speeds[-1]
        assert np.std(final_speeds) > 0.01  # Some variation
        
    def test_periodic_wrapping(self):
        """Particles should wrap around periodic boundaries."""
        rng = np.random.default_rng(42)
        
        config = {
            "sim": {
                "N": 10,
                "Lx": 5.0,
                "Ly": 5.0,
                "bc": "periodic",
                "T": 5.0,
                "dt": 0.1,
                "save_every": 10,
                "neighbor_rebuild": 1,
            },
            "model": {"speed": 1.0},  # High speed
            "params": {"R": 1.0},
            "noise": {"kind": "uniform", "eta": 0.1},
            "forces": {"enabled": False},
        }
        
        result = simulate_backend(config, rng)
        traj = result["traj"]
        
        # All positions should remain in domain
        assert np.all(traj[:, :, 0] >= 0.0)
        assert np.all(traj[:, :, 0] < 5.0)
        assert np.all(traj[:, :, 1] >= 0.0)
        assert np.all(traj[:, :, 1] < 5.0)
        
    def test_reflecting_boundaries(self):
        """Particles should bounce off reflecting boundaries."""
        rng = np.random.default_rng(42)
        
        config = {
            "sim": {
                "N": 10,
                "Lx": 5.0,
                "Ly": 5.0,
                "bc": "reflecting",
                "T": 2.0,
                "dt": 0.05,
                "save_every": 10,
                "neighbor_rebuild": 1,
            },
            "model": {"speed": 1.0},
            "params": {"R": 1.0},
            "noise": {"kind": "uniform", "eta": 0.1},
            "forces": {"enabled": False},
        }
        
        result = simulate_backend(config, rng)
        traj = result["traj"]
        
        # All positions should remain in domain
        assert np.all(traj[:, :, 0] >= 0.0)
        assert np.all(traj[:, :, 0] <= 5.0)
        assert np.all(traj[:, :, 1] >= 0.0)
        assert np.all(traj[:, :, 1] <= 5.0)


class TestContinuousVicsekDorsogna:
    """Test continuous Vicsek-D'Orsogna hybrid model with RK4."""
    
    def test_alignment_enabled(self):
        """RK4 model with alignment should produce coordinated motion."""
        config = {
            "seed": 42,
            "sim": {
                "N": 20,
                "Lx": 10.0,
                "Ly": 10.0,
                "bc": "periodic",
                "T": 5.0,
                "dt": 0.01,
                "save_every": 100,
                "neighbor_rebuild": 10,
                "integrator": "rk4",
            },
            "params": {
                "alpha": 1.5,
                "beta": 1.0,
                "Cr": 2.0,
                "Ca": 1.0,
                "lr": 0.5,
                "la": 1.5,
                "rcut_factor": 5.0,
                "alignment": {
                    "enabled": True,
                    "radius": 2.0,
                    "rate": 1.0,
                    "Dtheta": 0.001,
                }
            }
        }
        
        result = simulate(config)
        
        assert "traj" in result
        assert "vel" in result
        assert result["traj"].shape[1] == 20
        
        # Check speeds stabilize near alpha/beta
        speeds = np.linalg.norm(result["vel"], axis=2)
        mean_speed_final = speeds[-1].mean()
        expected_speed = config["params"]["alpha"] / config["params"]["beta"]
        
        # Should be within 50% of natural speed (loose tolerance)
        assert 0.5 * expected_speed < mean_speed_final < 1.5 * expected_speed
        
    def test_alignment_disabled(self):
        """Pure D'Orsogna without alignment."""
        config = {
            "seed": 42,
            "sim": {
                "N": 15,
                "Lx": 10.0,
                "Ly": 10.0,
                "bc": "periodic",
                "T": 2.0,
                "dt": 0.01,
                "save_every": 50,
                "neighbor_rebuild": 10,
                "integrator": "rk4",
            },
            "params": {
                "alpha": 1.5,
                "beta": 1.0,
                "Cr": 2.0,
                "Ca": 1.0,
                "lr": 0.5,
                "la": 1.5,
                "rcut_factor": 5.0,
                "alignment": {
                    "enabled": False,
                }
            }
        }
        
        result = simulate(config)
        
        assert "traj" in result
        assert result["traj"].shape[1] == 15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
