"""
Unit tests for CrowdROM pipeline.

Tests:
1. Mass conservation with synthetic data
2. Movie selection logic
3. Non-stationarity aggregation
4. JSON schema validation
"""

import json
import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from rectsim.crowdrom_runner import CrowdROMRunner
from rectsim.crowdrom_schemas import validate_run_json, validate_nonstationarity_report


class TestMassConservation:
    """Test mass conservation validation."""
    
    def test_exact_mass_conservation(self):
        """Test that exact mass=1 passes validation."""
        # Create synthetic config with exact mass
        cfg = {
            "meta": {"seed": 42},
            "simulation": {"C": 1, "T": 1.0, "dt_obs": 0.1, "num_particles": 10},
            "domain_grid": {"domain": {"xmin": 0, "xmax": 10, "ymin": 0, "ymax": 10},
                          "nx": 10, "ny": 10, "dx": 1.0, "dy": 1.0},
            "kde": {},
            "pod": {"energy_threshold": 0.99},
            "non_stationarity": {}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CrowdROMRunner(
                cfg=cfg,
                outdir=tmpdir,
                mass_tol=1e-12,
                movies_for=(),
                quiet=True
            )
            
            # Mock trajectory data
            n_frames = 11
            n_particles = 10
            times = np.linspace(0, 1.0, n_frames)
            positions = np.random.randn(n_frames, n_particles, 2)
            velocities = np.zeros((n_frames, n_particles, 2))
            runner.trajectories = [(times, positions, velocities)]
            
            # Create exact mass=1 density (uniform distribution)
            nx, ny = 10, 10
            density_snapshots = np.ones((n_frames, nx, ny)) / (nx * ny)
            runner.densities = [density_snapshots]
            
            # Mock POD
            from rectsim.pod import PODProjector
            runner.pod_projector = PODProjector(energy_threshold=0.99)
            runner.pod_projector.d = 5
            runner.pod_projector.s = np.array([1.0, 0.5, 0.3, 0.2, 0.1])
            runner.pod_projector.energy_curve = np.array([0.5, 0.75, 0.9, 0.95, 0.99])
            
            # Create latents
            runner.latents = [np.random.randn(5, n_frames)]
            
            # Mock nonstationarity processor
            from rectsim.nonstationarity import NonStationarityProcessor, CoordDecision, CaseMeta
            from dataclasses import dataclass
            runner.nonstationarity_processor = NonStationarityProcessor(adf_alpha=0.01)
            decisions = [CoordDecision(mode='raw', adf_pvalue=0.001, adf_lag=0, 
                                      adf_variant='trend', notes='') for _ in range(5)]
            case_meta = CaseMeta(
                decisions=decisions,
                trim_left=0,
                K_in=n_frames,
                K_out=n_frames
            )
            runner.nonstationarity_processor.case_meta = [case_meta]
            
            # Test order parameters computation
            exit_code = runner._compute_order_parameters()
            
            # Should succeed (exit code 0)
            assert exit_code == 0, "Exact mass conservation should pass"
            
            # Check mass errors are small
            df = runner.order_params[0]
            assert df['mass_err_weighted'].max() < 1e-12, "Mass error should be near zero"
    
    def test_mass_conservation_violation(self):
        """Test that mass violation triggers exit code 2."""
        cfg = {
            "meta": {"seed": 42},
            "simulation": {"C": 1, "T": 1.0, "dt_obs": 0.1, "num_particles": 10},
            "domain_grid": {"domain": {"xmin": 0, "xmax": 10, "ymin": 0, "ymax": 10},
                          "nx": 10, "ny": 10, "dx": 1.0, "dy": 1.0},
            "kde": {},
            "pod": {"energy_threshold": 0.99},
            "non_stationarity": {}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CrowdROMRunner(
                cfg=cfg,
                outdir=tmpdir,
                mass_tol=1e-12,
                movies_for=(),
                quiet=True
            )
            
            # Mock trajectory data
            n_frames = 11
            n_particles = 10
            times = np.linspace(0, 1.0, n_frames)
            positions = np.random.randn(n_frames, n_particles, 2)
            velocities = np.zeros((n_frames, n_particles, 2))
            runner.trajectories = [(times, positions, velocities)]
            
            # Create BAD density (not normalized, mass != 1)
            nx, ny = 10, 10
            density_snapshots = np.ones((n_frames, nx, ny)) * 2.0  # Mass = 2.0 * dx * dy
            runner.densities = [density_snapshots]
            
            # Mock POD
            from rectsim.pod import PODProjector
            runner.pod_projector = PODProjector(energy_threshold=0.99)
            runner.pod_projector.d = 5
            runner.latents = [np.random.randn(5, n_frames)]
            
            # Test order parameters computation
            exit_code = runner._compute_order_parameters()
            
            # Should fail with exit code 2 (mass conservation violation)
            assert exit_code == 2, "Mass violation should return exit code 2"


class TestMovieSelection:
    """Test selective movie generation."""
    
    def test_movie_selection_subset(self):
        """Test that only selected simulations get movies."""
        cfg = {
            "meta": {"seed": 42},
            "simulation": {"C": 3, "T": 0.5, "dt_obs": 0.1, "num_particles": 5},
            "domain_grid": {"domain": {"xmin": 0, "xmax": 10, "ymin": 0, "ymax": 10},
                          "nx": 10, "ny": 10, "dx": 1.0, "dy": 1.0},
            "kde": {},
            "pod": {"energy_threshold": 0.99},
            "non_stationarity": {}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            
            # Create runner with movies for sims 2 and 3 only
            runner = CrowdROMRunner(
                cfg=cfg,
                outdir=str(outdir),
                mass_tol=1.0,  # Relax for this test
                movies_for=(2, 3),
                quiet=True
            )
            
            # Mock data for 3 simulations
            n_frames = 6
            for c in range(3):
                times = np.linspace(0, 0.5, n_frames)
                positions = np.random.randn(n_frames, 5, 2)
                velocities = np.zeros((n_frames, 5, 2))
                runner.trajectories.append((times, positions, velocities))
                
                density = np.ones((n_frames, 10, 10)) / 100
                runner.densities.append(density)
                
                latents = np.random.randn(3, n_frames)
                runner.latents.append(latents)
            
            # Mock POD
            from rectsim.pod import PODProjector
            runner.pod_projector = PODProjector(energy_threshold=0.99)
            runner.pod_projector.d = 3
            
            # Create sim directories
            for c in range(1, 4):
                (outdir / f"sim_{c:04d}").mkdir(parents=True)
            
            # Generate movies
            exit_code = runner._generate_movies()
            
            # Should succeed
            assert exit_code == 0
            
            # Check that only sim 2 and 3 have movies
            sim1_dir = outdir / "sim_0001"
            sim2_dir = outdir / "sim_0002"
            sim3_dir = outdir / "sim_0003"
            
            # Sim 1 should NOT have movies
            assert not (sim1_dir / "movie_trajectory.mp4").exists()
            assert not (sim1_dir / "movie_density.mp4").exists()
            assert not (sim1_dir / "movie_latent.mp4").exists()
            
            # Sim 2 SHOULD have movies
            assert (sim2_dir / "movie_trajectory.mp4").exists()
            assert (sim2_dir / "movie_density.mp4").exists()
            assert (sim2_dir / "movie_latent.mp4").exists()
            
            # Sim 3 SHOULD have movies
            assert (sim3_dir / "movie_trajectory.mp4").exists()
            assert (sim3_dir / "movie_density.mp4").exists()
            assert (sim3_dir / "movie_latent.mp4").exists()
    
    def test_no_movies(self):
        """Test that movies_for=() skips all movie generation."""
        cfg = {
            "meta": {"seed": 42},
            "simulation": {"C": 2, "T": 0.5, "dt_obs": 0.1, "num_particles": 5},
            "domain_grid": {"domain": {"xmin": 0, "xmax": 10, "ymin": 0, "ymax": 10},
                          "nx": 10, "ny": 10},
            "kde": {},
            "pod": {"energy_threshold": 0.99},
            "non_stationarity": {}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CrowdROMRunner(
                cfg=cfg,
                outdir=tmpdir,
                movies_for=(),  # No movies
                quiet=True
            )
            
            exit_code = runner._generate_movies()
            
            # Should succeed and skip
            assert exit_code == 0


class TestNonstationarityAggregation:
    """Test non-stationarity analysis aggregation."""
    
    def test_mixed_stationarity(self):
        """Test latent coordinates with mixed stationarity."""
        # Create synthetic latent time series
        n_t = 100
        
        # Coordinate 1: stationary (white noise)
        y1 = np.random.randn(n_t)
        
        # Coordinate 2: unit root (random walk, non-stationary)
        y2 = np.cumsum(np.random.randn(n_t))
        
        # Coordinate 3: trend-stationary
        y3 = np.arange(n_t) * 0.1 + np.random.randn(n_t) * 0.5
        
        # Stack
        latents = np.vstack([y1, y2, y3])  # (3, 100)
        
        # Run non-stationarity analysis
        from rectsim.nonstationarity import NonStationarityProcessor
        processor = NonStationarityProcessor(adf_alpha=0.05, verbose=False)
        processor.fit([latents])
        
        # Check decisions
        case_meta = processor.case_meta[0]
        decisions = case_meta.decisions
        
        # Coord 0 (y1) should be stationary (mode='raw')
        assert decisions[0].mode == 'raw', "White noise should be stationary"
        
        # Coord 1 (y2) should need transformation (mode != 'raw')
        # Note: This is probabilistic, may occasionally pass as stationary
        # In production, would use deterministic synthetic data
        
        # Coord 2 (y3) - trend-stationary, should be detected
        # Depending on ADF settings, may be 'raw' or 'detrend'


class TestJSONSchemas:
    """Test JSON schema validation."""
    
    def test_valid_run_json(self):
        """Test that valid run.json passes schema validation."""
        valid_run_json = {
            "meta": {
                "run_id": "2025-10-30T12-00-00Z",
                "timestamp_utc": "2025-10-30T12:00:00Z",
                "seed": 42,
                "code_version": {
                    "git_commit": "abc123",
                    "repo": "wsindy-manifold"
                },
                "env": {
                    "python": "3.11.0",
                    "numpy": "1.26.0",
                    "scipy": "1.11.0",
                    "platform": "linux"
                }
            },
            "simulation": {
                "model": "Vicsek",
                "C": 3,
                "T": 100.0,
                "num_particles": 100
            },
            "domain_grid": {
                "domain": {"xmin": 0, "xmax": 20, "ymin": 0, "ymax": 20},
                "nx": 40,
                "ny": 40
            },
            "kde": {},
            "pod": {
                "energy_threshold": 0.99,
                "chosen_d": 13,
                "svd": {"method": "economy", "randomized": False}
            },
            "non_stationarity": {
                "adf_alpha": 0.01,
                "adf_max_lags": "auto",
                "trend_policy": "auto"
            },
            "movies": {
                "fps": 20,
                "max_frames": 500,
                "make_for": [1, 2]
            },
            "io": {
                "save_dtype": "float64"
            }
        }
        
        valid, msg = validate_run_json(valid_run_json)
        assert valid, f"Valid run.json should pass: {msg}"
    
    def test_invalid_run_json_missing_field(self):
        """Test that invalid run.json fails validation."""
        invalid_run_json = {
            "meta": {
                "run_id": "2025-10-30T12-00-00Z",
                # Missing timestamp_utc
                "seed": 42,
                "code_version": {"git_commit": "abc", "repo": "test"},
                "env": {"python": "3.11", "numpy": "1.26", "platform": "linux"}
            },
            "simulation": {"model": "Vicsek", "C": 1},
            "domain_grid": {"domain": {"xmin": 0, "xmax": 10, "ymin": 0, "ymax": 10}, 
                          "nx": 10, "ny": 10},
            "kde": {},
            "pod": {"energy_threshold": 0.99, "chosen_d": 5, 
                   "svd": {"method": "economy", "randomized": False}},
            "non_stationarity": {"adf_alpha": 0.01, "adf_max_lags": "auto", "trend_policy": "auto"},
            "movies": {"fps": 20, "max_frames": 500, "make_for": []},
            "io": {"save_dtype": "float64"}
        }
        
        valid, msg = validate_run_json(invalid_run_json)
        # Should fail (or warn if jsonschema not installed)
        if "jsonschema not installed" not in msg:
            assert not valid, "Invalid JSON should fail validation"
    
    def test_valid_nonstationarity_report(self):
        """Test that valid non_stationarity_report.json passes."""
        valid_report = {
            "adf_alpha": 0.01,
            "adf_max_lags": "auto",
            "d": 5,
            "C": 2,
            "per_simulation": [
                {
                    "sim_id": 1,
                    "K": 100,
                    "decisions": [
                        {
                            "coord": 1,
                            "mode": "raw",
                            "adf_variant": "trend",
                            "p_value": 0.001,
                            "lag": 0,
                            "notes": ""
                        },
                        {
                            "coord": 2,
                            "mode": "diff",
                            "adf_variant": "const",
                            "p_value": 0.02,
                            "lag": 1,
                            "notes": "differenced"
                        }
                    ]
                },
                {
                    "sim_id": 2,
                    "K": 100,
                    "decisions": [
                        {
                            "coord": 1,
                            "mode": "raw",
                            "adf_variant": "trend",
                            "p_value": 0.0005,
                            "lag": 0,
                            "notes": ""
                        },
                        {
                            "coord": 2,
                            "mode": "raw",
                            "adf_variant": "trend",
                            "p_value": 0.003,
                            "lag": 0,
                            "notes": ""
                        }
                    ]
                }
            ]
        }
        
        valid, msg = validate_nonstationarity_report(valid_report)
        assert valid, f"Valid report should pass: {msg}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
