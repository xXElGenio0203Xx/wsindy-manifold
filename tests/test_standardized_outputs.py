"""
Tests for standardized metrics and outputs.

This module tests:
- Order parameter computations (polarization, angular momentum, mean speed, density variance)
- CSV output functions
- Summary plot generation
- Animation creation (if ffmpeg available)
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from rectsim.standard_metrics import (
    polarization, angular_momentum, mean_speed, density_variance,
    compute_all_metrics, compute_metrics_series
)

from rectsim.io_outputs import (
    save_order_parameters_csv, save_trajectory_csv, save_density_csv,
    plot_order_summary, save_standardized_outputs
)


class TestPolarization:
    """Test polarization order parameter computation."""
    
    def test_perfect_alignment(self):
        """Perfect alignment should give phi = 1.0."""
        velocities = np.array([[1, 0], [1, 0], [1, 0]])
        phi = polarization(velocities)
        assert abs(phi - 1.0) < 1e-6
    
    def test_opposite_directions(self):
        """Opposite directions should cancel out."""
        velocities = np.array([[1, 0], [-1, 0]])
        phi = polarization(velocities)
        assert abs(phi) < 1e-6
    
    def test_random_directions(self):
        """Random directions should give low phi."""
        rng = np.random.default_rng(42)
        angles = rng.uniform(0, 2*np.pi, 100)
        velocities = np.column_stack([np.cos(angles), np.sin(angles)])
        phi = polarization(velocities)
        assert 0.0 <= phi <= 0.2  # Should be near zero for large N
    
    def test_zero_velocities(self):
        """Zero velocities should return 0."""
        velocities = np.zeros((10, 2))
        phi = polarization(velocities)
        assert phi == 0.0
    
    def test_varying_speeds(self):
        """Polarization should only depend on direction, not speed."""
        velocities1 = np.array([[1, 0], [1, 0], [1, 0]])
        velocities2 = np.array([[2, 0], [2, 0], [2, 0]])
        phi1 = polarization(velocities1)
        phi2 = polarization(velocities2)
        assert abs(phi1 - phi2) < 1e-6


class TestAngularMomentum:
    """Test angular momentum order parameter."""
    
    def test_circular_motion(self):
        """Particles in circular motion should have high L."""
        # Particles on a circle moving tangentially
        theta = np.linspace(0, 2*np.pi, 10, endpoint=False)
        positions = np.column_stack([np.cos(theta), np.sin(theta)])
        # Tangent velocities (perpendicular to radius)
        velocities = np.column_stack([-np.sin(theta), np.cos(theta)])
        
        L = angular_momentum(positions, velocities)
        assert L > 0.5  # Should have significant angular momentum
    
    def test_radial_motion(self):
        """Radial motion should have low L."""
        theta = np.linspace(0, 2*np.pi, 10, endpoint=False)
        positions = np.column_stack([np.cos(theta), np.sin(theta)])
        # Radial velocities (parallel to radius)
        velocities = np.column_stack([np.cos(theta), np.sin(theta)])
        
        L = angular_momentum(positions, velocities)
        assert L < 0.1  # Should have minimal angular momentum
    
    def test_zero_velocities(self):
        """Zero velocities should give L = 0."""
        positions = np.random.rand(10, 2)
        velocities = np.zeros((10, 2))
        L = angular_momentum(positions, velocities)
        assert L == 0.0


class TestMeanSpeed:
    """Test mean speed computation."""
    
    def test_constant_speed(self):
        """All particles with same speed."""
        velocities = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        speed = mean_speed(velocities)
        assert abs(speed - 1.0) < 1e-6
    
    def test_varying_speeds(self):
        """Particles with different speeds."""
        velocities = np.array([[1, 0], [2, 0], [3, 0]])
        speed = mean_speed(velocities)
        assert abs(speed - 2.0) < 1e-6


class TestDensityVariance:
    """Test density variance computation."""
    
    def test_uniform_distribution(self):
        """Uniformly distributed particles should have low variance."""
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 10, (100, 2))
        domain_bounds = (0, 10, 0, 10)
        
        var = density_variance(positions, domain_bounds, resolution=30)
        # Uniform distribution should have relatively low variance
        assert var > 0  # Not exactly zero due to sampling
    
    def test_clustered_distribution(self):
        """Clustered particles should have high variance."""
        # Two tight clusters
        cluster1 = np.random.randn(50, 2) * 0.2 + [2, 2]
        cluster2 = np.random.randn(50, 2) * 0.2 + [8, 8]
        positions = np.vstack([cluster1, cluster2])
        domain_bounds = (0, 10, 0, 10)
        
        var_clustered = density_variance(positions, domain_bounds, resolution=30)
        
        # Compare to uniform
        positions_uniform = np.random.uniform(0, 10, (100, 2))
        var_uniform = density_variance(positions_uniform, domain_bounds, resolution=30)
        
        # Clustered should have higher variance
        assert var_clustered > var_uniform


class TestComputeAllMetrics:
    """Test combined metrics computation."""
    
    def test_all_metrics_computed(self):
        """All metrics should be in output dict."""
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 10, (50, 2))
        velocities = rng.uniform(-1, 1, (50, 2))
        domain_bounds = (0, 10, 0, 10)
        
        metrics = compute_all_metrics(positions, velocities, domain_bounds)
        
        assert 'polarization' in metrics
        assert 'angular_momentum' in metrics
        assert 'mean_speed' in metrics
        assert 'density_variance' in metrics
        
        # All should be scalar floats
        for key, value in metrics.items():
            assert isinstance(value, float)


class TestMetricsSeries:
    """Test time series computation."""
    
    def test_time_series_shape(self):
        """Output arrays should have correct shape."""
        T = 10
        N = 20
        
        trajectory = np.random.rand(T, N, 2) * 10
        velocities = np.random.rand(T, N, 2)
        domain_bounds = (0, 10, 0, 10)
        
        metrics = compute_metrics_series(trajectory, velocities, domain_bounds,
                                        resolution=20, verbose=False)
        
        assert metrics['polarization'].shape == (T,)
        assert metrics['angular_momentum'].shape == (T,)
        assert metrics['mean_speed'].shape == (T,)
        assert metrics['density_variance'].shape == (T,)


class TestCSVOutputs:
    """Test CSV saving functions."""
    
    def test_save_order_parameters_csv(self, tmp_path):
        """Test order parameters CSV saving."""
        times = np.linspace(0, 10, 5)
        metrics = {
            'polarization': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            'angular_momentum': np.array([0.01, 0.02, 0.03, 0.04, 0.05]),
            'mean_speed': np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            'density_variance': np.array([0.001, 0.002, 0.003, 0.004, 0.005])
        }
        
        output_path = tmp_path / "order_parameters.csv"
        save_order_parameters_csv(times, metrics, output_path)
        
        assert output_path.exists()
        
        # Check content
        data = np.loadtxt(output_path, delimiter=',', skiprows=1)
        assert data.shape == (5, 5)  # 5 rows, 5 columns
        np.testing.assert_allclose(data[:, 0], times)
        np.testing.assert_allclose(data[:, 1], metrics['polarization'])
    
    def test_save_trajectory_csv(self, tmp_path):
        """Test trajectory CSV saving."""
        times = np.array([0, 1, 2])
        positions = np.random.rand(3, 5, 2)  # 3 times, 5 particles
        velocities = np.random.rand(3, 5, 2)
        
        output_path = tmp_path / "traj.csv"
        save_trajectory_csv(times, positions, velocities, output_path)
        
        assert output_path.exists()
        
        # Check that file has correct number of rows
        with open(output_path) as f:
            lines = f.readlines()
        # 1 header + 3 times × 5 particles = 16 lines
        assert len(lines) == 16


class TestPlotOutputs:
    """Test plot generation."""
    
    def test_plot_order_summary(self, tmp_path):
        """Test summary plot creation."""
        times = np.linspace(0, 10, 50)
        metrics = {
            'polarization': np.sin(times / 10 * np.pi),
            'angular_momentum': np.cos(times / 10 * np.pi),
            'mean_speed': np.ones_like(times),
            'density_variance': np.exp(-times / 5)
        }
        
        output_path = tmp_path / "order_summary.png"
        plot_order_summary(times, metrics, output_path)
        
        assert output_path.exists()
        assert output_path.stat().st_size > 1000  # Should be non-trivial size


class TestStandardizedOutputs:
    """Test complete standardized output pipeline."""
    
    def test_save_standardized_outputs_minimal(self, tmp_path):
        """Test with minimal config (no animations)."""
        T, N = 10, 20
        times = np.linspace(0, 10, T)
        positions = np.random.rand(T, N, 2) * 10
        velocities = np.random.rand(T, N, 2)
        domain_bounds = (0, 10, 0, 10)
        
        config_outputs = {
            'order_parameters': True,
            'animations': False,  # Skip animations for speed
            'save_csv': True,
            'fps': 20,
            'density_resolution': 20  # Low resolution for speed
        }
        
        metrics = save_standardized_outputs(
            times, positions, velocities, domain_bounds,
            tmp_path, config_outputs
        )
        
        # Check that files were created
        assert (tmp_path / 'order_parameters.csv').exists()
        assert (tmp_path / 'order_summary.png').exists()
        assert (tmp_path / 'traj.csv').exists()
        assert (tmp_path / 'density.csv').exists()
        
        # Check metrics were returned
        assert metrics is not None
        assert 'polarization' in metrics
        assert len(metrics['polarization']) == T
    
    def test_save_standardized_outputs_no_csv(self, tmp_path):
        """Test with CSV disabled."""
        T, N = 10, 20
        times = np.linspace(0, 10, T)
        positions = np.random.rand(T, N, 2) * 10
        velocities = np.random.rand(T, N, 2)
        domain_bounds = (0, 10, 0, 10)
        
        config_outputs = {
            'order_parameters': True,
            'animations': False,
            'save_csv': False,  # Disable CSV
            'density_resolution': 20
        }
        
        save_standardized_outputs(
            times, positions, velocities, domain_bounds,
            tmp_path, config_outputs
        )
        
        # CSV files should not exist
        assert not (tmp_path / 'traj.csv').exists()
        assert not (tmp_path / 'density.csv').exists()
        
        # But order parameters should still exist
        assert (tmp_path / 'order_parameters.csv').exists()
