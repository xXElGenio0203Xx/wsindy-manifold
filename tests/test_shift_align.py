"""
Tests for shift alignment module (src/rectsim/shift_align.py)
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from rectsim.shift_align import (
    fft_cross_correlation_shift,
    compute_reference_field,
    compute_shifts,
    apply_shifts,
    undo_shifts,
    align_training_data,
    predict_shifts_linear,
)


class TestFFTCrossCorrelation:
    """Test FFT-based cross-correlation shift detection."""

    def test_zero_shift(self):
        """Identical fields should give zero shift."""
        np.random.seed(42)
        field = np.random.rand(48, 48)
        dy, dx = fft_cross_correlation_shift(field, field)
        assert dy == 0
        assert dx == 0

    def test_known_shift(self):
        """Detect a known translational shift."""
        np.random.seed(42)
        ref = np.zeros((48, 48))
        # Place a Gaussian blob at center
        y, x = np.mgrid[0:48, 0:48]
        ref = np.exp(-((y - 24)**2 + (x - 24)**2) / (2 * 5**2))
        
        # Shift by (3, -5) using periodic roll
        target = np.roll(np.roll(ref, 3, axis=0), -5, axis=1)
        
        dy, dx = fft_cross_correlation_shift(ref, target)
        # The shift that aligns target to ref should be (-3, 5)
        # Because: applying (dy, dx) to target should recover ref
        # np.roll(target, -3, axis=0) undoes the +3 shift
        assert dy == -3, f"Expected dy=-3, got {dy}"
        assert dx == 5, f"Expected dx=5, got {dx}"

    def test_periodic_wrap(self):
        """Shift detection should handle periodic wrapping."""
        np.random.seed(42)
        ref = np.zeros((48, 48))
        y, x = np.mgrid[0:48, 0:48]
        ref = np.exp(-((y - 5)**2 + (x - 5)**2) / (2 * 3**2))
        
        # Large shift that wraps around
        target = np.roll(np.roll(ref, 20, axis=0), -22, axis=1)
        
        dy, dx = fft_cross_correlation_shift(ref, target)
        # Undo: need to roll target by -20 in y, +22 in x
        assert dy == -20, f"Expected dy=-20, got {dy}"
        assert dx == 22, f"Expected dx=22, got {dx}"


class TestAlignmentRoundtrip:
    """Test that apply + undo shifts is an identity operation."""

    def test_roundtrip(self):
        """apply_shifts then undo_shifts should recover original."""
        np.random.seed(42)
        densities = np.random.rand(10, 48, 48)
        shifts = np.array([[3, -2], [0, 5], [-4, 1], [2, 2], [-1, -3],
                           [0, 0], [6, -6], [-2, 4], [1, -1], [3, 3]], dtype=np.int32)
        
        aligned = apply_shifts(densities, shifts)
        recovered = undo_shifts(aligned, shifts)
        
        np.testing.assert_array_almost_equal(recovered, densities)

    def test_alignment_reduces_variance(self):
        """Aligning translated copies of same field should reduce variance."""
        np.random.seed(42)
        ref = np.zeros((48, 48))
        y, x = np.mgrid[0:48, 0:48]
        ref = np.exp(-((y - 24)**2 + (x - 24)**2) / (2 * 5**2))
        
        # Create shifted copies
        T = 20
        densities = np.zeros((T, 48, 48))
        for t in range(T):
            dy_t = t  # progressive shift
            densities[t] = np.roll(ref, dy_t, axis=0)
        
        # Before alignment: high variance across frames
        var_before = densities.var(axis=0).mean()
        
        # Align
        sa_result = align_training_data(densities, M=1, T_rom=T, ref_method='mean')
        aligned = sa_result['aligned']
        
        # After alignment: much lower variance
        var_after = aligned.var(axis=0).mean()
        
        assert var_after < var_before * 0.1, \
            f"Alignment should reduce variance: before={var_before:.6f}, after={var_after:.6f}"


class TestReferenceComputation:
    """Test reference field computation methods."""

    def test_mean_reference(self):
        densities = np.random.rand(5, 10, 10)
        ref = compute_reference_field(densities, method='mean')
        np.testing.assert_array_almost_equal(ref, densities.mean(axis=0))

    def test_first_reference(self):
        densities = np.random.rand(5, 10, 10)
        ref = compute_reference_field(densities, method='first')
        np.testing.assert_array_almost_equal(ref, densities[0])

    def test_median_reference(self):
        densities = np.random.rand(5, 10, 10)
        ref = compute_reference_field(densities, method='median')
        np.testing.assert_array_almost_equal(ref, np.median(densities, axis=0))


class TestShiftPrediction:
    """Test linear shift extrapolation."""

    def test_constant_velocity(self):
        """Linear extrapolation of constant-velocity shifts."""
        # Shifts with constant velocity: dy=2*t, dx=-1*t
        T = 10
        known = np.column_stack([
            2 * np.arange(T),
            -1 * np.arange(T)
        ]).astype(np.int32)
        
        predicted = predict_shifts_linear(known, n_forecast=5)
        
        expected_dy = np.round(2 * np.arange(T, T + 5)).astype(np.int32)
        expected_dx = np.round(-1 * np.arange(T, T + 5)).astype(np.int32)
        
        np.testing.assert_array_equal(predicted[:, 0], expected_dy)
        np.testing.assert_array_equal(predicted[:, 1], expected_dx)

    def test_single_point_fallback(self):
        """With 1 known point, should tile the last shift."""
        known = np.array([[5, -3]], dtype=np.int32)
        predicted = predict_shifts_linear(known, n_forecast=3)
        
        assert predicted.shape == (3, 2)
        for i in range(3):
            assert predicted[i, 0] == 5
            assert predicted[i, 1] == -3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
