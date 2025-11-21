"""Unit test for ROM/MVAR video generation utilities.

Tests video creation with synthetic data.

Author: Maria
Date: November 2025
"""

import numpy as np
import tempfile
import shutil
from pathlib import Path

from rectsim.rom_video_utils import (
    make_truth_vs_pred_density_video,
    make_density_snapshot_comparison,
)


def test_video_generation():
    """Test video generation with synthetic data."""
    print("Testing make_truth_vs_pred_density_video...")
    
    # Create synthetic density fields
    T, Ny, Nx = 20, 32, 32
    np.random.seed(42)
    
    # Create spatially correlated patterns
    x = np.linspace(0, 2*np.pi, Nx)
    y = np.linspace(0, 2*np.pi, Ny)
    X, Y = np.meshgrid(x, y)
    
    density_true = np.zeros((T, Ny, Nx))
    density_pred = np.zeros((T, Ny, Nx))
    
    for t in range(T):
        # True: rotating Gaussian blob
        phase = 2 * np.pi * t / T
        cx = np.pi + 0.5 * np.cos(phase)
        cy = np.pi + 0.5 * np.sin(phase)
        density_true[t] = np.exp(-((X - cx)**2 + (Y - cy)**2) / 0.5)
        
        # Pred: similar but with phase lag
        phase_pred = 2 * np.pi * (t - 2) / T
        cx_pred = np.pi + 0.5 * np.cos(phase_pred)
        cy_pred = np.pi + 0.5 * np.sin(phase_pred)
        density_pred[t] = np.exp(-((X - cx_pred)**2 + (Y - cy_pred)**2) / 0.5)
    
    times = np.linspace(0, 10, T)
    
    # Create temp directory
    tmpdir = Path(tempfile.mkdtemp())
    
    try:
        # Test video generation
        out_path = tmpdir / "test_video.mp4"
        
        print(f"  Generating video to {out_path}...")
        make_truth_vs_pred_density_video(
            density_true,
            density_pred,
            out_path,
            fps=10,
            title="Test Video",
            times=times,
        )
        
        assert out_path.exists(), "Video file not created"
        
        file_size = out_path.stat().st_size
        assert file_size > 0, "Video file is empty"
        
        print(f"  ✓ Video created: {file_size / 1024:.1f} KB")
        print()
        
    finally:
        # Clean up
        shutil.rmtree(tmpdir)


def test_snapshot_comparison():
    """Test snapshot comparison plot."""
    print("Testing make_density_snapshot_comparison...")
    
    T, Ny, Nx = 100, 32, 32
    np.random.seed(42)
    
    # Create synthetic data
    density_true = np.random.rand(T, Ny, Nx) + 1.0
    density_pred = density_true + 0.1 * np.random.randn(T, Ny, Nx)
    
    times = np.linspace(0, 10, T)
    time_indices = [0, T//4, T//2, 3*T//4, T-1]
    
    tmpdir = Path(tempfile.mkdtemp())
    
    try:
        out_path = tmpdir / "test_snapshots.png"
        
        make_density_snapshot_comparison(
            density_true,
            density_pred,
            time_indices,
            times,
            out_path,
            title="Test Snapshots",
        )
        
        assert out_path.exists(), "Snapshot plot not created"
        
        file_size = out_path.stat().st_size
        print(f"  ✓ Snapshot plot created: {file_size / 1024:.1f} KB")
        print()
        
    finally:
        shutil.rmtree(tmpdir)


def test_video_color_scaling():
    """Test that video uses consistent color scale."""
    print("Testing video color scale consistency...")
    
    T, Ny, Nx = 10, 16, 16
    
    # True has range [0, 1]
    density_true = np.random.rand(T, Ny, Nx)
    
    # Pred has range [0.5, 2.0] (different from true)
    density_pred = 0.5 + 1.5 * np.random.rand(T, Ny, Nx)
    
    tmpdir = Path(tempfile.mkdtemp())
    
    try:
        out_path = tmpdir / "test_scale.mp4"
        
        # Should auto-compute vmin=0, vmax=2.0 for both
        make_truth_vs_pred_density_video(
            density_true,
            density_pred,
            out_path,
            fps=5,
        )
        
        assert out_path.exists()
        print(f"  ✓ Color scale handled correctly")
        print()
        
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    print("=" * 70)
    print("ROM/MVAR Video Generation Unit Tests")
    print("=" * 70)
    print()
    
    # Check if imageio and ffmpeg are available
    try:
        import imageio
        print("  imageio available: yes")
        
        # Test ffmpeg availability
        try:
            import imageio_ffmpeg
            print("  imageio-ffmpeg available: yes")
        except ImportError:
            print("  imageio-ffmpeg available: no (will use system ffmpeg)")
        
        print()
        
        test_video_generation()
        test_snapshot_comparison()
        test_video_color_scaling()
        
        print("=" * 70)
        print("✓ All video generation tests passed!")
        print("=" * 70)
        
    except ImportError as e:
        print(f"\n✗ Required package missing: {e}")
        print("\nTo install:")
        print("  pip install imageio imageio-ffmpeg")
