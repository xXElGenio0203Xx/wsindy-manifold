# Simulation Outputs and Visualization

Complete guide to standardized outputs, animation generation, and data formats for all rectsim models.

---

## Table of Contents

1. [Overview](#overview)
2. [Output Directory Structure](#output-directory-structure)
3. [File Descriptions](#file-descriptions)
4. [Animation Generation](#animation-generation)
5. [Order Parameters](#order-parameters)
6. [Configuration](#configuration)
7. [Post-Processing](#post-processing)
8. [Troubleshooting](#troubleshooting)

---

## Overview

**Unified output system** for all rectsim models produces:

✅ **4 Standard Order Parameters**: Polarization, angular momentum, mean speed, density variance  
✅ **Multiple Output Formats**: NPZ (binary), CSV (text), PNG (plots), MP4 (animations)  
✅ **Backend Agnostic**: Same outputs for D'Orsogna, Vicsek, and hybrid models  
✅ **Fully Tested**: 19 comprehensive tests ensuring correctness  

**Key feature**: Consistent outputs regardless of model type (Vicsek discrete, D'Orsogna continuous, or hybrid).

---

## Output Directory Structure

Each simulation creates a directory (e.g., `outputs/vicsek_dorsogna_discrete/`) containing:

```
outputs/<run_name>/
├── traj_animation.mp4          # Video showing particle trajectories
├── density_animation.mp4       # Video showing density field evolution
├── order_parameters.csv        # Time series of all metrics
├── order_summary.png          # Summary plots of metrics vs time
├── traj.csv                   # Full trajectory data (positions + velocities)
├── density.csv                # Density field data
├── results.npz                # Binary data for Python analysis
├── config.yaml                # Configuration used for this run
└── metadata.json              # Run metadata (timestamp, seed, etc.)
```

---

## File Descriptions

### 1. Videos (MP4)

**`traj_animation.mp4`** - Particle trajectory animation
- Particles shown as arrows colored by heading angle
- Arrow length scaled by speed
- Ideal for visualizing collective motion patterns
- Default fps = 20 (configurable in `outputs.fps`)

**`density_animation.mp4`** - Density field animation
- Spatial distribution using kernel density estimation (KDE)
- Reveals clustering, waves, and density patterns
- Resolution configurable in `outputs.density_resolution`

### 2. Data Files (CSV)

**`order_parameters.csv`** - Metrics time series
```csv
time,polarization,angular_momentum,mean_speed,density_variance
0.0,0.1234,0.0567,0.5000,0.0023
0.1,0.1345,0.0589,0.5001,0.0022
...
```

Columns:
- **Polarization (Φ)**: Velocity alignment, range [0, 1]
- **Angular momentum (L)**: Collective rotation
- **Mean speed**: Average particle speed
- **Density variance**: Spatial clustering measure

**`traj.csv`** - Full trajectory data
```csv
time,particle_id,x,y,vx,vy
0.0,0,1.234,5.678,0.345,0.123
0.0,1,2.345,6.789,0.456,0.234
...
```
- Every particle, every saved frame
- Can be large for long simulations (estimate: ~100MB per 1000 particles × 1000 frames)

**`density.csv`** - Density field data
- Spatial grid of particle density over time
- Computed via Gaussian KDE
- Resolution: `(density_resolution, density_resolution, n_frames)`

### 3. Analysis Files

**`order_summary.png`** - Summary visualization
- 4 subplots showing time evolution:
  * Polarization Φ(t)
  * Angular momentum L(t)
  * Mean speed v̄(t)
  * Density variance σ²_ρ(t)
- Quick visual overview of simulation behavior

**`results.npz`** - Binary NumPy archive
```python
data = np.load('outputs/my_run/results.npz')
positions = data['positions']  # (n_frames, N, 2)
velocities = data['velocities']  # (n_frames, N, 2)
times = data['times']           # (n_frames,)
config = data['config'].item()  # dict
```

**`config.yaml`** - Exact configuration used for this run
- Stored for reproducibility
- Can be used to rerun identical simulation

**`metadata.json`** - Run metadata
```json
{
  "timestamp": "2025-01-20T14:30:00",
  "seed": 42,
  "model": "vicsek_discrete",
  "N": 200,
  "T_max": 100.0,
  "runtime_seconds": 12.34
}
```

---

## Animation Generation

### Requirements

**ffmpeg must be installed** for animation creation:

```bash
# macOS
brew install ffmpeg

# Linux (Ubuntu/Debian)
sudo apt-get install ffmpeg

# Linux (RHEL/CentOS)
sudo yum install ffmpeg
```

### Automatic Animation Creation

By default, animations are attempted when `outputs.animations: true`:

```yaml
outputs:
  animations: true  # Default
  fps: 20
```

**With ffmpeg installed:**
```bash
$ python scripts/run_standardized.py config.yaml

Creating trajectory animation...
✓ Saved outputs/my_run/traj_animation.mp4

Creating density animation...
✓ Saved outputs/my_run/density_animation.mp4
```

**Without ffmpeg installed:**
```bash
$ python scripts/run_standardized.py config.yaml

⚠️  Warning: ffmpeg not found!
   Animations require ffmpeg to be installed.
   Install with: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)
   You can create animations later using: python scripts/create_animations.py
   outputs/my_run/results.npz

Simulation continues normally...
✓ Saved outputs/my_run/order_parameters.csv
✓ Saved outputs/my_run/order_summary.png
```

### Post-Processing Animations

Create animations from existing results:

```bash
# Basic usage
python scripts/create_animations.py outputs/my_run/results.npz

# Custom fps and DPI
python scripts/create_animations.py outputs/my_run/results.npz 30 150

# Arguments: <npz_file> <fps> <dpi>
```

This is useful when:
- ffmpeg wasn't installed during initial run
- Want different fps/quality settings
- Reprocessing archived results

---

## Order Parameters

### 1. Polarization (Φ)

**Definition:**
```
Φ = (1/N) |∑ᵢ vᵢ/|vᵢ||
```

**Physical Meaning:**
- Measures global velocity alignment
- Range: [0, 1]
- Φ = 0: Disordered (random velocities)
- Φ = 1: Perfectly aligned (all velocities parallel)

**Code:**
```python
from rectsim.standard_metrics import polarization
phi = polarization(velocities)  # velocities: (N, 2) array
```

### 2. Angular Momentum (L)

**Definition:**
```
L = (1/N) ∑ᵢ (rᵢ - r_cm) × vᵢ

where r_cm = (1/N) ∑ᵢ rᵢ is center of mass
```

**Physical Meaning:**
- Measures collective rotation around center of mass
- L > 0: Counter-clockwise rotation
- L < 0: Clockwise rotation
- L ≈ 0: No net rotation

**Code:**
```python
from rectsim.standard_metrics import angular_momentum
L = angular_momentum(positions, velocities)
```

### 3. Mean Speed

**Definition:**
```
v̄ = (1/N) ∑ᵢ |vᵢ|
```

**Physical Meaning:**
- Average particle speed
- Tracks energy/activity level

**Code:**
```python
from rectsim.standard_metrics import mean_speed
v_mean = mean_speed(velocities)
```

### 4. Density Variance

**Definition:**
```
σ²_ρ = Var[ρ(x, y)]

where ρ(x, y) is KDE density field
```

**Physical Meaning:**
- Measures spatial clustering
- Low variance: Uniform distribution
- High variance: Strong clustering

**Code:**
```python
from rectsim.standard_metrics import density_variance
sigma2 = density_variance(positions, domain_bounds=(Lx, Ly))
```

### Computing All Metrics

```python
from rectsim.standard_metrics import compute_all_metrics

metrics = compute_all_metrics(
    positions,      # (N, 2)
    velocities,     # (N, 2)
    domain_bounds=(Lx, Ly)
)

# Returns dict: {'polarization': ..., 'angular_momentum': ..., 'mean_speed': ..., 'density_variance': ...}
```

### Time Series

```python
from rectsim.standard_metrics import compute_metrics_series

metrics_series = compute_metrics_series(
    positions_series,   # (n_frames, N, 2)
    velocities_series,  # (n_frames, N, 2)
    times,              # (n_frames,)
    domain_bounds=(Lx, Ly)
)

# Returns DataFrame with columns: time, polarization, angular_momentum, mean_speed, density_variance
```

---

## Configuration

### Example Output Configuration

```yaml
outputs:
  save_dir: "outputs"
  run_name: "my_experiment"
  
  # What to save
  save_trajectories: true      # traj.csv
  save_density: true           # density.csv
  save_order_params: true      # order_parameters.csv
  animations: true             # traj_animation.mp4, density_animation.mp4
  plots: true                  # order_summary.png
  
  # Animation settings
  fps: 20                      # Frames per second
  dpi: 100                     # Resolution (higher = larger file)
  
  # Density computation
  density_resolution: 64       # Grid size for KDE (64x64, 128x128, etc.)
  
  # Data saving frequency
  save_every: 10               # Save every 10 timesteps
```

### Minimal Configuration (Fast Tests)

```yaml
outputs:
  run_name: "quick_test"
  save_trajectories: false     # Skip CSV files
  save_density: false
  animations: false            # Skip animations
  plots: true                  # Keep summary plot
  save_every: 50               # Save less frequently
```

### High-Quality Configuration (Publication)

```yaml
outputs:
  run_name: "publication_run"
  save_trajectories: true
  save_density: true
  save_order_params: true
  animations: true
  plots: true
  fps: 30                      # Smoother animations
  dpi: 150                     # Higher resolution
  density_resolution: 128      # Finer density grid
  save_every: 1                # Save every timestep
```

---

## Post-Processing

### Loading Results

```python
import numpy as np
import pandas as pd

# Load binary data
data = np.load('outputs/my_run/results.npz')
positions = data['positions']  # (n_frames, N, 2)
velocities = data['velocities']
times = data['times']

# Load metrics
metrics = pd.read_csv('outputs/my_run/order_parameters.csv')
print(metrics.head())

# Load trajectory data
traj = pd.read_csv('outputs/my_run/traj.csv')
```

### Custom Analysis

```python
from rectsim.standard_metrics import polarization, angular_momentum

# Compute metrics for specific frame
frame_idx = 100
pos = positions[frame_idx]
vel = velocities[frame_idx]

phi = polarization(vel)
L = angular_momentum(pos, vel)

print(f"Frame {frame_idx}: Φ={phi:.3f}, L={L:.3f}")
```

### Reprocessing Animations

```bash
# Create animations with custom settings
python scripts/create_animations.py outputs/my_run/results.npz 60 200

# Arguments:
# - 60: fps (higher = smoother)
# - 200: dpi (higher = sharper)
```

---

## Troubleshooting

### Issue: Animations not created

**Symptoms:**
```
⚠️  Warning: ffmpeg not found!
```

**Solution:**
```bash
# Install ffmpeg
brew install ffmpeg  # macOS
sudo apt-get install ffmpeg  # Linux

# Then create animations
python scripts/create_animations.py outputs/my_run/results.npz
```

### Issue: Animation creation very slow

**Symptoms:**
- Animation generation takes > 1 minute

**Solutions:**
```yaml
# Reduce resolution
outputs:
  dpi: 80  # Lower from 100

# Save fewer frames
outputs:
  save_every: 20  # Higher number = fewer frames

# Reduce fps
outputs:
  fps: 10  # Lower from 20
```

### Issue: Large output files

**Symptoms:**
- `traj.csv` or `density.csv` > 1GB

**Solutions:**
```yaml
# Option 1: Save less frequently
outputs:
  save_every: 50  # Save only every 50 timesteps

# Option 2: Don't save CSV files
outputs:
  save_trajectories: false
  save_density: false
  # Use results.npz for analysis instead
```

### Issue: Density variance always low

**Symptoms:**
- `density_variance` column shows very small values (< 0.001)

**Cause:**
- Particles uniformly distributed (not clustered)

**Solutions:**
- Increase attraction forces (D'Orsogna)
- Decrease noise (Vicsek)
- This may be expected behavior for your parameters

### Issue: Missing output files

**Symptoms:**
- Expected files not created (e.g., `density.csv` missing)

**Check configuration:**
```yaml
outputs:
  save_density: true  # Must be true to create density.csv
  save_trajectories: true  # Must be true to create traj.csv
  animations: true  # Must be true to create MP4 files
```

---

## Performance Notes

### File Size Estimates

| File | Size (Typical) | Scales With |
|------|----------------|-------------|
| `results.npz` | 10-100 MB | N × n_frames |
| `traj.csv` | 50-500 MB | N × n_frames |
| `density.csv` | 10-100 MB | resolution² × n_frames |
| `order_parameters.csv` | < 1 MB | n_frames only |
| `traj_animation.mp4` | 1-10 MB | n_frames, fps, dpi |
| `density_animation.mp4` | 1-10 MB | n_frames, fps, dpi |

### Computation Time Estimates

| Task | Time (Typical) | Bottleneck |
|------|----------------|------------|
| Order parameters | < 1s | Negligible |
| Trajectory CSV | 1-10s | Disk I/O |
| Density CSV | 5-30s | KDE computation |
| Trajectory animation | 10-60s | Matplotlib rendering + ffmpeg |
| Density animation | 10-60s | Matplotlib rendering + ffmpeg |

### Optimization Tips

**For fast tests:**
```yaml
outputs:
  animations: false
  save_density: false
  save_every: 50
```

**For production runs:**
```yaml
outputs:
  animations: true
  fps: 20
  dpi: 100
  density_resolution: 64
  save_every: 10
```

---

## Implementation Details

### Core Modules

**`src/rectsim/standard_metrics.py`** (302 lines)
- `polarization(velocities)` - Global alignment Φ ∈ [0, 1]
- `angular_momentum(positions, velocities)` - Collective rotation L
- `mean_speed(velocities)` - Average particle speed
- `density_variance(positions, domain_bounds)` - Spatial clustering (KDE-based)
- `compute_all_metrics()` - Single frame, all metrics
- `compute_metrics_series()` - Time series over entire simulation

**`src/rectsim/io_outputs.py`** (457 lines)
- `save_order_parameters_csv()` - Time series CSV
- `save_trajectory_csv()` - Per-particle trajectory data
- `save_density_csv()` - KDE density field data
- `plot_order_summary()` - 4-panel summary plot
- `create_traj_animation()` - Particle motion MP4 with velocity arrows
- `create_density_animation()` - Density field MP4 heatmap
- `save_standardized_outputs()` - Main entry point, generates all outputs

### Testing

**`tests/test_standardized_outputs.py`** (318 lines)
- 19 tests covering all functionality
- Test classes:
  * `TestPolarization` (5 tests)
  * `TestAngularMomentum` (3 tests)
  * `TestMeanSpeed` (2 tests)
  * `TestDensityVariance` (2 tests)
  * `TestComputeAllMetrics` (1 test)
  * `TestMetricsSeries` (1 test)
  * `TestCSVOutputs` (2 tests)
  * `TestPlotOutputs` (1 test)
  * `TestStandardizedOutputs` (2 tests)

---

## Additional Resources

- **Main README**: [README.md](../README.md) - Installation and quickstart
- **Models Guide**: [MODELS.md](MODELS.md) - Model descriptions
- **ROM/MVAR Pipeline**: [ROM_MVAR.md](../ROM_MVAR.md) - Reduced-order modeling
- **Oscar Guide**: [OSCAR.md](../OSCAR.md) - HPC deployment
