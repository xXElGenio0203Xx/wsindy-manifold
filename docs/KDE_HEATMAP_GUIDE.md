# KDE Heatmap Generation Guide

This guide explains how to generate Gaussian kernel density estimation (KDE) heatmaps from particle trajectory data in the wsindy-manifold project.

## Overview

The KDE heatmap functionality allows you to convert discrete particle trajectories into continuous density fields on a regular grid. This is useful for:

- Visualizing collective motion patterns
- Creating input data for reduced-order models (ROM/MVAR)
- Analyzing spatial distributions over time
- Generating publication-quality visualizations

## Quick Start

### Demo Mode

Generate example heatmaps with synthetic data:

```bash
python examples/generate_kde_heatmaps.py --demo
```

This creates:
- `kde_output/kde_density.npz` - Density data with metadata
- `kde_output/kde_snapshots_magma.png` - Snapshot grid visualization  
- `kde_output/kde_animation_magma.gif` - Animated density evolution

### From Simulation Data

Generate heatmaps from your simulation trajectories:

```bash
python examples/generate_kde_heatmaps.py \
    --input outputs/simulation/trajectories.npz \
    --output kde_results/ \
    --Lx 20.0 --Ly 20.0 \
    --nx 128 --ny 128 \
    --hx 0.6 --hy 0.6 \
    --cmap magma
```

## Key Parameters

### Grid Resolution

- `--nx`, `--ny`: Number of grid cells (default: 64)
- Higher resolution = smoother heatmaps but slower computation
- Typical values: 64-128 for quick visualization, 128-256 for ROM training

### KDE Bandwidth

- `--hx`, `--hy`: Gaussian kernel bandwidth (default: 0.8)
- Larger bandwidth = smoother density fields
- Smaller bandwidth = more detailed features
- Rule of thumb: bandwidth ≈ 0.5-1.5 grid cells

### Colormap Options

- `--cmap`: Choose visualization colormap
- Available: `magma` (default), `viridis`, `hot`, `plasma`, `inferno`
- `magma` is recommended for density visualizations (perceptually uniform, colorblind-friendly)

### Boundary Conditions

- `--bc`: `periodic` (default) or `reflecting`
- Must match your simulation boundary conditions
- Affects how KDE wraps at domain edges

## Programmatic Usage

### Basic Example

```python
from wsindy_manifold.latent.kde import trajectories_to_density_movie
import numpy as np

# Load trajectory data (T, N, 2)
traj = np.load("trajectories.npz")['traj']

# Generate KDE density movie
Rho, meta = trajectories_to_density_movie(
    X_all=traj,
    Lx=20.0, Ly=20.0,
    nx=128, ny=128,
    hx=0.6, hy=0.6,
    bc="periodic"
)

# Rho has shape (T, nx*ny) - flat density at each time
# meta contains grid info: dx, dy, Xc, etc.
```

### Visualization Example

```python
import matplotlib.pyplot as plt

# Get frame at time t=10
t = 10
density_2d = Rho[t].reshape(meta['ny'], meta['nx'])

# Plot
plt.figure(figsize=(8, 8))
plt.imshow(
    density_2d,
    extent=(0, meta['Lx'], 0, meta['Ly']),
    origin='lower',
    cmap='magma',
    aspect='equal'
)
plt.colorbar(label='Density')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'KDE Density at t={t}')
plt.savefig('density_snapshot.png', dpi=150)
```

## Mass Conservation

The KDE implementation automatically normalizes density to conserve mass:

```python
dx = float(meta['dx'])
dy = float(meta['dy'])
mass_per_frame = Rho.sum(axis=1) * dx * dy
# mass_per_frame should equal 1.0 (normalized) for all frames
```

This ensures the density integrates correctly when used in ROM/MVAR pipelines.

## Integration with ROM/MVAR Workflow

The generated KDE density data can be directly used for reduced-order modeling:

```bash
# 1. Generate KDE heatmaps from simulation
python examples/generate_kde_heatmaps.py \
    --input simulation/trajectories.npz \
    --output kde_data/ \
    --nx 128 --ny 128 --hx 0.6 --hy 0.6

# 2. Train ROM/MVAR model on density data
python scripts/rom_mvar_train.py \
    --heatmap kde_data/kde_density.npz \
    --output rom_models/ \
    --energy_keep 0.95 \
    --mvar_order 4

# 3. Generate forecasts
python scripts/rom_mvar_eval.py \
    --experiment my_experiment \
    --steps 100

# 4. Visualize results
python scripts/rom_mvar_visualize.py \
    --experiment my_experiment
```

## API Reference

### Main Functions

#### `trajectories_to_density_movie`

Convert particle trajectories to density movie.

**Parameters:**
- `X_all` (ndarray): Trajectory array (T, N, 2)
- `Lx`, `Ly` (float): Domain extents
- `nx`, `ny` (int): Grid resolution
- `hx`, `hy` (float): KDE bandwidths
- `bc` (str): Boundary conditions ('periodic' or 'reflecting')

**Returns:**
- `Rho` (ndarray): Density movie (T, nx*ny)
- `meta` (dict): Grid metadata (dx, dy, Xc, nx, ny, etc.)

#### `kde_gaussian`

Single-frame KDE evaluation.

**Parameters:**
- `X` (ndarray): Particle positions (N, 2)
- `Xc` (ndarray): Grid cell centers (nc, 2)
- `hx`, `hy` (float): KDE bandwidths
- `Lx`, `Ly` (float): Domain extents
- `bc` (str): Boundary conditions

**Returns:**
- `rho` (ndarray): Density field (nc,)

#### `make_grid`

Create uniform grid for KDE evaluation.

**Parameters:**
- `Lx`, `Ly` (float): Domain extents
- `nx`, `ny` (int): Number of grid cells

**Returns:**
- `Xc` (ndarray): Cell centers (nx*ny, 2)
- `dx`, `dy` (float): Cell spacings

## Troubleshooting

### Memory Issues

If you run out of memory with large grids:
- Reduce grid resolution (`--nx`, `--ny`)
- Process trajectories in batches
- Use lower precision (float32 instead of float64)

### Poor Mass Conservation

If mass varies significantly across frames:
- Check that particles stay within domain bounds
- Verify boundary conditions match simulation
- Increase grid resolution if particles are clustered

### Visualization Issues

If heatmaps look pixelated or blocky:
- Increase grid resolution (`--nx`, `--ny`)
- Increase KDE bandwidth (`--hx`, `--hy`)
- Use interpolation in imshow: `interpolation='bilinear'`

If density features are too smooth:
- Decrease KDE bandwidth
- Increase grid resolution
- Check that particles aren't too sparse

## References

- Bhaskar & Ziegelmeier (2019), *Chaos* 29, 123125 - Original parameter sweep paper
- D'Orsogna et al. (2006) - Morse swarming model
- Silverman (1986) - KDE theory and bandwidth selection

## Examples in the Repository

- `examples/generate_kde_heatmaps.py` - Standalone KDE generation script
- `examples/quickstart_rect2d.py` - Full ROM pipeline with KDE
- `tests/test_kde.py` - Unit tests for KDE functions
- `scripts/rom_mvar_visualize.py` - Visualization with density heatmaps
