# Simulation Outputs Guide

## Overview

All simulations now produce **standardized outputs** in a consistent format, regardless of whether you run:
- Pure Vicsek (alignment only)
- Hybrid Vicsek-D'Orsogna (alignment + forces)
- Pure D'Orsogna continuous (forces + optional alignment)

The output system is consistent across all models.

## Output Directory Structure

Each simulation creates a directory (e.g., `outputs/vicsek_dorsogna_discrete/`) containing:

```
outputs/vicsek_dorsogna_discrete/
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

## File Descriptions

### Videos (MP4)

1. **`traj_animation.mp4`** - Particle trajectory animation
   - Shows particles as arrows colored by heading angle
   - Arrow length scaled by speed
   - Ideal for visualizing collective motion patterns
   - fps = 20 (configurable in `outputs.fps`)

2. **`density_animation.mp4`** - Density field animation
   - Shows spatial distribution using kernel density estimation
   - Reveals clustering, waves, and density patterns
   - Resolution configurable in `outputs.density_resolution`

### Data Files (CSV)

3. **`order_parameters.csv`** - Metrics time series
   ```csv
   time,polarization,angular_momentum,mean_speed,density_variance
   0.0,0.1234,0.0567,0.5000,0.0023
   0.1,0.1345,0.0589,0.5001,0.0022
   ...
   ```
   - **Polarization (Φ)**: Velocity alignment, range [0, 1]
   - **Angular momentum (L)**: Collective rotation
   - **Mean speed**: Average particle speed
   - **Density variance**: Spatial clustering measure

4. **`traj.csv`** - Full trajectory data
   ```csv
   time,particle_id,x,y,vx,vy
   0.0,0,1.234,5.678,0.345,0.123
   0.0,1,2.345,6.789,0.456,0.234
   ...
   ```
   - Every particle, every saved frame
   - Can be large for long simulations

5. **`density.csv`** - Density field data
   - Spatial grid of particle density over time
   - Computed via Gaussian KDE

### Analysis Files

6. **`order_summary.png`** - Summary visualization
   - 4 subplots showing time evolution:
     * Polarization Φ(t)
     * Angular momentum L(t)
     * Mean speed v̄(t)
     * Density variance σ²_ρ(t)
   - Quick visual overview of simulation behavior

7. **`results.npz`** - Binary data for Python
   ```python
   import numpy as np
   data = np.load('outputs/vicsek_dorsogna_discrete/results.npz', allow_pickle=True)
   times = data['times']        # shape: (T,)
   positions = data['positions']  # shape: (T, N, 2)
   velocities = data['velocities']  # shape: (T, N, 2)
   config = data['config'].item()  # dict
   ```

### Metadata

8. **`config.yaml`** - Exact configuration used
   - Copy of the input config file
   - Ensures reproducibility

9. **`metadata.json`** - Run information
   ```json
   {
     "model": "vicsek_discrete",
     "forces_enabled": true,
     "timestamp": "2025-10-20T16:15:23.456789",
     "seed": 42,
     "N": 100,
     "T": 100.0,
     "frames_saved": 1001
   }
   ```

## Running Simulations

### Discrete Vicsek Models

Use the script `scripts/run_vicsek_discrete.py`:

```bash
# Pure Vicsek (no forces)
python scripts/run_vicsek_discrete.py examples/configs/vicsek_pure.yaml

# Hybrid Vicsek-D'Orsogna (with forces)
python scripts/run_vicsek_discrete.py examples/configs/vicsek_dorsogna_discrete.yaml

# Custom output directory
python scripts/run_vicsek_discrete.py config.yaml -o outputs/my_run

# Override seed
python scripts/run_vicsek_discrete.py config.yaml --seed 12345
```

### Continuous D'Orsogna Models

Use the script `scripts/run_dorsogna.py`:

```bash
# Continuous D'Orsogna with optional alignment
python scripts/run_dorsogna.py examples/configs/vicsek_dorsogna_continuous.yaml
```

## Configuration: Outputs Section

Control what gets generated via the `outputs` section in your YAML config:

```yaml
outputs:
  directory: outputs/my_simulation  # Where to save
  
  # Flags
  order_parameters: true   # Compute metrics (CSV + plots)
  animations: true         # Create videos (requires ffmpeg)
  save_csv: true          # Save full trajectory/density CSVs
  
  # Settings
  fps: 20                 # Video frame rate
  density_resolution: 50  # Grid size for density field
```

## Comparing Results

### Example 1: Pure Vicsek vs Hybrid

```python
import numpy as np
import matplotlib.pyplot as plt

# Load both simulations
pure = np.load('outputs/vicsek_pure/results.npz', allow_pickle=True)
hybrid = np.load('outputs/vicsek_dorsogna_discrete/results.npz', allow_pickle=True)

# Compare final states
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Pure Vicsek
axes[0].scatter(pure['positions'][-1, :, 0], 
               pure['positions'][-1, :, 1], alpha=0.6)
axes[0].set_title('Pure Vicsek (alignment only)')
axes[0].set_aspect('equal')

# Hybrid
axes[1].scatter(hybrid['positions'][-1, :, 0],
               hybrid['positions'][-1, :, 1], alpha=0.6)
axes[1].set_title('Hybrid (alignment + forces)')
axes[1].set_aspect('equal')

plt.show()
```

### Example 2: Speed Distribution Comparison

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load metrics
pure_metrics = pd.read_csv('outputs/vicsek_pure/order_parameters.csv')
hybrid_metrics = pd.read_csv('outputs/vicsek_dorsogna_discrete/order_parameters.csv')

# Plot mean speeds
plt.figure(figsize=(10, 4))
plt.plot(pure_metrics['time'], pure_metrics['mean_speed'], 
         label='Pure Vicsek', linewidth=2)
plt.plot(hybrid_metrics['time'], hybrid_metrics['mean_speed'],
         label='Hybrid (with forces)', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Mean Speed')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Key Differences Between Models

### Pure Vicsek
- **Fixed speed**: All particles move at v₀ = 0.5 exactly
- **Alignment only**: Coordination via heading synchronization
- **Mean speed variance**: ≈ 0 (constant speed constraint)
- **Output**: `outputs/vicsek_pure/`

### Hybrid Vicsek-D'Orsogna
- **Variable speed**: Forces can speed up or slow down particles
- **Alignment + spacing**: Both coordination and repulsion/attraction
- **Mean speed variance**: > 0 (forces affect speeds)
- **Output**: `outputs/vicsek_dorsogna_discrete/`

### Continuous D'Orsogna
- **Natural speed**: Self-propulsion at v₀ = α/β
- **Force-dominated**: Morse potential drives dynamics
- **Optional alignment**: Can add Vicsek-style coordination
- **Integration**: RK4 (4th order accuracy)

## Troubleshooting

### No videos generated?

Check if ffmpeg is installed:
```bash
which ffmpeg
```

Install if needed:
```bash
# macOS
brew install ffmpeg

# Linux
sudo apt-get install ffmpeg
```

Or disable animations in config:
```yaml
outputs:
  animations: false  # Skip video creation
```

### Files too large?

Reduce save frequency:
```yaml
sim:
  save_every: 50  # Save less frequently
```

Or disable CSV output:
```yaml
outputs:
  save_csv: false  # Only keep NPZ + plots
```

### Want to recreate videos later?

Use the standalone animation script:
```bash
python scripts/create_animations.py outputs/vicsek_dorsogna_discrete/results.npz
```

## Summary

✅ **Consistent outputs** across all model types  
✅ **Videos + CSVs + plots** in every simulation  
✅ **Reproducible** via saved configs and seeds  
✅ **Analyzable** with standard Python tools (pandas, numpy, matplotlib)  

The output system has **not changed** - only the models were extended to support hybrid dynamics!
