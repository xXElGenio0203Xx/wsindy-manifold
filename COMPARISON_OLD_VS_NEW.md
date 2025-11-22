# Comparison: Old Pipeline (commit 67655d3) vs Current Pipeline

## Folder Structure

### OLD VERSION (Working):
```
/Users/maria_1/Desktop/wsindy-manifold-OLD/
```
- Located at Desktop level
- Commit: 67655d38c709f9ce1dc89311c0afb8124bc0b596
- Date: Earlier version with working production pipeline

### CURRENT VERSION:
```
/Users/maria_1/Desktop/wsindy-manifold/
```
- Current working directory
- Latest commits with recent refactoring

---

## Key Differences in Pipeline Scripts

### OLD: `scripts/run_mvar_rom_production.py` (658 lines)
**Features:**
- Loads pre-generated simulations from `simulations/<sim_name>__<run_id>/`
- Generates outputs in `mvar_outputs/<sim_name>__<run_id>__<exp_name>/`
- Uses `wsindy_manifold` module:
  - `wsindy_manifold.io` - for manifest, arrays, CSV, video generation
  - `wsindy_manifold.pod` - fit_pod, restrict, lift
  - `wsindy_manifold.standard_metrics` - rel_errors, r2_score, tolerance_horizon
  - `wsindy_manifold.latent.mvar` - fit_mvar, rollout
- **Video generation**: `side_by_side_video()` function
- Structured output:
  - manifest.json
  - config.yaml
  - pod/ (Ud.npy, xbar.npy, energy_curve.npy, energy.png)
  - model/ (A0.npy, Astack.npy, summary.json)
  - forecast/ (latent_pred.npy, density_pred.npz, videos/)
  - eval/ (metrics_over_time.csv, summary.json, plots/)

### CURRENT: `run_production_pipeline.py` (514 lines)
**Features:**
- Generates simulations inline (100 training, 20 test)
- Uses `rectsim` module (consolidated version)
- Outputs in `outputs/production_pipeline/`
- Different structure:
  - train/ - training runs
  - test/ - test runs with trajectories
  - mvar/ - trained model
  - videos/ - comparison videos
  - plots/ - summary plots

---

## Key Module Differences

### OLD: `src/wsindy_manifold/`
```
wsindy_manifold/
├── density.py (173 lines) - kde_density_movie with metadata
├── io.py (341 lines) - save_manifest, side_by_side_video, etc.
├── pod.py (204 lines) - fit_pod, restrict, lift
├── standard_metrics.py (349 lines) - comprehensive metrics
├── mvar_rom.py (1025 lines) - full ROM implementation
└── latent/
    └── mvar.py (628 lines) - fit_mvar, rollout
```

### CURRENT: `src/rectsim/`
```
rectsim/
├── density.py - compute_density_grid, density_movie_kde
├── pod.py - compute_pod (different API)
├── rom_eval_metrics.py - compute_relative_errors_timeseries
├── mvar_models.py - fit_mvar_from_runs
└── (consolidated from multiple modules)
```

---

## Video Generation Comparison

### OLD: `wsindy_manifold.io.side_by_side_video()`
Located in: `wsindy-manifold-OLD/src/wsindy_manifold/io.py`

**What it does:**
- Creates side-by-side comparison videos
- Shows density heatmaps (true | predicted)
- Includes error metrics plot below
- Uses imageio for video generation
- Well-tested, production-ready

### CURRENT: Custom video generation in pipeline script
**What it does:**
- Top: True particles (scatter) | Predicted density (heatmap)
- Bottom: Error over time graph with threshold
- Using matplotlib figure → frame conversion
- New implementation, not fully tested

---

## Density Computation Comparison

### OLD: `kde_density_movie()` in `wsindy_manifold.density`
```python
def kde_density_movie(
    traj: Array,
    Lx: float, Ly: float,
    nx: int, ny: int,
    bandwidth: float,
    bc: str = "periodic"
) -> Tuple[Array, Dict]:
    """Returns (rho, meta) where meta contains full grid info"""
    # Returns density movie + metadata dict
    # Metadata: bandwidth, nx, ny, Lx, Ly, extent, bc, N_particles, T_frames
```

### CURRENT: `compute_density_grid()` and `density_movie_kde()` in `rectsim.density`
```python
def compute_density_grid(pos, nx, ny, Lx, Ly, bandwidth, bc):
    """Single frame density - returns (rho, x_edges, y_edges)"""
    # No metadata dictionary
    # Must be called per frame
    
def density_movie_kde(traj, Lx, Ly, nx, ny, bandwidth, bc):
    """Returns only density array, no metadata"""
```

---

## How to Run Comparisons

### 1. Run Old Pipeline
```bash
cd /Users/maria_1/Desktop/wsindy-manifold-OLD

# First generate simulations
python scripts/run_sim_production.py --config configs/gentle_clustering.yaml --n_runs 20

# Then run MVAR-ROM
python scripts/run_mvar_rom_production.py \
  --sim_dir simulations/gentle_clustering__001 \
  --exp_name test_old_pipeline \
  --pod_energy 0.90
```

### 2. Run Current Pipeline
```bash
cd /Users/maria_1/Desktop/wsindy-manifold
python run_production_pipeline.py
```

### 3. Compare Outputs
- OLD outputs: `wsindy-manifold-OLD/mvar_outputs/`
- NEW outputs: `wsindy-manifold/outputs/production_pipeline/`

---

## What You Requested

> "I want to have similar code that we had to this repo version"

The old version is now available at:
```
/Users/maria_1/Desktop/wsindy-manifold-OLD/
```

You can:
1. **Browse the old code** to see what worked before
2. **Copy functions** from old → new (especially video generation)
3. **Compare side-by-side** the two implementations
4. **Run both pipelines** to see visual differences
5. **Cherry-pick fixes** from the old version

---

## Recommended Actions

### Option 1: Restore Old Video Function
Copy `side_by_side_video()` from:
```
wsindy-manifold-OLD/src/wsindy_manifold/io.py
```
Into current pipeline.

### Option 2: Restore Old Density Function
Use `kde_density_movie()` from:
```
wsindy-manifold-OLD/src/wsindy_manifold/density.py
```
Which returns proper metadata.

### Option 3: Restore Full Old Pipeline
Use the old scripts directly:
```bash
cp -r wsindy-manifold-OLD/scripts/run_mvar_rom_production.py wsindy-manifold/scripts/
cp -r wsindy-manifold-OLD/src/wsindy_manifold wsindy-manifold/src/
```

---

## Quick Commands to Explore Old Version

```bash
# Navigate to old version
cd /Users/maria_1/Desktop/wsindy-manifold-OLD

# See structure
tree -L 2 -I '__pycache__|*.pyc'

# View old video generation code
less src/wsindy_manifold/io.py

# View old density computation
less src/wsindy_manifold/density.py

# View old MVAR-ROM pipeline
less scripts/run_mvar_rom_production.py

# List available configs
ls configs/

# Check what simulations exist (if any)
ls simulations/ 2>/dev/null || echo "No simulations yet - run run_sim_production.py first"
```

Let me know which approach you'd like to take!
