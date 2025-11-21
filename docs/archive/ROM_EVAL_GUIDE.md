# ROM/MVAR Evaluation Pipeline Guide

Complete guide for evaluating trained ROM/MVAR models on unseen initial condition (IC) simulations.

**Author:** Maria  
**Date:** November 2025

---

## Overview

This pipeline evaluates ROM/MVAR (Proper Orthogonal Decomposition + Multivariate AutoRegressive) models on test simulations with different initial conditions. It computes forecast metrics, generates visualizations, and creates comparison videos.

### What Gets Evaluated

- **Input:** Trained ROM model (POD basis + MVAR coefficients)
- **Test Data:** Unseen simulations organized by IC type (ring, gaussian, uniform, cluster, etc.)
- **Outputs:** Metrics (R², RMSE, errors), plots (error vs time, order parameters), videos (truth vs prediction)

### Pipeline Steps

1. **Load Model:** Load trained ROM/MVAR from disk
2. **Load Simulations:** Load test simulations organized by IC type
3. **Run Predictions:** Encode → forecast in latent space → decode
4. **Compute Metrics:** R², RMSE, L¹/L²/L∞ errors, mass conservation, τ (threshold crossing)
5. **Aggregate:** Compute statistics per IC type and overall
6. **Select Best Runs:** Choose best simulation per IC type (by R²)
7. **Generate Plots:** Error vs time, order parameters
8. **Generate Videos:** Side-by-side truth vs prediction comparison

---

## Quick Start

### Single Command (Full Pipeline)

```bash
python scripts/rom_mvar_full_eval_local.py \
  --rom_dir rom_mvar/vicsek_morse_base/model \
  --unseen_root simulations_unseen \
  --out_root rom_mvar/vicsek_morse_base/unseen_eval \
  --train_frac 0.8 \
  --tol 0.1 \
  --fps 20
```

### Output Structure

```
rom_mvar/vicsek_morse_base/unseen_eval/
├── metrics_per_sim.csv          # Per-simulation metrics
├── metrics_per_sim.json
├── metrics_aggregated.json      # Aggregated by IC type + overall
├── ring/
│   ├── best_error.png           # Error vs time for best run
│   ├── best_order_params.png    # Order parameters (polarization, speed)
│   └── best_truth_vs_pred.mp4   # Side-by-side video comparison
├── gaussian/
│   └── ...
└── uniform/
    └── ...
```

---

## Detailed Usage

### Prerequisites

**Python Packages:**
```bash
pip install numpy pandas matplotlib imageio imageio-ffmpeg
```

**Directory Structure:**

Training data (ROM model):
```
rom_mvar/vicsek_morse_base/
└── model/
    ├── pod_basis.npz           # POD spatial basis functions
    ├── mvar_params.npz         # MVAR coefficients (A0, A1, ..., Ap)
    └── train_summary.json      # Metadata (latent_dim, mvar_order, etc.)
```

Test simulations:
```
simulations_unseen/
├── ring/
│   ├── sim_000/
│   │   ├── density.npz         # Required: density field (T, Ny, Nx)
│   │   ├── traj.npz           # Optional: trajectories for order params
│   │   └── run.json           # Optional: metadata
│   └── sim_001/
│       └── ...
├── gaussian/
│   └── ...
└── uniform/
    └── ...
```

### Step-by-Step Execution

#### 1. Run Predictions and Compute Metrics

```bash
python scripts/rom_mvar_eval_unseen.py \
  --rom_dir rom_mvar/exp1/model \
  --unseen_root simulations_unseen \
  --out_dir results/eval_unseen \
  --train_frac 0.8 \
  --tol 0.1 \
  --save_predictions
```

**Outputs:**
- `metrics_per_sim.csv` / `.json` - Metrics for each simulation
- `metrics_aggregated.json` - Aggregated statistics
- `predictions/*.npz` - Density predictions and error timeseries (if `--save_predictions`)

**Metrics Computed:**
- **R²:** Coefficient of determination (1 - MSE/var)
- **RMSE:** Root mean squared error (mean over time)
- **L¹/L²/L∞:** Median norms per time step
- **Mass error:** Relative mass conservation error
- **τ (tau):** Time when error exceeds threshold (default: 10%)

#### 2. Generate Best Run Plots

```bash
python scripts/rom_mvar_best_plots.py \
  --eval_dir results/eval_unseen \
  --unseen_root simulations_unseen \
  --out_dir results/eval_unseen/best_plots \
  --metric r2
```

**Outputs:**
- `ic_type/best_error.png` - Error vs time (L², relative L², L¹, L∞, mass)
- `ic_type/best_order_params.png` - Polarization and speed statistics

**Selection Criteria:**
- Default: Maximum R² per IC type
- Can use `--metric rmse_mean --no-maximize` to select by min RMSE

#### 3. Generate Videos (Optional)

Videos are generated automatically by `rom_mvar_full_eval_local.py`, or manually:

```python
from rectsim.rom_video_utils import make_truth_vs_pred_density_video

make_truth_vs_pred_density_video(
    density_true,   # (T, Ny, Nx)
    density_pred,   # (T, Ny, Nx)
    out_path=Path("video.mp4"),
    fps=20,
    title="Truth vs Prediction"
)
```

---

## Configuration Options

### Full Evaluation Script

```bash
python scripts/rom_mvar_full_eval_local.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--rom_dir` | *required* | ROM model directory |
| `--unseen_root` | *required* | Test simulations root |
| `--out_root` | *required* | Output directory |
| `--train_frac` | 0.8 | Fraction for initialization (T0 = train_frac * T) |
| `--tol` | 0.1 | Error tolerance for τ computation (10%) |
| `--ic_types` | auto-detect | Comma-separated IC types to evaluate |
| `--fps` | 20 | Video frames per second |
| `--metric` | r2 | Metric for best run selection |
| `--no-videos` | - | Skip video generation |
| `--no-plots` | - | Skip plot generation |

### Evaluation Only (No Plots/Videos)

```bash
python scripts/rom_mvar_eval_unseen.py \
  --rom_dir rom_mvar/exp1/model \
  --unseen_root simulations_unseen \
  --out_dir results/eval_unseen \
  --train_frac 0.8
```

---

## Understanding the Metrics

### Prediction Protocol

For each test simulation:

1. **Encode:** Project density field to latent space: `Y_true = encode(ρ_true)`
2. **Split:** Choose T0 = int(0.8 * T) as forecast start
3. **Initialize:** Use last p states as history: `Y_hist = Y_true[T0-p:T0]`
4. **Forecast:** Run MVAR prediction: `Y_pred = forecast(Y_hist, n_steps=T-T0)`
5. **Decode:** Reconstruct density: `ρ_pred = decode(Y_pred)`
6. **Compare:** Compute errors against `ρ_true[T0:]`

### Metric Definitions

**R² (Coefficient of Determination):**
```
R² = 1 - (SS_res / SS_tot)
SS_res = Σ (ρ_true - ρ_pred)²
SS_tot = Σ (ρ_true - mean(ρ_true))²
```
- Range: (-∞, 1], where 1 = perfect prediction
- Values < 0 indicate prediction worse than mean baseline

**RMSE (Root Mean Squared Error):**
```
RMSE(t) = sqrt(mean((ρ_true[t] - ρ_pred[t])²))
RMSE_mean = mean over time
```
- Lower is better
- Same units as density

**Relative L² Error:**
```
rel_e2(t) = ||ρ_pred[t] - ρ_true[t]||_2 / ||ρ_true[t]||_2
```
- Normalized error (dimensionless)
- τ = first time when rel_e2 > tol

**Mass Conservation:**
```
mass(t) = Σ ρ(t, x, y)
mass_error(t) = |mass_pred(t) - mass_true(t)| / mass_true(t)
```
- Measures if total mass is preserved
- Should be << 1 for good models

---

## File Formats

### Input: ROM Model Bundle

**pod_basis.npz:**
```python
{
    "basis": (d, Ny*Nx),     # POD spatial modes (row vectors)
    "grid_shape": (Ny, Nx),  # Spatial grid dimensions
    "mean_field": (Ny*Nx,)   # Mean field (optional)
}
```

**mvar_params.npz:**
```python
{
    "A0": (d,),              # Intercept
    "A1": (d, d),           # Lag-1 coefficient
    "A2": (d, d),           # Lag-2 coefficient (if p >= 2)
    ...
    "latent_dim": int,
    "mvar_order": int
}
```

**train_summary.json:**
```json
{
  "latent_dim": 20,
  "mvar_order": 3,
  "grid_shape": [64, 64],
  "n_train": 100,
  "train_r2": 0.95
}
```

### Input: Test Simulation

**density.npz:**
```python
{
    "density": (T, Ny, Nx),  # Required
    "times": (T,),           # Optional
    "dt": float,             # Optional metadata
    "Lx": float,
    "Ly": float
}
```

**traj.npz (optional, for order parameters):**
```python
{
    "x": (T, N, 2),          # Positions
    "v": (T, N, 2),          # Velocities
    "times": (T,)
}
```

### Output: Metrics

**metrics_per_sim.csv:**
```csv
ic_type,name,r2,rmse_mean,e1_median,e2_median,einf_median,mass_error_mean,mass_error_max,tau,n_forecast,train_frac
ring,sim_000,0.9523,0.0234,0.0189,0.0234,0.0567,0.0012,0.0023,None,80,0.8
ring,sim_001,0.9401,0.0289,0.0221,0.0289,0.0678,0.0015,0.0028,7.2,80,0.8
...
```

**metrics_aggregated.json:**
```json
{
  "overall": {
    "r2_mean": 0.9234,
    "r2_median": 0.9301,
    "r2_std": 0.0456,
    "rmse_mean": 0.0345,
    "n": 150
  },
  "by_ic_type": {
    "ring": {
      "r2_mean": 0.9523,
      "rmse_mean": 0.0234,
      "n": 30
    },
    "gaussian": { ... }
  }
}
```

---

## Advanced Usage

### Custom Metric Selection

Select best runs by minimum RMSE instead of maximum R²:

```bash
python scripts/rom_mvar_best_plots.py \
  --eval_dir results/eval_unseen \
  --unseen_root simulations_unseen \
  --metric rmse_mean \
  --no-maximize
```

### Programmatic Access

```python
from rectsim.rom_eval_pipeline import evaluate_unseen_rom, aggregate_metrics
from rectsim.rom_eval_viz import select_best_runs

# Run evaluation
metrics_list, predictions = evaluate_unseen_rom(
    rom_dir=Path("rom_mvar/exp1/model"),
    unseen_root=Path("simulations_unseen"),
    train_frac=0.8,
    return_predictions=True
)

# Aggregate
aggregated = aggregate_metrics(metrics_list)
print(f"Overall R²: {aggregated['overall']['r2_mean']:.4f}")

# Select best
best = select_best_runs(metrics_list, key="r2", maximize=True)
for ic_type, metrics in best.items():
    print(f"{ic_type}: {metrics.name} (R²={metrics.r2:.4f})")
```

### Smoke Test

Verify installation and basic functionality:

```bash
# Test metrics computation
python tests/test_rom_eval_metrics.py

# Test visualization
python tests/test_rom_eval_viz.py

# Test video generation
python tests/test_rom_video_utils.py
```

---

## Troubleshooting

### "No simulations found"

- Check directory structure: `simulations_unseen/ic_type/sim_xxx/density.npz`
- Verify `density.npz` contains `"density"` or `"rho"` array

### "Grid shape mismatch"

- ROM model expects specific (Ny, Nx) grid
- Check `train_summary.json` for expected dimensions
- All test simulations must use same grid as training

### "Trajectory too short"

- Simulation must have at least `T >= mvar_order + 10` time steps
- Increase simulation duration or reduce MVAR order

### Video generation fails

```bash
# Install video dependencies
pip install imageio imageio-ffmpeg

# Test manually
python tests/test_rom_video_utils.py
```

### Out of memory

- Reduce number of simulations with `--ic_types`
- Use `--no-videos` to skip video generation
- Process IC types separately

---

## Performance Tips

- **Parallel Evaluation:** Process IC types separately and combine results
- **Skip Videos:** Use `--no-videos` for faster metrics-only runs (5-10x speedup)
- **Reduced FPS:** Use `--fps 10` for smaller video files
- **Local Execution:** This pipeline is designed for local machines, not HPC clusters

---

## Citation

If you use this evaluation pipeline, please cite:

```
[Your paper/repository citation here]
```

---

## Support

For issues or questions:
- Check documentation: `docs/ROM_MVAR_GUIDE.md`
- Run smoke tests: `tests/test_rom_eval_*.py`
- Review examples: `examples/rom_mvar_eval_example.py` (if exists)
