# Global POD + MVAR Pipeline Guide

Complete guide for training Multivariate AutoRegressive (MVAR) models on density field ensembles using global Proper Orthogonal Decomposition (POD).

## Overview

The EF-ROM (Empirical Flow Reduced-Order Model) pipeline consists of three phases:

1. **Ensemble Generation** (Prompt 1): Generate multiple simulations with varied initial conditions
2. **Global POD Basis** (Prompt 2): Compute low-dimensional basis across all runs
3. **MVAR Training** (Prompt 3): Train linear autoregressive model in latent space

This guide covers **Phase 3: MVAR Training and Evaluation**.

## Quick Start

```bash
# 1. Generate ensemble (10 runs, varied ICs)
rectsim ensemble --config configs/mvar_example.yaml

# 2. Train MVAR model
python scripts/run_mvar_global.py \
  --sim_root simulations/social_force_N100_T500_... \
  --out_root MVAR_outputs/test_run1 \
  --order 4 \
  --ridge 1e-6 \
  --train_frac 0.8

# 3. Inspect results
ls MVAR_outputs/test_run1/
# pod/       - POD basis (Phi.npy, S.npy, mean.npy)
# model/     - MVAR model (mvar_model.npz, train_info.json)
# metrics/   - Evaluation (mvar_metrics.json, rmse_time_series.csv)
# plots/     - Visualizations (pod_energy.png, rmse_time_series.png)
```

## Pipeline Steps

### Step 1: Load Density Movies

The pipeline scans for `density.npz` files in the simulation directory tree.

**Expected structure:**
```
simulations/social_force_N100_T500.../
├── run_0001/
│   ├── density.npz      # (T, ny, nx) density movie
│   ├── run.json         # Optional metadata
│   └── ...
├── run_0002/
│   ├── density.npz
│   └── ...
└── ...
```

**Function:** `load_density_movies(run_dirs)`
- Returns: `density_dict[run_name] = {'rho': (T, ny, nx), 'times': (T,), 'meta': {}}`

### Step 2: Build Global Snapshot Matrix

Concatenate flattened density fields from all runs into a single matrix.

**Function:** `build_global_snapshot_matrix(density_dict, subtract_mean=True)`
- Flatten: `(T_r, ny, nx)` → `(T_r, d)` where `d = ny * nx`
- Concatenate: All runs → `(T_total, d)` where `T_total = sum(T_r)`
- Center: Subtract global spatial mean if `subtract_mean=True`
- Returns: `X, run_slices, global_mean_flat`

### Step 3: Compute Global POD Basis

Perform SVD on the snapshot matrix to extract dominant spatial modes.

**Function:** `compute_pod(X, r=None, energy_threshold=0.995)`
- SVD: `X = U @ diag(S) @ Vt`
- Energy: `E_k = sum(S[:k]**2) / sum(S**2)`
- Mode selection: Choose `r` such that `E_r >= energy_threshold`
- Returns: `pod_basis = {'Phi': (d, r), 'S': all singular values, 'U': (T_total, r), 'r': int, 'energy': (k,)}`

**Output:**
- `pod/Phi.npy`: Spatial modes (d, r)
- `pod/S.npy`: Singular values
- `pod/mean.npy`: Global spatial mean
- `pod/pod_info.json`: Metadata (r, energy_threshold, energy_captured)
- `plots/pod_energy.png`: Cumulative energy plot

### Step 4: Project to Latent Space

Project density fields onto POD basis to obtain time-varying latent coefficients.

**Function:** `project_to_pod(density_dict, Phi, global_mean_flat)`
- For each run: `Y_r = (rho_flat - mean) @ Phi`, shape `(T_r, r)`
- Returns: `latent_dict[run_name] = {'Y': (T_r, r), 'times': (T_r,)}`

### Step 5: Fit MVAR Model

Train linear autoregressive model on latent time series using ridge regression.

**Function:** `fit_mvar_from_runs(latent_dict, order=4, ridge=1e-6, train_frac=0.8)`

**Model:**
```
y(t) = A0 + A[0] @ y(t-1) + A[1] @ y(t-2) + ... + A[order-1] @ y(t-order)
```

**Training:**
- Use first `train_frac` (default 80%) of each run for training
- Concatenate regression samples from all runs
- Ridge regression: `W = (X^T X + ridge * I)^{-1} X^T Y`
- Returns: `MVARModel(order, A0, A, ridge, latent_dim)`, `train_info`

**Output:**
- `model/mvar_model.npz`: MVAR coefficients (A0, A)
- `model/train_info.json`: Training metadata (num_samples, num_runs, train_splits)

### Step 6: Evaluate on Test Segments

Evaluate multi-step forecast accuracy on held-out test segments.

**Function:** `evaluate_mvar_on_runs(...)`

**Metrics:**
- **Per-run:**
  - R²: Coefficient of determination at density field level
  - RMSE time series: Root-mean-square error over forecast horizon
  - Latent RMSE: RMSE in latent space
- **Aggregate:**
  - Mean/median/std of R² across runs
  - Mean/median/percentiles of RMSE

**Output:**
- `metrics/mvar_metrics.json`: Full evaluation results
- `metrics/rmse_time_series.csv`: Per-run RMSE over time
- `plots/rmse_time_series.png`: RMSE evolution for all runs

## Command-Line Arguments

```bash
python scripts/run_mvar_global.py [OPTIONS]
```

**Required:**
- `--sim_root PATH`: Root directory containing simulation runs
- `--out_root PATH`: Output directory for all results

**Optional:**
- `--pattern STR`: Glob pattern to match density.npz files (default: `"run_*/density.npz"`)
- `--order INT`: MVAR model order (default: 4)
- `--ridge FLOAT`: Ridge regularization parameter (default: 1e-6)
- `--train_frac FLOAT`: Training fraction (0.0-1.0, default: 0.8)
- `--max_runs INT`: Limit number of runs for testing (default: None)
- `--energy_threshold FLOAT`: POD energy threshold (default: 0.995)

## Example Workflows

### Local Testing (Fast)

```bash
# Generate small ensemble
rectsim ensemble --config configs/mvar_example.yaml \
  --sim.N 50 --sim.T 300 --ensemble.n_runs 5 --density.nx 32 --density.ny 32

# Train MVAR
python scripts/run_mvar_global.py \
  --sim_root simulations/social_force_N50_T300_... \
  --out_root MVAR_outputs/local_test \
  --order 3 --ridge 1e-6 --max_runs 5
```

### Oscar HPC (Production)

```bash
# Generate large ensemble (20 runs, N=200, T=1000)
rectsim ensemble --config configs/mvar_production.yaml

# Train MVAR with higher-order model
python scripts/run_mvar_global.py \
  --sim_root /users/emaciaso/src/wsindy-manifold/simulations/social_force_N200_T1000_... \
  --out_root /users/emaciaso/src/wsindy-manifold/MVAR_outputs/production_run1 \
  --order 6 \
  --ridge 1e-6 \
  --train_frac 0.8 \
  --energy_threshold 0.995
```

### Parameter Sweep

```bash
# Test different MVAR orders
for order in 2 4 6 8; do
  python scripts/run_mvar_global.py \
    --sim_root simulations/... \
    --out_root MVAR_outputs/order_${order} \
    --order ${order} \
    --ridge 1e-6
done

# Compare results
for dir in MVAR_outputs/order_*; do
  echo "$dir:"
  grep "mean_R2" $dir/metrics/mvar_metrics.json
done
```

## Output Directory Structure

```
MVAR_outputs/global_run1/
├── pod/
│   ├── Phi.npy              # POD spatial modes (d, r)
│   ├── S.npy                # Singular values (k,)
│   ├── mean.npy             # Global spatial mean (d,)
│   └── pod_info.json        # Metadata (r, energy_threshold, energy_captured)
├── model/
│   ├── mvar_model.npz       # MVAR coefficients (order, A0, A, ridge, latent_dim)
│   └── train_info.json      # Training metadata
├── metrics/
│   ├── mvar_metrics.json    # Per-run and aggregate evaluation
│   └── rmse_time_series.csv # RMSE over time for each run
└── plots/
    ├── pod_energy.png       # Cumulative energy curve
    └── rmse_time_series.png # RMSE evolution
```

## Python API Usage

```python
from pathlib import Path
from rectsim.mvar import (
    load_density_movies,
    build_global_snapshot_matrix,
    compute_pod,
    project_to_pod,
    fit_mvar_from_runs,
    mvar_forecast,
    evaluate_mvar_on_runs,
)

# Load data
run_dirs = list(Path("simulations/model_id").glob("run_*"))
density_dict = load_density_movies(run_dirs)

# Compute POD
X, run_slices, mean = build_global_snapshot_matrix(density_dict)
pod_basis = compute_pod(X, r=None, energy_threshold=0.995)

# Project to latent space
latent_dict = project_to_pod(density_dict, pod_basis["Phi"], mean)

# Fit MVAR
model, info = fit_mvar_from_runs(latent_dict, order=4, ridge=1e-6)

# Forecast
Y_init = latent_dict["run_0001"]["Y"][:4]  # Last 4 states
Y_pred = mvar_forecast(model, Y_init, steps=100)

# Evaluate
results = evaluate_mvar_on_runs(
    model, latent_dict, density_dict, pod_basis, mean, ny, nx
)
print(f"Mean R²: {results['aggregate']['mean_R2']:.4f}")
```

## Troubleshooting

### Issue: No density.npz files found

**Solution:** Ensure ensemble was generated with `save_density: true` in config:
```yaml
output:
  save_density: true
```

### Issue: POD uses too many/few modes

**Adjust energy threshold:**
```bash
python scripts/run_mvar_global.py ... --energy_threshold 0.99  # More modes
python scripts/run_mvar_global.py ... --energy_threshold 0.98  # Even more
```

### Issue: MVAR forecast explodes

**Increase ridge regularization:**
```bash
python scripts/run_mvar_global.py ... --ridge 1e-5  # Stronger regularization
```

**Or reduce model order:**
```bash
python scripts/run_mvar_global.py ... --order 3  # Simpler model
```

### Issue: Low R² on test segments

**Possible causes:**
- **Too little training data**: Increase `--train_frac 0.85`
- **Insufficient POD modes**: Decrease `--energy_threshold 0.99`
- **Model too simple**: Increase `--order 6`
- **Model overfitting**: Increase `--ridge 1e-5`

### Issue: Out of memory

**Reduce data:**
```bash
python scripts/run_mvar_global.py ... --max_runs 10  # Use fewer runs
```

**Or downsample densities** (modify ensemble config):
```yaml
density:
  nx: 64  # Reduce from 128
  ny: 64
  t_skip: 2  # Downsample time
```

## Performance Notes

**Typical runtimes (Oscar HPC):**
- Load 20 runs (T=1000, 128x128): ~30 seconds
- Build snapshot matrix: ~5 seconds
- Compute POD: ~1-2 minutes (depends on T_total and d)
- Fit MVAR: ~10-30 seconds (depends on order and T_total)
- Evaluate: ~20-40 seconds (depends on forecast horizon)

**Total:** ~3-5 minutes for 20 runs with 128x128 grids

**Memory requirements:**
- Snapshot matrix: `T_total * d * 8 bytes` (float64)
  - Example: 20,000 timesteps × 16,384 dimensions = ~2.5 GB
- POD modes: `d * r * 8 bytes`
  - Example: 16,384 × 50 modes = ~6 MB

## References

- **POD methodology**: Lumley, J. L. (1967). "The structure of inhomogeneous turbulent flows"
- **MVAR forecasting**: Lütkepohl, H. (2005). "New Introduction to Multiple Time Series Analysis"
- **EF-ROM framework**: Alvarez et al. (2024). "Autoregressive ROM for collective motion forecasting"
- **D'Orsogna swarms**: Bhaskar & Ziegelmeier (2023). "KDE-based density representation for social force models"

## See Also

- [ENSEMBLE_GUIDE.md](../ENSEMBLE_GUIDE.md): Phase 1 - Ensemble generation
- [configs/mvar_example.yaml](../configs/mvar_example.yaml): Example configuration
- [tests/test_mvar.py](../tests/test_mvar.py): Test suite with examples
