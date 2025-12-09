

# ROM Pipeline Guide

Complete guide for the Reduced-Order Model (ROM) pipeline using global POD + MVAR for collective motion density forecasting.

## Overview

The ROM pipeline is now unified into a single command that handles all stages:

**Main Pipeline:** `run_unified_mvar_pipeline.py`
- Automatically runs all 4 stages from a single config file
- Handles training simulations → POD basis → MVAR training → Test evaluation
- Supports multiple IC types, custom experiments, and stability enforcement
- Generates comprehensive outputs and metrics

**Supporting Pipelines:**
- `run_visualizations.py` - Generate plots, videos, and analysis
- `run_parameter_experiment.py` - Quick parameter testing (no training)

**Pipeline Stages (automated):**
1. **Simulation**: Generate training and test simulations with varied ICs
2. **POD Basis**: Compute global POD basis from training runs only
3. **MVAR Training**: Fit autoregressive model in latent space
4. **Evaluation**: Generate forecasts and compute comprehensive metrics

This unified design provides:
- Single config file controls everything
- Automatic train/test separation
- Consistent evaluation across experiments
- Modular components in `src/rectsim/` for customization

## Directory Structure

```
project_root/
├── configs/
│   └── your_experiment.yaml    # Single config for entire pipeline
│
├── oscar_output/               # Pipeline outputs
│   └── <experiment_name>/
│       ├── train/
│       │   ├── simulations.csv      # Training run metadata
│       │   ├── density_rXXXX.npz    # Density fields
│       │   └── trajectory_rXXXX.npz # Trajectories
│       ├── test/
│       │   ├── simulations.csv      # Test run metadata
│       │   ├── density_rXXXX.npz
│       │   └── trajectory_rXXXX.npz
│       ├── pod/
│       │   ├── basis.npz            # POD modes, singular values, mean
│       │   ├── pod_energy.png
│       │   └── metadata.json
│       └── mvar/
│           ├── model.npz            # MVAR coefficients
│           ├── training_metadata.json
│           ├── test_metrics.csv     # Per-run metrics
│           ├── predictions_rXXXX.npz
│           └── forecast_metadata.json
│
├── predictions/                # Visualization outputs
│   └── <experiment_name>/
│       ├── pod_analysis/
│       ├── best_runs/
│       ├── summary_plots/
│       ├── time_resolved/
│       └── summary.json
│
└── experiments/                # Parameter testing outputs
    └── <test_name>/
        └── simulations/
            └── <ic_type>/
                ├── trajectory.mp4
## Quick Start

### Local Testing

**1. Run complete pipeline:**
```bash
python run_unified_mvar_pipeline.py \
  --config configs/unified_quick_test.yaml \
  --experiment_name my_test
```

This single command runs all 4 stages automatically:
- Generates training simulations (e.g., 8 runs with varied ICs)
- Generates test simulations (e.g., 2 runs)
- Builds POD basis from training data only
- Trains MVAR model on latent trajectories
- Evaluates forecasts on test runs
- Saves all outputs to `oscar_output/my_test/`

**2. Visualize results:**
```bash
python run_visualizations.py --experiment_name my_test
```

Generates:
- POD energy plots
- Per-run visualizations (best/worst)
- Summary statistics and boxplots
- Time-resolved R² degradation analysis
- Comprehensive summary JSON

**3. Quick parameter testing (optional):**
```bash
python run_parameter_experiment.py \
  --config configs/test_params.yaml \
  --experiment_name param_test
```

### Oscar HPC Workflow

**1. Create SLURM script:**
```bash
#!/bin/bash
#SBATCH --job-name=rom_experiment
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

source ~/wsindy-manifold/.venv/bin/activate
cd ~/wsindy-manifold

python run_unified_mvar_pipeline.py \
  --config configs/production_experiment.yaml \
  --experiment_name production_v1
```

**2. Submit job:**
```bash
sbatch my_job.slurm
```

**3. Monitor:**
```bash
squeue -u $USER
tail -f slurm-JOBID.out
```

**4. Download results and visualize locally:**
```bash
# On local machine
scp -r user@oscar:~/wsindy-manifold/oscar_output/production_v1 .
python run_visualizations.py --experiment_name production_v1
```

**See:** `slurm_scripts/` folder for example SLURM scripts.

**Example config structure:**
```yaml
# Base simulation parameters
sim:
  N: 200                    # Number of particles
  T: 400.0                  # Duration (seconds)
  dt: 0.1                   # Time step
  Lx: 30.0                  # Domain size X
  Ly: 30.0                  # Domain size Y
  v0: 1.0                   # Speed
  R: 1.0                    # Interaction radius
  bc: "periodic"            # Boundary conditions

# Density computation
density:
  nx: 64                    # Grid resolution X
  ny: 64                    # Grid resolution Y
  bandwidth: 0.7            # KDE bandwidth

# Training simulations
train_ic:
  n_train_per_type: 100     # Simulations per IC type
  distributions:
    - type: "gaussian_cluster"
      params: {std: 3.0}
    - type: "uniform"
      params: {}
    - type: "ring"
## Pipeline Stages (Automated)

### Stage 1: Simulation Generation

The pipeline automatically generates training and test simulations based on config.

**Handled by:** `rectsim.ic_generator` and `rectsim.simulation_runner`

**What happens:**
1. Generate initial conditions for each IC type and count
2. Run simulations in parallel using `joblib`
3. Compute density fields using KDE
4. Save outputs: `oscar_output/<experiment>/train/` and `test/`

**Outputs:**
- `simulations.csv` - Metadata (IC type, params, seeds)
- `density_rXXXX.npz` - Density fields (T, ny, nx)
- `trajectory_rXXXX.npz` - Particle trajectories (T, N, 2)

---

### Stage 2: POD Basis Construction

Build global POD basis from training runs only, project all runs to latent space.

**Handled by:** `rectsim.pod_builder`

**What happens:**
1. Load density fields from training runs only
2. Flatten spatial dimensions: (T, ny, nx) → (T, ny×nx)
3. Stack all training snapshots into matrix X
4. Center data: X_centered = X - mean
5. Compute SVD: X_centered = U @ diag(S) @ Vt
6. Select modes by energy threshold or fixed count
7. POD modes: Phi = U[:, :r]
8. Project all runs (train + test) to latent space: Y = (X - mean) @ Phi

**Outputs:**
- `pod/basis.npz` - Modes (Phi), singular values (S), mean, energy
- `pod/pod_energy.png` - Scree plot
- `pod/metadata.json` - Configuration and diagnostics

**Key principle:** POD basis computed ONLY from training data to prevent leakage.

---

### Stage 3: MVAR Training

Fit multivariate autoregressive model on training latent trajectories.

**Handled by:** `rectsim.mvar_trainer`

**What happens:**
1. Load latent trajectories for training runs
2. Apply time-based split: first 80% for training, last 20% reserved
3. Build autoregressive design matrix with specified lag order
4. Fit MVAR model via ridge regression:
   ```python
   Y(t) = A1·Y(t-1) + A2·Y(t-2) + ... + Ap·Y(t-p) + ε
   ```
5. Optional: Enforce stability by scaling eigenvalues < 1
6. Compute training metrics (residuals, R²)

**Outputs:**
- `mvar/model.npz` - Coefficient matrices (A1, A2, ..., Ap)
- `mvar/training_metadata.json` - Order, regularization, stability info

---

### Stage 4: Test Evaluation

Generate forecasts on test runs and compute comprehensive metrics.

**Handled by:** `rectsim.test_evaluator`

**What happens:**
1. Load test latent trajectories
2. For each test run:
   - Use first 20% as initial condition
   - Generate MVAR forecast for remaining time
   - Reconstruct density: ρ_pred = mean + Y_pred @ Phi.T
   - Compute metrics: MSE, MAE, R², mass conservation
   - Calculate time-resolved degradation
3. Aggregate statistics across all test runs

**Outputs:**
- `mvar/predictions_rXXXX.npz` - Predicted latent and density
- `mvar/test_metrics.csv` - Per-run metrics
- `mvar/forecast_metadata.json` - Aggregate statisticsim_root`: Directory containing simulation run folders
- `--train_runs`: Space-separated indices for training (used to build POD)
- `--test_runs`: Space-separated indices for testing (projected with fixed basis)
- `--energy_threshold`: POD energy threshold (default: 0.995 = 99.5%)
- `--latent_dim`: Fixed number of modes (overrides energy threshold)
- `--rom_root`: Output directory (default: `rom`)

**What it does:**
1. Load density movies from training runs only
2. Build global snapshot matrix: flatten + concatenate + center
3. Compute POD via SVD: `X = U @ diag(S) @ Vt`
4. Select modes based on cumulative energy threshold
5. Project all runs (train + test) to latent space: `Y = (X - mean) @ Phi`
6. Save POD basis, latent trajectories, and metadata

**Outputs:**
- `rom/<experiment>/pod/basis.npz`: Phi (d, r), S, mean, energy
- `rom/<experiment>/pod/pod_energy.png`: Scree plot
- `rom/<experiment>/latent/run_XXXX_latent.npz`: Y (T, r), times
- `rom/<experiment>/config.json`: Experiment configuration

**Key principle:** POD basis is computed ONLY from training data to avoid leakage.

---

### Stage 3: MVAR Training

Fit multivariate autoregressive model on latent trajectories.

**Command:**
```bash
python scripts/rom_train_mvar.py \
  --experiment_name <NAME> \
  --mvar_order 4 \
  --ridge 1e-6 \
  --train_frac 0.8
```

**Arguments:**
- `--experiment_name`: Same as Stage 2
- `--mvar_order`: Number of lags (default: 4)
- `--ridge`: Ridge regularization parameter (default: 1e-6)
- `--train_frac`: Fraction of each run's time for training (default: 0.8)
- `--rom_root`: ROM directory (default: `rom`)

**What it does:**
1. Load latent trajectories for training runs
2. Apply time-based split within each run (first 80% for training)
3. Fit MVAR model via ridge regression:
   ```
   y(t) = A0 + A1 @ y(t-1) + A2 @ y(t-2) + ... + Ap @ y(t-p)
   ```
4. Save model coefficients and metadata

**Outputs:**
- `rom/<experiment>/mvar/mvar_model.npz`: MVAR coefficients
- `rom/<experiment>/mvar/train_info.json`: Training metadata

**Time-based split:** Within each training run, use first `train_frac` (e.g., 80%) for fitting MVAR. Remaining 20% can be used for validation.

---

### Stage 4: Evaluation

Generate forecasts on test runs and compute comprehensive metrics.

**Command:**
```bash
python scripts/rom_evaluate.py \
  --experiment_name <NAME> \
  --sim_root <SIM_ROOT> \
  --no_videos  # Optional: skip video generation
```

**Arguments:**
- `--experiment_name`: Same as previous stages
- `--sim_root`: Directory with original simulation runs
- `--rom_root`: ROM directory (default: `rom`)
- `--no_videos`: Skip video generation (faster)
- `--snapshot_times`: Specific time indices for snapshots

**What it does:**
1. Load POD basis and MVAR model
2. For each test run:
   - Load latent trajectory and true density
   - Split at `train_frac` (e.g., t=0-800 vs t=800-1000)
   - Generate MVAR forecast from t=800 onwards using last `p` states
   - Reconstruct predicted density: `rho_pred = mean + Y @ Phi.T`
   - Compute metrics and visualizations
3. Aggregate statistics across all test runs

**Outputs per test run:**

*Numeric data:*
- `forecast_run_XXXX.npz`: density_true, density_pred, Y_true, Y_pred, errors, mass
- `metrics_run_XXXX.json`: All summary statistics
- `order_params_run_XXXX.csv`: Time-resolved order parameters

*Plots:*
- `errors_time_run_XXXX.png`: 3-panel dashboard (L2/RMSE, normalized RMSE, mass error)
- `order_params_run_XXXX.png`: True vs predicted order parameters
- `snapshot_grid_run_XXXX.png`: Side-by-side snapshots at start/middle/end

*Videos (if not --no_videos):*
- `density_true_run_XXXX.mp4`: True density evolution
- `density_pred_run_XXXX.mp4`: Predicted density evolution
- `density_comparison_run_XXXX.mp4`: Side-by-side comparison

*Aggregate:*
- `aggregate_metrics.json`: Mean/median R², RMSE, mass drift across test runs

---

## Evaluation Metrics

### Pointwise Errors (per time step)

- **L1 error**: `e1(t) = ||ρ_pred(t) - ρ_true(t)||_1`
- **L2 error**: `e2(t) = ||ρ_pred(t) - ρ_true(t)||_2`
- **L∞ error**: `e_inf(t) = ||ρ_pred(t) - ρ_true(t)||_∞`
- **RMSE**: `rmse(t) = e2(t) / sqrt(n_grid)`
- **Normalized RMSE**: `rmse_norm(t) = e2(t) / ||ρ_true(t)||_2`

### Summary Statistics (over forecast window)

- Median of e1, e2, e_inf
- 10th and 90th percentiles of e2
- Mean and median RMSE
- **R² score**: Coefficient of determination over all (t, i, j) samples

### Mass Conservation

- `mass(t) = ∑_{i,j} ρ(t, i, j) * ΔA`
- `mass_error(t) = mass_pred(t) - mass_true(t)`
- `mass_drift_max = max_t |mass_error(t)|`
- `mass_conservation_ok = (mass_drift_max < 1e-6)`

### Order Parameters

- System-specific order parameter computed from density (e.g., polarization, clustering)
- Tracked for both true and predicted densities
- `order_error(t) = |order_pred(t) - order_true(t)|`

### Diagnostics

- `nan_count`: Total NaN values in forecasts (should be 0)
- `compression_ratio`: `(ny * nx) / r` (dimensionality reduction)

---

## Oscar Workflow

### SLURM Scripts

Create a workflow with three sequential SLURM jobs:

**1. Ensemble generation (`job_ensemble.sh`):**

```bash
#!/bin/bash
#SBATCH --job-name=rom_ensemble
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem=64G
#SBATCH --output=logs/ensemble_%j.out

module load python/3.11
source ~/envs/wsindy/bin/activate

# Generate 20 runs in parallel
rectsim ensemble --config configs/rom_production.yaml
```

**2. POD + MVAR training (`job_train.sh`):**

```bash
#!/bin/bash
#SBATCH --job-name=rom_train
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --output=logs/train_%j.out

module load python/3.11
source ~/envs/wsindy/bin/activate

EXP_NAME="production_run1"
SIM_ROOT="simulations/social_force_N200_T1000.../runs"

# Stage 2: Build POD
python scripts/rom_build_pod.py \
  --experiment_name $EXP_NAME \
  --sim_root $SIM_ROOT \
  --train_runs 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
  --test_runs 16 17 18 19 \
  --energy_threshold 0.995

# Stage 3: Train MVAR
python scripts/rom_train_mvar.py \
  --experiment_name $EXP_NAME \
  --mvar_order 6 \
  --ridge 1e-6 \
  --train_frac 0.8
```

**3. Evaluation (`job_evaluate.sh`):**

```bash
#!/bin/bash
#SBATCH --job-name=rom_eval
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --output=logs/eval_%j.out

module load python/3.11
source ~/envs/wsindy/bin/activate

EXP_NAME="production_run1"
SIM_ROOT="simulations/social_force_N200_T1000.../runs"

# Stage 4: Evaluate
python scripts/rom_evaluate.py \
  --experiment_name $EXP_NAME \
  --sim_root $SIM_ROOT
  # Videos will be generated (remove --no_videos if not wanted)
```

**Workflow:**

```bash
# Submit jobs with dependencies
JOB1=$(sbatch job_ensemble.sh | awk '{print $4}')
JOB2=$(sbatch --dependency=afterok:$JOB1 job_train.sh | awk '{print $4}')
JOB3=$(sbatch --dependency=afterok:$JOB2 job_evaluate.sh | awk '{print $4}')

echo "Submitted job chain: $JOB1 -> $JOB2 -> $JOB3"
```

### Data Transfer

```bash
# From Oscar to local machine
scp -r <user>@oscar.ccv.brown.edu:/path/to/rom/<experiment>/ ./rom_results/

# Review locally
python -m json.tool rom_results/<experiment>/mvar/forecast/aggregate_metrics.json
```

---

## Parameter Tuning Guide

### POD Parameters

**Energy threshold** (`--energy_threshold`):
- Higher (0.999): More modes, better reconstruction, slower
- Lower (0.95): Fewer modes, more compression, potential information loss
- **Recommendation:** Start with 0.995 (99.5%)

**Fixed latent dim** (`--latent_dim`):
- Override energy-based selection
- Useful for comparing models with same dimensionality
- **Recommendation:** Use energy threshold for initial exploration

### MVAR Parameters

**Model order** (`--mvar_order`):
- Higher: Captures longer-range temporal dependencies
- Lower: Faster, more stable, less expressive
- **Recommendation:** Try 4-8, validate on held-out segment

**Ridge regularization** (`--ridge`):
- Higher (1e-5): More regularization, more stable, potentially underfits
- Lower (1e-7): Less regularization, can overfit
- **Recommendation:** Start with 1e-6, increase if forecast explodes

**Training fraction** (`--train_frac`):
- Higher (0.9): More training data, less test horizon
- Lower (0.7): More test data, less training
- **Recommendation:** 0.8 provides good balance

### Train/Test Split

**Across runs:**
- Use majority for training (e.g., 16 out of 20 runs)
- Reserve 3-4 runs for held-out testing
- **Recommendation:** 80/20 split

**Within runs:**
- Use `--train_frac 0.8` within each run
- Validate MVAR hyperparameters on remaining 20%
- **Recommendation:** 80/20 split

---

## Troubleshooting

### Issue: POD uses too many/too few modes

**Symptom:** `r` very large or very small

**Solutions:**
```bash
# Too many modes (r > 100)
python scripts/rom_build_pod.py ... --energy_threshold 0.98  # Fewer modes

# Too few modes (r < 10)
python scripts/rom_build_pod.py ... --energy_threshold 0.999  # More modes

# Fixed dimension
python scripts/rom_build_pod.py ... --latent_dim 50  # Exactly 50 modes
```

### Issue: MVAR forecast explodes (NaN or huge values)

**Symptom:** `nan_count > 0` or density values >> 1

**Solutions:**
```bash
# Increase regularization
python scripts/rom_train_mvar.py ... --ridge 1e-5  # Stronger regularization

# Reduce model order
python scripts/rom_train_mvar.py ... --mvar_order 3  # Simpler model

# Use more POD modes
python scripts/rom_build_pod.py ... --energy_threshold 0.998  # Better reconstruction
```

### Issue: Low R² on test runs

**Symptom:** `mean_r2 < 0.5`

**Possible causes:**
1. **Insufficient POD modes**: Increase `--energy_threshold` or `--latent_dim`
2. **Model too simple**: Increase `--mvar_order`
3. **Model overfitting**: Increase `--ridge`, use more training runs
4. **Test runs very different from training**: Check IC diversity

**Diagnosis:**
```bash
# Check per-run R²
cat rom/<experiment>/mvar/forecast/metrics_run_*.json | grep '"r2"'

# Check POD energy captured
cat rom/<experiment>/pod/pod_info.json | grep 'energy_captured'

# Review error plots
open rom/<experiment>/mvar/forecast/errors_time_run_*.png
```

### Issue: Mass not conserved

**Symptom:** `mass_conservation_ok = false`

**Explanation:** POD + MVAR does not inherently conserve mass. This is expected for truncated POD.

**Solutions:**
- Use more POD modes
- Post-process predictions to enforce mass constraint
- Consider mass-preserving ROM methods (future work)

### Issue: Videos fail to generate

**Symptom:** Error during video creation

**Solutions:**
```bash
# Skip videos for faster runs
python scripts/rom_evaluate.py ... --no_videos

# Check matplotlib backend
python -c "import matplotlib; print(matplotlib.get_backend())"

# Ensure ffmpeg is available
which ffmpeg  # Should show path

# On Oscar, load multimedia module
module load ffmpeg
```

### Issue: Out of memory

**Symptom:** Killed by SLURM or kernel

**Solutions:**
- **Reduce data:** Use `--max_runs` in POD stage (for testing)
- **Downsample time:** Use `t_skip` in simulation config
- **Lower resolution:** Reduce `nx, ny` in density estimation
- **Request more memory:** Increase `#SBATCH --mem` in SLURM script

**Memory estimates:**
- POD snapshot matrix: `T_total * d * 8 bytes`
- Example: 20,000 steps × 16,384 grid = ~2.5 GB
- Recommendation: Request 2-3x data size

---

## Python API Usage

For custom workflows or batch processing:

```python
from pathlib import Path
import numpy as np

from rectsim.mvar import (
    load_density_movies,
    build_global_snapshot_matrix,
    compute_pod,
    project_to_pod,
    fit_mvar_from_runs,
    mvar_forecast,
    reconstruct_from_pod,
)
from rectsim.rom_eval import (
    ROMConfig,
    setup_rom_directories,
    split_runs_train_test,
    get_forecast_split_indices,
    compute_pointwise_errors,
    compute_r2_score,
    check_mass_conservation,
)

# Setup
config = ROMConfig(
    experiment_name="custom_exp",
    train_runs=[0, 1, 2, 3],
    test_runs=[4, 5],
    mvar_order=4,
    ridge=1e-6,
    train_frac=0.8,
)
paths = setup_rom_directories(config)

# Load and split data
all_runs = list(Path("simulations/runs").glob("run_*"))
train_dirs, test_dirs = split_runs_train_test(all_runs, config.train_runs, config.test_runs)

# Build POD
train_density = load_density_movies(train_dirs)
X, _, mean = build_global_snapshot_matrix(train_density)
pod_basis = compute_pod(X, r=None, energy_threshold=0.995)

# Project to latent
all_density = load_density_movies(all_runs)
latent_dict = project_to_pod(all_density, pod_basis["Phi"], mean)

# Train MVAR
model, info = fit_mvar_from_runs(latent_dict, order=config.mvar_order, ridge=config.ridge)

# Forecast on test run
test_latent = latent_dict["run_0004"]
Y_full = test_latent["Y"]
T_train, _ = get_forecast_split_indices(Y_full.shape[0], config.train_frac)

Y_init = Y_full[T_train - model.order : T_train]
Y_pred = mvar_forecast(model, Y_init, steps=Y_full.shape[0] - T_train)

# Reconstruct
rho_pred = reconstruct_from_pod(Y_pred, pod_basis["Phi"], mean, ny, nx)

# Evaluate
rho_true = all_density["run_0004"]["rho"][T_train:]
errors = compute_pointwise_errors(rho_true, rho_pred)
r2 = compute_r2_score(rho_true, rho_pred)

print(f"R² = {r2:.4f}, Mean RMSE = {errors['rmse'].mean():.6f}")
```

---

## Comparison with Other Methods

The standardized evaluation framework allows easy comparison:

| Method | Module | Evaluation | Notes |
|--------|--------|------------|-------|
| POD + MVAR | `rectsim.mvar` | `rom_evaluate.py` | Linear dynamics |
| POD + LSTM | *(future)* | Same evaluation | Nonlinear dynamics |
| Koopman | *(future)* | Same evaluation | Operator-based |
| DMD variants | *(future)* | Same evaluation | Dynamic modes |

All methods use:
- Same train/test splits
- Same POD basis (for fair comparison)
- Same metrics (R², RMSE, mass conservation, order parameters)

---

## References

- **POD methodology**: Lumley, J. L. (1967). "The structure of inhomogeneous turbulent flows"
- **MVAR forecasting**: Lütkepohl, H. (2005). "New Introduction to Multiple Time Series Analysis"
- **EF-ROM framework**: Alvarez et al. (2024). "Autoregressive ROM for collective motion forecasting"
- **Mass conservation**: Note that truncated POD does not preserve mass exactly. For mass-preserving methods, see Carlberg et al. (2013).

---

## See Also

- [ENSEMBLE_GUIDE.md](../ENSEMBLE_GUIDE.md): Stage 1 - Ensemble generation
- [MVAR_GUIDE.md](../MVAR_GUIDE.md): Original MVAR pipeline (deprecated, use ROM pipeline)
- [configs/](../configs/): Example configuration files
- [tests/test_rom_eval.py](../tests/test_rom_eval.py): Test suite

---

**Questions or issues?** Check the troubleshooting section or review example outputs in `rom/test_exp/`.
