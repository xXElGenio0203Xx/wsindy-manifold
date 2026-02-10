# Experimental Protocol and Metrics

**Document Status**: Technical Reference for Thesis Chapter  
**Context**: Compute infrastructure, dataset scale, train/test splits, and evaluation metrics  
**Primary Modules**: `src/rectsim/simulation_runner.py`, `src/rectsim/test_evaluator.py`, `src/rectsim/standard_metrics.py`  
**Config**: `configs/alvarez_style_production.yaml`  
**Author**: Maria  
**Date**: February 2, 2026  

---

## Table of Contents

1. [Compute Infrastructure and Dataset Scale](#1-compute-infrastructure-and-dataset-scale)
2. [Storage Format](#2-storage-format)
3. [Train/Test Splits Across IC Families](#3-traintest-splits-across-ic-families)
4. [Evaluation Metrics](#4-evaluation-metrics)
5. [Parameter Counts and Identifiability](#5-parameter-counts-and-identifiability)

---

## 1. Compute Infrastructure and Dataset Scale

### 1.1 Compute Resources

**Primary Infrastructure**: Brown University's Oscar/CCV High-Performance Computing Cluster

**Hardware Specifications**:
- **CPU Nodes**: Intel Xeon processors (2.4-3.0 GHz)
- **Memory**: 128-512 GB RAM per node
- **Storage**: High-performance Lustre filesystem
- **Network**: InfiniBand interconnect for parallel jobs

**Batch System**: SLURM Workload Manager
- Job submission via `sbatch` scripts
- Array jobs for parallel simulation runs
- Partition: `batch` (general compute nodes)

**Acknowledgement Language** (required for thesis):

> This research was conducted using computational resources and services at the Center for Computation and Visualization (CCV), Brown University. We gratefully acknowledge the CCV staff for their technical support and infrastructure maintenance.

**Citations**:
- Brown CCV Overview: `https://ccv.brown.edu`
- Oscar Documentation: `https://docs.ccv.brown.edu/oscar/`

### 1.2 Dataset Scale

**Training Dataset**:
- **Number of simulations**: $M = 408$ independent runs
- **Trajectory duration**: $T_{\text{train}} = 8.0$ seconds
- **Temporal resolution**: $\Delta t = 0.1$ seconds
- **Timesteps per run**: $K = 80$ snapshots
- **Total snapshots**: $M \times K = 408 \times 80 = 32,640$

**Spatial Discretization**:
- **Density grid**: $64 \times 64 = 4,096$ spatial points
- **Domain size**: $L_x \times L_y = 15.0 \times 15.0$ (square)
- **Grid spacing**: $\Delta x = \Delta y \approx 0.234$ units

**Data Volume** (per training run):
- Particle trajectories: $\sim 25$ KB (compressed npz)
- Density fields: $\sim 1.2$ MB (compressed npz)
- Total training data: $\sim 500$ MB (all 408 runs)

**Test Dataset**:
- **Number of test runs**: $n_{\text{test}} = 31$ independent ICs
- **Trajectory duration**: $T_{\text{test}} = 20.0$ seconds
- **Timesteps per test run**: $K_{\text{test}} = 200$ snapshots
- **Forecast period**: $[8.0, 20.0]$ seconds (120 timesteps)
- **Total test snapshots**: $31 \times 200 = 6,200$

**Computational Cost**:
- **Simulation time**: ~5 seconds per run (80 timesteps)
- **Total simulation time**: $408 \times 5 = 2,040$ seconds $\approx$ 34 minutes (serial)
- **Parallel execution**: ~3-5 minutes (using 100 cores via SLURM array)
- **POD computation**: ~10 seconds (SVD on 32,640 × 4,096 matrix)
- **MVAR training**: ~2 seconds (Ridge regression)
- **LSTM training**: ~5-10 minutes (CPU), ~2-3 minutes (GPU)

**Total Pipeline Runtime** (end-to-end):
- **Training + POD + MVAR**: ~10 minutes (parallel simulations + sequential ROM)
- **Test evaluation**: ~2 minutes (31 test runs)
- **Complete experiment**: ~15 minutes (simulation to metrics)

### 1.3 SLURM Job Configuration

**Typical SLURM Script** (`run_rectsim_single.slurm`):

```bash
#!/bin/bash
#SBATCH -J wsindy_mvar           # Job name
#SBATCH -N 1                     # Number of nodes
#SBATCH -n 1                     # Number of tasks (cores)
#SBATCH -t 01:00:00              # Time limit (1 hour)
#SBATCH --mem=16G                # Memory per node
#SBATCH -o slurm_logs/%j.out     # Standard output
#SBATCH -e slurm_logs/%j.err     # Standard error

# Load environment
module load miniconda3/23.11.0s
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate wsindy

# Run unified MVAR pipeline
python scripts/run_unified.py \
    --config configs/alvarez_style_production.yaml \
    --output oscar_output/production_run
```

**Array Job** (parallel simulations):

```bash
#SBATCH --array=0-407%50         # 408 jobs, max 50 concurrent
```

Each array task processes one training simulation independently.

---

## 2. Storage Format

### 2.1 File Hierarchy

**Standard Output Structure**:

```
oscar_output/
└── <experiment_name>/
    ├── config.yaml                      # Experiment configuration
    ├── summary.json                     # Pipeline-level metrics
    ├── train/                           # Training simulations
    │   ├── train_000/
    │   │   ├── trajectory.npz           # Particle positions & velocities
    │   │   ├── density.npz              # KDE density fields
    │   │   ├── order_params.csv         # Order parameters vs time
    │   │   └── metadata.json            # Run metadata
    │   ├── train_001/
    │   └── ...
    ├── test/                            # Test simulations
    │   ├── test_000/
    │   │   ├── trajectory.npz           # Ground truth trajectory
    │   │   ├── density_true.npz         # Ground truth density
    │   │   ├── density_pred.npz         # ROM prediction
    │   │   ├── metrics_summary.json     # R², RMSE, mass error
    │   │   ├── r2_vs_time.csv           # Time-resolved R²(t)
    │   │   ├── density_metrics.csv      # Density variance, mass
    │   │   └── order_params.csv         # Order parameters
    │   ├── test_001/
    │   └── ...
    ├── pod/                             # POD basis
    │   ├── pod_basis.npz                # U_r, singular values, mean
    │   ├── pod_summary.json             # Variance explained, rank
    │   └── latent_data.npz              # Training latent trajectories
    └── mvar/                            # MVAR model
        ├── mvar_model.npz               # Coefficient matrices A_τ
        └── mvar_summary.json            # Training R², hyperparameters
```

### 2.2 File Format Specifications

#### **trajectory.npz** (Particle-Level Data)

**Contents**:
```python
{
    'traj': np.ndarray,      # (T, N, 2) - Positions [x, y]
    'vel': np.ndarray,       # (T, N, 2) - Velocities [v_x, v_y]
    'times': np.ndarray      # (T,) - Timestamps
}
```

**Dimensions**:
- $T$: Timesteps (80 for training, 200 for test)
- $N$: Number of particles (40 in production)
- Positions: $\mathbf{x}_i(t) \in [0, L_x] \times [0, L_y]$
- Velocities: $\mathbf{v}_i(t) \in \mathbb{R}^2$

**Usage**:
- Visualization (trajectory animations)
- Order parameter computation (polarization, angular momentum)
- Density field generation (KDE input)

#### **density.npz** / **density_true.npz** (Density Fields)

**Contents**:
```python
{
    'rho': np.ndarray,       # (T, Ny, Nx) - Density fields
    'xgrid': np.ndarray,     # (Nx,) - X-axis grid points
    'ygrid': np.ndarray,     # (Ny,) - Y-axis grid points
    'times': np.ndarray      # (T,) - Timestamps
}
```

**Dimensions**:
- $T$: Timesteps (80 training, 200 test)
- $N_x = N_y = 64$: Spatial grid resolution
- Grid spacing: $\Delta x = L_x / N_x \approx 0.234$

**Density Field** $\rho(x, y, t)$:
$$
\rho(x, y, t) = \sum_{i=1}^N K_h(x - x_i(t), y - y_i(t))
$$

where $K_h$ is Gaussian kernel with bandwidth $h = 3.0$ grid cells.

**Normalization**: Mass-preserving KDE
$$
\int_{\Omega} \rho(x, y, t) \, dx \, dy = N \quad \text{(particle count)}
$$

#### **density_pred.npz** (ROM Predictions)

**Contents**:
```python
{
    'rho': np.ndarray,       # (T_forecast, Ny, Nx) - Predicted density
    'xgrid': np.ndarray,     # (Nx,) - X-axis grid
    'ygrid': np.ndarray,     # (Ny,) - Y-axis grid
    'times': np.ndarray      # (T_forecast,) - Forecast timestamps
}
```

**Dimensions**:
- $T_{\text{forecast}} = 120$: Forecast period timesteps
- Same spatial grid as ground truth (64 × 64)

**Generation**:
1. MVAR forecast in latent space: $\hat{\mathbf{y}}(t)$
2. POD reconstruction: $\hat{\boldsymbol{\rho}}(t) = \boldsymbol{\Phi}_r \hat{\mathbf{y}}(t) + \bar{\boldsymbol{\rho}}$
3. Reshape to grid: $(4096,) \to (64, 64)$

#### **pod_basis.npz** (POD Decomposition)

**Contents**:
```python
{
    'U_r': np.ndarray,       # (d_full, r) - Truncated POD basis
    'singular_values': np.ndarray,  # (r,) - Leading singular values
    'X_mean': np.ndarray,    # (d_full,) - Mean density field
    'energy_ratio': float,   # Variance captured by r modes
    'R_POD': int,            # Truncation rank r
    'M': int,                # Number of training runs
    'T_rom': int             # Timesteps per run
}
```

**Dimensions**:
- $d_{\text{full}} = 4,096$: Spatial DOF (flattened 64 × 64 grid)
- $r = 35$: Latent dimension (fixed)
- $M = 408$: Training runs
- $T_{\text{rom}} = 80$: Timesteps per run

**POD Basis Matrix** $\boldsymbol{\Phi}_r \in \mathbb{R}^{d_{\text{full}} \times r}$:
- Left singular vectors from SVD: $\mathbf{X} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$
- Orthonormal columns: $\boldsymbol{\Phi}_r^T \boldsymbol{\Phi}_r = \mathbf{I}_r$

#### **mvar_model.npz** (MVAR Coefficients)

**Contents**:
```python
{
    'A_matrices': np.ndarray,    # (p, r, r) - Coefficient tensors
    'A_companion': np.ndarray,   # (r, p*r) - Flattened sklearn format
    'p': int,                    # Lag order (5)
    'r': int,                    # Latent dimension (35)
    'alpha': float,              # Ridge regularization (1e-4)
    'train_r2': float,           # Training R² (~0.95)
    'train_rmse': float,         # Training RMSE (~0.02)
    'rho_before': float,         # Spectral radius before scaling
    'rho_after': float           # Spectral radius after scaling
}
```

**MVAR Model**:
$$
\mathbf{y}(t + \Delta t) = \sum_{\tau=1}^{w} \mathbf{A}_\tau \mathbf{y}(t - (\tau-1)\Delta t) + \mathbf{c}
$$

where:
- $\mathbf{A}_\tau \in \mathbb{R}^{35 \times 35}$: Coefficient matrix for lag $\tau$
- $w = 5$: Lag order
- $\mathbf{c} \in \mathbb{R}^{35}$: Intercept (stored in sklearn model)

#### **metrics_summary.json** (Evaluation Metrics)

**Contents**:
```json
{
    "r2_recon": 0.8523,          // R² (physical space)
    "r2_latent": 0.8845,         // R² (latent space)
    "r2_pod": 0.9012,            // R² (POD reconstruction)
    "rmse_recon": 0.0342,        // RMSE (physical space)
    "rmse_latent": 0.0215,       // RMSE (latent space)
    "rmse_pod": 0.0189,          // RMSE (POD reconstruction)
    "rel_error_recon": 0.0284,   // Relative error (reconstructed)
    "rel_error_pod": 0.0157,     // Relative error (POD)
    "max_mass_violation": 0.0023 // Maximum mass error (%)
}
```

#### **order_params.csv** (Order Parameters)

**Columns**:
```
t,phi,mean_speed,angular_momentum,density_variance,total_mass
0.0,0.245,0.987,0.012,0.134,40.0
0.1,0.312,0.991,0.018,0.128,40.0
...
```

**Order Parameters**:
- **Polarization** $\phi(t)$: Global alignment measure
  $$
  \phi(t) = \frac{1}{N} \left\| \sum_{i=1}^N \frac{\mathbf{v}_i(t)}{|\mathbf{v}_i(t)|} \right\|
  $$
  
- **Mean speed** $\bar{v}(t)$: Average particle speed
  $$
  \bar{v}(t) = \frac{1}{N} \sum_{i=1}^N |\mathbf{v}_i(t)|
  $$
  
- **Angular momentum** $L(t)$: Rotational order
  $$
  L(t) = \frac{1}{N} \sum_{i=1}^N (\mathbf{r}_i - \bar{\mathbf{r}}) \times \mathbf{v}_i
  $$

### 2.3 Metadata Standards

**Run-Level Metadata** (`metadata.json`):
```json
{
    "run_id": 0,
    "distribution": "gaussian",
    "ic_params": {
        "mean_x": 7.5,
        "mean_y": 7.5,
        "std": 1.0
    },
    "seed": 1000,
    "N": 40,
    "T": 8.0,
    "dt": 0.1,
    "Lx": 15.0,
    "Ly": 15.0,
    "eta": 0.3
}
```

**Experiment-Level Summary** (`summary.json`):
```json
{
    "experiment_name": "alvarez_production",
    "n_train": 408,
    "n_test": 31,
    "pod_modes": 35,
    "mvar_lag": 5,
    "ridge_alpha": 1e-4,
    "mean_r2_recon": 0.8501,
    "mean_r2_latent": 0.8823,
    "training_time_s": 602.3,
    "forecast_time_s": 1.2
}
```

---

## 3. Train/Test Splits Across IC Families

### 3.1 Initial Condition (IC) Families

**Four IC Families** (systematic coverage of configuration space):

1. **Gaussian Clusters** ($n_1 = 108$ training, $n_1' = 10$ test)
2. **Uniform Random** ($n_2 = 100$ training, $n_2' = 10$ test)
3. **Ring Configurations** ($n_3 = 100$ training, $n_3' = 10$ test)
4. **Two-Cluster Separation** ($n_4 = 100$ training, $n_4' = 1$ test)

**Total**: $M = 408$ training runs, $n_{\text{test}} = 31$ test runs

### 3.2 Training Set (408 Runs)

#### **Family 1: Gaussian Clusters** ($n_1 = 108$)

**Configuration Grid**:
- **Center positions**: $(x_c, y_c) \in \{3.75, 7.5, 11.25\}^2$ (9 locations)
- **Cluster widths**: $\sigma \in \{0.5, 1.0, 2.0, 3.0\}$ (4 scales)
- **Samples per config**: 3 independent realizations

**Total**: $9 \times 4 \times 3 = 108$ runs

**Probability Density**:
$$
p(\mathbf{x}) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{|\mathbf{x} - \boldsymbol{\mu}|^2}{2\sigma^2}\right)
$$

**Parameter Space**:
| Parameter | Training Values | Test Values |
|-----------|----------------|-------------|
| $x_c$ | 3.75, 7.5, 11.25 | 5.5, 9.0, 2.0 |
| $y_c$ | 3.75, 7.5, 11.25 | 5.5, 9.0, 2.0 |
| $\sigma$ | 0.5, 1.0, 2.0, 3.0 | 1.5 |

**Rationale**: Tests interpolation (novel cluster positions) and extrapolation (intermediate cluster widths).

#### **Family 2: Uniform Random** ($n_2 = 100$)

**Configuration**:
- Particles sampled uniformly: $\mathbf{x}_i \sim \mathcal{U}([0, L_x] \times [0, L_y])$
- Initial velocities: Random unit vectors

**Total**: 100 independent samples

**Expected Density**:
$$
\bar{\rho}(\mathbf{x}) = \frac{N}{L_x L_y} = \frac{40}{225} \approx 0.178 \text{ particles/unit}^2
$$

**Rationale**: Baseline IC (no spatial structure). Tests model on maximally disordered state.

#### **Family 3: Ring Configurations** ($n_3 = 100$)

**Configuration Grid**:
- **Ring radii**: $R \in \{2.0, 3.0, 4.0, 5.0\}$ (4 scales)
- **Ring widths**: $\sigma_r \in \{0.3, 0.6\}$ (thin/thick)
- **Samples per config**: $\approx 25$ realizations

**Total**: $4 \times 25 = 100$ runs

**Probability Density** (polar):
$$
p(r, \theta) = \frac{1}{2\pi R} \cdot \frac{1}{\sqrt{2\pi\sigma_r^2}} \exp\left(-\frac{(r - R)^2}{2\sigma_r^2}\right)
$$

**Parameter Space**:
| Parameter | Training Values | Test Values |
|-----------|----------------|-------------|
| $R$ | 2.0, 3.0, 4.0, 5.0 | 2.5, 3.5, 4.5 |
| $\sigma_r$ | 0.3, 0.6 | 0.45 |

**Rationale**: Tests rotational symmetry breaking and intermediate ring radii.

#### **Family 4: Two-Cluster Separation** ($n_4 = 100$)

**Configuration Grid**:
- **Cluster separation**: $d_{\text{sep}} \in \{3.0, 4.5, 6.0, 7.5\}$ (4 distances)
- **Cluster widths**: $\sigma \in \{0.8, 1.5\}$ (compact/diffuse)
- **Samples per config**: 25 realizations

**Total**: $4 \times 25 = 100$ runs

**Cluster Centers**:
- Cluster 1: $\boldsymbol{\mu}_1 = (L_x/2 - d_{\text{sep}}/2, L_y/2)$
- Cluster 2: $\boldsymbol{\mu}_2 = (L_x/2 + d_{\text{sep}}/2, L_y/2)$

**Parameter Space**:
| Parameter | Training Values | Test Values |
|-----------|----------------|-------------|
| $d_{\text{sep}}$ | 3.0, 4.5, 6.0, 7.5 | 3.75, 5.25, 6.75 |
| $\sigma$ | 0.8, 1.5 | 1.1 |

**Rationale**: Tests cluster merging dynamics and intermediate separations.

### 3.3 Test Set (31 Runs)

**Out-of-Training Parameters**: All test ICs use parameter values **not seen during training**.

**Test IC Distribution**:
- Gaussian: 10 runs (new positions/widths)
- Uniform: 10 runs (different seeds)
- Ring: 10 runs (intermediate radii/widths)
- Two-cluster: 1 run (intermediate separation/width)

**Test Trajectory Duration**: $T_{\text{test}} = 20.0$ seconds (2.5× training length)

**Evaluation Protocol**:
1. **Warmup period**: Use first 8.0s (training horizon) for conditioning
2. **Forecast period**: $[8.0, 20.0]$ seconds (12s extrapolation)
3. **Metrics**: Computed only on forecast period (120 timesteps)

### 3.4 Train/Test Split Philosophy

**Principles**:
1. **Parameter Interpolation**: Test on unseen combinations of known parameter ranges
2. **Temporal Extrapolation**: Forecast beyond training horizon (8s → 20s)
3. **Distributional Coverage**: Span diverse IC families (clustered, uniform, structured)
4. **Identifiability**: Ensure $\rho = N_{\text{train}} / p \approx 5$ (well-conditioned)

**Contrast with Alvarez et al. (2024)**:
- **Alvarez**: Single IC family (Gaussian), sweep noise levels
- **This work**: Four IC families, fixed noise level ($\eta = 0.3$)
- **Benefit**: Tests model generalization across IC topologies, not just noise sensitivity

---

## 4. Evaluation Metrics

### 4.1 R²(t) Over Time and Forecast Horizon

#### **Coefficient of Determination** $R^2$

**Definition** (over forecast period $t \in [T_{\text{train}}, T_{\text{test}}]$):

$$
R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}
$$

where:
$$
\begin{aligned}
\text{SS}_{\text{res}} &= \sum_{t=T_{\text{train}}}^{T_{\text{test}}} \|\boldsymbol{\rho}_{\text{true}}(t) - \hat{\boldsymbol{\rho}}(t)\|^2 \quad &&\text{(residual sum of squares)} \\
\text{SS}_{\text{tot}} &= \sum_{t=T_{\text{train}}}^{T_{\text{test}}} \|\boldsymbol{\rho}_{\text{true}}(t) - \bar{\boldsymbol{\rho}}_{\text{true}}\|^2 \quad &&\text{(total sum of squares)}
\end{aligned}
$$

**Interpretation**:
- $R^2 = 1$: Perfect prediction (SS$_{\text{res}} = 0$)
- $R^2 = 0$: Prediction no better than mean predictor
- $R^2 < 0$: Prediction worse than mean (model failure)

**Typical Values** (production results):
- Training R²: $\approx 0.95$ (MVAR one-step-ahead)
- Test R² (forecast): $\approx 0.85$ (12s closed-loop)

#### **Time-Resolved R²(t)** (Optional)

**Cumulative R² up to time $t$**:

$$
R^2(t) = 1 - \frac{\sum_{\tau=T_{\text{train}}}^{t} \|\boldsymbol{\rho}_{\text{true}}(\tau) - \hat{\boldsymbol{\rho}}(\tau)\|^2}{\sum_{\tau=T_{\text{train}}}^{t} \|\boldsymbol{\rho}_{\text{true}}(\tau) - \bar{\boldsymbol{\rho}}_{\text{true}}\|^2}
$$

**Usage**:
- Track forecast degradation over time
- Identify useful forecast horizon (where $R^2(t)$ drops below threshold)
- Saved to `r2_vs_time.csv` if `save_time_resolved: true`

**Forecast Horizon Definition**:

**95% Accuracy Horizon** $T_{0.95}$:
$$
T_{0.95} = \max\{t : R^2(t) \geq 0.95\}
$$

**Typical Horizons** (production):
- $T_{0.95} \approx 2.0$ seconds (excellent accuracy)
- $T_{0.85} \approx 8.0$ seconds (good accuracy)
- $T_{0.70} \approx 15.0$ seconds (acceptable accuracy)

### 4.2 Three R² Variants

#### **(1) R² Reconstructed** (Physical Space Accuracy)

**Formula**:
$$
R^2_{\text{recon}} = 1 - \frac{\sum_{t} \|\boldsymbol{\rho}_{\text{true}}(t) - \hat{\boldsymbol{\rho}}_{\text{pred}}(t)\|^2}{\sum_{t} \|\boldsymbol{\rho}_{\text{true}}(t) - \bar{\boldsymbol{\rho}}_{\text{true}}\|^2}
$$

**Interpretation**: Overall density prediction quality (includes POD truncation error + MVAR/LSTM error).

**Typical Value**: $R^2_{\text{recon}} \approx 0.85$

#### **(2) R² Latent** (ROM Space Accuracy)

**Formula**:
$$
R^2_{\text{latent}} = 1 - \frac{\sum_{t} \|\mathbf{y}_{\text{true}}(t) - \hat{\mathbf{y}}_{\text{pred}}(t)\|^2}{\sum_{t} \|\mathbf{y}_{\text{true}}(t) - \bar{\mathbf{y}}_{\text{true}}\|^2}
$$

where $\mathbf{y}_{\text{true}}(t) = \boldsymbol{\Phi}_r^T (\boldsymbol{\rho}_{\text{true}}(t) - \bar{\boldsymbol{\rho}})$ (true latent state).

**Interpretation**: Pure MVAR/LSTM forecasting accuracy in reduced coordinates (no POD truncation error).

**Typical Value**: $R^2_{\text{latent}} \approx 0.88$ (higher than $R^2_{\text{recon}}$)

#### **(3) R² POD** (POD Reconstruction Quality)

**Formula**:
$$
R^2_{\text{POD}} = 1 - \frac{\sum_{t} \|\boldsymbol{\rho}_{\text{true}}(t) - \boldsymbol{\rho}_{\text{POD}}(t)\|^2}{\sum_{t} \|\boldsymbol{\rho}_{\text{true}}(t) - \bar{\boldsymbol{\rho}}_{\text{true}}\|^2}
$$

where $\boldsymbol{\rho}_{\text{POD}}(t) = \boldsymbol{\Phi}_r \mathbf{y}_{\text{true}}(t) + \bar{\boldsymbol{\rho}}$ (reconstruct true latent, no forecast error).

**Interpretation**: Upper bound on $R^2_{\text{recon}}$ (best possible with 35 modes). Measures information loss from POD truncation.

**Typical Value**: $R^2_{\text{POD}} \approx 0.90$

**Relationship**:
$$
R^2_{\text{recon}} \leq R^2_{\text{latent}} \leq R^2_{\text{POD}}
$$

(ROM cannot outperform POD basis; latent space has no truncation error.)

### 4.3 Snapshot-Wise Errors

#### **Root Mean Squared Error (RMSE)**

**Physical Space RMSE**:
$$
\text{RMSE}_{\text{recon}} = \sqrt{\frac{1}{T_{\text{forecast}} \cdot N_x N_y} \sum_{t=T_{\text{train}}}^{T_{\text{test}}} \sum_{i,j} \left(\rho_{\text{true}}^{ij}(t) - \hat{\rho}_{\text{pred}}^{ij}(t)\right)^2}
$$

**Latent Space RMSE**:
$$
\text{RMSE}_{\text{latent}} = \sqrt{\frac{1}{T_{\text{forecast}} \cdot r} \sum_{t=T_{\text{train}}}^{T_{\text{test}}} \|\mathbf{y}_{\text{true}}(t) - \hat{\mathbf{y}}_{\text{pred}}(t)\|^2}
$$

**Interpretation**: Average pointwise error across space-time.

**Typical Values**:
- $\text{RMSE}_{\text{recon}} \approx 0.03$ (physical space)
- $\text{RMSE}_{\text{latent}} \approx 0.02$ (latent space)

#### **Relative Error**

**Formula**:
$$
\text{RelErr} = \frac{\text{RMSE}_{\text{recon}}}{\bar{\rho}_{\text{ref}}}
$$

where $\bar{\rho}_{\text{ref}} = \frac{1}{T_{\text{forecast}} \cdot N_x N_y} \sum_{t,i,j} |\rho_{\text{true}}^{ij}(t)|$ (mean absolute density).

**Interpretation**: Error as percentage of typical density magnitude.

**Typical Value**: $\text{RelErr} \approx 2.8\%$ (relative to mean density)

#### **Pointwise Absolute Error**

**Formula** (per timestep $t$):
$$
e_{\text{abs}}(t) = \frac{1}{N_x N_y} \sum_{i,j} |\rho_{\text{true}}^{ij}(t) - \hat{\rho}_{\text{pred}}^{ij}(t)|
$$

**Mean Absolute Error (MAE)**:
$$
\text{MAE} = \frac{1}{T_{\text{forecast}}} \sum_{t=T_{\text{train}}}^{T_{\text{test}}} e_{\text{abs}}(t)
$$

**Typical Value**: $\text{MAE} \approx 0.025$ (similar to RMSE for well-behaved errors)

#### **Infinity Norm (Maximum Error)**

**Formula**:
$$
e_{\infty} = \max_{t, i, j} |\rho_{\text{true}}^{ij}(t) - \hat{\rho}_{\text{pred}}^{ij}(t)|
$$

**Interpretation**: Worst-case pointwise error (useful for identifying outliers).

**Typical Value**: $e_{\infty} \approx 0.15$ (localized errors in high-density regions)

### 4.4 Mass Conservation Violation

**Total Mass** (per timestep $t$):
$$
m(t) = \sum_{i=1}^{N_x} \sum_{j=1}^{N_y} \rho_{ij}(t) \cdot \Delta x \cdot \Delta y
$$

**Mass Conservation Violation**:
$$
\Delta m(t) = \frac{|m_{\text{pred}}(t) - m_{\text{true}}(t)|}{m_{\text{true}}(t)} \times 100\%
$$

**Maximum Mass Violation** (over forecast period):
$$
\Delta m_{\max} = \max_{t \in [T_{\text{train}}, T_{\text{test}}]} \Delta m(t)
$$

**Expected Behavior**:
- **Ideal**: $\Delta m_{\max} = 0\%$ (exact mass conservation)
- **Acceptable**: $\Delta m_{\max} < 1\%$ (good physical consistency)
- **Concerning**: $\Delta m_{\max} > 5\%$ (potential POD/MVAR error accumulation)

**Typical Value** (production): $\Delta m_{\max} \approx 0.23\%$ (excellent conservation)

**Why It Matters**:
- POD truncation can break mass conservation (modes not constrained)
- MVAR closed-loop forecast can amplify small violations
- Mass drift indicates model instability

### 4.5 Summary Metrics Table

| Metric | Formula | Typical Value | Interpretation |
|--------|---------|---------------|----------------|
| **R² Reconstructed** | $1 - \text{SS}_{\text{res}} / \text{SS}_{\text{tot}}$ | 0.85 | Overall physical space accuracy |
| **R² Latent** | $1 - \text{SS}_{\text{res,lat}} / \text{SS}_{\text{tot,lat}}$ | 0.88 | Pure ROM forecast accuracy |
| **R² POD** | $1 - \text{SS}_{\text{res,POD}} / \text{SS}_{\text{tot}}$ | 0.90 | POD truncation error ceiling |
| **RMSE (recon)** | $\sqrt{\text{mean}((\rho_{\text{true}} - \hat{\rho})^2)}$ | 0.034 | Average pointwise error |
| **RMSE (latent)** | $\sqrt{\text{mean}((y_{\text{true}} - \hat{y})^2)}$ | 0.022 | Latent space forecast error |
| **RelErr** | $\text{RMSE} / \bar{\rho}_{\text{ref}}$ | 2.8% | Error relative to mean density |
| **MAE** | $\text{mean}(\|\rho_{\text{true}} - \hat{\rho}\|)$ | 0.025 | Mean absolute error |
| **$e_\infty$** | $\max(\|\rho_{\text{true}} - \hat{\rho}\|)$ | 0.150 | Worst-case pointwise error |
| **$\Delta m_{\max}$** | $\max(\|m_{\text{pred}} - m_{\text{true}}\| / m_{\text{true}})$ | 0.23% | Maximum mass violation |

---

## 5. Parameter Counts and Identifiability

### 5.1 Alvarez Design Principle

**Core Insight** (Alvarez et al., 2024):

> For ROM forecasting of collective behavior, **model parsimony** (low parameter count $p$) is more important than **variance capture** (high POD truncation rank $r$).

**Rationale**:
- Too many POD modes → underdetermined MVAR (overfitting)
- Too few POD modes → poor reconstruction (high truncation error)
- **Sweet spot**: $\rho = N_{\text{train}} / p \approx 5-10$ (well-conditioned)

### 5.2 MVAR Parameter Count

**Model**: MVAR($w$) with latent dimension $r$

**Total Parameters**:
$$
p_{\text{MVAR}} = r^2 w + r
$$

where:
- $r^2 w$: Coefficient matrices $\{\mathbf{A}_1, \ldots, \mathbf{A}_w\}$ (each $r \times r$)
- $r$: Intercept vector $\mathbf{c}$

**Production Configuration** ($w = 5$, $r = 35$):
$$
p_{\text{MVAR}} = 35^2 \times 5 + 35 = 6,125 + 35 = 6,160
$$

**Breakdown**:
- $\mathbf{A}_1, \ldots, \mathbf{A}_5$: $5 \times 1,225 = 6,125$ coefficients
- $\mathbf{c}$: 35 coefficients

### 5.3 LSTM Parameter Count

**Model**: 2-layer LSTM with hidden dimension $N_h = 64$

**Parameter Formula** (per layer):
$$
p_{\text{LSTM,layer}} = 4 \times (N_h \times (r + N_h) + N_h)
$$

where:
- $4$: Number of gates (forget, input, candidate, output)
- $N_h \times (r + N_h)$: Weight matrices $\mathbf{W}_{\{f,i,c,o\}} \in \mathbb{R}^{N_h \times (r + N_h)}$
- $N_h$: Bias vectors $\mathbf{b}_{\{f,i,c,o\}} \in \mathbb{R}^{N_h}$

**Total LSTM Parameters** ($L = 2$ layers, $N_h = 64$, $r = 35$):
$$
\begin{aligned}
p_{\text{LSTM}} &= L \times p_{\text{LSTM,layer}} + (N_h \times r + r) \\
&= 2 \times 4 \times (64 \times 99 + 64) + (64 \times 35 + 35) \\
&= 2 \times 4 \times (6,336 + 64) + (2,240 + 35) \\
&= 2 \times 25,600 + 2,275 \\
&= 51,200 + 2,275 \\
&\approx 53,475
\end{aligned}
$$

### 5.4 Identifiability Analysis

#### **Parameter-to-Data Ratio**

**Definition**:
$$
\rho = \frac{N_{\text{train}}}{p}
$$

where:
- $N_{\text{train}}$: Number of training windows
- $p$: Model parameter count

**Rule of Thumb** (statistical learning):
- $\rho < 1$: **Severely underdetermined** (overfitting inevitable)
- $1 \leq \rho < 3$: **Marginal** (requires strong regularization)
- $3 \leq \rho < 10$: **Well-conditioned** (regularization recommended) ✅
- $\rho \geq 10$: **Well-determined** (regularization optional)

#### **Production Values**

**Training Windows**:
$$
N_{\text{train}} = M \times (T_{\text{rom}} - w) = 408 \times (80 - 5) = 30,600
$$

**MVAR Identifiability**:
$$
\rho_{\text{MVAR}} = \frac{30,600}{6,160} \approx 4.97 \quad \text{(well-conditioned)}
$$

**LSTM Identifiability**:
$$
\rho_{\text{LSTM}} = \frac{30,600}{53,475} \approx 0.57 \quad \text{(underdetermined!)}
$$

#### **Alvarez Comparison**

**Alvarez et al. (2024)** (Table 3):
- **Dataset**: 1,008 runs × 75 windows = 75,600 samples
- **MVAR**: $r = 35$, $w = 5$ → $p = 6,160$ → $\rho \approx 12.3$
- **LSTM**: $N_h = 50$, $L = 2$ → $p \approx 28,000$ → $\rho \approx 2.7$

**This Work** (production):
- **Dataset**: 408 runs × 75 windows = 30,600 samples
- **MVAR**: $r = 35$, $w = 5$ → $p = 6,160$ → $\rho \approx 5.0$ ✅
- **LSTM**: $N_h = 64$, $L = 2$ → $p \approx 53,000$ → $\rho \approx 0.58$ ⚠️

**Observation**:
- **MVAR**: Similar identifiability to Alvarez ($\rho \approx 5$ vs $\rho \approx 12$)
- **LSTM**: Severely underdetermined ($\rho < 1$) → requires implicit regularization (early stopping, dropout)

**Design Choice**: Use **MVAR as default** due to favorable $\rho \approx 5$. LSTM enabled only for high-capacity experiments.

### 5.5 Effect of POD Truncation on Identifiability

**Counterfactual**: If we used energy threshold (99% variance) instead of fixed $r = 35$:

**Energy Threshold Bug** ($\tau = 0.99$):
- Typical rank: $r \approx 287$ modes
- MVAR parameters: $p = 287^2 \times 5 + 287 = 412,532$
- Identifiability: $\rho = 30,600 / 412,532 \approx 0.074$ (severely underdetermined!)
- Result: Training $R^2 \approx 0.99$, Test $R^2 < 0$ (catastrophic overfitting)

**Fixed Modes** ($r = 35$):
- MVAR parameters: $p = 6,160$
- Identifiability: $\rho \approx 5.0$ (well-conditioned)
- Result: Training $R^2 \approx 0.95$, Test $R^2 \approx 0.85$ (excellent generalization) ✅

**Alvarez Principle Validated**:
> Fixed low-dimensional latent space ($r = 35$, capturing 85% variance) outperforms high-dimensional representation ($r = 287$, capturing 99% variance) due to superior identifiability.

### 5.6 Parameter Count Summary Table

| Model | Configuration | Parameters | Training Samples | Ratio $\rho$ | Status |
|-------|--------------|------------|------------------|--------------|--------|
| **MVAR** | $w=5$, $r=35$ | 6,160 | 30,600 | 4.97 | ✅ Well-conditioned |
| **LSTM** | $N_h=64$, $L=2$, $r=35$ | 53,475 | 30,600 | 0.57 | ⚠️ Underdetermined |
| **MVAR (Alvarez)** | $w=5$, $r=35$ | 6,160 | 75,600 | 12.28 | ✅ Well-determined |
| **LSTM (Alvarez)** | $N_h=50$, $L=2$, $r=35$ | ~28,000 | 75,600 | 2.70 | ✅ Marginal |
| **MVAR (bug)** | $w=5$, $r=287$ | 412,532 | 30,600 | 0.074 | ❌ Catastrophic |

**Key Takeaway**: Achieving $\rho \approx 5-10$ requires careful balance of:
1. POD truncation rank ($r$)
2. Lag order ($w$)
3. Training dataset size ($M \times (T - w)$)

---

## Summary

This document provides complete experimental protocol reference:

1. **Compute Infrastructure**: Brown University Oscar/CCV cluster (SLURM, ~34 minutes total runtime)
2. **Dataset Scale**: 408 training runs (32,640 snapshots), 31 test runs (6,200 snapshots)
3. **Storage Format**: Standardized npz files (trajectories, densities, POD, MVAR models, metrics)
4. **Train/Test Splits**: Four IC families (Gaussian, uniform, ring, two-cluster) with out-of-distribution test parameters
5. **Evaluation Metrics**:
   - Three R² variants (reconstructed, latent, POD): $\approx 0.85$, $0.88$, $0.90$
   - RMSE: $\approx 0.03$ (physical), $0.02$ (latent)
   - Mass conservation: $\Delta m_{\max} < 0.25\%$
   - Time-resolved R²(t) for forecast horizon analysis
6. **Parameter Counts**:
   - MVAR: 6,160 parameters → $\rho \approx 5.0$ (well-conditioned) ✅
   - LSTM: 53,475 parameters → $\rho \approx 0.57$ (underdetermined) ⚠️
   - Alvarez principle: Fixed $r = 35$ outperforms energy threshold ($r = 287$)

**Key Finding**: Achieving favorable identifiability ($\rho \approx 5$) requires fixed low-dimensional POD truncation, not variance-based thresholding.

---

**Document Version**: 1.0  
**Last Updated**: February 2, 2026  
**Author**: Maria  
**Status**: Complete ✓
