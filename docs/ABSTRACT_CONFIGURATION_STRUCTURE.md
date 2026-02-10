# Abstract Configuration Structure

**Document Status**: Technical Reference  
**Purpose**: Complete configuration parameter space with mathematical symbols  
**Context**: Theoretical parameter specification for thesis documentation  
**Author**: Maria  
**Date**: February 2, 2026  

---

## Overview

This document provides the **abstract configuration structure** for the unified MVAR pipeline, with all parameters represented by their mathematical symbols rather than specific numerical values. This structure corresponds to the YAML configuration files used in production (e.g., `configs/alvarez_style_production.yaml`).

---

## Complete Configuration Schema

```yaml
# ==============================================================================
# EXPERIMENT METADATA
# ==============================================================================
experiment:
  name: <experiment_identifier>
  description: <experiment_description>
  output_dir: <path_to_output_directory>
  seed: s_global                          # Global random seed for reproducibility

# ==============================================================================
# VICSEK MODEL PARAMETERS (Discrete-Time Dynamics)
# ==============================================================================
vicsek:
  # Particle system
  N: N                                    # Number of particles
  v0: v_0                                 # Constant speed (magnitude)
  
  # Domain geometry
  Lx: L_x                                 # Domain width (x-direction)
  Ly: L_y                                 # Domain height (y-direction)
  boundary: <boundary_type>               # "periodic" or "reflecting"
  
  # Interaction parameters
  R: R                                    # Interaction radius
  eta: η                                  # Angular noise level (radians)
  
  # Temporal discretization
  dt: Δt                                  # Timestep size (seconds)
  T_train: T_train                        # Training trajectory duration (seconds)
  T_test: T_test                          # Test trajectory duration (seconds)
  
  # Morse potential (optional attraction/repulsion)
  morse:
    enabled: <boolean>                    # Enable/disable Morse forces
    C_a: C_a                              # Attraction strength
    l_a: ℓ_a                              # Attraction length scale
    C_r: C_r                              # Repulsion strength
    l_r: ℓ_r                              # Repulsion length scale

# ==============================================================================
# INITIAL CONDITION FAMILIES
# ==============================================================================
initial_conditions:
  train:
    # ------------------------------------------------------------------------
    # Family 1: Gaussian Clusters
    # ------------------------------------------------------------------------
    - family: gaussian
      n_samples: n_gaussian               # Number of training samples
      parameters:
        centers:                          # Grid of cluster positions
          x_c: [x_c^(1), x_c^(2), ..., x_c^(n_x)]
          y_c: [y_c^(1), y_c^(2), ..., y_c^(n_y)]
        std: [σ_1, σ_2, ..., σ_n_σ]      # Cluster width scales
        samples_per_config: n_rep         # Replicates per parameter combo
    
    # ------------------------------------------------------------------------
    # Family 2: Uniform Random
    # ------------------------------------------------------------------------
    - family: uniform
      n_samples: n_uniform                # Number of uniform IC samples
      parameters:
        seeds: [s_1, s_2, ..., s_n_uniform]  # Random seeds for each sample
    
    # ------------------------------------------------------------------------
    # Family 3: Ring Configurations
    # ------------------------------------------------------------------------
    - family: ring
      n_samples: n_ring                   # Number of ring IC samples
      parameters:
        radii: [R_1, R_2, ..., R_n_R]    # Ring radii
        widths: [σ_r^(1), σ_r^(2), ..., σ_r^(n_σ)]  # Ring thickness scales
        samples_per_config: n_rep         # Replicates per parameter combo
    
    # ------------------------------------------------------------------------
    # Family 4: Two-Cluster Separation
    # ------------------------------------------------------------------------
    - family: two_cluster
      n_samples: n_two_cluster            # Number of two-cluster samples
      parameters:
        separations: [d_sep^(1), d_sep^(2), ..., d_sep^(n_d)]  # Cluster distances
        std: [σ_c^(1), σ_c^(2), ..., σ_c^(n_σ)]  # Cluster widths
        samples_per_config: n_rep         # Replicates per parameter combo
  
  # ----------------------------------------------------------------------------
  # Test Set (Out-of-Training Parameters)
  # ----------------------------------------------------------------------------
  test:
    # Out-of-distribution Gaussian clusters
    - family: gaussian
      n_samples: n_gaussian_test
      parameters:
        centers:
          x_c: [x_c^test_1, x_c^test_2, ..., x_c^test_n]
          y_c: [y_c^test_1, y_c^test_2, ..., y_c^test_n]
        std: [σ_test_1, σ_test_2, ..., σ_test_n]
    
    # Out-of-distribution uniform
    - family: uniform
      n_samples: n_uniform_test
      parameters:
        seeds: [s_test_1, s_test_2, ..., s_test_n]
    
    # Out-of-distribution rings
    - family: ring
      n_samples: n_ring_test
      parameters:
        radii: [R_test_1, R_test_2, ..., R_test_n]
        widths: [σ_r^test_1, σ_r^test_2, ..., σ_r^test_n]
    
    # Out-of-distribution two-cluster
    - family: two_cluster
      n_samples: n_two_cluster_test
      parameters:
        separations: [d_sep^test_1, d_sep^test_2, ..., d_sep^test_n]
        std: [σ_c^test_1, σ_c^test_2, ..., σ_c^test_n]

# ==============================================================================
# DENSITY FIELD COMPUTATION (Kernel Density Estimation)
# ==============================================================================
density:
  # Spatial discretization
  nx: n_x                                 # Grid points in x-direction
  ny: n_y                                 # Grid points in y-direction
  
  # KDE parameters
  kde:
    bandwidth: h_KDE                      # Gaussian kernel bandwidth (grid cells)
    normalize: <boolean>                  # Mass-preserving normalization
    
  # Output format
  output:
    save_density: <boolean>               # Save density fields to npz
    save_trajectory: <boolean>            # Save particle trajectories to npz

# ==============================================================================
# PROPER ORTHOGONAL DECOMPOSITION (POD)
# ==============================================================================
pod:
  # Truncation strategy
  method: <fixed_modes|energy_threshold>  # POD truncation method
  
  # Fixed modes (recommended, Alvarez principle)
  r: r                                    # Latent space dimension (fixed)
  
  # Energy threshold (alternative, not recommended)
  energy_threshold: τ_energy              # Variance capture threshold (e.g., 0.99)
  
  # Centering
  center: <boolean>                       # Subtract temporal mean (always True)
  
  # Output
  save_basis: <boolean>                   # Save POD basis to npz
  save_latent: <boolean>                  # Save latent trajectories to npz

# ==============================================================================
# MVAR MODEL CONFIGURATION
# ==============================================================================
mvar:
  # Model architecture
  lag: w                                  # Lag order (window size)
  
  # Ridge regularization
  regularization:
    alpha: λ_ridge                        # Ridge penalty coefficient
    method: <ridge|lasso|elasticnet>      # Regularization type
  
  # Training data preparation
  training:
    t_start: t_start                      # Start time for training window (seconds)
    t_end: t_end                          # End time for training window (seconds)
    use_all_runs: <boolean>               # Pool all training runs
  
  # Stability enforcement
  stability:
    enforce: <boolean>                    # Apply eigenvalue scaling
    max_spectral_radius: ρ_max            # Maximum spectral radius (typically 0.95-0.99)
    method: <scale|clip>                  # Eigenvalue correction method
  
  # Output
  save_model: <boolean>                   # Save MVAR coefficients to npz
  save_metrics: <boolean>                 # Save training metrics to JSON

# ==============================================================================
# LSTM MODEL CONFIGURATION (Optional)
# ==============================================================================
lstm:
  enabled: <boolean>                      # Enable LSTM training
  
  # Architecture
  architecture:
    n_layers: L                           # Number of LSTM layers
    hidden_size: N_h                      # Hidden state dimension
    dropout: p_dropout                    # Dropout probability (0-1)
  
  # Training hyperparameters
  training:
    batch_size: B                         # Mini-batch size
    learning_rate: α_lr                   # Adam learning rate
    n_epochs: E                           # Maximum epochs
    validation_split: ν_val               # Validation set fraction (0-1)
    early_stopping:
      enabled: <boolean>                  # Enable early stopping
      patience: P_early                   # Epochs without improvement
      min_delta: δ_min                    # Minimum improvement threshold
  
  # Loss function
  loss: <mse|mae|huber>                   # Loss function type
  
  # Output
  save_model: <boolean>                   # Save LSTM weights to file
  save_metrics: <boolean>                 # Save training curves to JSON

# ==============================================================================
# FORECASTING CONFIGURATION
# ==============================================================================
forecasting:
  # Warmup period (conditioning on ground truth)
  warmup:
    enabled: <boolean>                    # Use warmup period
    duration: T_warmup                    # Warmup duration (seconds)
    # Typically T_warmup = T_train for test trajectories
  
  # Closed-loop autoregressive rollout
  rollout:
    method: <closed_loop|teacher_forcing> # Forecasting mode
    horizon: T_horizon                    # Forecast duration (seconds)
    # Typically T_horizon = T_test - T_train
  
  # Initial condition strategies
  initial_state:
    source: <last_warmup|average_window>  # How to initialize latent state
    window_size: w_init                   # Averaging window (if applicable)

# ==============================================================================
# EVALUATION METRICS
# ==============================================================================
evaluation:
  # Metric computation
  metrics:
    # Coefficient of determination
    r2:
      compute: <boolean>
      variants: [reconstructed, latent, pod]  # All three R² types
    
    # Root mean squared error
    rmse:
      compute: <boolean>
      variants: [reconstructed, latent, pod]
    
    # Relative error
    relative_error:
      compute: <boolean>
      reference: <mean|max|l2_norm>       # Normalization reference
    
    # Mass conservation
    mass_conservation:
      compute: <boolean>
      tolerance: ε_mass                   # Acceptable violation threshold
    
    # Time-resolved metrics
    time_resolved:
      compute: <boolean>                  # Compute R²(t) over time
      save_csv: <boolean>                 # Save to r2_vs_time.csv
  
  # Forecast horizons (thresholds for useful predictions)
  horizons:
    r2_thresholds: [τ_1, τ_2, ..., τ_k]  # R² thresholds (e.g., [0.95, 0.85, 0.70])
    compute: <boolean>                    # Compute T_{τ_i} for each threshold
  
  # Order parameters (from particle trajectories)
  order_parameters:
    compute: <boolean>
    types: [polarization, mean_speed, angular_momentum, density_variance]
    save_csv: <boolean>

# ==============================================================================
# VISUALIZATION
# ==============================================================================
visualization:
  # Density field animations
  animations:
    create: <boolean>                     # Generate MP4 animations
    fps: f_fps                            # Frames per second
    dpi: d_dpi                            # Resolution (dots per inch)
    colormap: <colormap_name>             # Matplotlib colormap
  
  # Static plots
  plots:
    # POD spectrum
    pod_spectrum: <boolean>               # Singular values vs mode index
    
    # Training curves
    training_curves: <boolean>            # MVAR/LSTM training loss
    
    # Forecast comparison
    forecast_heatmaps: <boolean>          # True vs predicted density
    
    # Time series
    latent_timeseries: <boolean>          # Latent coordinates over time
    r2_vs_time: <boolean>                 # R²(t) curves
  
  # Output format
  format: <png|pdf|svg>                   # Figure format
  save_dir: <path_to_figures>             # Figure output directory

# ==============================================================================
# COMPUTATIONAL RESOURCES
# ==============================================================================
compute:
  # Parallelization
  parallel:
    enabled: <boolean>                    # Use parallel simulation execution
    n_workers: W                          # Number of parallel workers
    backend: <multiprocessing|joblib|ray> # Parallelization backend
  
  # SLURM configuration (HPC cluster)
  slurm:
    enabled: <boolean>                    # Submit as SLURM array job
    partition: <partition_name>           # SLURM partition (e.g., "batch")
    time_limit: T_slurm                   # Wall time limit (HH:MM:SS)
    memory: M_mem                         # Memory per node (GB)
    n_nodes: N_nodes                      # Number of compute nodes
    n_tasks: N_tasks                      # Tasks per node
  
  # GPU acceleration (LSTM only)
  gpu:
    enabled: <boolean>                    # Use GPU for LSTM training
    device_id: d_gpu                      # CUDA device ID

# ==============================================================================
# LOGGING AND OUTPUT
# ==============================================================================
logging:
  # Verbosity
  level: <debug|info|warning|error>       # Logging level
  
  # Log files
  save_logs: <boolean>                    # Save logs to file
  log_file: <path_to_logfile>             # Log file path
  
  # Progress tracking
  progress_bar: <boolean>                 # Show tqdm progress bars
  
  # Checkpointing
  checkpoint:
    enabled: <boolean>                    # Save intermediate results
    frequency: f_checkpoint               # Checkpoint frequency (iterations)

# ==============================================================================
# REPRODUCIBILITY
# ==============================================================================
reproducibility:
  # Random seeds (for each component)
  seeds:
    simulation: s_sim                     # Vicsek dynamics
    ic_generation: s_ic                   # Initial conditions
    pod: s_pod                            # POD (if stochastic)
    mvar: s_mvar                          # MVAR training
    lstm: s_lstm                          # LSTM training
  
  # Deterministic algorithms
  deterministic: <boolean>                # Force deterministic operations
  
  # Version tracking
  track_versions: <boolean>               # Save package versions to JSON
```

---

## Parameter Glossary

### Vicsek Model Parameters

| Symbol | Name | Description | Typical Range | Units |
|--------|------|-------------|---------------|-------|
| $N$ | Particle count | Number of self-propelled particles | 40-400 | - |
| $v_0$ | Constant speed | Magnitude of velocity vector | 0.5-2.0 | length/time |
| $L_x$ | Domain width | Spatial extent (x-direction) | 10.0-20.0 | length |
| $L_y$ | Domain height | Spatial extent (y-direction) | 10.0-20.0 | length |
| $R$ | Interaction radius | Alignment neighborhood size | 1.0-3.0 | length |
| $\eta$ | Angular noise | Noise amplitude in alignment rule | 0.0-1.0 | radians |
| $\Delta t$ | Timestep | Temporal discretization | 0.05-0.2 | time |
| $T_{\text{train}}$ | Training duration | Trajectory length for ROM training | 5.0-10.0 | time |
| $T_{\text{test}}$ | Test duration | Trajectory length for evaluation | 15.0-30.0 | time |

### Morse Potential Parameters

| Symbol | Name | Description | Typical Range |
|--------|------|-------------|---------------|
| $C_a$ | Attraction strength | Magnitude of attractive force | 0.0-2.0 |
| $\ell_a$ | Attraction scale | Length scale of attraction | 2.0-5.0 |
| $C_r$ | Repulsion strength | Magnitude of repulsive force | 0.0-5.0 |
| $\ell_r$ | Repulsion scale | Length scale of repulsion | 0.5-2.0 |

### Initial Condition Parameters

| Symbol | Name | Description |
|--------|------|-------------|
| $n_{\text{gaussian}}$ | Gaussian sample count | Number of Gaussian IC training runs |
| $n_{\text{uniform}}$ | Uniform sample count | Number of uniform IC training runs |
| $n_{\text{ring}}$ | Ring sample count | Number of ring IC training runs |
| $n_{\text{two\_cluster}}$ | Two-cluster sample count | Number of two-cluster training runs |
| $x_c^{(i)}$ | Cluster x-position | x-coordinate of cluster $i$ center |
| $y_c^{(i)}$ | Cluster y-position | y-coordinate of cluster $i$ center |
| $\sigma_i$ | Cluster width | Standard deviation of Gaussian cluster $i$ |
| $R_i$ | Ring radius | Radius of ring configuration $i$ |
| $\sigma_r^{(i)}$ | Ring width | Radial thickness of ring $i$ |
| $d_{\text{sep}}^{(i)}$ | Cluster separation | Distance between two clusters |
| $n_{\text{rep}}$ | Replicates | Number of samples per parameter combination |

### Density Field Parameters

| Symbol | Name | Description | Typical Range |
|--------|------|-------------|---------------|
| $n_x$ | Grid points (x) | Spatial resolution (x-direction) | 32-128 |
| $n_y$ | Grid points (y) | Spatial resolution (y-direction) | 32-128 |
| $h_{\text{KDE}}$ | KDE bandwidth | Gaussian kernel width | 2.0-5.0 grid cells |

### POD Parameters

| Symbol | Name | Description | Typical Range |
|--------|------|-------------|---------------|
| $r$ | Latent dimension | Number of POD modes (fixed) | 20-50 |
| $\tau_{\text{energy}}$ | Energy threshold | Variance capture threshold | 0.85-0.99 |

### MVAR Parameters

| Symbol | Name | Description | Typical Range |
|--------|------|-------------|---------------|
| $w$ | Lag order | Number of past timesteps | 3-10 |
| $\lambda_{\text{ridge}}$ | Ridge penalty | Regularization strength | 1e-6 to 1e-2 |
| $\rho_{\text{max}}$ | Max spectral radius | Stability constraint | 0.90-0.99 |
| $t_{\text{start}}$ | Training start | Warmup period cutoff | 0.5-2.0 |
| $t_{\text{end}}$ | Training end | Training window cutoff | $T_{\text{train}}$ |

### LSTM Parameters

| Symbol | Name | Description | Typical Range |
|--------|------|-------------|---------------|
| $L$ | Number of layers | LSTM depth | 1-3 |
| $N_h$ | Hidden size | LSTM state dimension | 32-128 |
| $p_{\text{dropout}}$ | Dropout probability | Regularization dropout | 0.0-0.5 |
| $B$ | Batch size | Mini-batch size | 16-128 |
| $\alpha_{\text{lr}}$ | Learning rate | Adam optimizer step size | 1e-4 to 1e-2 |
| $E$ | Max epochs | Training iteration limit | 50-500 |
| $\nu_{\text{val}}$ | Validation split | Fraction of data for validation | 0.1-0.3 |
| $P_{\text{early}}$ | Early stop patience | Epochs without improvement | 10-50 |
| $\delta_{\text{min}}$ | Min improvement | Early stopping threshold | 1e-5 to 1e-3 |

### Forecasting Parameters

| Symbol | Name | Description |
|--------|------|-------------|
| $T_{\text{warmup}}$ | Warmup duration | Conditioning period (usually $T_{\text{train}}$) |
| $T_{\text{horizon}}$ | Forecast horizon | Prediction duration beyond warmup |
| $w_{\text{init}}$ | Init window | Averaging window for initial state |

### Evaluation Parameters

| Symbol | Name | Description |
|--------|------|-------------|
| $\epsilon_{\text{mass}}$ | Mass tolerance | Acceptable mass violation | 0.01-0.05 (1%-5%) |
| $\tau_i$ | R² threshold | Accuracy threshold $i$ | 0.95, 0.85, 0.70 |

### Computational Parameters

| Symbol | Name | Description |
|--------|------|-------------|
| $W$ | Workers | Number of parallel processes |
| $T_{\text{slurm}}$ | Time limit | SLURM wall time |
| $M_{\text{mem}}$ | Memory | RAM per node (GB) |
| $N_{\text{nodes}}$ | Nodes | Number of compute nodes |
| $N_{\text{tasks}}$ | Tasks | Tasks per node |
| $d_{\text{gpu}}$ | GPU device | CUDA device ID |
| $f_{\text{checkpoint}}$ | Checkpoint freq | Iterations per checkpoint |

### Visualization Parameters

| Symbol | Name | Description |
|--------|------|-------------|
| $f_{\text{fps}}$ | Frame rate | Frames per second (video) |
| $d_{\text{dpi}}$ | Resolution | Dots per inch (figures) |

### Random Seeds

| Symbol | Name | Description |
|--------|------|-------------|
| $s_{\text{global}}$ | Global seed | Master random seed |
| $s_{\text{sim}}$ | Simulation seed | Vicsek dynamics seed |
| $s_{\text{ic}}$ | IC seed | Initial condition seed |
| $s_{\text{pod}}$ | POD seed | POD algorithm seed |
| $s_{\text{mvar}}$ | MVAR seed | MVAR training seed |
| $s_{\text{lstm}}$ | LSTM seed | LSTM training seed |

---

## Derived Quantities

The following quantities are **computed from** the configuration parameters:

### Dataset Scale

$$
\begin{aligned}
M &= n_{\text{gaussian}} + n_{\text{uniform}} + n_{\text{ring}} + n_{\text{two\_cluster}} \quad &&\text{(total training runs)} \\
n_{\text{test}} &= n_{\text{gaussian}}^{\text{test}} + n_{\text{uniform}}^{\text{test}} + n_{\text{ring}}^{\text{test}} + n_{\text{two\_cluster}}^{\text{test}} \quad &&\text{(total test runs)} \\
K &= \lfloor T_{\text{train}} / \Delta t \rfloor \quad &&\text{(timesteps per training run)} \\
K_{\text{test}} &= \lfloor T_{\text{test}} / \Delta t \rfloor \quad &&\text{(timesteps per test run)} \\
\end{aligned}
$$

### Spatial Grid

$$
\begin{aligned}
d_{\text{full}} &= n_x \times n_y \quad &&\text{(flattened density field dimension)} \\
\Delta x &= L_x / n_x \quad &&\text{(grid spacing, x-direction)} \\
\Delta y &= L_y / n_y \quad &&\text{(grid spacing, y-direction)} \\
\end{aligned}
$$

### Training Windows

$$
\begin{aligned}
K_{\text{valid}} &= K - w \quad &&\text{(valid timesteps per run after lag window)} \\
N_{\text{train}} &= M \times K_{\text{valid}} \quad &&\text{(total training samples)} \\
\end{aligned}
$$

### Parameter Counts

$$
\begin{aligned}
p_{\text{MVAR}} &= r^2 w + r \quad &&\text{(MVAR parameters)} \\
p_{\text{LSTM}} &= L \times 4(N_h(r + N_h) + N_h) + N_h r + r \quad &&\text{(LSTM parameters)} \\
\end{aligned}
$$

### Identifiability Ratio

$$
\rho = \frac{N_{\text{train}}}{p} \quad \text{(parameter-to-data ratio)}
$$

**Design Goal**: $\rho \approx 5-10$ (well-conditioned)

### Forecast Duration

$$
\begin{aligned}
T_{\text{forecast}} &= T_{\text{test}} - T_{\text{warmup}} \quad &&\text{(forecast duration)} \\
K_{\text{forecast}} &= \lfloor T_{\text{forecast}} / \Delta t \rfloor \quad &&\text{(forecast timesteps)} \\
\end{aligned}
$$

---

## Production Configuration Example

**Symbolic representation** of `configs/alvarez_style_production.yaml`:

```yaml
experiment:
  name: alvarez_production
  seed: s_global = 42

vicsek:
  N: N = 40
  v0: v_0 = 1.0
  Lx: L_x = 15.0
  Ly: L_y = 15.0
  R: R = 1.0
  eta: η = 0.3
  dt: Δt = 0.1
  T_train: T_train = 8.0       # → K = 80 timesteps
  T_test: T_test = 20.0         # → K_test = 200 timesteps

initial_conditions:
  train:
    - family: gaussian
      n_samples: n_gaussian = 108
      parameters:
        centers:
          x_c: [x_c^(1), x_c^(2), x_c^(3)] = [3.75, 7.5, 11.25]
          y_c: [y_c^(1), y_c^(2), y_c^(3)] = [3.75, 7.5, 11.25]
        std: [σ_1, σ_2, σ_3, σ_4] = [0.5, 1.0, 2.0, 3.0]
        samples_per_config: n_rep = 3
    
    - family: uniform
      n_samples: n_uniform = 100
    
    - family: ring
      n_samples: n_ring = 100
      parameters:
        radii: [R_1, R_2, R_3, R_4] = [2.0, 3.0, 4.0, 5.0]
        widths: [σ_r^(1), σ_r^(2)] = [0.3, 0.6]
    
    - family: two_cluster
      n_samples: n_two_cluster = 100
      parameters:
        separations: [d_sep^(1), ..., d_sep^(4)] = [3.0, 4.5, 6.0, 7.5]
        std: [σ_c^(1), σ_c^(2)] = [0.8, 1.5]

# Total: M = 108 + 100 + 100 + 100 = 408 training runs

density:
  nx: n_x = 64
  ny: n_y = 64                  # → d_full = 4096
  kde:
    bandwidth: h_KDE = 3.0

pod:
  method: fixed_modes
  r: r = 35                      # Fixed latent dimension (Alvarez principle)

mvar:
  lag: w = 5
  regularization:
    alpha: λ_ridge = 1e-4
  stability:
    enforce: true
    max_spectral_radius: ρ_max = 0.95

# Derived quantities:
# K_valid = 80 - 5 = 75
# N_train = 408 × 75 = 30,600
# p_MVAR = 35² × 5 + 35 = 6,160
# ρ = 30,600 / 6,160 ≈ 4.97 ✓ (well-conditioned)

forecasting:
  warmup:
    duration: T_warmup = T_train = 8.0
  rollout:
    horizon: T_horizon = T_test - T_train = 12.0  # → K_forecast = 120

evaluation:
  metrics:
    r2:
      variants: [reconstructed, latent, pod]
    mass_conservation:
      tolerance: ε_mass = 0.01  # 1% violation threshold
  horizons:
    r2_thresholds: [τ_1, τ_2, τ_3] = [0.95, 0.85, 0.70]
```

---

## Identifiability Constraint

**Key Design Principle** (Alvarez et al., 2024):

Given fixed computational budget $M$ (training runs), choose POD rank $r$ and MVAR lag $w$ such that:

$$
\rho = \frac{M \times (K - w)}{r^2 w + r} \geq 5
$$

**Solved for $r$** (quadratic constraint):

$$
r^2 w + r \leq \frac{M \times (K - w)}{5}
$$

**Example** (production):
- $M = 408$, $K = 80$, $w = 5$
- Required: $r^2 \times 5 + r \leq 30,600 / 5 = 6,120$
- Solution: $r = 35$ → $p = 6,160$ → $\rho \approx 4.97$ ✓

**Counterfactual** (energy threshold):
- If $r = 287$ (99% variance): $p = 412,532$ → $\rho \approx 0.074$ ✗ (catastrophic)

---

## Summary

This abstract configuration structure provides:

1. **Complete parameter space** for unified MVAR pipeline
2. **Mathematical symbols** for all tunable quantities
3. **Derived quantities** computed from base parameters
4. **Identifiability constraint** relating $(M, K, w, r) \to \rho$
5. **Production example** with symbolic notation

**Usage**:
- Thesis documentation (abstract parameter descriptions)
- Sensitivity analysis (vary symbols systematically)
- New experiment design (constraint-based parameter selection)

---

**Document Version**: 1.0  
**Last Updated**: February 2, 2026  
**Author**: Maria  
**Status**: Complete ✓
