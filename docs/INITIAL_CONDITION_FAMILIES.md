# Initial Condition Families: Dataset Augmentation Strategy

**Document Status**: Complete Technical Reference  
**Primary Module**: `src/rectsim/ic.py`, `src/rectsim/ic_generator.py`  
**Related Config**: `configs/alvarez_style_production.yaml`  
**Author**: Maria  
**Date**: February 2026  

---

## Table of Contents

1. [Overview and Motivation](#1-overview-and-motivation)
2. [Mathematical Formulations](#2-mathematical-formulations)
3. [Implementation Details](#3-implementation-details)
4. [Pipeline Integration](#4-pipeline-integration)
5. [Evaluation and Visualization](#5-evaluation-and-visualization)
6. [Dataset Statistics](#6-dataset-statistics)
7. [References](#7-references)

---

## 1. Overview and Motivation

### 1.1 Purpose of IC Diversity

Our ROM-MVAR pipeline requires **robust generalization** across different initial collective states. Following the approach of Alvarez et al., we train on a **diverse ensemble** of initial conditions (ICs) to ensure the reduced-order model can:

1. **Capture varied dynamics**: Different ICs lead to different transient behaviors (ordered vs disordered states)
2. **Improve POD basis quality**: Rich data manifold → more representative spatial modes
3. **Enable generalization testing**: Hold-out IC types for interpolation/extrapolation tests

### 1.2 Four IC Families

We implement four IC families inspired by collective behavior literature:

| **IC Type** | **Spatial Distribution** | **Purpose** | **Code Name** |
|------------|--------------------------|-------------|---------------|
| **Uniform** | Homogeneous random distribution | Baseline, maximum entropy state | `uniform` |
| **Single Gaussian** | Concentrated blob, random center | Localized clustering | `gaussian` |
| **Ring/Annulus** | Circular arrangement | Rotational symmetry, vortex-like states | `ring` |
| **Two-Cluster Gaussian** | Bimodal distribution | Multi-group interactions, merging dynamics | `cluster` (or `two_clusters`) |

These families provide:
- **Training diversity**: ~400 training simulations across all types
- **Generalization tests**: Hold-out parameter values for interpolation/extrapolation
- **Physical relevance**: Each represents distinct collective behavior regimes

---

## 2. Mathematical Formulations

All ICs sample $N = 200$ particle positions in domain $\Omega = [0, L_x] \times [0, L_y]$ with $L_x = L_y = 20$.

### 2.1 Uniform IC

**Description**: Particles uniformly distributed across domain (maximum entropy state).

**Mathematical Definition**:
$$
\mathbf{x}_i \sim \mathcal{U}([0, L_x] \times [0, L_y]), \quad i = 1, \ldots, N
$$

Each particle position is independently sampled:
$$
x_i \sim \mathcal{U}(0, L_x), \quad y_i \sim \mathcal{U}(0, L_y)
$$

**Probability Density**:
$$
p(\mathbf{x}) = \frac{1}{L_x L_y} = \frac{1}{400} = 0.0025
$$

**Expected Density Field**:
$$
\mathbb{E}[\rho(\mathbf{r})] = \frac{N}{L_x L_y} = \frac{200}{400} = 0.5 \text{ particles/unit area}
$$

**Implementation** (`src/rectsim/ic.py`, lines 78-80):
```python
def _uniform_ic(N: int, Lx: float, Ly: float, rng: np.random.Generator) -> np.ndarray:
    """Uniform distribution in [0,Lx]×[0,Ly]."""
    return rng.uniform(low=[0.0, 0.0], high=[Lx, Ly], size=(N, 2))
```

**Properties**:
- **Spatial variance**: $\sigma_x^2 = \sigma_y^2 = L_x^2 / 12 = 33.3$
- **Standard deviation**: $\sigma \approx 5.77$ (uniform distribution property)
- **No clustering**: Homogeneous distribution, minimal spatial correlation

---

### 2.2 Single Gaussian IC

**Description**: Particles concentrated in a Gaussian blob with random center.

**Mathematical Definition**:
$$
\mathbf{x}_i \sim \mathcal{N}(\boldsymbol{\mu}, \sigma^2 \mathbf{I}), \quad i = 1, \ldots, N
$$

**Center Selection**:
To avoid edge effects, center is randomly chosen with margin:
$$
\mu_x \sim \mathcal{U}(0.2 L_x, 0.8 L_x) = \mathcal{U}(4, 16)
$$
$$
\mu_y \sim \mathcal{U}(0.2 L_y, 0.8 L_y) = \mathcal{U}(4, 16)
$$

**Variance Parameter**:
Default: $\sigma = 0.1 \cdot \min(L_x, L_y) = 2.0$

For dataset augmentation, we vary:
$$
\text{Variance } V \in \{0.25, 1.0, 4.0, 9.0\} \implies \sigma \in \{0.5, 1.0, 2.0, 3.0\}
$$

**Clipping to Domain**:
Since Gaussian is unbounded, we clip positions:
$$
x_i' = \min(\max(x_i, 0), L_x), \quad y_i' = \min(\max(y_i, 0), L_y)
$$

**Implementation** (`src/rectsim/ic.py`, lines 83-125):
```python
def _gaussian_blob_ic(
    N: int,
    Lx: float,
    Ly: float,
    rng: np.random.Generator,
    sigma_factor: float = 0.1,
) -> np.ndarray:
    """Single Gaussian blob centered at a random location."""
    # Choose random center, avoiding edges
    margin = 0.2
    center_x = rng.uniform(margin * Lx, (1 - margin) * Lx)
    center_y = rng.uniform(margin * Ly, (1 - margin) * Ly)
    center = np.array([center_x, center_y])

    # Standard deviation based on domain size
    sigma = sigma_factor * min(Lx, Ly)

    # Sample from 2D Gaussian
    positions = rng.normal(loc=center, scale=sigma, size=(N, 2))

    # Clip to domain
    positions[:, 0] = np.clip(positions[:, 0], 0.0, Lx)
    positions[:, 1] = np.clip(positions[:, 1], 0.0, Ly)

    return positions
```

**Probability Density (2D Gaussian)**:
$$
p(\mathbf{x}; \boldsymbol{\mu}, \sigma) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{\|\mathbf{x} - \boldsymbol{\mu}\|^2}{2\sigma^2}\right)
$$

**Properties**:
- **Concentration**: ~68% of particles within $\sigma$ of center
- **Effective support**: $\approx 3\sigma$ contains 99.7% of mass
- **Clustering measure**: Local density $\rho(\boldsymbol{\mu}) \gg \bar{\rho}$

**Production Config** (`configs/alvarez_style_production.yaml`):
```yaml
train_ic:
  gaussian:
    enabled: true
    positions_x: [3.75, 7.5, 11.25]     # 3 x-centers (off-center + center)
    positions_y: [3.75, 7.5, 11.25]     # 3 y-centers
    variances: [0.25, 1.0, 4.0, 9.0]    # 4 variance levels
    n_samples_per_config: 3             # 3 random seeds per config
    # Total: 3 × 3 × 4 × 3 = 108 training runs
```

**Test ICs** (held-out configurations):
```yaml
test_ic:
  gaussian:
    enabled: true
    test_positions_x: [5.5, 9.0, 2.0]   # Interpolation + extrapolation
    test_positions_y: [5.5, 9.0, 2.0]
    test_variances: [2.25]              # Intermediate variance (σ = 1.5)
```

---

### 2.3 Ring / Annulus IC

**Description**: Particles arranged around a ring with small radial noise.

**Mathematical Definition**:
Particles distributed on annulus with:
- **Mean radius**: $R$
- **Radial spread**: $\Delta R$ (width)

**Polar Coordinates**:
$$
\theta_i \sim \mathcal{U}(0, 2\pi)
$$
$$
r_i \sim \mathcal{N}(R, (\Delta R \cdot R)^2)
$$

**Cartesian Conversion**:
$$
x_i = c_x + r_i \cos(\theta_i)
$$
$$
y_i = c_y + r_i \sin(\theta_i)
$$

where $(c_x, c_y) = (L_x/2, L_y/2) = (10, 10)$ is the domain center.

**Default Parameters**:
- $R = 0.3 \cdot \min(L_x, L_y) = 6.0$
- $\Delta R = 0.05$ (5% radial noise)

**Implementation** (`src/rectsim/ic.py`, lines 135-180):
```python
def _ring_ic(
    N: int,
    Lx: float,
    Ly: float,
    rng: np.random.Generator,
    radius_factor: float = 0.3,
    radial_noise: float = 0.05,
) -> np.ndarray:
    """Particles arranged around a ring with small radial noise."""
    # Center of the domain
    center = np.array([Lx / 2, Ly / 2])

    # Ring radius
    radius = radius_factor * min(Lx, Ly)

    # Angular positions uniformly distributed
    angles = rng.uniform(0.0, 2 * np.pi, size=N)

    # Add radial noise
    radii = radius + rng.normal(0.0, radial_noise * radius, size=N)

    # Convert to Cartesian coordinates
    positions = np.empty((N, 2))
    positions[:, 0] = center[0] + radii * np.cos(angles)
    positions[:, 1] = center[1] + radii * np.sin(angles)

    # Wrap or clip to domain
    positions[:, 0] = np.clip(positions[:, 0], 0.0, Lx)
    positions[:, 1] = np.clip(positions[:, 1], 0.0, Ly)

    return positions
```

**Probability Density** (polar form):
$$
p(r, \theta; R, \Delta R) = \frac{1}{2\pi} \cdot \frac{1}{\sqrt{2\pi (\Delta R \cdot R)^2}} \exp\left(-\frac{(r - R)^2}{2(\Delta R \cdot R)^2}\right)
$$

**Production Config**:
```yaml
train_ic:
  ring:
    enabled: true
    radii: [2.0, 3.0, 4.0, 5.0]         # 4 radius values
    widths: [0.3, 0.3, 0.6, 0.6]        # Paired: tight (0.3) and loose (0.6)
    n_samples_per_config: 25            # 25 random angular arrangements
    # Total: 4 × 25 = 100 training runs
```

**Test ICs** (interpolation):
```yaml
test_ic:
  ring:
    enabled: true
    test_radii: [2.5, 3.5, 4.5]         # Intermediate radii
    test_widths: [0.45, 0.45, 0.45]     # Intermediate width
```

**Properties**:
- **Rotational symmetry**: Invariant under rotation about center
- **Angular uniformity**: No preferred direction (before velocity assignment)
- **Physical relevance**: Models vortex-like collective states
- **Distance from center**: $d_i = \|(\mathbf{x}_i - \mathbf{c})\| \approx R$ for all $i$

---

### 2.4 Two-Cluster Gaussian Mixture IC

**Description**: Two separate Gaussian clusters (bimodal distribution).

**Mathematical Definition**:
Mixture of two Gaussians:
$$
p(\mathbf{x}) = \frac{1}{2} \mathcal{N}(\boldsymbol{\mu}_1, \sigma^2 \mathbf{I}) + \frac{1}{2} \mathcal{N}(\boldsymbol{\mu}_2, \sigma^2 \mathbf{I})
$$

**Cluster Centers**:
Horizontally separated:
$$
\boldsymbol{\mu}_1 = \left(\frac{L_x}{2} - \frac{d}{2}, \frac{L_y}{2}\right)
$$
$$
\boldsymbol{\mu}_2 = \left(\frac{L_x}{2} + \frac{d}{2}, \frac{L_y}{2}\right)
$$

where $d$ is the **separation distance** between clusters.

**Particle Assignment**:
$$
N_1 = \lfloor N / 2 \rfloor = 100, \quad N_2 = N - N_1 = 100
$$

Each particle is assigned to cluster 1 or 2, then sampled from corresponding Gaussian.

**Default Parameters**:
- Separation: $d = L_x / 3 \approx 6.67$
- Cluster std: $\sigma = \min(L_x, L_y) / 8 = 2.5$

**Implementation** (`src/rectsim/initial_conditions.py`, lines 73-120):
```python
def two_clusters(N: int, Lx: float, Ly: float, rng: np.random.Generator,
                 separation: float = None, sigma: float = None) -> np.ndarray:
    """Generate particle positions in two separated Gaussian clusters."""
    if separation is None:
        separation = Lx / 3
    if sigma is None:
        sigma = min(Lx, Ly) / 8
    
    # Split particles between two clusters
    N1 = N // 2
    N2 = N - N1
    
    # Cluster centers (horizontally separated)
    center1 = (Lx / 2 - separation / 2, Ly / 2)
    center2 = (Lx / 2 + separation / 2, Ly / 2)
    
    # Generate positions for each cluster
    cluster1 = rng.normal(loc=center1, scale=sigma, size=(N1, 2))
    cluster2 = rng.normal(loc=center2, scale=sigma, size=(N2, 2))
    
    # Combine and wrap to domain
    positions = np.vstack([cluster1, cluster2])
    positions[:, 0] = positions[:, 0] % Lx
    positions[:, 1] = positions[:, 1] % Ly
    
    return positions
```

**Production Config**:
```yaml
train_ic:
  two_clusters:
    enabled: true
    separations: [3.0, 4.5, 6.0, 7.5]   # 4 separation distances
    sigmas: [0.8, 0.8, 1.5, 1.5]        # Paired: tight (0.8) and loose (1.5)
    n_samples_per_config: 25            # 25 random velocity initializations
    # Total: 4 × 25 = 100 training runs
```

**Test ICs** (interpolation):
```yaml
test_ic:
  two_clusters:
    enabled: true
    test_separations: [3.75, 5.25, 6.75]  # Intermediate separations
    test_sigmas: [1.1, 1.1, 1.1]          # Intermediate cluster size
```

**Properties**:
- **Bimodality**: Mixture distribution has two modes
- **Cluster overlap**: Decreases with increasing separation $d$
- **Merging dynamics**: Small $d$ → rapid merging; large $d$ → prolonged separation
- **Symmetry**: Reflection symmetry about vertical midline
- **Separation parameter range**: Training uses $d \in [3.0, 4.5, 6.0, 7.5]$, testing uses intermediate values $[3.75, 5.25, 6.75]$

---

## 3. Implementation Details

### 3.1 Module Structure

**Primary Module**: `src/rectsim/ic.py` (300 lines)

```python
"""Initial condition generation for ensemble simulations.

This module provides utilities to generate varied initial particle
configurations for large ensemble runs. It supports multiple IC types
inspired by the Alvarez et al. autoregressive ROM paper.
"""

from typing import Literal
import numpy as np

ICType = Literal["uniform", "gaussian", "ring", "cluster"]

def sample_initial_positions(
    ic_type: str,
    N: int,
    Lx: float,
    Ly: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate initial particle positions according to the specified distribution.
    
    Parameters
    ----------
    ic_type : str
        Type of initial condition: "uniform", "gaussian", "ring", "cluster"
    N : int
        Number of particles.
    Lx, Ly : float
        Domain dimensions.
    rng : np.random.Generator
        Random number generator for reproducibility.
        
    Returns
    -------
    positions : np.ndarray, shape (N, 2)
        Initial positions in [0,Lx]×[0,Ly].
    """
    if ic_type == "uniform":
        return _uniform_ic(N, Lx, Ly, rng)
    elif ic_type == "gaussian":
        return _gaussian_blob_ic(N, Lx, Ly, rng)
    elif ic_type == "ring":
        return _ring_ic(N, Lx, Ly, rng)
    elif ic_type == "cluster":
        return _cluster_ic(N, Lx, Ly, rng)
    else:
        raise ValueError(
            f"Unknown ic_type '{ic_type}'. "
            f"Supported types: uniform, gaussian, ring, cluster"
        )
```

**Helper Module**: `src/rectsim/ic_generator.py` (300 lines)

Generates configuration lists for training/test runs:

```python
def generate_training_configs(train_ic_config, base_config):
    """Generate list of training run configurations.
    
    Returns
    -------
    list
        List of configuration dictionaries with keys:
        - run_id: unique run identifier
        - distribution: IC type (gaussian_cluster, uniform, ring, two_clusters)
        - ic_params: parameters for IC generation
        - label: human-readable label
    """
    configs = []
    run_id = 0
    
    # Gaussian configurations
    if train_ic_config.get('gaussian', {}).get('enabled', False):
        gauss_cfg = train_ic_config['gaussian']
        positions_x = gauss_cfg.get('positions_x', [])
        positions_y = gauss_cfg.get('positions_y', [])
        variances = gauss_cfg.get('variances', [])
        n_samples = gauss_cfg.get('n_samples_per_config', 1)
        
        for px in positions_x:
            for py in positions_y:
                for var in variances:
                    for sample in range(n_samples):
                        configs.append({
                            'run_id': run_id,
                            'distribution': 'gaussian_cluster',
                            'ic_params': {
                                'center': (float(px), float(py)),
                                'sigma': float(np.sqrt(var))
                            },
                            'label': f'gauss_x{px:.1f}_y{py:.1f}_var{var:.1f}_s{sample}'
                        })
                        run_id += 1
    
    # Similar logic for uniform, ring, two_clusters...
    
    return configs
```

### 3.2 Velocity Initialization

All IC types use the **same velocity initialization** to isolate position effects:

**Uniform random directions**:
$$
\theta_i \sim \mathcal{U}(0, 2\pi)
$$
$$
\mathbf{v}_i = v_0 \begin{pmatrix} \cos(\theta_i) \\ \sin(\theta_i) \end{pmatrix}, \quad v_0 = 1.0
$$

This ensures:
- **Constant speed**: $\|\mathbf{v}_i\| = v_0$ for all particles
- **Random headings**: No initial velocity alignment
- **Zero polarization**: $\mathbf{P}(t=0) = \frac{1}{N}\sum_i \mathbf{v}_i \approx \mathbf{0}$

**Implementation** (`src/rectsim/initial_conditions.py`):
```python
def initialize_velocities(N: int, v0: float, rng: np.random.Generator,
                         distribution: str = 'random') -> np.ndarray:
    """Generate initial velocities."""
    if distribution == 'random':
        # Random angles, constant speed
        theta = rng.uniform(0, 2 * np.pi, size=N)
        vx = v0 * np.cos(theta)
        vy = v0 * np.sin(theta)
        return np.column_stack([vx, vy])
    else:
        raise ValueError(f"Unknown velocity distribution: {distribution}")
```

### 3.3 Reproducibility

All IC generation uses **NumPy's new RNG system** for reproducibility:

```python
# In pipeline script
for cfg in training_configs:
    seed = cfg['run_id'] + 1000  # Offset to avoid collisions
    rng = np.random.default_rng(seed)
    
    # Sample positions
    pos = sample_initial_positions(
        ic_type=cfg['distribution'],
        N=200,
        Lx=20.0,
        Ly=20.0,
        rng=rng,
    )
    
    # Sample velocities
    vel = initialize_velocities(N=200, v0=1.0, rng=rng)
```

**Reproducibility guarantees**:
- Same `run_id` → identical IC
- Different `run_id` → independent random sample
- Test ICs use `run_id >= 1000` to avoid overlap

---

## 4. Pipeline Integration

### 4.1 Three-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: Training Data Generation                          │
│ ─────────────────────────────────────────────────────────── │
│ • Generate N_train = 400 diverse ICs                        │
│ • Simulate each for T_train = 8s (80 timesteps)             │
│ • Compute density fields ρ(x,y,t) on 64×64 grid             │
│ • Save: {x, v, times, density} per run                      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: POD + MVAR Training                               │
│ ─────────────────────────────────────────────────────────── │
│ • Stack all density snapshots: X ∈ ℝ^(4096 × 32000)        │
│ • Compute POD: X ≈ Φ Y (Φ: 4096×35 spatial modes)          │
│ • Extract latent trajectories: y(t) = Φ^T ρ(t)             │
│ • Train MVAR(5): y(t+Δt) = Σ A_τ y(t-τΔt) + ε              │
│ • Save: Φ, {A_0, ..., A_4}, ridge_alpha                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: Generalization Testing                            │
│ ─────────────────────────────────────────────────────────── │
│ • Generate N_test = 40 held-out ICs                         │
│ • Simulate T_test = 20s (200 timesteps)                     │
│ • Warmup: Use t ∈ [0, 8s] for initial condition             │
│ • Forecast: Predict t ∈ [8s, 20s] using MVAR               │
│ • Evaluate: R², RMSE, τ (time to tol exceedance)           │
│ • Visualize: Error plots, order parameters, best runs       │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Training IC Distribution

**Production Configuration** (`configs/alvarez_style_production.yaml`):

```yaml
train_ic:
  gaussian:
    enabled: true
    positions_x: [3.75, 7.5, 11.25]     # 3 × 3 centers
    positions_y: [3.75, 7.5, 11.25]
    variances: [0.25, 1.0, 4.0, 9.0]    # 4 variances
    n_samples_per_config: 3             # 3 samples
    # Subtotal: 3 × 3 × 4 × 3 = 108 runs
  
  uniform:
    enabled: true
    n_runs: 100                         # 100 uniform ICs
  
  ring:
    enabled: true
    radii: [2.0, 3.0, 4.0, 5.0]         # 4 radii
    widths: [0.3, 0.3, 0.6, 0.6]
    n_samples_per_config: 25            # 25 samples
    # Subtotal: 4 × 25 = 100 runs
  
  two_clusters:
    enabled: true
    separations: [3.0, 4.5, 6.0, 7.5]   # 4 separations
    sigmas: [0.8, 0.8, 1.5, 1.5]
    n_samples_per_config: 25            # 25 samples
    # Subtotal: 4 × 25 = 100 runs

# Total: 108 + 100 + 100 + 100 = 408 training simulations
```

**Training Horizon**:
```yaml
sim:
  T: 8.0          # 8 seconds
  dt: 0.1         # 0.1s timestep
  # → 80 snapshots per trajectory
  # → Total: 408 × 80 = 32,640 density snapshots
```

### 4.3 Test IC Distribution

**Test Configuration** (held-out parameters):

```yaml
test_ic:
  gaussian:
    enabled: true
    test_positions_x: [5.5, 9.0, 2.0]   # Interpolation (5.5, 9.0) + extrapolation (2.0)
    test_positions_y: [5.5, 9.0, 2.0]
    test_variances: [2.25]              # Intermediate variance
    # Subtotal: 3 × 3 × 1 = 9 test runs
  
  uniform:
    enabled: true
    n_runs: 10                          # 10 uniform test ICs
  
  ring:
    enabled: true
    test_radii: [2.5, 3.5, 4.5]         # Interpolation radii
    test_widths: [0.45, 0.45, 0.45]
    n_samples_per_config: 1
    # Subtotal: 3 × 1 = 3 test runs
  
  two_clusters:
    enabled: true
    test_separations: [3.75, 5.25, 6.75]  # Interpolation
    test_sigmas: [1.1, 1.1, 1.1]
    n_samples_per_config: 3
    # Subtotal: 3 × 3 = 9 test runs

# Total: 9 + 10 + 3 + 9 = 31 test simulations
```

**Test Horizon** (longer than training):
```yaml
test_sim:
  T: 20.0         # 20 seconds (2.5× training duration)
  dt: 0.1         # Same timestep
  # → 200 snapshots per test trajectory
  # → Forecast window: [8s, 20s] = 120 timesteps (12s forecast!)
```

### 4.4 Warmup and Forecast Windows

**Temporal Split**:
- **Warmup window**: $t \in [0, T_{\text{train}}] = [0, 8\text{s}]$
  - Use ground truth trajectory to initialize MVAR
  - Extract $w = 5$ lagged latent states: $\{y(t-4\Delta t), \ldots, y(t)\}$
  
- **Forecast window**: $t \in [T_{\text{train}}, T_{\text{test}}] = [8\text{s}, 20\text{s}]$
  - Autoregressive prediction: $y(t+\Delta t) = \sum_{\tau=0}^{w-1} A_\tau y(t - \tau\Delta t)$
  - Density reconstruction: $\rho_{\text{pred}}(t) = \Phi \, y(t)$
  - Compare with ground truth $\rho_{\text{true}}(t)$

**Mathematical Formulation**:

Given:
- POD basis $\Phi \in \mathbb{R}^{n_s \times d}$ (4096 × 35)
- MVAR coefficients $\{A_0, \ldots, A_{w-1}\}$ with $A_\tau \in \mathbb{R}^{d \times d}$ (35 × 35)
- Initial condition window: $\{y(t_0), y(t_0 + \Delta t), \ldots, y(t_0 + (w-1)\Delta t)\}$

**Autoregressive Forecast**:
$$
y(t_{k+1}) = \sum_{\tau=0}^{w-1} A_\tau \, y(t_k - \tau\Delta t), \quad k = w, w+1, \ldots, n_{\text{forecast}}
$$

**Density Reconstruction**:
$$
\rho_{\text{pred}}(t_k) = \Phi \, y(t_k) \in \mathbb{R}^{n_s}
$$

---

## 5. Evaluation and Visualization

### 5.1 Evaluation Metrics

For each test IC, we compute metrics over the **forecast window** $t \in [8\text{s}, 20\text{s}]$.

#### 5.1.1 Coefficient of Determination (R²)

**Definition**:
$$
R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}
$$

where:
$$
SS_{\text{res}} = \sum_{t=t_0}^{T} \sum_{i,j} (\rho_{\text{pred}}(x_i, y_j, t) - \rho_{\text{true}}(x_i, y_j, t))^2
$$
$$
SS_{\text{tot}} = \sum_{t=t_0}^{T} \sum_{i,j} (\rho_{\text{true}}(x_i, y_j, t) - \bar{\rho}_{\text{true}})^2
$$

**Interpretation**:
- $R^2 = 1$: Perfect prediction
- $R^2 = 0$: Prediction no better than mean
- $R^2 < 0$: Prediction worse than mean

**Implementation** (`src/rectsim/rom_eval_metrics.py`, lines 94-101):
```python
def compute_forecast_metrics(
    density_true: np.ndarray,   # (T, Ny, Nx)
    density_pred: np.ndarray,
    times: Optional[np.ndarray] = None,
    tol: float = 0.1,
) -> Dict[str, Any]:
    """Compute forecast error metrics."""
    # Flatten spatial dimensions
    rho_true_flat = density_true.reshape(T, -1)  # (T, Ny*Nx)
    rho_pred_flat = density_pred.reshape(T, -1)
    
    # R² computation
    ss_res = np.sum((rho_true_flat - rho_pred_flat) ** 2)
    ss_tot = np.sum((rho_true_flat - rho_true_flat.mean()) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {"r2": float(r2), ...}
```

#### 5.1.2 Root Mean Squared Error (RMSE)

**Per-Time-Step RMSE**:
$$
\text{RMSE}(t) = \sqrt{\frac{1}{n_x \cdot n_y} \sum_{i,j} (\rho_{\text{pred}}(x_i, y_j, t) - \rho_{\text{true}}(x_i, y_j, t))^2}
$$

**Mean RMSE** (averaged over forecast window):
$$
\overline{\text{RMSE}} = \frac{1}{n_t} \sum_{t=t_0}^{T} \text{RMSE}(t)
$$

**Implementation**:
```python
# Per-time RMSE
e2_t = np.sqrt(np.mean((density_true - density_pred) ** 2, axis=(1, 2)))

# Mean RMSE
rmse_mean = e2_t.mean()
```

#### 5.1.3 L¹, L², and L∞ Norms

**L¹ (Mean Absolute Error per time)**:
$$
\|e\|_1(t) = \frac{1}{n_x \cdot n_y} \sum_{i,j} |\rho_{\text{pred}}(x_i, y_j, t) - \rho_{\text{true}}(x_i, y_j, t)|
$$

**L² (already defined as RMSE)**:
$$
\|e\|_2(t) = \text{RMSE}(t)
$$

**L∞ (Maximum absolute error per time)**:
$$
\|e\|_\infty(t) = \max_{i,j} |\rho_{\text{pred}}(x_i, y_j, t) - \rho_{\text{true}}(x_i, y_j, t)|
$$

**Summary Statistics**:
We report **median** values to be robust to outliers:
```python
e1_median = np.median(e1_t)
e2_median = np.median(e2_t)
einf_median = np.median(einf_t)
```

#### 5.1.4 Mass Conservation Error

**Relative Mass Error per Time**:
$$
\epsilon_{\text{mass}}(t) = \frac{|M_{\text{pred}}(t) - M_{\text{true}}(t)|}{|M_{\text{true}}(t)|}
$$

where:
$$
M(t) = \sum_{i,j} \rho(x_i, y_j, t) \cdot \Delta x \cdot \Delta y
$$

For uniform grid, $\Delta x \cdot \Delta y = 1$:
$$
M(t) = \sum_{i,j} \rho(x_i, y_j, t)
$$

**Reported Metrics**:
- Mean mass error: $\overline{\epsilon}_{\text{mass}} = \frac{1}{n_t}\sum_t \epsilon_{\text{mass}}(t)$
- Max mass error: $\max_t \epsilon_{\text{mass}}(t)$

**Implementation**:
```python
mass_true = density_true.sum(axis=(1, 2))
mass_pred = density_pred.sum(axis=(1, 2))

mass_error_t = np.abs(mass_pred - mass_true) / (np.abs(mass_true) + 1e-12)
mass_error_mean = mass_error_t.mean()
mass_error_max = mass_error_t.max()
```

#### 5.1.5 Time to Tolerance Exceedance (τ)

**Definition**: First time when relative L² error exceeds tolerance $\epsilon_{\text{tol}} = 0.1$:

$$
\tau = \min \left\{ t \geq t_0 : \frac{\|\rho_{\text{pred}}(t) - \rho_{\text{true}}(t)\|_2}{\|\rho_{\text{true}}(t)\|_2} > \epsilon_{\text{tol}} \right\}
$$

**If never exceeds**: $\tau = \text{None}$ (excellent long-horizon accuracy)

**Implementation**:
```python
# Relative L² error per time
norm_true_t = np.sqrt(np.sum(density_true ** 2, axis=(1, 2)))
norm_error_t = np.sqrt(np.sum((density_pred - density_true) ** 2, axis=(1, 2)))
rel_error_t = norm_error_t / (norm_true_t + 1e-12)

# Find first time exceeding tolerance
exceed_mask = rel_error_t > tol
if np.any(exceed_mask):
    idx = np.where(exceed_mask)[0][0]
    tau = times[idx].item()
else:
    tau = None
```

### 5.2 Visualization Outputs

#### 5.2.1 Error Timeseries Plots

**Purpose**: Visualize temporal evolution of errors over forecast window.

**Generated Plot** (`scripts/rom_evaluate.py` → `src/rectsim/rom_eval_viz.py`):

```
┌────────────────────────────────────────────────────────────┐
│  Forecast Error vs Time (IC Type: gaussian)               │
├─────────────────────────┬──────────────────────────────────┤
│ L² Error (RMSE)         │ Relative L² Error                │
│ [Plot: RMSE vs time]    │ [Plot: rel_e2 vs time]           │
│                         │ [Horizontal line: tol = 0.1]     │
├─────────────────────────┼──────────────────────────────────┤
│ L¹ and L∞ Errors        │ Mass Conservation Error          │
│ [Plot: L1, Linf vs time]│ [Plot: mass_error vs time]       │
└─────────────────────────┴──────────────────────────────────┘
```

**Implementation** (`src/rectsim/rom_eval_viz.py`, lines 91-170):
```python
def plot_error_time_series(
    times: np.ndarray,
    errors: Dict[str, np.ndarray],
    out_path: Path,
    title: str = "Forecast Error vs Time",
    ic_type: Optional[str] = None,
) -> None:
    """Plot error metrics vs time."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # L² error (RMSE)
    axes[0, 0].plot(times, errors["e2"], 'b-', linewidth=2)
    axes[0, 0].set_ylabel("L² Error (RMSE)")
    axes[0, 0].set_title("L² Error")
    
    # Relative L² error with tolerance
    axes[0, 1].plot(times, errors["rel_e2"], 'r-', linewidth=2)
    axes[0, 1].axhline(0.1, color='k', linestyle='--', label='tol=0.1')
    axes[0, 1].set_ylabel("Relative L² Error")
    
    # L¹ and L∞
    axes[1, 0].plot(times, errors["e1"], 'g-', label='L¹ (MAE)')
    axes[1, 0].plot(times, errors["einf"], 'm-', label='L∞ (max)')
    axes[1, 0].legend()
    
    # Mass conservation
    axes[1, 1].plot(times, errors["mass_error"], 'orange', linewidth=2)
    axes[1, 1].set_ylabel("Relative Mass Error")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
```

**Output Directory Structure**:
```
outputs/
├── gaussian/
│   ├── best_error.png          # Error plot for best R² run
│   └── test_gauss_x5.5_y5.5_var2.25_s0_error.png
├── uniform/
│   ├── best_error.png
│   └── test_uniform_s0_error.png
├── ring/
│   ├── best_error.png
│   └── test_ring_r2.5_w0.45_s0_error.png
└── two_cluster/
    ├── best_error.png
    └── test_two_cluster_sep3.75_sig1.1_s0_error.png
```

#### 5.2.2 Order Parameter Plots

**Purpose**: Contextualize prediction quality with physical observables.

**Order Parameters Computed**:

1. **Polarization** (global velocity alignment):
$$
\Phi(t) = \frac{\left\| \frac{1}{N} \sum_{i=1}^N \mathbf{v}_i(t) \right\|}{\frac{1}{N} \sum_{i=1}^N \|\mathbf{v}_i(t)\|} = \frac{\|\langle \mathbf{v} \rangle\|}{\langle \|\mathbf{v}\| \rangle}
$$

2. **Mean Speed**:
$$
\bar{v}(t) = \frac{1}{N} \sum_{i=1}^N \|\mathbf{v}_i(t)\|
$$

3. **Speed Standard Deviation**:
$$
\sigma_v(t) = \sqrt{\frac{1}{N} \sum_{i=1}^N (\|\mathbf{v}_i(t)\| - \bar{v}(t))^2}
$$

**Generated Plot**:
```
┌────────────────────────────────────────────────────────────┐
│  Order Parameters vs Time (IC Type: ring)                 │
├────────────────────────────────────────────────────────────┤
│  Polarization Φ(t)                                         │
│  [Plot: 0 ≤ Φ ≤ 1]                                         │
│  [Vertical line at T0 = 8s: forecast start]                │
├────────────────────────────────────────────────────────────┤
│  Mean Speed ± 1 std                                        │
│  [Plot: mean speed with shaded region]                     │
│  [Should be ≈ v0 = 1.0 for Vicsek]                         │
└────────────────────────────────────────────────────────────┘
```

**Implementation** (`src/rectsim/rom_eval_viz.py`, lines 173-234):
```python
def compute_order_params_from_sample(
    sample: SimulationSample,
    T0: Optional[int] = None,
) -> pd.DataFrame:
    """Compute order parameters from simulation trajectories."""
    v = sample.traj_true["v"]  # (T, N, 2)
    times = sample.traj_true.get("times", np.arange(v.shape[0]))
    
    if T0 is not None:
        v = v[T0:]
        times = times[T0:]
    
    T, N, _ = v.shape
    
    polarization = np.zeros(T)
    speed_mean = np.zeros(T)
    speed_std = np.zeros(T)
    
    for t in range(T):
        vt = v[t]  # (N, 2)
        
        # Speed per agent
        speeds = np.linalg.norm(vt, axis=1)
        speed_mean[t] = speeds.mean()
        speed_std[t] = speeds.std()
        
        # Polarization: ||<v>|| / <||v||>
        v_mean = vt.mean(axis=0)
        polarization[t] = np.linalg.norm(v_mean) / (speeds.mean() + 1e-12)
    
    return pd.DataFrame({
        "time": times,
        "polarization": polarization,
        "speed_mean": speed_mean,
        "speed_std": speed_std,
    })
```

**Physical Interpretation**:
- **High polarization** ($\Phi \approx 1$): Ordered collective motion (flocking)
- **Low polarization** ($\Phi \approx 0$): Disordered motion (gas-like)
- **Speed fluctuations**: Should be minimal for constant-speed Vicsek ($\sigma_v \approx 0$)

#### 5.2.3 Best Run Selection

**Strategy**: For each IC type, select the run with **highest R²** for detailed visualization.

**Implementation** (`src/rectsim/rom_eval_viz.py`, lines 28-77):
```python
def select_best_runs(
    metrics_list: List[SimulationMetrics],
    key: str = "r2",
    maximize: bool = True,
) -> Dict[str, SimulationMetrics]:
    """Select best simulation per IC type based on a metric."""
    # Group by IC type
    by_ic_type: Dict[str, List[SimulationMetrics]] = {}
    for m in metrics_list:
        if m.ic_type not in by_ic_type:
            by_ic_type[m.ic_type] = []
        by_ic_type[m.ic_type].append(m)
    
    # Select best for each IC type
    best_runs = {}
    for ic_type, sims in by_ic_type.items():
        values = [getattr(m, key) for m in sims]
        
        if maximize:
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        
        best_runs[ic_type] = sims[best_idx]
    
    return best_runs
```

**Usage in Evaluation Script**:
```python
# Load all test metrics
metrics_list = [...]  # List[SimulationMetrics]

# Select best run per IC type
best_runs = select_best_runs(metrics_list, key="r2", maximize=True)

# Generate plots for best runs only
for ic_type, metrics in best_runs.items():
    print(f"\nBest {ic_type}: {metrics.name} (R² = {metrics.r2:.4f})")
    
    # Error plot
    plot_error_time_series(...)
    
    # Order parameter plot
    plot_order_params(...)
```

#### 5.2.4 Aggregate Metrics CSV

**Purpose**: Summarize all test runs for statistical analysis.

**Output File**: `outputs/aggregate_metrics.csv`

**Columns**:
```csv
ic_type,name,r2,rmse_mean,e1_median,e2_median,einf_median,mass_error_mean,mass_error_max,tau,n_forecast,train_frac
gaussian,test_gauss_x5.5_y5.5_var2.25_s0,0.9234,0.0145,0.0089,0.0142,0.0567,0.0012,0.0045,None,120,0.4
gaussian,test_gauss_x9.0_y9.0_var2.25_s0,0.8901,0.0198,0.0123,0.0195,0.0789,0.0018,0.0067,15.3,120,0.4
uniform,test_uniform_s0,0.9456,0.0112,0.0067,0.0109,0.0432,0.0008,0.0029,None,120,0.4
...
```

**Generation** (`ROM_pipeline.py`):
```python
# Collect all metrics
all_metrics = []
for sample in test_samples:
    metrics = evaluate_single_simulation(sample, pod_basis, mvar_model, ...)
    all_metrics.append(metrics)

# Save to CSV
df = pd.DataFrame([m.to_dict() for m in all_metrics])
df.to_csv(output_dir / "aggregate_metrics.csv", index=False)
```

**Statistical Analysis**:
```python
import pandas as pd

df = pd.read_csv("outputs/aggregate_metrics.csv")

# Group by IC type
by_ic = df.groupby("ic_type")

print("\nMean R² by IC Type:")
print(by_ic["r2"].mean())

print("\nMedian RMSE by IC Type:")
print(by_ic["rmse_mean"].median())

print("\nFraction with τ = None (never exceeds tol):")
print(by_ic["tau"].apply(lambda x: (x.isna()).sum() / len(x)))
```

**Example Output**:
```
Mean R² by IC Type:
ic_type
gaussian       0.9012
ring           0.8845
two_cluster    0.8723
uniform        0.9367
Name: r2, dtype: float64

Fraction with τ = None:
ic_type
gaussian       0.667
ring           0.333
two_cluster    0.444
uniform        0.800
dtype: float64
```

### 5.3 Density Heatmap Comparisons

**Purpose**: Visually compare predicted vs ground truth density fields.

**Generated Plots** (via external scripts):
```
outputs/
├── gaussian/
│   ├── best_heatmap_t10.0.png      # Snapshot at t=10s
│   ├── best_heatmap_t15.0.png
│   └── best_heatmap_t20.0.png
└── ring/
    └── best_heatmap_t12.0.png
```

**Each Heatmap Shows**:
```
┌────────────────────────────────────────────────────────────┐
│  Density Comparison at t = 12.0s (IC: ring)               │
├─────────────────────────┬──────────────────────────────────┤
│  Ground Truth ρ_true    │  Prediction ρ_pred               │
│  [Heatmap: 64×64]       │  [Heatmap: 64×64]                │
│  [Colorbar: 0-2]        │  [Colorbar: 0-2]                 │
├─────────────────────────┴──────────────────────────────────┤
│  Absolute Error |ρ_pred - ρ_true|                         │
│  [Heatmap: 64×64]                                          │
│  [Colorbar: 0-0.5]                                         │
│  [Text: RMSE = 0.0234, Mass error = 0.0012]               │
└────────────────────────────────────────────────────────────┘
```

**Script**: `scripts/make_heatmap.py` (custom visualization tool)

---

## 6. Dataset Statistics

### 6.1 Training Dataset Summary

**Total Simulations**: 408

| IC Type | Count | Parameter Space | Total Snapshots |
|---------|-------|-----------------|-----------------|
| Gaussian | 108 | 3 centers × 3 centers × 4 variances × 3 samples | 8,640 |
| Uniform | 100 | 100 random seeds | 8,000 |
| Ring | 100 | 4 radii × 25 samples | 8,000 |
| Two-Cluster | 100 | 4 separations × 25 samples | 8,000 |
| **Total** | **408** | — | **32,640** |

**Spatial Resolution**: 64 × 64 = 4,096 grid cells

**POD Data Matrix**: $X \in \mathbb{R}^{4096 \times 32640}$
- Rows: Spatial locations (grid cells)
- Columns: Spatiotemporal snapshots (408 runs × 80 timesteps)

**Storage Requirements** (approximate):
- Dense numpy array: 4096 × 32640 × 8 bytes = 1.07 GB
- Compressed (npz): ~400 MB
- POD basis (4096 × 35): ~1.1 MB

### 6.2 Test Dataset Summary

**Total Simulations**: 31

| IC Type | Count | Parameter Space | Test Horizon |
|---------|-------|-----------------|--------------|
| Gaussian | 9 | 3 × 3 interpolation centers × 1 variance | 20s (200 steps) |
| Uniform | 10 | 10 random seeds | 20s |
| Ring | 3 | 3 interpolation radii | 20s |
| Two-Cluster | 9 | 3 interpolation separations × 3 samples | 20s |
| **Total** | **31** | — | **Forecast: 12s** |

**Forecast Window**: $t \in [8\text{s}, 20\text{s}]$ (120 timesteps)

**Total Forecast Snapshots**: 31 runs × 120 steps = 3,720 predictions

### 6.3 Interpolation vs Extrapolation

**Training Parameter Ranges** (for reference):

| IC Type | Parameter | Training Range | Test Range | Type |
|---------|-----------|----------------|------------|------|
| Gaussian | Center X | [3.75, 7.5, 11.25] | [5.5, 9.0, **2.0**] | Interp + **Extrap** |
| Gaussian | Variance | [0.25, 1.0, 4.0, 9.0] | [2.25] | Interpolation |
| Ring | Radius | [2.0, 3.0, 4.0, 5.0] | [2.5, 3.5, 4.5] | Interpolation |
| Ring | Width | [0.3, 0.6] | [0.45] | Interpolation |
| Two-Cluster | Separation | [3.0, 4.5, 6.0, 7.5] | [3.75, 5.25, 6.75] | Interpolation |
| Two-Cluster | Sigma | [0.8, 1.5] | [1.1] | Interpolation |

**Key Observations**:
- **Most tests are interpolation**: Parameters lie within training range
- **One extrapolation test**: Gaussian center at (2.0, 2.0) lies outside training range
- **Temporal extrapolation**: All test horizons (20s) exceed training horizon (8s)

### 6.4 POD Mode Analysis

**POD Truncation**: $d = 35$ modes (fixed, following Alvarez et al.)

**Energy Capture** (typical):
$$
\frac{\sum_{k=1}^{35} \lambda_k}{\sum_{k=1}^{n} \lambda_k} \approx 0.998 \text{ (99.8% of variance)}
$$

**Mode Structure** (qualitative):
- **Low modes** ($k = 1\text{-}5$): Global density variations, large-scale clustering
- **Mid modes** ($k = 6\text{-}20$): Localized features, cluster boundaries
- **High modes** ($k = 21\text{-}35$): Fine details, noise-like fluctuations

**Implementation Note**: The 35-mode truncation is **fixed** in production config, not determined by energy threshold. This ensures consistent latent dimension across all experiments.

---

## 7. References

### 7.1 Related Documentation

- **Domain and Boundary Conditions**: `docs/DOMAIN_AND_BOUNDARY_CONDITIONS.md`
  - Periodic boundary conditions, minimal image convention
  
- **Numerical Integration**: `docs/NUMERICAL_SIMULATION_AND_DATA_GENERATION.md`
  - Explicit Euler timestepping, stability criteria
  
- **ROM/MVAR Training**: `docs/ROM_MVAR_GUIDE.md`
  - POD basis computation, MVAR parameter estimation

### 7.2 Primary Code Files

| File | Purpose | Lines |
|------|---------|-------|
| `src/rectsim/ic.py` | Core IC generation functions | 300 |
| `src/rectsim/ic_generator.py` | Config-based IC enumeration | 300 |
| `src/rectsim/initial_conditions.py` | Legacy IC functions (two_clusters) | 280 |
| `src/rectsim/rom_eval_data.py` | Test data loading by IC type | 400 |
| `src/rectsim/rom_eval_metrics.py` | Metric computation | 300 |
| `src/rectsim/rom_eval_viz.py` | Visualization utilities | 400 |
| `ROM_pipeline.py` | Main pipeline orchestrator | 800 |

### 7.3 Configuration Files

| Config | Purpose | Train ICs | Test ICs |
|--------|---------|-----------|----------|
| `configs/alvarez_style_production.yaml` | Full production pipeline | 408 | 31 |
| `configs/gaussians_test.yaml` | Gaussian-only experiment | 72 | 9 |
| `configs/alvarez_local_test.yaml` | Quick local test | 40 | 10 |

### 7.4 Key Papers

1. **Alvarez et al. (2023)**: "Autoregressive Reduced-Order Models for Collective Behavior"
   - Inspiration for IC diversity strategy
   - Fixed 35-mode POD truncation principle
   
2. **Vicsek et al. (1995)**: "Novel Type of Phase Transition in a System of Self-Driven Particles"
   - Original Vicsek model definition
   - Polarization order parameter
   
3. **D'Orsogna et al. (2006)**: "Self-Propelled Particles with Soft-Core Interactions"
   - Gaussian cluster and ring IC precedents
   
4. **Berkooz et al. (1993)**: "The Proper Orthogonal Decomposition in the Analysis of Turbulent Flows"
   - POD theory and energy capture

### 7.5 Testing and Validation

**Test Suite**: `tests/test_ic.py` (200 lines)

**Test Coverage**:
- ✅ Uniform IC: domain bounds, reproducibility
- ✅ Gaussian IC: clustering, variance scaling
- ✅ Ring IC: radial distribution, centering
- ✅ Cluster IC: multiple clusters, particle assignment
- ✅ Invalid IC type handling

**Run Tests**:
```bash
pytest tests/test_ic.py -v
```

**Expected Output**:
```
tests/test_ic.py::TestUniformIC::test_uniform_within_bounds PASSED
tests/test_ic.py::TestGaussianIC::test_gaussian_clustered PASSED
tests/test_ic.py::TestRingIC::test_ring_shape PASSED
tests/test_ic.py::TestClusterIC::test_cluster_multiple_groups PASSED
tests/test_ic.py::TestICTypeValidation::test_all_valid_types PASSED

========================== 15 passed in 2.31s ==========================
```

---

## Summary

This document provides a complete technical reference for the four IC families used in our ROM-MVAR pipeline:

1. **Uniform**: Baseline homogeneous distribution
2. **Single Gaussian**: Concentrated clustering with varied centers and variances
3. **Ring/Annulus**: Circular arrangement with varied radii
4. **Two-Cluster Gaussian**: Bimodal distribution with varied separations

**Key Takeaways**:
- **408 training simulations** across 4 IC types provide robust dataset augmentation
- **31 held-out test simulations** enable interpolation/extrapolation generalization tests
- **12s forecast window** (from 8s warmup to 20s horizon) tests long-term prediction
- **Comprehensive metrics** (R², RMSE, mass error, τ) and **visualizations** (error plots, order parameters) enable thorough evaluation

**For Thesis Writing**:
- Cite this document for IC methodology
- Include sample visualizations (heatmaps, error plots)
- Report aggregate statistics (mean R² by IC type)
- Discuss generalization performance (interpolation success, extrapolation challenges)

---

**Document Version**: 1.0  
**Last Updated**: February 2, 2026  
**Author**: Maria  
**Status**: Complete ✓
