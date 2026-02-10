# ROM-MVAR Production Pipeline: Import Dependency Map

**Generated:** 2 February 2026  
**Pipeline Entry Point:** `run_unified_mvar_pipeline.py`  
**Purpose:** Identify ACTIVE vs DEPRECATED modules in `src/rectsim/`

---

## Executive Summary

The production ROM-MVAR pipeline uses **10 active modules** from `src/rectsim/` and has **~28 deprecated modules** that are NOT imported by the pipeline.

### Dependency Chain Overview

```
run_unified_mvar_pipeline.py
├── config_loader.py (no internal deps)
├── ic_generator.py (no internal deps)
├── simulation_runner.py
│   ├── vicsek_discrete.py
│   │   └── domain.py
│   └── legacy_functions.py
├── pod_builder.py (no internal deps)
├── mvar_trainer.py (no internal deps)
└── test_evaluator.py
    └── standard_metrics.py
```

**Total Active Files:** 10  
**Total Deprecated Files:** ~28 (listed at end)

---

## 1. ACTIVE MODULES (Used in Production Pipeline)

### 1.1 `config_loader.py` ⭐ Level 0 Dependency

**Purpose:** Loads and parses YAML configuration files

**Functions Defined:**
- `load_config(config_path)` - Main loader, returns 9-tuple of config components

**Internal Imports:** NONE

**External Imports:**
- `yaml` (PyYAML)
- `pathlib.Path`

**Called By:** Main pipeline (Step 0: Configuration)

**Key Role:** Parses experiment YAML into structured config dictionaries for simulation, ROM, and evaluation parameters.

---

### 1.2 `ic_generator.py` ⭐ Level 0 Dependency

**Purpose:** Generates training and test initial condition configurations

**Functions Defined:**
1. `generate_training_configs(train_ic_config, base_config)` - Creates list of training run configs
2. `generate_test_configs(test_ic_config, base_config)` - Creates list of test run configs

**Internal Imports:** NONE

**External Imports:**
- `numpy`

**Called By:** Main pipeline (Steps 1 & 3: Training/Test IC Generation)

**Key Role:** Converts high-level IC specifications (Gaussian centers, ring radii, etc.) into explicit run configurations. Supports:
- Gaussian clusters (multiple centers/variances)
- Uniform distributions
- Ring structures
- Two-cluster configurations

**Output Format:** List of dicts with keys:
```python
{
    'run_id': int,
    'distribution': str,  # 'gaussian_cluster', 'uniform', 'ring', 'two_clusters'
    'ic_params': dict,
    'label': str
}
```

---

### 1.3 `simulation_runner.py` ⭐ Level 1 Dependency

**Purpose:** Parallel simulation execution for training and test runs

**Functions Defined:**
1. `simulate_single_run(args_tuple)` - Worker function for single simulation
2. `run_simulations_parallel(configs, base_config, output_dir, ...)` - Parallel orchestration

**Internal Imports:**
```python
from rectsim.vicsek_discrete import simulate_backend
from rectsim.legacy_functions import kde_density_movie
```

**External Imports:**
- `numpy`, `pandas`, `json`
- `pathlib.Path`
- `multiprocessing.Pool`, `tqdm`

**Called By:** Main pipeline (Steps 1 & 3: Run Training/Test Simulations)

**Key Role:** 
1. Distributes simulations across CPU cores
2. Calls `simulate_backend()` for particle trajectories
3. Calls `kde_density_movie()` for density field computation
4. Saves structured outputs:
   - `train_XXX/trajectory.npz` (traj, vel, times)
   - `train_XXX/density.npz` (rho, xgrid, ygrid, times)
   - `test_XXX/trajectory.npz`
   - `test_XXX/density_true.npz` (note: different filename for test)
   - `metadata.json`, `index_mapping.csv`

**Parallelization:** Uses `multiprocessing.Pool` with default `min(cpu_count(), 16)` workers

---

### 1.4 `vicsek_discrete.py` ⭐ Level 2 Dependency (via simulation_runner)

**Purpose:** Discrete-time Vicsek model simulation backend

**Functions Defined:**
1. `rotation(phi)` - 2D rotation matrix
2. `headings_from_angles(theta)` - Convert angles to unit vectors
3. `angles_from_headings(p)` - Convert unit vectors to angles
4. `compute_neighbors(x, Lx, Ly, R, bc, cell_list)` - Neighbor search
5. `_apply_noise(theta_mean, noise_cfg, rng)` - Noise application (Gaussian/uniform)
6. `step_vicsek_discrete(x, theta, v0, Lx, Ly, R, bc, noise_cfg, rng, ...)` - Single timestep
7. `simulate_vicsek(cfg)` - High-level simulation wrapper
8. `simulate_backend(config, rng)` - **MAIN ENTRY POINT** (called by simulation_runner)

**Internal Imports:**
```python
from .domain import CellList, apply_bc, build_cells, neighbor_indices_from_celllist
```

**External Imports:**
- `numpy`
- `typing` (type hints)

**Called By:** `simulation_runner.simulate_single_run()`

**Key Role:** 
- Implements discrete Vicsek model dynamics (alignment + noise)
- Returns `{"times": ndarray, "traj": (T,N,2), "vel": (T,N,2), "order_params": dict}`
- Supports periodic/reflecting BC, Gaussian/uniform noise
- Uses efficient cell-list neighbor search from `domain.py`

**Physics:** Classic Vicsek model (Phys. Rev. Lett. 75, 1226, 1995)

---

### 1.5 `domain.py` ⭐ Level 3 Dependency (via vicsek_discrete)

**Purpose:** Domain utilities for boundary conditions and neighbor searches

**Functions Defined:**
1. `apply_bc(x, Lx, Ly, bc)` - Apply periodic/reflecting boundary conditions
2. `pair_displacements(x, Lx, Ly, bc)` - Compute pairwise displacements
3. `build_cells(x, Lx, Ly, cell_size, bc)` - Build cell-list data structure
4. `iter_neighbors(...)` - Iterator for neighbor pairs
5. `neighbor_indices_from_celllist(...)` - Extract neighbor indices from cell list

**Data Structure:**
```python
@dataclass
class CellList:
    cells: Dict[Tuple[int, int], List[int]]
    cell_size: float
    ncellx: int
    ncelly: int
```

**Internal Imports:** NONE

**External Imports:**
- `numpy`
- `dataclasses`
- `typing`

**Called By:** `vicsek_discrete.py` (for neighbor searches)

**Key Role:** Low-level spatial utilities used by simulation backend. Performance-critical for large N.

---

### 1.6 `legacy_functions.py` ⭐ Level 2 Dependency (via simulation_runner)

**Purpose:** KDE density computation and visualization utilities

**Functions Defined (13 total, key ones):**
1. `kde_density_movie(traj, Lx, Ly, nx, ny, bandwidth, bc)` - **MAIN: KDE density computation**
2. `estimate_bandwidth(Lx, Ly, N, nx, ny)` - Automatic bandwidth selection
3. `polarization(vel)` - Order parameter
4. `mean_speed(vel)` - Average speed
5. `nematic_order(vel)` - Nematic order parameter
6. `compute_order_params(...)` - Batch order parameter computation
7. `save_video(...)` - Single density field video
8. `side_by_side_video(...)` - True vs predicted comparison video
9. `trajectory_video(...)` - Particle trajectory video

**Internal Imports:** NONE

**External Imports:**
- `numpy`, `scipy.ndimage.gaussian_filter`
- `matplotlib`, `matplotlib.animation.FFMpegWriter`
- `pathlib.Path`

**Called By:** `simulation_runner.simulate_single_run()`

**Key Role:**
- Converts particle trajectories `(T,N,2)` to density fields `(T,ny,nx)`
- Uses 2D histogram + Gaussian smoothing (KDE approximation)
- Returns `(rho, metadata)` where metadata includes grid info
- Handles periodic boundary conditions properly (wraps particle positions)

**Critical for:** Generating density data that POD operates on

**Note:** Called "legacy" but actively used - contains working KDE from old pipeline

---

### 1.7 `pod_builder.py` ⭐ Level 0 Dependency

**Purpose:** Constructs POD basis from training density data

**Functions Defined:**
1. `build_pod_basis(train_dir, n_train, rom_config, density_key='rho')` - Build POD basis
2. `save_pod_basis(pod_data, mvar_dir)` - Save POD basis to disk

**Internal Imports:** NONE

**External Imports:**
- `numpy`
- `pathlib.Path`

**Called By:** Main pipeline (Step 2: Build POD Basis)

**Key Role:**
1. Loads all training density data
2. Subsamples in time if `rom_config['subsample'] > 1`
3. Flattens to data matrix `X_all: (M*T, n_spatial)`
4. Centers data: `X_centered = X_all - X_mean`
5. SVD: `U, S, Vt = np.linalg.svd(X_centered.T)`
6. Determines `R_POD` modes via:
   - **Priority 1:** Fixed mode count (`fixed_modes` or `fixed_d`)
   - **Priority 2:** Energy threshold (`pod_energy` or `energy_threshold`, default 0.995)
7. Saves POD basis with standardized keys

**Output Dictionary:**
```python
{
    'U_r': (n_spatial, R_POD),  # POD basis
    'S': (n_modes,),            # All singular values
    'X_mean': (n_spatial,),     # Mean field
    'X_centered': (M*T, n_spatial),
    'X_latent': (M*T, R_POD),   # Projected training data
    'R_POD': int,
    'energy_captured': float,
    'cumulative_energy': (n_modes,),
    'total_energy': float,
    'M': int,                   # Number of training runs
    'T_rom': int                # Timesteps per run
}
```

**Saved Files:**
- `mvar/X_train_mean.npy`
- `mvar/pod_basis.npz` (U, singular_values, energy metrics)

---

### 1.8 `mvar_trainer.py` ⭐ Level 0 Dependency

**Purpose:** Trains MVAR model on POD latent space

**Functions Defined:**
1. `train_mvar_model(pod_data, rom_config)` - Train Ridge-regularized MVAR
2. `save_mvar_model(mvar_data, mvar_dir)` - Save MVAR model to disk

**Internal Imports:** NONE

**External Imports:**
- `numpy`
- `pathlib.Path`
- `sklearn.linear_model.Ridge`

**Called By:** Main pipeline (Step 2: Train MVAR Model)

**Key Role:**
1. Extracts latent data from `pod_data['X_latent']`
2. Reshapes to `(M, T_rom, R_POD)`
3. Builds autoregressive training pairs:
   ```
   X_train: (N_samples, p*d) - [x(t-p), ..., x(t-1)]
   Y_train: (N_samples, d)   - x(t)
   ```
4. Fits Ridge regression: `Y = X @ coef + intercept`
5. Computes training R² and RMSE
6. **Optional:** Eigenvalue stability enforcement (if `eigenvalue_threshold` specified)
7. Saves coefficient matrices in two formats:
   - `A_matrices: (p, d, d)` - tensor form
   - `A_companion: (d, p*d)` - flat form (sklearn coef_)

**Config Support:** Backward compatible with both:
- New: `rom.models.mvar.lag`, `rom.models.mvar.ridge_alpha`
- Old: `rom.mvar_lag`, `rom.ridge_alpha`

**Output Dictionary:**
```python
{
    'model': Ridge model object,
    'P_LAG': int,
    'RIDGE_ALPHA': float,
    'r2_train': float,
    'train_rmse': float,
    'A_matrices': (p, d, d),
    'rho_before': float,  # Spectral radius before scaling
    'rho_after': float,   # Spectral radius after scaling
    'R_POD': int
}
```

**Saved Files:**
- `mvar/mvar_model.npz` (A_matrices, p, r, alpha, metrics)

---

### 1.9 `test_evaluator.py` ⭐ Level 1 Dependency

**Purpose:** Evaluates ROM-MVAR model on test data, saves predictions and metrics

**Functions Defined:**
1. `evaluate_test_runs(test_dir, n_test, base_config_test, pod_data, forecast_fn, lag, ...)` - **MAIN**
2. `_compute_time_resolved_r2(...)` - Helper for time-resolved R² computation

**Internal Imports:**
```python
from rectsim.standard_metrics import compute_metrics_series
```

**External Imports:**
- `numpy`, `pandas`, `json`
- `pathlib.Path`
- `tqdm`

**Called By:** Main pipeline (Step 4: Evaluate Test Runs)

**Key Role:**
1. For each test run:
   - Loads `density_true.npz`
   - Projects to latent space via POD
   - Generates forecast using `forecast_fn(y_init_window, n_steps)`
   - Reconstructs to physical space
2. Computes **3 types of R²:**
   - `r2_reconstructed`: Physical space (true vs predicted density)
   - `r2_latent`: Latent space (true vs predicted coefficients)
   - `r2_pod`: POD reconstruction quality (true vs POD(latent_true))
3. Computes additional metrics:
   - RMSE (reconstructed, latent, POD)
   - Relative error
   - Mass conservation violation
4. **Optional:** Time-resolved R² evolution (if `eval_config['save_time_resolved']=True`)
5. Saves outputs:
   - `test_XXX/density_pred.npz` (rho, xgrid, ygrid, times)
   - `test_XXX/metrics_summary.json`
   - `test_XXX/r2_vs_time.csv` (if time-resolved)
   - `test_XXX/order_params.csv` (from trajectory)
   - `test_XXX/density_metrics.csv`
   - `test/test_results.csv` (aggregate)

**Generic Forecasting:** Uses function signature `forecast_fn(y_init_window, n_steps) -> ys_pred`, making it model-agnostic (works with MVAR, LSTM, etc.)

**Forecast Window:** 
- Uses last `P_LAG` timesteps from training period as initial condition
- Forecasts from `T_train` to `T_test`
- Default: `T_train = forecast_start` from config, or training trajectory length

---

### 1.10 `standard_metrics.py` ⭐ Level 2 Dependency (via test_evaluator)

**Purpose:** Standardized order parameters and metrics for all simulations

**Functions Defined:**
1. `polarization(velocities)` - Φ = ||⟨v̂ᵢ⟩||
2. `angular_momentum(positions, velocities, center=None)` - Collective rotation
3. `mean_speed(velocities)` - Average particle speed
4. `total_mass(positions, domain_bounds, resolution, ...)` - Total mass from density
5. `density_variance(positions, domain_bounds, resolution, ...)` - Spatial density variance
6. `compute_all_metrics(positions, velocities, domain_bounds, ...)` - All metrics for one frame
7. `compute_metrics_series(trajectory, velocities, domain_bounds, ...)` - **MAIN: Time series**

**Internal Imports:** NONE

**External Imports:**
- `numpy`
- `scipy.stats.gaussian_kde`

**Called By:** `test_evaluator.evaluate_test_runs()` (to compute order parameters from trajectories)

**Key Role:**
- Provides unified metric computation across different model types
- Used to extract order parameters from test trajectories
- Saved to `order_params.csv` for visualization/analysis

**Metrics Computed:**
- Polarization (alignment)
- Angular momentum (rotation)
- Mean speed
- Density variance (spatial heterogeneity)
- Total mass (conservation check)

---

## 2. DEPENDENCY GRAPH (Bottom-Up)

### Level 0 (No Internal Dependencies)
- `config_loader.py`
- `ic_generator.py`
- `pod_builder.py`
- `mvar_trainer.py`

### Level 1 (Depends on Level 0)
- `simulation_runner.py` → calls `vicsek_discrete`, `legacy_functions`
- `test_evaluator.py` → calls `standard_metrics`

### Level 2 (Depends on Level 1)
- `vicsek_discrete.py` → calls `domain`
- `legacy_functions.py` (independent)
- `standard_metrics.py` (independent)

### Level 3 (Depends on Level 2)
- `domain.py` (leaf node)

### Main Pipeline (Depends on Level 1)
- `run_unified_mvar_pipeline.py` → orchestrates all Level 0 & Level 1 modules

---

## 3. FUNCTION INVENTORY FOR THESIS DOCUMENTATION

### Core Pipeline Functions (Must Document)

#### Configuration & Setup
1. `config_loader.load_config()` - YAML → structured config
2. `ic_generator.generate_training_configs()` - Training IC specs
3. `ic_generator.generate_test_configs()` - Test IC specs

#### Simulation
4. `simulation_runner.run_simulations_parallel()` - Parallel orchestration
5. `simulation_runner.simulate_single_run()` - Worker function
6. `vicsek_discrete.simulate_backend()` - Vicsek dynamics
7. `legacy_functions.kde_density_movie()` - Trajectory → density field

#### ROM Construction
8. `pod_builder.build_pod_basis()` - SVD-based POD
9. `pod_builder.save_pod_basis()` - Persistence
10. `mvar_trainer.train_mvar_model()` - Ridge-MVAR training
11. `mvar_trainer.save_mvar_model()` - Persistence

#### Evaluation
12. `test_evaluator.evaluate_test_runs()` - Forecast & metrics
13. `standard_metrics.compute_metrics_series()` - Order parameters

### Supporting Functions (Reference as Needed)

#### Spatial & Neighbor Search
- `domain.apply_bc()` - Boundary condition enforcement
- `domain.build_cells()` - Cell-list construction
- `domain.neighbor_indices_from_celllist()` - Neighbor extraction

#### Dynamics & Noise
- `vicsek_discrete.step_vicsek_discrete()` - Single timestep update
- `vicsek_discrete._apply_noise()` - Angular noise
- `vicsek_discrete.compute_neighbors()` - Neighbor search wrapper

#### Order Parameters
- `standard_metrics.polarization()` - Alignment
- `standard_metrics.angular_momentum()` - Rotation
- `standard_metrics.density_variance()` - Spatial heterogeneity

---

## 4. DEPRECATED FILES (NOT IMPORTED BY PIPELINE)

The following 28 files exist in `src/rectsim/` but are **NOT** imported by the production pipeline:

### 4.1 Old Configuration Systems
1. `config.py` - Old config loader (replaced by `config_loader.py`)
2. `unified_config.py` - Experimental unified config (unused)

### 4.2 Old Initial Condition Generators
3. `ic.py` - Old IC generator (replaced by `ic_generator.py`)
4. `initial_conditions.py` - Alternate IC functions (duplicate of ic.py)

### 4.3 Old Simulation Backends (Pre-unification)
5. `dynamics.py` - Old dynamics module
6. `integrators.py` - RK integrators (not used in discrete Vicsek)

### 4.4 Old ROM/MVAR Implementations
7. `rom_mvar.py` - Old monolithic ROM-MVAR pipeline (723 lines, replaced by modular pipeline)
8. `rom_mvar_model.py` - Old ROM-MVAR model class
9. `mvar.py` - Old MVAR implementation (replaced by `mvar_trainer.py`)
10. `pod.py` - Old POD implementation (replaced by `pod_builder.py`)

### 4.5 Old Evaluation & Metrics
11. `rom_eval.py` - Old evaluation module
12. `rom_eval_data.py` - Old data loading utilities
13. `rom_eval_metrics.py` - Old metrics computation
14. `rom_eval_pipeline.py` - Old evaluation pipeline
15. `rom_eval_smoke_test.py` - Old smoke tests
16. `rom_eval_viz.py` - Old visualization utilities
17. `metrics.py` - Old metrics module

### 4.6 Utility Modules (Unclear Status)
18. `io.py` - Old I/O utilities
19. `io_outputs.py` - Old output utilities
20. `utils.py` - General utilities (may have scattered uses)
21. `rom_data_utils.py` - Old ROM data utilities
22. `rom_video_utils.py` - Old video utilities
23. `forecast_utils.py` - Old forecasting utilities

### 4.7 Specialized Features (Not Used in Standard Pipeline)
24. `morse.py` - Morse potential forces (not used in Vicsek)
25. `noise.py` - Noise generation utilities (duplicated in vicsek_discrete)
26. `density.py` - Density computation (replaced by legacy_functions.kde_density_movie)

### 4.8 Command-Line Interface
27. `cli.py` - CLI utilities (not used by YAML-based pipeline)

### 4.9 Python Package Metadata
28. `__init__.py` - Package initialization (may re-export functions)

---

## 5. RECOMMENDATIONS FOR THESIS

### 5.1 Architecture Diagram
Create a flowchart with 5 boxes:

```
[Config YAML]
     ↓
[IC Generation] → [Parallel Simulation] → [Density KDE]
                       ↓
              [POD Basis Construction]
                       ↓
              [MVAR Training (Ridge)]
                       ↓
        [Test Evaluation & Forecasting]
                       ↓
          [Metrics & Visualizations]
```

### 5.2 Key Code Sections to Include

**Chapter: Data Generation**
- `ic_generator.generate_training_configs()` - Show how IC diversity is created
- `simulation_runner.run_simulations_parallel()` - Explain parallel workflow
- `legacy_functions.kde_density_movie()` - Show KDE algorithm

**Chapter: ROM Construction**
- `pod_builder.build_pod_basis()` - SVD decomposition
- Show energy-based mode selection vs fixed-d

**Chapter: MVAR Model**
- `mvar_trainer.train_mvar_model()` - Ridge regression formulation
- Discuss lag selection and regularization

**Chapter: Evaluation**
- `test_evaluator.evaluate_test_runs()` - Forecast generation
- Explain 3 types of R² metrics (reconstructed, latent, POD)

### 5.3 Key Parameters to Document

| Parameter | Location | Purpose |
|-----------|----------|---------|
| `N` | sim config | Number of particles |
| `T`, `dt` | sim config | Simulation duration & timestep |
| `nx`, `ny` | outputs | Density grid resolution |
| `bandwidth` | outputs | KDE smoothing parameter |
| `subsample` | rom | Temporal downsampling |
| `fixed_modes` or `pod_energy` | rom | POD mode selection |
| `mvar_lag` (p) | rom | MVAR order |
| `ridge_alpha` (α) | rom | Regularization strength |
| `eigenvalue_threshold` | rom | Stability enforcement |

### 5.4 Key Equations to Present

**POD Decomposition:**
```
X_centered = X_all - X_mean
U, S, V = SVD(X_centered^T)
x̃(t) ≈ X_mean + U_r @ a(t)
```

**MVAR Model:**
```
a(t) = ∑_{j=1}^p A_j @ a(t-j) + ε(t)
```

**Ridge Regression:**
```
min_{A} ||Y - X @ A||² + α ||A||²
```

**R² Metrics:**
```
R²_reconstructed = 1 - ||ρ_true - ρ_pred||² / ||ρ_true - mean(ρ_true)||²
R²_latent = 1 - ||a_true - a_pred||² / ||a_true - mean(a_true)||²
R²_POD = 1 - ||ρ_true - POD(a_true)||² / ||ρ_true - mean(ρ_true)||²
```

---

## 6. FILE SIZE & COMPLEXITY METRICS

| File | Lines | Functions | Complexity | Role |
|------|-------|-----------|------------|------|
| `config_loader.py` | 68 | 1 | Low | Config parsing |
| `ic_generator.py` | 276 | 2 | Medium | IC generation |
| `simulation_runner.py` | 161 | 2 | Medium | Parallel sim |
| `vicsek_discrete.py` | 715 | 8 | High | Vicsek dynamics |
| `domain.py` | 316 | 5 | Medium | Spatial utils |
| `legacy_functions.py` | 2062 | 13+ | High | KDE + viz |
| `pod_builder.py` | 158 | 2 | Low | POD SVD |
| `mvar_trainer.py` | 179 | 2 | Low | MVAR training |
| `test_evaluator.py` | 370 | 2 | High | Evaluation |
| `standard_metrics.py` | 323 | 7 | Medium | Order params |

**Total Active Code:** ~4,628 lines across 10 files

---

## 7. EXTERNAL DEPENDENCIES

### Core Scientific Stack
- `numpy` - Arrays, linear algebra
- `scipy` - KDE, Gaussian filtering
- `sklearn` - Ridge regression
- `pandas` - CSV/JSON I/O
- `yaml` - Config parsing

### Parallel & Progress
- `multiprocessing` - Parallel simulation
- `tqdm` - Progress bars

### Visualization (in legacy_functions, not core pipeline)
- `matplotlib` - Plotting
- `matplotlib.animation` - Video generation

### Standard Library
- `pathlib` - Path handling
- `json` - JSON I/O
- `time` - Timing
- `typing` - Type hints
- `dataclasses` - Data structures

---

## 8. OUTPUT FILE STRUCTURE

### Training Phase
```
oscar_output/<experiment>/
├── config_used.yaml                    # Saved config
├── train/
│   ├── metadata.json                   # All run metadata
│   ├── index_mapping.csv               # Run index table
│   ├── train_000/
│   │   ├── trajectory.npz              # (traj, vel, times)
│   │   └── density.npz                 # (rho, xgrid, ygrid, times)
│   ├── train_001/
│   └── ...
└── mvar/
    ├── X_train_mean.npy                # POD mean field
    ├── pod_basis.npz                   # POD basis + metrics
    └── mvar_model.npz                  # MVAR coefficients + metrics
```

### Test Phase
```
oscar_output/<experiment>/
└── test/
    ├── metadata.json
    ├── index_mapping.csv
    ├── test_results.csv                # Aggregate R² metrics
    ├── test_000/
    │   ├── trajectory.npz              # Ground truth trajectory
    │   ├── density_true.npz            # Ground truth density
    │   ├── density_pred.npz            # Predicted density
    │   ├── metrics_summary.json        # R², RMSE, mass violation
    │   ├── r2_vs_time.csv              # Time-resolved R² (optional)
    │   ├── order_params.csv            # Φ, L, mean_speed, etc.
    │   └── density_metrics.csv         # Spatial variance, mass
    └── test_001/
        └── ...
```

---

## 9. SUMMARY: ACTIVE vs DEPRECATED

### ✅ ACTIVE (10 files, 4,628 lines)
**Direct Pipeline Imports (6):**
1. `config_loader.py`
2. `ic_generator.py`
3. `simulation_runner.py`
4. `pod_builder.py`
5. `mvar_trainer.py`
6. `test_evaluator.py`

**Indirect Dependencies (4):**
7. `vicsek_discrete.py` (via simulation_runner)
8. `legacy_functions.py` (via simulation_runner)
9. `standard_metrics.py` (via test_evaluator)
10. `domain.py` (via vicsek_discrete)

### ❌ DEPRECATED (~28 files)
All other files in `src/rectsim/` are legacy code not used by the production pipeline. These include:
- Old config systems (config.py, unified_config.py)
- Old IC generators (ic.py, initial_conditions.py)
- Old ROM implementations (rom_mvar.py, pod.py, mvar.py)
- Old evaluation modules (rom_eval*.py, metrics.py)
- Unused utilities (io.py, rom_data_utils.py, forecast_utils.py)
- Specialized features (morse.py, noise.py, density.py)

**Recommendation:** Archive deprecated files to `src/rectsim/archive/` to clarify codebase structure.

---

## 10. CHANGE LOG & VERSION CONTROL

**Pipeline Evolution:**
- **Pre-Nov 2025:** Monolithic `rom_mvar.py` (723 lines)
- **Nov 2025:** Modular refactoring into 6 core modules
- **Current:** Unified pipeline supporting multiple IC types, flexible ROM config, optional stability enforcement

**Key Improvements:**
1. Separation of concerns (IC gen, simulation, ROM, eval)
2. Support for mixed distributions (Gaussian + uniform + ring + two-cluster)
3. Flexible POD mode selection (fixed vs energy threshold)
4. Generic forecast interface (model-agnostic evaluation)
5. Comprehensive metrics (3 types of R², mass conservation, order params)
6. Time-resolved analysis (optional R² evolution)

---

## APPENDIX: Quick Reference Commands

### Run Pipeline
```bash
python run_unified_mvar_pipeline.py \
    --config configs/oscar_production.yaml \
    --experiment_name my_experiment
```

### Check Dependencies
```bash
grep -r "^from rectsim" src/rectsim/*.py
grep -r "^import rectsim" src/rectsim/*.py
```

### Count Active Code
```bash
wc -l src/rectsim/{config_loader,ic_generator,simulation_runner,pod_builder,mvar_trainer,test_evaluator,vicsek_discrete,domain,legacy_functions,standard_metrics}.py
```

### List All Functions
```bash
grep "^def " src/rectsim/config_loader.py
# Repeat for each file
```

---

**END OF DEPENDENCY MAP**

**Author:** GitHub Copilot (Claude Sonnet 4.5)  
**Generated:** 2 February 2026  
**Verified Against:** Production pipeline `run_unified_mvar_pipeline.py` commit state
