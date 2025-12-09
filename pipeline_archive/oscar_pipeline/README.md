# Oscar Pipeline - Training & Data Generation

This directory contains scripts for running the **training phase** on Oscar HPC cluster.

---

## 🎯 Oscar Workflow

### Phase 1: Data Generation (Batch Jobs)
Generate all simulation data in parallel:

1. **M training simulations** (100 sims, 4 IC types)
   - Trajectories: `train_{id:03d}_traj.npz`
   - Densities: `train_{id:03d}_density.npz`
   
2. **N test simulations** (20 sims, 4 IC types)
   - Trajectories: `test_{id:03d}_traj.npz`
   - Densities: `test_{id:03d}_density.npz`
   - Order parameters: `test_{id:03d}_order_params.npz`

### Phase 2: Model Training (Single Job)
Train global POD and MVAR models:

3. **Global POD** from all M training densities
   - Output: `pod_model.npz` (U, S, mean)
   
4. **MVAR model** from latent trajectories
   - Output: `mvar_model.npz` (A matrices, metadata)

---

## 📁 Oscar Output Structure

```
oscar_outputs/
├── training/
│   ├── train_000/
│   │   ├── traj.npz              # (T, N, 2) positions
│   │   ├── density.npz           # (T, ny, nx) KDE density
│   │   └── metadata.json         # IC type, seed, config
│   ├── train_001/
│   └── ...
│
├── test/
│   ├── test_000/
│   │   ├── traj.npz
│   │   ├── density.npz
│   │   ├── order_params.npz      # Polarization, speed, nematic
│   │   └── metadata.json
│   ├── test_001/
│   └── ...
│
└── models/
    ├── pod_model.npz
    │   - U: (n_space, R_POD) - POD modes
    │   - S: (R_POD,) - Singular values
    │   - mean: (n_space,) - Mean density
    │   - energy: float - Captured energy
    │   - R_POD: int - Number of modes
    │
    ├── mvar_model.npz
    │   - A_matrices: (p, R_POD, R_POD) - MVAR coefficients
    │   - p: int - Lag order
    │   - alpha: float - Ridge regularization
    │   - train_r2: float - Training R²
    │
    ├── training_metadata.json
    │   - N_train: 100
    │   - IC_distribution: {uniform: 25, ...}
    │   - Config: simulation parameters
    │
    └── index_mapping.csv
        - global_idx, run_name, ic_type, time_idx
```

---

## 🚀 Oscar Scripts

### `01_generate_training_data.sh`
**SLURM array job** to generate M training simulations in parallel

```bash
#!/bin/bash
#SBATCH --array=0-99
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --output=logs/train_%a.out

python oscar_pipeline/generate_single_sim.py \
    --mode train \
    --sim_id $SLURM_ARRAY_TASK_ID \
    --output_dir oscar_outputs/training
```

### `02_generate_test_data.sh`
**SLURM array job** to generate N test simulations in parallel

```bash
#!/bin/bash
#SBATCH --array=0-19
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --output=logs/test_%a.out

python oscar_pipeline/generate_single_sim.py \
    --mode test \
    --sim_id $SLURM_ARRAY_TASK_ID \
    --output_dir oscar_outputs/test \
    --compute_order_params
```

### `03_train_pod_mvar.sh`
**Single job** to train POD + MVAR after all data is generated

```bash
#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/training.out

python oscar_pipeline/train_pod_mvar.py \
    --train_dir oscar_outputs/training \
    --output_dir oscar_outputs/models
```

---

## 📥 Local Pipeline Usage

After Oscar jobs complete, rsync the outputs:

```bash
# Sync from Oscar
rsync -avz oscar:/path/to/oscar_outputs/ ./oscar_outputs/

# Run local prediction & visualization
python run_local_pipeline.py --oscar_dir oscar_outputs
```

The local pipeline will:
1. Load POD + MVAR models
2. Load test simulation densities
3. Run predictions
4. Compute metrics
5. Generate all visualizations (videos, plots)

**No simulation or training on local machine!**

---

## 🔄 Complete Workflow

```
[OSCAR] Generate train data (parallel) → oscar_outputs/training/
[OSCAR] Generate test data (parallel)  → oscar_outputs/test/
[OSCAR] Train POD + MVAR (single)      → oscar_outputs/models/
[LOCAL] Rsync from Oscar               → ./oscar_outputs/
[LOCAL] Predict + visualize            → outputs/pipeline_results/
```

---

## 📋 File Formats

### Trajectory NPZ
```python
{
    'positions': (T, N, 2),  # Particle positions over time
    'times': (T,),           # Time array
    'config': dict           # Simulation config
}
```

### Density NPZ
```python
{
    'rho': (T, ny, nx),     # KDE density field
    'times': (T,),          # Time array
    'x_edges': (nx+1,),     # Grid edges
    'y_edges': (ny+1,),     # Grid edges
    'extent': [xmin, xmax, ymin, ymax]
}
```

### Order Parameters NPZ
```python
{
    'polarization': (T,),   # Order parameter φ
    'mean_speed': (T,),     # Average speed
    'nematic': (T,)         # Nematic order
}
```

### POD Model NPZ
```python
{
    'U': (n_space, R_POD),  # POD modes
    'S': (R_POD,),          # Singular values
    'mean': (n_space,),     # Mean field
    'energy': float,        # Captured energy
    'R_POD': int,           # Number of modes
    'target_energy': float  # Target (0.995)
}
```

### MVAR Model NPZ
```python
{
    'A_matrices': (p, R_POD, R_POD),  # Coefficients
    'p': int,                         # Lag order
    'alpha': float,                   # Ridge param
    'train_r2': float,                # Training R²
    'train_rmse': float               # Training RMSE
}
```

---

## 🎛️ Configuration

All scripts use the **same base configuration**:

```python
BASE_CONFIG = {
    "sim": {
        "N": 40,
        "Lx": 15.0, "Ly": 15.0,
        "bc": "periodic",
        "T": 2.0, "dt": 0.1,
        "save_every": 1
    },
    "model": {"speed": 1.0},
    "params": {"R": 2.0},
    "noise": {"kind": "gaussian", "eta": 0.3}
}

IC_TYPES = ["uniform", "gaussian_cluster", "ring", "two_clusters"]
N_TRAIN = 100  # 25 per IC type
M_TEST = 20    # 5 per IC type
DENSITY_NX = 64, DENSITY_NY = 64, BANDWIDTH = 2.0
TARGET_ENERGY = 0.995, P_LAG = 4, RIDGE_ALPHA = 1e-6
```

---

## 🔍 Monitoring

Check Oscar job status:
```bash
# View active jobs
squeue -u $USER

# Check array job status
sacct -j JOBID --format=JobID,State,ExitCode

# View logs
tail -f logs/train_*.out
tail -f logs/test_*.out
tail -f logs/training.out
```

---

## ✅ Validation

Before running local pipeline, verify Oscar outputs:

```bash
# Check all training sims generated
ls oscar_outputs/training/train_*/traj.npz | wc -l
# Expected: 100

# Check all test sims generated
ls oscar_outputs/test/test_*/traj.npz | wc -l
# Expected: 20

# Check models trained
ls oscar_outputs/models/*.npz
# Expected: pod_model.npz, mvar_model.npz
```

---

## 📝 Notes

- **Batch parallelization** makes Oscar runs fast (all sims in parallel)
- **Standardized NPZ format** ensures compatibility with local pipeline
- **No visualization on Oscar** - saves compute time and storage
- **Local pipeline unchanged** - uses exact same prediction/plotting functions
- **Clean separation** - Oscar = data generation, Local = analysis/viz
