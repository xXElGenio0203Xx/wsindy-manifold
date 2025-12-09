# Oscar-Local Pipeline Architecture

**Complete separation of training (Oscar HPC) and visualization (local machine)**

---

## 🎯 Philosophy

**Oscar (HPC):** Heavy computation
- Generate all M training simulations (parallel)
- Generate all N test simulations (parallel)  
- Train global POD + MVAR models
- **Output:** NPZ files + models (no videos/plots)

**Local (laptop):** Lightweight analysis
- Load Oscar outputs (rsync)
- Run predictions using trained models
- Generate all visualizations
- **NO simulation, NO training**

---

## 📊 Complete Workflow

### Step 1: Oscar - Data Generation (Parallel)

```bash
# On Oscar HPC
cd /gpfs/data/user/wsindy-manifold

# Submit array jobs (runs in ~30 min with 100 cores)
bash oscar_pipeline/run_oscar_pipeline.sh

# This submits:
# - Job 1: 100 training sims (array 0-99)
# - Job 2: 20 test sims (array 0-19)
# - Job 3: Train POD+MVAR (waits for Job 1)
```

**Oscar Output:**
```
oscar_outputs/
├── training/
│   ├── train_000/ {traj.npz, density.npz, metadata.json}
│   ├── train_001/
│   └── ... (100 total)
├── test/
│   ├── test_000/ {traj.npz, density.npz, order_params.npz, metadata.json}
│   ├── test_001/
│   └── ... (20 total)
└── models/
    ├── pod_model.npz
    ├── mvar_model.npz
    ├── training_metadata.json
    └── index_mapping.csv
```

### Step 2: Sync to Local

```bash
# On local machine
rsync -avz --progress \
    oscar:/gpfs/data/user/wsindy-manifold/oscar_outputs/ \
    ./oscar_outputs/

# Check sync
du -sh oscar_outputs/
ls -la oscar_outputs/{training,test,models}/
```

### Step 3: Local - Prediction & Visualization

```bash
# On local machine
python run_local_pipeline.py --oscar_dir oscar_outputs

# Generates:
# - All predictions (no re-simulation!)
# - All metrics (R², errors, by IC type)
# - All videos (trajectory, density comparison)
# - All plots (errors, order params, summaries)
```

**Local Output:**
```
outputs/local_pipeline/
├── metrics_all_runs.csv
├── best_runs/
│   ├── uniform/ {traj_truth.mp4, density_truth_vs_pred.mp4, error_time.png, ...}
│   ├── gaussian_cluster/
│   ├── ring/
│   └── two_clusters/
└── plots/
    ├── r2_by_ic_type.png
    ├── error_by_ic_type.png
    └── ...
```

---

## 🔑 Key Features

### Oscar Pipeline (`oscar_pipeline/`)

**Scripts:**
1. `config.py` - Shared configuration (ensures consistency)
2. `generate_single_sim.py` - Generate one simulation (array job worker)
3. `train_pod_mvar.py` - Train POD + MVAR models
4. `01_generate_training_data.sh` - SLURM array job (100 sims)
5. `02_generate_test_data.sh` - SLURM array job (20 sims)
6. `03_train_pod_mvar.sh` - SLURM single job (training)
7. `run_oscar_pipeline.sh` - Master script (submit all)

**Features:**
- ✅ Batch parallelization (all sims run simultaneously)
- ✅ Standardized NPZ format
- ✅ Minimal I/O (no videos/plots on cluster)
- ✅ Job dependencies (training waits for data)
- ✅ Automatic logging

### Local Pipeline (`run_local_pipeline.py`)

**Features:**
- ✅ Loads Oscar NPZ files directly
- ✅ NO simulation (uses saved trajectories/densities)
- ✅ NO training (uses saved POD/MVAR models)
- ✅ Fast predictions (only matrix operations)
- ✅ Complete visualization suite
- ✅ Uses existing `rectsim.legacy_functions` (no code changes!)

---

## 📁 File Formats (NPZ Specifications)

### Trajectory (`traj.npz`)
```python
{
    'positions': (T, N, 2),  # float64, particle positions
    'times': (T,),           # float64, time array
    'config': dict           # simulation config (saved as allow_pickle)
}
```

### Density (`density.npz`)
```python
{
    'rho': (T, ny, nx),      # float64, KDE density field
    'times': (T,),           # float64, time array
    'x_edges': (nx+1,),      # float64, grid edges
    'y_edges': (ny+1,),      # float64, grid edges
    'extent': [float]*4      # [xmin, xmax, ymin, ymax]
}
```

### Order Parameters (`order_params.npz`)
```python
{
    'polarization': (T,),    # float64, order parameter φ
    'mean_speed': (T,),      # float64, average speed
    'nematic': (T,),         # float64, nematic order
    'times': (T,)            # float64, time array
}
```

### POD Model (`pod_model.npz`)
```python
{
    'U': (n_space, R_POD),   # float64, POD modes
    'S': (R_POD,),           # float64, singular values
    'mean': (n_space,),      # float64, mean density field
    'energy': float,         # captured energy fraction
    'R_POD': int,            # number of modes
    'target_energy': float   # target (0.995)
}
```

### MVAR Model (`mvar_model.npz`)
```python
{
    'A_matrices': (p, R_POD, R_POD),  # float64, MVAR coefficients
    'p': int,                         # lag order
    'r': int,                         # latent dimension
    'alpha': float,                   # ridge regularization
    'train_r2': float,                # training R²
    'train_rmse': float               # training RMSE
}
```

---

## ⚙️ Configuration (Shared)

Both Oscar and local pipelines use **identical configuration**:

```python
# From oscar_pipeline/config.py
BASE_CONFIG = {
    "sim": {"N": 40, "Lx": 15.0, "Ly": 15.0, "T": 2.0, "dt": 0.1, ...},
    "model": {"speed": 1.0},
    "params": {"R": 2.0},
    "noise": {"kind": "gaussian", "eta": 0.3},
}

IC_TYPES = ["uniform", "gaussian_cluster", "ring", "two_clusters"]
N_TRAIN = 100  # 25 per IC type
M_TEST = 20    # 5 per IC type
DENSITY_NX = 64, DENSITY_NY = 64, BANDWIDTH = 2.0
TARGET_ENERGY = 0.995, P_LAG = 4, RIDGE_ALPHA = 1e-6
```

---

## 🚀 Quick Start

### Full Workflow (Oscar → Local)

```bash
# 1. On Oscar: Generate data + train models
ssh oscar
cd /gpfs/data/user/wsindy-manifold
bash oscar_pipeline/run_oscar_pipeline.sh

# Monitor jobs
squeue -u $USER
tail -f logs/*.out

# 2. On Local: Sync results
rsync -avz oscar:/gpfs/data/user/wsindy-manifold/oscar_outputs/ ./oscar_outputs/

# 3. On Local: Predict + visualize
python run_local_pipeline.py --oscar_dir oscar_outputs

# Done! Check outputs/local_pipeline/
```

### Development/Testing (Local Only)

For testing the workflow locally (without Oscar), you can generate small datasets:

```python
# Generate small test dataset locally
python oscar_pipeline/generate_single_sim.py --mode train --sim_id 0 --output_dir test_data/training
python oscar_pipeline/generate_single_sim.py --mode test --sim_id 0 --output_dir test_data/test --compute_order_params

# Train models
python oscar_pipeline/train_pod_mvar.py --train_dir test_data/training --output_dir test_data/models

# Run local pipeline
python run_local_pipeline.py --oscar_dir test_data
```

---

## 📊 Resource Usage

### Oscar (HPC)
- **Training data generation:** 100 jobs × 30 min = ~30 min wall time (parallel)
- **Test data generation:** 20 jobs × 30 min = ~30 min wall time (parallel)
- **Model training:** 1 job × 30-60 min
- **Total wall time:** ~90 min
- **Storage:** ~500 MB (NPZ files only, no videos)

### Local (Laptop)
- **Rsync:** ~5 min (depends on network)
- **Predictions:** ~5 min (pure matrix operations)
- **Visualizations:** ~10 min (video generation)
- **Total time:** ~20 min
- **Storage:** ~2 GB (includes videos/plots)

---

## ✅ Advantages

### vs. Old Approach (everything local)
- ✅ **100× faster** data generation (parallel on 100 cores)
- ✅ **Scalable** to N=1000 simulations easily
- ✅ **Clean separation** of concerns
- ✅ **Reproducible** (same NPZ files for all analyses)

### vs. Full Oscar Pipeline (everything on Oscar)
- ✅ **Interactive visualization** on local machine
- ✅ **Fast iteration** on plot styles
- ✅ **No cluster queue wait** for analysis
- ✅ **Saves Oscar storage** (no videos on cluster)

---

## 🔍 Validation

### Check Oscar Outputs

```bash
# On Oscar (or after rsync)
cd oscar_outputs

# Check training data (should be 100)
ls training/train_*/traj.npz | wc -l

# Check test data (should be 20)
ls test/test_*/traj.npz | wc -l

# Check models
ls models/*.npz
# Should see: pod_model.npz, mvar_model.npz

# Verify NPZ files
python -c "
import numpy as np
pod = np.load('models/pod_model.npz')
print(f'POD: {pod[\"R_POD\"]} modes, {pod[\"energy\"]*100:.2f}% energy')
mvar = np.load('models/mvar_model.npz')
print(f'MVAR: lag={mvar[\"p\"]}, R²={mvar[\"train_r2\"]:.4f}')
"
```

---

## 🛠️ Troubleshooting

### Oscar Jobs Failing?
```bash
# Check logs
tail -20 logs/train_*.err
tail -20 logs/test_*.err
tail -20 logs/training.err

# Check job status
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed

# Resubmit failed jobs
sbatch --array=5,12,23 oscar_pipeline/01_generate_training_data.sh
```

### Local Pipeline Errors?
```bash
# Verify Oscar outputs synced
ls -la oscar_outputs/{training,test,models}/

# Check NPZ files are valid
python -c "import numpy as np; np.load('oscar_outputs/models/pod_model.npz')"

# Run with verbose output
python run_local_pipeline.py --oscar_dir oscar_outputs --verbose
```

---

## 📝 Notes

- **No code changes to existing functions!** Uses `rectsim.legacy_functions` as-is
- **Backwards compatible:** Old `run_complete_pipeline.py` still works for local-only runs
- **Flexible:** Can run Oscar pipeline alone, or local pipeline alone, or both
- **Scalable:** Easy to increase N_TRAIN/M_TEST for production runs
- **Clean:** Oscar = heavy compute, Local = lightweight analysis/viz

---

## 🎓 For Paper/Reproducibility

The Oscar-local split provides clear **methods section** content:

> *"Training data generation and model training were performed on the Oscar HPC cluster 
> (Brown University), utilizing 100-core parallel batch jobs for efficient simulation 
> generation. All models and simulation data were stored in standardized NPZ format for 
> reproducibility. Post-processing, prediction, and visualization were performed on a 
> local workstation using the trained models, enabling rapid iteration on analysis and 
> figure generation without cluster resource usage."*

All Oscar scripts and configurations are version-controlled and documented for full reproducibility.
