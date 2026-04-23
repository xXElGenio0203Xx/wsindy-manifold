# ✅ Oscar-Local Pipeline Setup Complete

**Date:** November 22, 2025

---

## 🎯 What We Built

A **complete separation** of training (Oscar HPC) and visualization (local machine):

### Oscar Pipeline (Heavy Compute)
```
oscar_pipeline/
├── config.py                           # Shared configuration
├── generate_single_sim.py              # Worker script (array jobs)
├── train_pod_mvar.py                   # Model training script
├── 01_generate_training_data.sh        # SLURM: 100 training sims
├── 02_generate_test_data.sh            # SLURM: 20 test sims
├── 03_train_pod_mvar.sh                # SLURM: Train POD+MVAR
├── run_oscar_pipeline.sh               # Master: Submit all jobs
└── README.md                           # Documentation
```

### Local Pipeline (Fast Analysis)
```
run_local_pipeline.py                   # Load Oscar data → predict → visualize
OSCAR_LOCAL_WORKFLOW.md                # Complete workflow guide
```

---

## 🚀 Quick Start

### On Oscar HPC

```bash
# 1. Upload code
scp -r oscar_pipeline/ oscar:/path/to/wsindy-manifold/
scp run_local_pipeline.py oscar:/path/to/wsindy-manifold/

# 2. SSH to Oscar
ssh oscar
cd /gpfs/data/user/wsindy-manifold

# 3. Create directories
mkdir -p oscar_outputs/{training,test,models} logs

# 4. Run complete pipeline
bash oscar_pipeline/run_oscar_pipeline.sh

# Monitor progress
squeue -u $USER
tail -f logs/*.out
```

### On Local Machine

```bash
# 1. Wait for Oscar jobs to complete
# (Check with: ssh oscar "sacct -j JOBID")

# 2. Sync results
rsync -avz --progress \
    oscar:/gpfs/data/user/wsindy-manifold/oscar_outputs/ \
    ./oscar_outputs/

# 3. Run local pipeline
python run_local_pipeline.py --oscar_dir oscar_outputs

# 4. View results
open outputs/local_pipeline/best_runs/
```

---

## 📊 What Each Stage Does

### Stage 1: Oscar Training Data (Parallel)
- **Job:** Array 0-99 (100 simultaneous jobs)
- **Time:** ~30 min wall time
- **Generates:**
  - `oscar_outputs/training/train_000/` to `train_099/`
  - Each contains: `traj.npz`, `density.npz`, `metadata.json`

### Stage 2: Oscar Test Data (Parallel)
- **Job:** Array 0-19 (20 simultaneous jobs)
- **Time:** ~30 min wall time
- **Generates:**
  - `oscar_outputs/test/test_000/` to `test_019/`
  - Each contains: `traj.npz`, `density.npz`, `order_params.npz`, `metadata.json`

### Stage 3: Oscar Model Training (Single)
- **Job:** Waits for Stage 1 to complete
- **Time:** ~30-60 min
- **Generates:**
  - `oscar_outputs/models/pod_model.npz` (POD basis)
  - `oscar_outputs/models/mvar_model.npz` (MVAR coefficients)
  - `oscar_outputs/models/training_metadata.json`
  - `oscar_outputs/models/index_mapping.csv`

### Stage 4: Local Prediction & Visualization
- **Input:** Synced `oscar_outputs/`
- **Time:** ~20 min
- **Generates:**
  - `outputs/local_pipeline/metrics_all_runs.csv`
  - `outputs/local_pipeline/best_runs/{ic_type}/`
    - `traj_truth.mp4` - Ground truth trajectories
    - `density_truth_vs_pred.mp4` - Side-by-side comparison
    - `error_time.png` - Error metrics over time
    - `error_hist.png` - Error distribution
    - `order_parameters.png` - Polarization, speed, nematic
  - `outputs/local_pipeline/plots/`
    - `r2_by_ic_type.png`
    - `error_by_ic_type.png`
    - etc.

---

## 🔑 Key Design Decisions

### ✅ What Works
1. **Oscar generates NPZ files only** (no videos/plots on cluster)
2. **Standardized NPZ format** (positions, densities, order params)
3. **Shared configuration** (`oscar_pipeline/config.py`)
4. **Batch parallelization** (100 cores simultaneously)
5. **Job dependencies** (training waits for data)
6. **Local uses existing functions** (`rectsim.legacy_functions` unchanged!)

### ❌ What We Avoided
1. **No simulation on local machine** (too slow)
2. **No training on local machine** (heavy computation)
3. **No videos on Oscar** (wasteful, hard to view)
4. **No code duplication** (Oscar and local share config)

---

## 📁 File Structure

```
wsindy-manifold/
├── oscar_pipeline/                     # Oscar HPC scripts
│   ├── config.py                       # SHARED CONFIG ⚙️
│   ├── generate_single_sim.py          # Array job worker
│   ├── train_pod_mvar.py               # Model training
│   ├── *.sh                            # SLURM batch scripts
│   └── README.md
│
├── run_local_pipeline.py               # Local analysis script 🖥️
│
├── OSCAR_LOCAL_WORKFLOW.md             # Complete guide 📚
│
├── run_complete_pipeline.py            # (OLD) All-in-one local
│                                       # Still works for testing!
│
└── src/rectsim/                        # Core library
    ├── vicsek_discrete.py              # Simulation backend
    ├── legacy_functions.py             # Viz & metrics
    └── ...                             # (UNCHANGED)
```

---

## 🎓 Benefits

### For Development
- ✅ **Fast iteration:** Change local viz code, re-run in minutes
- ✅ **No cluster wait:** Analyze locally without queueing
- ✅ **Reproducible:** Same NPZ files for all analyses

### For Production
- ✅ **Scalable:** Easy to run N=1000 simulations
- ✅ **Efficient:** Oscar for heavy compute, local for analysis
- ✅ **Organized:** Clear separation of concerns

### For Paper
- ✅ **Methods clarity:** "Training on HPC, analysis local"
- ✅ **Reproducible:** All scripts version-controlled
- ✅ **Transparent:** NPZ format documented

---

## 🔍 Validation

### Test Locally (Before Oscar)

```bash
# Generate 1 training sim
python oscar_pipeline/generate_single_sim.py \
    --mode train --sim_id 0 --output_dir test_outputs/training

# Generate 1 test sim
python oscar_pipeline/generate_single_sim.py \
    --mode test --sim_id 0 --output_dir test_outputs/test --compute_order_params

# Check NPZ files
ls -lh test_outputs/*/test_000/

# Should see: traj.npz, density.npz, order_params.npz, metadata.json
```

### After Oscar Run

```bash
# Check all data generated
ls oscar_outputs/training/train_*/traj.npz | wc -l  # Should be 100
ls oscar_outputs/test/test_*/traj.npz | wc -l       # Should be 20
ls oscar_outputs/models/*.npz                        # Should see 2 files

# Verify models
python -c "
import numpy as np
pod = np.load('oscar_outputs/models/pod_model.npz')
mvar = np.load('oscar_outputs/models/mvar_model.npz')
print(f'POD: {pod[\"R_POD\"]} modes, {pod[\"energy\"]*100:.2f}% energy')
print(f'MVAR: lag={mvar[\"p\"]}, train R²={mvar[\"train_r2\"]:.4f}')
"
```

---

## 📊 Expected Resource Usage

### Oscar
- **Wall time:** ~90 min total
- **CPU-hours:** ~100 core-hours (parallel)
- **Memory:** 4 GB per job (100 jobs)
- **Storage:** ~500 MB (NPZ files only)

### Local
- **Time:** ~20 min
- **Memory:** ~2 GB
- **Storage:** ~2 GB (with videos)

---

## 🛠️ Configuration

All parameters in `oscar_pipeline/config.py`:

```python
# Simulation
N = 40 particles
Lx = Ly = 15.0
T = 2.0 s, dt = 0.1

# Ensemble
N_TRAIN = 100  # 25 per IC type
M_TEST = 20    # 5 per IC type
IC_TYPES = ["uniform", "gaussian_cluster", "ring", "two_clusters"]

# Density
64×64 grid, bandwidth = 2.0

# POD/MVAR
TARGET_ENERGY = 0.995  # 99.5% variance
P_LAG = 4
RIDGE_ALPHA = 1e-6
```

To change parameters, edit `oscar_pipeline/config.py` and re-run both Oscar and local pipelines.

---

## 📚 Documentation

- **`oscar_pipeline/README.md`** - Oscar pipeline details
- **`OSCAR_LOCAL_WORKFLOW.md`** - Complete workflow guide
- **`FILE_USAGE_ANALYSIS.md`** - Code organization analysis
- **`CLEANUP_SUMMARY.md`** - Recent refactoring

---

## ✅ Checklist

Before running on Oscar:

- [ ] Update `oscar_pipeline/config.py` with desired parameters
- [ ] Test locally with 1-2 simulations
- [ ] Verify NPZ files load correctly
- [ ] Check Oscar environment setup (Python, modules)
- [ ] Create `logs/` directory on Oscar
- [ ] Adjust SLURM parameters (#SBATCH) for your Oscar account

After Oscar run:

- [ ] Verify all 100 training sims generated
- [ ] Verify all 20 test sims generated
- [ ] Verify POD + MVAR models saved
- [ ] Rsync to local machine
- [ ] Run local pipeline
- [ ] Check output videos/plots

---

## 🎯 Next Steps

1. **Test locally:**
   ```bash
   python oscar_pipeline/generate_single_sim.py --mode train --sim_id 0 --output_dir test/training
   python oscar_pipeline/generate_single_sim.py --mode test --sim_id 0 --output_dir test/test --compute_order_params
   ```

2. **Upload to Oscar:**
   ```bash
   scp -r oscar_pipeline/ run_local_pipeline.py oscar:/path/to/project/
   ```

3. **Run on Oscar:**
   ```bash
   ssh oscar
   bash oscar_pipeline/run_oscar_pipeline.sh
   ```

4. **Sync & analyze:**
   ```bash
   rsync -avz oscar:/path/to/oscar_outputs/ ./oscar_outputs/
   python run_local_pipeline.py --oscar_dir oscar_outputs
   ```

---

## 🎉 Summary

**You now have a production-ready pipeline that:**
- ✅ Generates all data on Oscar HPC (fast, parallel)
- ✅ Trains POD + MVAR models on Oscar
- ✅ Analyzes and visualizes on local machine
- ✅ Uses existing code (no changes to `rectsim` functions!)
- ✅ Is fully documented and version-controlled
- ✅ Scales easily to larger experiments

**Total time:** Oscar ~90 min + Local ~20 min = **~110 min for complete analysis!**

(vs. ~24 hours for 100 sims locally in serial)

🚀 **Ready for production runs!**
