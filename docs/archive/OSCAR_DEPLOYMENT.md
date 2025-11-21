# Oscar Deployment Guide - ROM/MVAR Pipeline

**Date**: November 16, 2025  
**Config**: `configs/vicsek_morse_base.yaml`  
**Model**: Discrete Vicsek with Morse forces (variable speed mode)  
**Estimated Runtime**: ~6-8 minutes total (50 training runs + POD + MVAR)

---

## Pre-Flight Checklist ✅

- [x] Config validated: `vicsek_discrete` with forces
- [x] 50 ensemble runs configured
- [x] Discrete integrator: `euler_semiimplicit`
- [x] Oscar-optimized: No videos/plots during training
- [x] SLURM scripts executable
- [x] Output directories will auto-create

---

## Step 1: Push to GitHub

```bash
cd /Users/maria_1/Desktop/wsindy-manifold

# Stage all changes
git add configs/vicsek_morse_base.yaml
git add OSCAR_DEPLOYMENT.md

# Commit
git commit -m "Configure discrete Vicsek-Morse model for ROM/MVAR training (50 runs)"

# Push to GitHub
git push origin main
```

---

## Step 2: SSH to Oscar

```bash
ssh <your_username>@ssh.ccv.brown.edu
```

---

## Step 3: Pull Latest Code on Oscar

```bash
# Navigate to your workspace
cd ~/data/<your_directory>/wsindy-manifold

# Pull latest changes
git pull origin main

# Verify config is present
cat configs/vicsek_morse_base.yaml | head -25
```

---

## Step 4: Activate Conda Environment

```bash
# Load miniconda module
module load miniconda3/23.11.0s

# Activate environment
conda activate wsindy

# Verify Python and packages
python -c "import rectsim; print('✅ rectsim loaded')"
```

---

## Step 5: Submit Training Job (Quick Test)

### Option A: Training Only (Recommended First)

```bash
# Create logs directory
mkdir -p logs

# Submit training job
sbatch scripts/slurm/job_mvar_train.sh configs/vicsek_morse_base.yaml

# Check queue
squeue -u $USER

# Monitor output (replace JOBID with actual job ID)
tail -f logs/rom_train_JOBID.log
```

**Expected Output Structure**:
```
rom_mvar/
└── vicsek_morse_base/
    └── model/
        ├── pod_basis.npz          # Global POD basis (nx*ny × r)
        ├── mvar_params.npz        # MVAR coefficient matrices
        └── train_summary.json     # Training metadata
```

---

## Step 6: Submit Full Pipeline (Train → Eval)

After verifying training works, run the full pipeline with job dependencies:

```bash
# Submit both training and evaluation (eval waits for training)
bash scripts/slurm/submit_mvar_pipeline.sh configs/vicsek_morse_base.yaml

# This will output:
#   Train job ID: 12345678
#   Eval job ID:  12345679 (depends on 12345678)

# Check both jobs
squeue -u $USER

# Monitor training
tail -f logs/rom_train_12345678.log

# After training completes, monitor evaluation
tail -f logs/rom_eval_12345679.log
```

**Expected Output Structure**:
```
rom_mvar/
└── vicsek_morse_base/
    ├── model/                     # Training artifacts
    │   ├── pod_basis.npz
    │   ├── mvar_params.npz
    │   └── train_summary.json
    ├── test_ics/                  # Evaluation results
    │   ├── ic_0000/
    │   │   ├── density_true.npz
    │   │   ├── density_pred.npz
    │   │   └── metrics.json
    │   └── ic_0001/
    │       └── ...
    └── aggregate_metrics/         # Cross-IC statistics
        ├── metrics_summary.csv
        └── aggregate_stats.json
```

---

## Step 7: Monitor Jobs

```bash
# Check job status
squeue -u $USER

# Check detailed job info
scontrol show job JOBID

# View logs
tail -f logs/rom_train_JOBID.log
tail -f logs/rom_eval_JOBID.log

# Cancel job if needed
scancel JOBID
```

---

## Step 8: Download Results to Local Machine

Once jobs complete, download results for visualization:

```bash
# On your local machine (not Oscar!)
cd /Users/maria_1/Desktop/wsindy-manifold

# Download ROM results (excluding large simulation files)
rsync -avz --progress \
    <your_username>@ssh.ccv.brown.edu:~/data/<your_directory>/wsindy-manifold/rom_mvar/ \
    ./rom_mvar/

# Check what was downloaded
ls -lh rom_mvar/vicsek_morse_base/
```

---

## Step 9: Generate Visualizations Locally

```bash
# On your local machine
cd /Users/maria_1/Desktop/wsindy-manifold

# Activate local conda environment
conda activate wsindy

# Generate videos and plots
python scripts/rom_mvar_visualize.py \
    --experiment vicsek_morse_base \
    --config configs/vicsek_morse_base.yaml \
    --test-ics 0 1

# Results will be in rom_mvar/vicsek_morse_base/test_ics/ic_XXXX/
#   - density_comparison.mp4
#   - error_dashboard.png
```

---

## Troubleshooting

### Job Fails Immediately
```bash
# Check error log
cat logs/rom_train_JOBID.err

# Common issues:
#   - Config path wrong: Check path is relative to repo root
#   - Conda env not activated: Check module load in SLURM script
#   - Permission denied: chmod +x scripts/slurm/*.sh
```

### Out of Memory
```bash
# For 200 particles, 64x64 grid, 50 runs:
#   - Training needs ~16-32GB (current: 32GB ✅)
#   - Evaluation needs ~8-16GB (current: 16GB ✅)

# If still OOM, reduce grid resolution:
sbatch scripts/slurm/job_mvar_train.sh configs/vicsek_morse_base.yaml \
    outputs.grid_density.nx=32 outputs.grid_density.ny=32
```

### Runtime Too Long
```bash
# Current estimate: ~6-8 minutes for training
# If exceeds 4 hours (unlikely), reduce ensemble:
sbatch scripts/slurm/job_mvar_train.sh configs/vicsek_morse_base.yaml \
    ensemble.n_runs=20 rom.num_train_ics=20
```

---

## Quick Reference Commands

```bash
# Submit training only
sbatch scripts/slurm/job_mvar_train.sh configs/vicsek_morse_base.yaml

# Submit full pipeline
bash scripts/slurm/submit_mvar_pipeline.sh configs/vicsek_morse_base.yaml

# Check jobs
squeue -u $USER

# Monitor log
tail -f logs/rom_train_*.log

# Cancel job
scancel JOBID

# Download results
rsync -avz <user>@ssh.ccv.brown.edu:~/data/path/wsindy-manifold/rom_mvar/ ./rom_mvar/

# Visualize locally
python scripts/rom_mvar_visualize.py --experiment vicsek_morse_base --test-ics 0 1
```

---

## Expected Timeline

1. **Push to GitHub**: ~30 seconds
2. **SSH + Pull on Oscar**: ~1 minute
3. **Submit jobs**: ~10 seconds
4. **Queue wait**: 0-10 minutes (depends on cluster load)
5. **Training execution**: ~6-8 minutes
6. **Evaluation execution**: ~2-3 minutes
7. **Download results**: ~1-2 minutes (small files, no videos)
8. **Generate visualizations**: ~2-3 minutes per IC

**Total**: ~15-30 minutes from submission to visualizations

---

## Success Indicators

✅ **Training Complete**:
- Log shows: "Training complete! Check model/ directory"
- Files exist: `pod_basis.npz`, `mvar_params.npz`, `train_summary.json`
- Summary JSON shows: `num_train_ics: 50`, `latent_dim: ~10-20`

✅ **Evaluation Complete**:
- Log shows: "Evaluation complete! Results saved to test_ics/"
- Directories exist: `test_ics/ic_0000/`, `test_ics/ic_0001/`
- Metrics JSON shows: `r2 > 0.8`, `tau > 50` (good predictions)

✅ **Visualization Complete**:
- Files exist: `density_comparison.mp4`, `error_dashboard.png`
- Video shows true vs predicted density side-by-side
- Dashboard shows error metrics over time with tolerance horizon τ

---

**Ready to deploy!** 🚀
