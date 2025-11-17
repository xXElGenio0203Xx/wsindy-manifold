# Oscar HPC Quick Start Guide: Vicsek-Morse ROM/MVAR Pipeline

## Overview

This guide shows you how to run the full ROM/MVAR pipeline on Oscar using **Slurm job arrays** for massive parallelization. Instead of 50-75 minutes sequential runtime, you get **~20-25 minutes end-to-end** with everything running in parallel.

## Key Speed-Up: Slurm Arrays

**Old approach (sequential):**
```python
for seed in range(50):
    run_simulation(seed)  # Takes 1 minute each = 50 minutes total
```

**New approach (parallel):**
```bash
sbatch --array=0-49 run_ensemble.slurm  # All 50 run simultaneously = 1-2 minutes total
```

## Setup (One-Time)

### 1. Create Oscar environment helper

On Oscar:
```bash
cd ~/src/wsindy-manifold
cp scripts/oscar_env.sh ~/
echo 'alias wsindy-oscar="source ~/oscar_env.sh"' >> ~/.bashrc
```

Now every login:
```bash
wsindy-oscar  # Loads conda, navigates to repo
```

### 2. Make scripts executable

```bash
chmod +x scripts/slurm/*.sh
```

## Running the Pipeline

### Option A: One-Command Full Pipeline (Recommended)

This submits all three stages with automatic job dependencies:

```bash
cd ~/src/wsindy-manifold
bash scripts/slurm/submit_vicsek_morse_pipeline.sh
```

**What it does:**
1. **Ensemble (1-2 min):** Runs 50 simulations in parallel (array job)
2. **ROM Training (5-10 min):** Global POD + MVAR fitting (waits for ensemble)
3. **Evaluation (5 min):** 2 test ICs + forecasting (waits for training)

**Total runtime:** ~20-25 minutes

**Monitor progress:**
```bash
squeue -u $USER                      # Check job status
tail -f logs/vicsek_*.out            # Watch live output
```

### Option B: Step-by-Step (Manual Control)

If you want to run stages separately:

#### Step 1: Ensemble Generation (50 parallel simulations)

```bash
sbatch scripts/slurm/run_vicsek_morse_ensemble.slurm
```

**Output:** `simulations/vicsek_morse_base/run_*/density.npz`

**Monitor array job:**
```bash
squeue -u $USER                               # Shows all 50 tasks
sacct -j <job_id> --format=JobID,State,Elapsed  # Detailed status
```

#### Step 2: ROM/MVAR Training

After ensemble completes:

```bash
sbatch scripts/slurm/run_vicsek_morse_rom.slurm
```

**Or with automatic dependency:**
```bash
jid=$(sbatch --parsable scripts/slurm/run_vicsek_morse_ensemble.slurm)
sbatch --dependency=afterok:${jid} scripts/slurm/run_vicsek_morse_rom.slurm
```

**Output:** `rom_mvar/vicsek_morse_base/model/pod_basis.npz`, `mvar_params.npz`

#### Step 3: ROM/MVAR Evaluation

After training completes:

```bash
sbatch scripts/slurm/run_vicsek_morse_eval.slurm
```

**Output:** `rom_mvar/vicsek_morse_base/test_ics/ic_*/metrics.json`

## Resource Allocation Explained

### Ensemble (`run_vicsek_morse_ensemble.slurm`)
- `--array=0-49`: 50 tasks run in parallel
- `--cpus-per-task=1`: Each simulation uses 1 CPU
- `--mem=4G`: 4GB RAM per simulation
- `OMP_NUM_THREADS=1`: Prevent BLAS oversubscription

**Why this works:** Simulations are independent, so Oscar can run many simultaneously on different nodes.

### ROM Training (`run_vicsek_morse_rom.slurm`)
- `--cpus-per-task=4`: 4 CPUs for POD/MVAR linear algebra
- `--mem=16G`: Enough for SVD on 50 × 3000 snapshots
- `OMP_NUM_THREADS=4`: Let BLAS use all 4 cores

**Why this works:** POD (SVD) and MVAR (matrix solve) benefit from multi-threaded BLAS.

### Evaluation (`run_vicsek_morse_eval.slurm`)
- `--cpus-per-task=2`: Light linear algebra
- `--mem=8G`: Smaller than training

## After Completion: Get Results

### On Oscar (check output)
```bash
cd ~/src/wsindy-manifold

# Check model artifacts
ls -lh rom_mvar/vicsek_morse_base/model/
cat rom_mvar/vicsek_morse_base/model/train_summary.json

# Check evaluation metrics
cat rom_mvar/vicsek_morse_base/aggregate_metrics/summary.json
```

### On Local Machine (visualization)

```bash
# Sync results from Oscar
rsync -avz emaciaso@ssh.ccv.brown.edu:~/src/wsindy-manifold/rom_mvar/ ./rom_mvar/

# Generate videos and plots
python scripts/rom_mvar_visualize.py \
  --experiment vicsek_morse_base \
  --test-ics 0 1

# View results
open rom_mvar/vicsek_morse_base/test_ics/ic_1000/density_comparison.mp4
```

**Why visualize locally?**
- Video encoding with ffmpeg is easier on local machine
- Avoids I/O load on Oscar filesystem
- Faster iteration for making figures

## Troubleshooting

### Job pending forever
```bash
squeue -u $USER  # Check job status

# If pending, check reason:
scontrol show job <job_id> | grep Reason
```

Common reasons:
- `Priority`: Normal queue behavior, just wait
- `Resources`: Cluster busy, job will start when nodes available
- `QOSMaxJobsPerUserLimit`: Too many jobs, wait for some to finish

### Array job failures

Check which tasks failed:
```bash
sacct -j <job_id> --format=JobID,State,ExitCode

# View logs for specific failed task
cat logs/vicsek_<job_id>_<task_id>.err
```

### Out of memory

If jobs fail with OOM (out of memory):
- Increase `--mem` in SLURM script
- Or reduce grid resolution: `outputs.grid_density.nx=32`

### Timeout

If jobs hit time limit:
- Increase `--time` in SLURM script
- Or reduce simulation time: `sim.T=200`

## Best Practices

### ✅ DO:
- Use job arrays for ensemble generation
- Set `OMP_NUM_THREADS` to match `--cpus-per-task`
- Disable videos/plots during Oscar runs
- Use job dependencies to chain pipeline stages
- Visualize locally after rsync

### ❌ DON'T:
- Run ensemble loops on login nodes (use `sbatch`)
- Request more CPUs than you use (wastes resources)
- Generate videos on Oscar (slow I/O, finicky ffmpeg)
- Over-allocate memory (limits scheduling)

## Advanced: Custom Configurations

### Run with different parameters

Override config values at submission:

```bash
# Smaller test run (10 simulations, shorter time)
sbatch scripts/slurm/run_vicsek_morse_ensemble.slurm \
  --array=0-9 \
  --export=EXTRA_ARGS="sim.T=100"

# Different force parameters
sbatch scripts/slurm/run_vicsek_morse_ensemble.slurm \
  --export=EXTRA_ARGS="forces.params.Ca=2.0 forces.params.Cr=1.0"
```

### Modify array range

Edit the `#SBATCH --array=` line:
- `--array=0-99`: 100 simulations
- `--array=0-49:2`: 25 simulations (every other seed)
- `--array=0-9`: 10 simulations (quick test)

## Summary of Scripts

| Script | Purpose | Runtime | Resources |
|--------|---------|---------|-----------|
| `run_vicsek_morse_ensemble.slurm` | 50 parallel simulations | 1-2 min | 50×1 CPU, 4GB each |
| `run_vicsek_morse_rom.slurm` | Global POD + MVAR training | 5-10 min | 4 CPUs, 16GB |
| `run_vicsek_morse_eval.slurm` | Test ICs + forecasting | 5 min | 2 CPUs, 8GB |
| `submit_vicsek_morse_pipeline.sh` | Full pipeline with dependencies | 20-25 min | (chains above) |

## Questions?

See also:
- [Oscar Documentation](https://docs.ccv.brown.edu/oscar/)
- [Slurm Arrays Guide](https://slurm.schedmd.com/job_array.html)
- [ROM_EVAL_GUIDE.md](../ROM_EVAL_GUIDE.md) for pipeline details
