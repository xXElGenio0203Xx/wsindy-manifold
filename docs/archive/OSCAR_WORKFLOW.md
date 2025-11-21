# Oscar Cluster Workflow Guide

## Quick Start

### 1. Connect to Oscar from VS Code
1. Press `Cmd+Shift+P`
2. Select `Remote-SSH: Connect to Host...`
3. Choose `oscar`
4. Approve Duo 2FA
5. Open folder: `/users/emaciaso/src/wsindy-manifold`

### 2. Activate Environment in Terminal
In VS Code's integrated terminal on Oscar:
```bash
source setup_oscar_env.sh
```

Or manually:
```bash
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate wsindy
```

---

## Interactive Testing

### Run a quick test interactively (short simulation)
```bash
# Activate environment first
source setup_oscar_env.sh

# Run a 2-second test
rectsim single \
  --config configs/gentle_clustering.yaml \
  --sim.N 50 \
  --sim.T 2.0 \
  --outputs.animate false
```

### Check outputs
```bash
ls -lh outputs/single/
```

---

## SLURM Job Submission

### Submit a job with default parameters
```bash
sbatch run_rectsim_single.slurm
```

### Submit with custom config
```bash
sbatch --export=CONFIG=configs/long_loose_N200_T400.yaml run_rectsim_single.slurm
```

### Submit with custom parameters
```bash
sbatch --export=CONFIG=configs/gentle_clustering.yaml,RECTSIM_ARGS="--sim.N 400 --sim.T 200.0 --params.Ca 1.5" run_rectsim_single.slurm
```

### Monitor jobs
```bash
# Check your job queue
squeue -u emaciaso

# Check job details
scontrol show job <JOB_ID>

# Cancel a job
scancel <JOB_ID>

# Check recent job history
sacct -u emaciaso --format=JobID,JobName,State,Elapsed,MaxRSS
```

### View job output (while running or after completion)
```bash
# Real-time monitoring
tail -f slurm_logs/rectsim_<JOB_ID>.out

# Check for errors
tail -f slurm_logs/rectsim_<JOB_ID>.err

# View completed job output
cat slurm_logs/rectsim_<JOB_ID>.out
```

---

## SLURM Script Parameters

Edit `run_rectsim_single.slurm` to adjust:

- **Time limit**: `#SBATCH --time=02:00:00` (HH:MM:SS)
- **CPUs**: `#SBATCH --cpus-per-task=4`
- **Memory**: `#SBATCH --mem=8G`
- **Partition**: `#SBATCH --partition=batch`
  - Options: `batch` (default), `gpu` (for GPU jobs)

---

## Common Tasks

### Install/update Python packages
```bash
source setup_oscar_env.sh
pip install -r requirements.txt
pip install -e .
```

### Check package installation
```bash
source setup_oscar_env.sh
python -c "import rectsim; print(rectsim.__file__)"
rectsim --help
```

### Clean up old outputs
```bash
rm -rf outputs/single/*
rm -rf slurm_logs/*.out slurm_logs/*.err
```

---

## Typical Workflow

1. **Edit code/configs locally in VS Code** (connected via Remote-SSH)
2. **Test interactively** with short runs:
   ```bash
   rectsim single --config configs/gentle_clustering.yaml --sim.T 2.0
   ```
3. **Submit production job** via SLURM:
   ```bash
   sbatch run_rectsim_single.slurm
   ```
4. **Monitor progress**:
   ```bash
   squeue -u emaciaso
   tail -f slurm_logs/rectsim_<JOB_ID>.out
   ```
5. **Analyze outputs**:
   ```bash
   ls -lh outputs/single/
   ```

---

## Troubleshooting

### "rectsim: command not found"
- Make sure conda environment is activated:
  ```bash
  source setup_oscar_env.sh
  which python  # Should show wsindy env path
  ```

### Job fails immediately
- Check error log: `cat slurm_logs/rectsim_<JOB_ID>.err`
- Check if environment activation failed
- Verify config file path exists

### Out of memory
- Increase `#SBATCH --mem=` in SLURM script
- Reduce `--sim.N` (number of particles)

### Job timeout
- Increase `#SBATCH --time=` in SLURM script
- Or reduce `--sim.T` (simulation time)

---

## Next Steps (Future)

### Parameter sweeps with SLURM arrays
Will create `run_rectsim_grid.slurm` for array jobs using `rectsim grid`

### EF-ROM / MVAR pipelines
Will adapt existing scripts:
- `scripts/run_mvar_rom_production.py`
- `scripts/train_latent_heatmap.py`

### Output organization
```
simulations/
  <sim_name>__<timestamp>_<hash>_<seed>/
    traj.npz
    density.npz
    ...

mvar_outputs/
  <experiment_name>/
    pod/
    model/
    eval/
```
