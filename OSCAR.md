# Oscar Cluster Guide

**Brown University's Oscar HPC Cluster**  
**Last Updated:** November 21, 2025

Complete guide for running wsindy-manifold simulations and ROM/MVAR pipelines on Oscar.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [SSH Setup](#ssh-setup)
3. [Environment Setup](#environment-setup)
4. [Interactive Testing](#interactive-testing)
5. [SLURM Job Submission](#slurm-job-submission)
6. [ROM/MVAR Pipeline](#rommvar-pipeline)
7. [SLURM Arrays (Speed-Up)](#slurm-arrays-speed-up)
8. [Git Sync Workflow](#git-sync-workflow)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

### First-Time Setup (One-Time)

1. **SSH to Oscar:**
   ```bash
   ssh your_username@ssh.ccv.brown.edu
   ```

2. **Clone repository:**
   ```bash
   cd ~/data/your_directory
   git clone https://github.com/xXElGenio0203Xx/wsindy-manifold.git
   cd wsindy-manifold
   ```

3. **Set up environment:**
   ```bash
   chmod +x setup_oscar_env.sh
   source setup_oscar_env.sh
   ```

4. **Verify installation:**
   ```bash
   python -c "from src.rectsim import config; print('✅ rectsim loaded')"
   ```

### Daily Workflow

1. Connect via VS Code Remote-SSH
2. Open folder: `/users/your_username/data/wsindy-manifold`
3. Activate environment: `source setup_oscar_env.sh`
4. Submit jobs: `sbatch run_rectsim_single.slurm`

---

## SSH Setup

### SSH Keys

If not already configured:

```bash
# On your Mac
ssh-keygen -t ed25519 -C "your_email@brown.edu"
ssh-copy-id your_username@ssh.ccv.brown.edu
```

### SSH Config

Add to `~/.ssh/config` on your Mac:

```
# Oscar HPC
Host oscar
    HostName ssh.ccv.brown.edu
    User your_username
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

**Usage:**
```bash
ssh oscar  # Quick connection
```

### VS Code Remote-SSH

1. Install "Remote - SSH" extension
2. Press `Cmd+Shift+P`
3. Select `Remote-SSH: Connect to Host...`
4. Choose `oscar`
5. Approve Duo 2FA
6. Open folder: `/users/your_username/data/wsindy-manifold`

---

## Environment Setup

### setup_oscar_env.sh

Automatically loads miniconda and activates environment:

```bash
#!/bin/bash
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate wsindy
```

**Usage:**
```bash
source setup_oscar_env.sh
```

### Manual Setup

```bash
# Load module
module load miniconda3/23.11.0s

# Activate conda
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

# Activate environment
conda activate wsindy

# Verify
python -c "import rectsim; print('✅ OK')"
```

### Installing Dependencies

```bash
# Activate environment first
source setup_oscar_env.sh

# Install requirements
pip install -r requirements.txt

# Or install individual packages
pip install numpy scipy matplotlib pyyaml tqdm
```

---

## Interactive Testing

### Quick Test Run

```bash
# Activate environment
source setup_oscar_env.sh

# Run 2-second test simulation
python -m rectsim.cli single \
  --config configs/vicsek_morse_base.yaml \
  --sim.N 50 \
  --sim.T 2.0 \
  --outputs.animate_traj false

# Check outputs
ls -lh simulations/vicsek_morse_base/
```

### Test Config Loading

```bash
python -c "
from src.rectsim.config import load_config
cfg = load_config('configs/vicsek_morse_base.yaml')
print(f'✅ Config loaded: N={cfg[\"sim\"][\"N\"]}, T={cfg[\"sim\"][\"T\"]}')
"
```

---

## SLURM Job Submission

### Basic Single Simulation

```bash
# Submit with default config
sbatch run_rectsim_single.slurm

# Submit with specific config
sbatch --export=CONFIG=configs/gentle_clustering.yaml run_rectsim_single.slurm

# Submit with parameter overrides
sbatch --export=CONFIG=configs/vicsek_morse_base.yaml,RECTSIM_ARGS="--sim.N 400 --sim.T 200" run_rectsim_single.slurm
```

### Job Monitoring

```bash
# Check queue
squeue -u $USER

# Check specific job
scontrol show job <JOB_ID>

# Cancel job
scancel <JOB_ID>

# View live output
tail -f slurm_logs/rectsim_<JOB_ID>.out

# Check for errors
tail -f slurm_logs/rectsim_<JOB_ID>.err

# Job history
sacct -u $USER --format=JobID,JobName,State,Elapsed,MaxRSS -S $(date -d '7 days ago' +%Y-%m-%d)
```

### SLURM Script Configuration

Edit `run_rectsim_single.slurm`:

```bash
#SBATCH --time=02:00:00        # Time limit
#SBATCH --cpus-per-task=4      # CPU cores
#SBATCH --mem=8G               # Memory
#SBATCH --partition=batch      # Partition
#SBATCH --output=slurm_logs/rectsim_%j.out
#SBATCH --error=slurm_logs/rectsim_%j.err
```

---

## ROM/MVAR Pipeline

### Step 1: Generate Ensemble

```bash
sbatch --export=CONFIG=configs/vicsek_morse_base.yaml scripts/slurm/job_ensemble.sh
```

**Expected outputs:**
```
simulations/vicsek_morse_base/
├── ic_000/
│   ├── trajectories.npz
│   ├── density.npz
│   └── order_parameter.csv
├── ic_001/
...
└── ic_049/
```

### Step 2: Train ROM/MVAR

```bash
sbatch scripts/slurm/job_mvar_train.sh configs/vicsek_morse_base.yaml
```

**Expected outputs:**
```
rom_mvar/vicsek_morse_base/model/
├── pod_basis.npz          # Global POD basis
├── mvar_coeffs.npz        # MVAR model coefficients
└── train_metrics.json     # Training diagnostics
```

### Step 3: Evaluate ROM

```bash
sbatch scripts/slurm/job_mvar_eval.sh configs/vicsek_morse_base.yaml
```

**Expected outputs:**
```
rom_mvar/vicsek_morse_base/eval/
├── forecast_metrics.json  # Quantitative results
├── ic_000_forecast.png    # Visualization
└── ic_000_error.png       # Error plots
```

### Complete Pipeline (One Command)

```bash
# Sequential pipeline
python scripts/run_mvar_rom_production.py --config configs/vicsek_morse_base.yaml
```

**Estimated Runtime:**
- 50 ensemble runs: ~4-5 minutes
- POD computation: ~30 seconds
- MVAR training: ~1 minute
- Evaluation: ~1 minute
- **Total: ~6-8 minutes**

---

## SLURM Arrays (Speed-Up)

### Parallel Ensemble Generation

Use SLURM job arrays to run ensemble simulations in parallel:

```bash
# Submit 50 runs in parallel (one job per IC)
sbatch --array=0-49 scripts/slurm/job_ensemble_array.sh configs/vicsek_morse_base.yaml
```

**Benefits:**
- 50x speedup for ensemble generation
- Each IC runs independently
- Automatic load balancing

**Example Array Script:**
```bash
#!/bin/bash
#SBATCH --array=0-49
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

IC_ID=$SLURM_ARRAY_TASK_ID
python -m rectsim.cli single \
  --config $1 \
  --seed $IC_ID \
  --outputs.directory simulations/ensemble/ic_$(printf "%03d" $IC_ID)
```

### Array Job Management

```bash
# Submit array
sbatch --array=0-49 job_array.sh config.yaml

# Check array jobs
squeue -u $USER --array

# Cancel specific array task
scancel <JOB_ID>_<ARRAY_INDEX>

# Cancel entire array
scancel <JOB_ID>
```

---

## Git Sync Workflow

### Local Development (Mac) → Oscar

```bash
# On your Mac
cd /Users/your_name/Desktop/wsindy-manifold

# 1. Make changes in VS Code
# 2. Test locally
python -m rectsim.cli single --config configs/test.yaml --sim.T 5

# 3. Commit and push
git add .
git commit -m "Add new feature"
git push origin main
```

### Pull Changes on Oscar

```bash
# SSH to Oscar
ssh oscar

# Navigate to repo
cd ~/data/wsindy-manifold

# Pull latest changes
git pull origin main

# Verify changes
git log --oneline -5
```

### Sync Helper Script

```bash
# sync_and_test.sh
#!/bin/bash
echo "Pulling latest changes..."
git pull origin main

echo "Activating environment..."
source setup_oscar_env.sh

echo "Testing config loading..."
python -c "from src.rectsim.config import load_config; load_config('configs/vicsek_morse_base.yaml'); print('✅ OK')"

echo "Ready to submit jobs!"
```

**Usage:**
```bash
chmod +x sync_and_test.sh
./sync_and_test.sh
```

---

## Troubleshooting

### Issue: "Module not found: rectsim"

**Solution:**
```bash
# Check PYTHONPATH
echo $PYTHONPATH

# Add repo to path
export PYTHONPATH=/users/your_username/data/wsindy-manifold:$PYTHONPATH

# Or install in dev mode
pip install -e .
```

### Issue: "Config loading fails"

**Solution:**
```bash
# Check config syntax
python -c "import yaml; yaml.safe_load(open('configs/your_config.yaml'))"

# Test loading
python -c "from src.rectsim.config import load_config; load_config('configs/your_config.yaml')"
```

### Issue: "SLURM job fails immediately"

**Solution:**
```bash
# Check error log
cat slurm_logs/rectsim_<JOB_ID>.err

# Verify script permissions
chmod +x run_rectsim_single.slurm

# Test script interactively
bash -x run_rectsim_single.slurm
```

### Issue: "Out of memory"

**Solution:**
```bash
# Increase memory in SLURM script
#SBATCH --mem=16G  # or higher

# Or reduce problem size
--sim.N 200  # fewer agents
--sim.save_every 50  # save less frequently
```

### Issue: "Job timeout"

**Solution:**
```bash
# Increase time limit
#SBATCH --time=04:00:00  # 4 hours

# Or reduce simulation time
--sim.T 100  # shorter simulation
```

### Issue: "Conda environment activation fails"

**Solution:**
```bash
# Manually initialize conda
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

# Recreate environment
conda create -n wsindy python=3.11
conda activate wsindy
pip install -r requirements.txt
```

---

## Best Practices

### Resource Requests

- **Short tests (<10 min):** `--time=00:10:00 --mem=2G --cpus-per-task=1`
- **Single simulations:** `--time=01:00:00 --mem=4G --cpus-per-task=2`
- **Ensemble runs:** `--time=02:00:00 --mem=8G --cpus-per-task=4`
- **ROM training:** `--time=00:30:00 --mem=16G --cpus-per-task=4`

### Output Management

- Keep videos/plots disabled during training (`animate_traj: false`)
- Use `save_every` to reduce output size
- Clean up old outputs regularly: `rm -rf simulations/old_experiment/`

### Job Organization

```
slurm_logs/           # Job outputs
simulations/          # Simulation results
rom_mvar/            # ROM model outputs
mvar_outputs/        # Legacy MVAR outputs
```

---

## Quick Reference

### Commands

```bash
# Setup
source setup_oscar_env.sh

# Submit job
sbatch run_rectsim_single.slurm

# Check queue
squeue -u $USER

# Monitor job
tail -f slurm_logs/rectsim_<JOB_ID>.out

# Cancel job
scancel <JOB_ID>

# Pull updates
git pull origin main

# Test config
python -m rectsim.cli single --config FILE --sim.T 5
```

### Partitions

- `batch` - General compute (default)
- `gpu` - GPU nodes
- `bigmem` - High-memory nodes

### Storage Locations

- Home: `/users/your_username/` (limited quota)
- Data: `/users/your_username/data/` (larger quota, use this!)
- Scratch: `/scratch/your_username/` (temporary, fast)

---

**For more details, see Oscar documentation: https://docs.ccv.brown.edu/oscar/**
