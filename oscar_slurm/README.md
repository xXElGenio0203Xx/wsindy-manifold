# Oscar SLURM Scripts

Production-ready SLURM scripts for running ROM-MVAR pipelines on Oscar.

## Active Scripts

### ✅ run_vicsek_forces_oscar.sh
**Current production script** - Vicsek + Morse forces with constant speed mode

**Configuration:**
- N=400 particles
- T=40s duration
- Uniform IC distribution
- 200 training + 50 test runs
- 16 CPUs parallel
- Runtime: ~50-60 minutes

**Usage:**
```bash
cd ~/wsindy-manifold
sbatch oscar_slurm/run_vicsek_forces_oscar.sh
```

**Key Features:**
- Uses `module load python/3.11`
- Activates `~/wsindy_env` virtual environment
- Pulls latest code from GitHub before running
- Saves pipeline_summary.json with metrics

---

### run_nice_params_production.sh
Previous production script - Nice parameters configuration

**Configuration:**
- N=200 particles
- T=30s duration
- 200 training + 50 test runs
- 16 CPUs parallel
- Runtime: ~30-40 minutes

**Results:**
- MVAR R²=0.9799 (training)
- Prediction R²=0.0804 (test) - overfitting observed

---

## Template Structure

All scripts follow this pattern:

```bash
#!/bin/bash
#SBATCH --job-name=<name>
#SBATCH --output=slurm_logs/<name>_%j.out
#SBATCH --error=slurm_logs/<name>_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=batch
#SBATCH --nodes=1

# Pull latest code
cd ~/wsindy-manifold
git pull origin main

# Load environment
module load python/3.11
source ~/wsindy_env/bin/activate

# Run pipeline
python run_data_generation.py \
    --config configs/<config>.yaml \
    --experiment_name <name> \
    --n_train 200 \
    --n_test 50
```

---

## Monitoring Commands

```bash
# Check job status
squeue -u $USER

# View logs
tail -f slurm_logs/<job_name>_*.out

# Check CPU usage (replace NODE)
ssh <NODE> "top -b -n 1 -u $USER | head -20"

# Count generated files
ls ~/wsindy-manifold/oscar_output/<experiment>/train/ | wc -l
```

---

## Output Structure

```
oscar_output/<experiment_name>/
├── train/           # 200 training simulations
├── test/            # 50 test simulations
├── mvar/            # POD basis + MVAR model
└── pipeline_summary.json  # Metrics and timing
```
