# Oscar Production Run - Quick Reference

## Configuration
- **Config**: `configs/oscar_production.yaml`
- **Particles**: N=200
- **Duration**: T=50s (1000 timesteps)
- **Training sims**: 100
- **Test sims**: 50
- **Estimated time**: 30 min - 1 hour

## Steps to Run

### 1. Connect to Oscar
```bash
ssh -X emaciaso@ssh.ccv.brown.edu
# Enter Brown password + 2FA code
```

### 2. Setup (first time only)
```bash
cd ~/wsindy-manifold
git pull
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wsindy
```

### 3. Submit Job
```bash
sbatch run_oscar_production.sh
```

### 4. Monitor Progress
```bash
# Check job status
squeue -u emaciaso

# Watch output in real-time
tail -f slurm_logs/production_*.out

# Check if job finished
ls oscar_output/oscar_production/
```

### 5. Download Results (when complete)
```bash
# On your local machine:
scp -r emaciaso@ssh.ccv.brown.edu:~/wsindy-manifold/oscar_output/oscar_production ./
```

### 6. Visualize Locally
```bash
# On your local machine:
cd ~/Desktop/wsindy-manifold
python run_visualizations.py --experiment_name oscar_production
```

## Troubleshooting

**Connection refused?**
- Wait 30 min if you failed password attempts
- Check Oscar status: support@ccv.brown.edu
- Make sure you're using Brown password (not old Oscar password)

**Job failed?**
```bash
# Check error log
cat slurm_logs/production_*.err

# Check output log
cat slurm_logs/production_*.out
```

**Need to cancel job?**
```bash
scancel JOBID
```

## What Gets Generated

**On Oscar** (`oscar_output/oscar_production/`):
- `train/` - 100 training simulations with trajectories + densities
- `test/` - 50 test simulations with trajectories + densities  
- `mvar/` - POD basis and MVAR model
- `mvar/predictions/` - ROM predictions for test cases

**After visualization locally** (`predictions/oscar_production/`):
- Videos comparing truth vs predictions
- Error plots and metrics
- POD energy spectrum
- Summary statistics

## Ready to Scale Up?

To run more simulations, just change the sbatch command:
```bash
# Edit the script or pass new arguments
python run_data_generation.py \
  --config configs/oscar_production.yaml \
  --experiment_name oscar_production_large \
  --n_train 500 \
  --n_test 100
```

Time will scale roughly linearly with number of simulations!
