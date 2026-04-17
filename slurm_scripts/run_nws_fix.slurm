#!/bin/bash
#SBATCH --job-name=nws_fix
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH -p batch
#SBATCH --array=0-1
#SBATCH -o slurm_logs/nws_fix_%A_%a.out
#SBATCH -e slurm_logs/nws_fix_%A_%a.err

# ============================================================================
# Noise Sweep WSINDy — rerun tasks 0,1 with corrected configs
# Task 0 = gas_eta0p15, Task 1 = gas_eta0p50
# ============================================================================

cd ~/wsindy-manifold
module load python/3.11.0
source .venv/bin/activate

CONFIGS=(
  "configs/noise_sweep_ws/NDYN04_gas_eta0p15.yaml"
  "configs/noise_sweep_ws/NDYN04_gas_eta0p50.yaml"
)

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
EXPNAME=$(python3 -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('experiment_name', 'unknown'))
")

echo "============================================"
echo "Noise-sweep WSINDy fix"
echo "Task ID:     $SLURM_ARRAY_TASK_ID"
echo "Config:      $CONFIG"
echo "Experiment:  $EXPNAME"
echo "============================================"

python3 ROM_WSINDY_pipeline.py \
    --config "$CONFIG" \
    --experiment_name "$EXPNAME" \
    --wsindy-only

echo "Done: exit code $?"
