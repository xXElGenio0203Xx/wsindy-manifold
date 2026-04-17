#!/bin/bash
#SBATCH --job-name=t2s_ws
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -p batch
#SBATCH --array=0-5
#SBATCH -o slurm_logs/t2s_ws_%A_%a.out
#SBATCH -e slurm_logs/t2s_ws_%A_%a.err

# ============================================================================
# TIER 2 SLIM — WSINDy-only rerun with corrected configs
# ============================================================================
# Training data already on disk from previous tier2_slim runs.
# Now using production-matched params: n_ell=12, B=0, p=[2,2,2]
#
# Usage:
#   sbatch slurm_scripts/run_t2s_wsindy_only.slurm

cd ~/wsindy-manifold
module load python/3.11.0
source .venv/bin/activate

MANIFEST=configs/noise_sweep_slim/manifest.txt
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
EXPNAME=$(python3 -c "
import yaml, sys
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('experiment_name', 'unknown'))
")

echo "============================================"
echo "TIER2_SLIM WSINDy-only rerun"
echo "Task ID:     $SLURM_ARRAY_TASK_ID"
echo "Config:      $CONFIG"
echo "Experiment:  $EXPNAME"
echo "============================================"

python3 ROM_WSINDY_pipeline.py \
    --config "$CONFIG" \
    --experiment_name "$EXPNAME" \
    --wsindy-only

echo "Done: exit code $?"
