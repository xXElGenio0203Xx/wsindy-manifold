#!/bin/bash
#SBATCH --job-name=vdyn_clamp
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH -p batch
#SBATCH --array=0-13
#SBATCH -o slurm_logs/vdyn_clamp_%A_%a.out
#SBATCH -e slurm_logs/vdyn_clamp_%A_%a.err

# ============================================================================
# VDYN CLAMP-MODE COMPARISON — SLURM Array Job
# ============================================================================
# Runs 14 configs: 7 VDYN dynamics × {C0, C2} clamp modes.
# Purpose: determine optimal clamp_mode before the full systematic sweep.
#
# Usage:
#   sbatch slurm_scripts/run_vdyn_clamp.slurm
#   sbatch --array=0-6 slurm_scripts/run_vdyn_clamp.slurm   # C0 only
#   sbatch --array=7-13 slurm_scripts/run_vdyn_clamp.slurm  # C2 only
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs oscar_output

MANIFEST="configs/vdyn_clamp/manifest.txt"
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: $MANIFEST not found. Run shell_scripts/submit_vdyn_clamp.sh first."
    exit 1
fi

# Pick the config for this array task
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
if [ -z "$CONFIG" ]; then
    echo "ERROR: No config for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

EXP_NAME=$(basename "$CONFIG" .yaml)

echo "============================================"
echo "VDYN CLAMP-MODE COMPARISON"
echo "============================================"
echo "Array task:  $SLURM_ARRAY_TASK_ID / 13"
echo "Config:      $CONFIG"
echo "Experiment:  $EXP_NAME"
echo "Start time:  $(date)"
echo "Node:        $(hostname)"
echo "Python:      $(which python)"
echo ""

# Skip if already completed
if [ -f "oscar_output/$EXP_NAME/summary.json" ]; then
    echo "SKIP: $EXP_NAME already has summary.json — already completed."
    exit 0
fi

python ROM_pipeline.py \
    --config "$CONFIG" \
    --experiment_name "$EXP_NAME"

EXIT_CODE=$?
echo ""
echo "End time:  $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
