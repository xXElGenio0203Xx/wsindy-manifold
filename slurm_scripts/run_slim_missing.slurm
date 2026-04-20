#!/bin/bash
#SBATCH --job-name=slim_miss
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH -p batch
#SBATCH --array=0-3
#SBATCH -o slurm_logs/slim_missing_%A_%a.out
#SBATCH -e slurm_logs/slim_missing_%A_%a.err

# ============================================================================
# MISSING SLIM NOISE SWEEP — 4 ROM (MVAR+LSTM) jobs
# ============================================================================
# Gas η=0.15, Gas η=1.00, Blackhole η=0.50, Blackhole η=1.50
# N=100 particles, 340 training sims, MVAR+LSTM+WSINDy, lag=5
#
# Time estimate: completed slim jobs used <24h each at N=100.
# Allocating 36h for safety margin.
#
# Usage:
#   sbatch slurm_scripts/run_slim_missing.slurm
# ============================================================================

set -euo pipefail

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
export PYTHONPATH=src
mkdir -p slurm_logs oscar_output

MANIFEST="configs/noise_sweep_slim/manifest_missing.txt"
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: $MANIFEST not found."
    exit 1
fi

CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
if [ -z "$CONFIG" ]; then
    echo "ERROR: No config for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

EXP_NAME=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['experiment_name'])")

echo "============================================"
echo "SLIM NOISE SWEEP — Missing ROM jobs"
echo "============================================"
echo "Manifest:    $MANIFEST"
echo "Array task:  $SLURM_ARRAY_TASK_ID / 3"
echo "Config:      $CONFIG"
echo "Experiment:  $EXP_NAME"
echo "Start time:  $(date)"
echo "Node:        $(hostname)"
echo ""

# Remove stale partial data for clean re-run
OUTPUT_DIR="oscar_output/${EXP_NAME}"
if [ -d "$OUTPUT_DIR" ]; then
    echo "Removing stale $OUTPUT_DIR for clean re-run"
    rm -rf "$OUTPUT_DIR"
fi

python ROM_WSINDY_pipeline.py \
    --config "$CONFIG" \
    --experiment_name "$EXP_NAME"

EXIT_CODE=$?
echo ""
echo "End time:  $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
