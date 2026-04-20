#!/bin/bash
#SBATCH --job-name=ws_miss
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=96G
#SBATCH --time=18:00:00
#SBATCH -p batch
#SBATCH --array=0-3
#SBATCH -o slurm_logs/ws_missing_%A_%a.out
#SBATCH -e slurm_logs/ws_missing_%A_%a.err

# ============================================================================
# MISSING WSINDy-ONLY NOISE SWEEP — 4 jobs
# ============================================================================
# Gas η=0.15, Blackhole η=0.05, Blackhole η=0.15, Supernova η=0.50
# N=300 particles, 340 training sims, WSINDy only (no ROM)
#
# Time estimate: 16 of 20 WSINDy-only noise jobs finished within 12h.
# The 4 that failed may have timed out → allocating 18h (50% margin).
#
# Usage:
#   sbatch slurm_scripts/run_ws_missing.slurm
# ============================================================================

set -euo pipefail

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
export PYTHONPATH=src
mkdir -p slurm_logs oscar_output

MANIFEST="configs/noise_sweep_ws/manifest_missing.txt"
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: $MANIFEST not found."
    exit 1
fi

CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
if [ -z "$CONFIG" ]; then
    echo "ERROR: No config for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

EXP_NAME=$(basename "$CONFIG" .yaml)
OUTPUT_DIR="oscar_output/${EXP_NAME}_ws"

echo "============================================"
echo "WSINDy-ONLY NOISE SWEEP — Missing jobs"
echo "============================================"
echo "Manifest:    $MANIFEST"
echo "Array task:  $SLURM_ARRAY_TASK_ID / 3"
echo "Config:      $CONFIG"
echo "Experiment:  $EXP_NAME"
echo "Output dir:  $OUTPUT_DIR"
echo "Start time:  $(date)"
echo "Node:        $(hostname)"
echo ""

# Remove stale partial data for clean re-run
if [ -d "$OUTPUT_DIR" ]; then
    echo "Removing stale $OUTPUT_DIR for clean re-run"
    rm -rf "$OUTPUT_DIR"
fi

python ROM_WSINDY_pipeline.py \
    --config "$CONFIG" \
    --experiment_name "$EXP_NAME" \
    --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?
echo ""
echo "End time:  $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
