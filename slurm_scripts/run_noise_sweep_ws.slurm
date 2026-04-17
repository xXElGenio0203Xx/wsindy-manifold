#!/bin/bash
#SBATCH --job-name=noise_ws
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH -p batch
#SBATCH --array=0-19%7
#SBATCH -o slurm_logs/noise_ws_%A_%a.out
#SBATCH -e slurm_logs/noise_ws_%A_%a.err

# ============================================================================
# Noise Sweep — WSINDy only (LSTM=off, MVAR=off)
# 4 regimes × 5 noise levels = 20 tasks
# ============================================================================

set -euo pipefail

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
export PYTHONPATH=src
mkdir -p slurm_logs oscar_output

MANIFEST="configs/noise_sweep_ws/manifest.txt"

CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
if [ -z "$CONFIG" ]; then
    echo "ERROR: No config for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

EXP_NAME=$(basename "$CONFIG" .yaml)
OUTPUT_DIR="oscar_output/${EXP_NAME}_ws"

echo "============================================"
echo "NOISE SWEEP — WSINDy only"
echo "============================================"
echo "Manifest:    $MANIFEST"
echo "Array task:  $SLURM_ARRAY_TASK_ID"
echo "Config:      $CONFIG"
echo "Experiment:  $EXP_NAME"
echo "Output dir:  $OUTPUT_DIR"
echo "Start time:  $(date)"
echo "Node:        $(hostname)"
echo ""

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
