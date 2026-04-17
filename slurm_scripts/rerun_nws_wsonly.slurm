#!/bin/bash
#SBATCH --job-name=nws_wso
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH -p batch
#SBATCH --array=0-4%3
#SBATCH -o slurm_logs/nws_wso_%A_%a.out
#SBATCH -e slurm_logs/nws_wso_%A_%a.err

# ============================================================================
# Noise Sweep WSINDy-only rerun — 5 configs that timed out with data intact
# (gas eta1p00/1p50/2p00, blackhole eta0p05/0p15)
# Training data exists on disk; runs WSINDy identification only.
# ============================================================================

set -euo pipefail

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
export PYTHONPATH=src
mkdir -p slurm_logs oscar_output

MANIFEST="configs/rerun/nws_wsonly_manifest.txt"

CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
if [ -z "$CONFIG" ]; then
    echo "ERROR: No config for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

EXP_NAME=$(basename "$CONFIG" .yaml)
OUTPUT_DIR="oscar_output/${EXP_NAME}_ws"

echo "============================================"
echo "NOISE SWEEP — WSINDy-only rerun"
echo "============================================"
echo "Array task:  $SLURM_ARRAY_TASK_ID"
echo "Config:      $CONFIG"
echo "Experiment:  $EXP_NAME"
echo "Output dir:  $OUTPUT_DIR"
echo "Start time:  $(date)"
echo "Node:        $(hostname)"
echo ""

if [ ! -d "$OUTPUT_DIR/train" ]; then
    echo "ERROR: Training data not found at $OUTPUT_DIR/train"
    exit 1
fi

python ROM_WSINDY_pipeline.py \
    --config "$CONFIG" \
    --experiment_name "$EXP_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --wsindy-only

EXIT_CODE=$?
echo ""
echo "End time:  $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
