#!/bin/bash
#SBATCH --job-name=nws_orph
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH -p batch
#SBATCH --array=0-6%4
#SBATCH -o slurm_logs/nws_orph_%A_%a.out
#SBATCH -e slurm_logs/nws_orph_%A_%a.err

# ============================================================================
# Noise Sweep WSINDy-only rerun - 7 orphaned configs (24h time limit)
# Previous 6h and 12h runs timed out; training data exists on disk.
# Configs: gas eta={1.00,1.50,2.00}, blackhole eta={0.50,1.00,1.50},
#          supernova eta=0.15
# ============================================================================

set -euo pipefail

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
export PYTHONPATH=src
mkdir -p slurm_logs oscar_output

MANIFEST="configs/rerun/nws_orphan_manifest.txt"

CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
if [ -z "$CONFIG" ]; then
    echo "ERROR: No config for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

EXP_NAME=$(basename "$CONFIG" .yaml)
OUTPUT_DIR="oscar_output/${EXP_NAME}_ws"

echo "============================================"
echo "NOISE SWEEP — WSINDy-only rerun (24h)"
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
