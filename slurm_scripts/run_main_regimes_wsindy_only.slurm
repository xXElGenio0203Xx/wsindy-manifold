#!/bin/bash
#SBATCH --job-name=mainwsyv2
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH -p batch
#SBATCH --array=0-6%4
#SBATCH -o slurm_logs/mainwsyv2_%A_%a.out
#SBATCH -e slurm_logs/mainwsyv2_%A_%a.err

set -euo pipefail

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs oscar_output

MANIFEST="${MANIFEST_PATH:-configs/systematic/main_regimes/manifest.txt}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-wsindy_v2}"

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: $MANIFEST not found."
    exit 1
fi

N_CONFIGS=$(wc -l < "$MANIFEST" | tr -d ' ')
LAST_INDEX=$((N_CONFIGS - 1))

CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
if [ -z "$CONFIG" ]; then
    echo "ERROR: No config for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

EXP_NAME=$(basename "$CONFIG" .yaml)
OUTPUT_DIR="oscar_output/${EXP_NAME}_${OUTPUT_SUFFIX}"

echo "============================================"
echo "MAIN REGIMES WSINDY-ONLY SWEEP"
echo "============================================"
echo "Manifest:    $MANIFEST"
echo "Array task:  $SLURM_ARRAY_TASK_ID / $LAST_INDEX"
echo "Config:      $CONFIG"
echo "Experiment:  $EXP_NAME"
echo "Output dir:  $OUTPUT_DIR"
echo "Start time:  $(date)"
echo "Node:        $(hostname)"
echo "Python:      $(which python)"
echo ""

if [ -d "$OUTPUT_DIR" ]; then
    echo "FORCE RE-RUN: removing stale $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
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
