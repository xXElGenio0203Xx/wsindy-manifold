#!/bin/bash
#SBATCH --job-name=ncv_wso
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH -p batch
#SBATCH --array=0-4%3
#SBATCH -o slurm_logs/ncv_wso_%A_%a.out
#SBATCH -e slurm_logs/ncv_wso_%A_%a.err

# ============================================================================
# N-Convergence WSINDy-only — 5 configs (N=50,100,200,300,500)
# Training data exists in oscar_output/NDYN08_pure_vicsek_N{XXXX}/
# We override --output-dir to point at existing data (strip _ws suffix).
# ============================================================================

set -euo pipefail

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
export PYTHONPATH=src
mkdir -p slurm_logs oscar_output

MANIFEST="configs/rerun/nconv_wsonly_manifest.txt"

CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
if [ -z "$CONFIG" ]; then
    echo "ERROR: No config for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Config experiment name has _ws suffix; strip it to point at existing data
EXP_NAME_WS=$(basename "$CONFIG" .yaml)
EXP_NAME="${EXP_NAME_WS%_ws}"
OUTPUT_DIR="oscar_output/${EXP_NAME}"

echo "============================================"
echo "N-CONVERGENCE — WSINDy-only rerun"
echo "============================================"
echo "Array task:  $SLURM_ARRAY_TASK_ID"
echo "Config:      $CONFIG"
echo "Experiment:  $EXP_NAME (was $EXP_NAME_WS)"
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
