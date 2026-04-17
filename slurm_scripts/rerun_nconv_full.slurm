#!/bin/bash
#SBATCH --job-name=ncv_full
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH -p batch
#SBATCH --array=0-0
#SBATCH -o slurm_logs/ncv_full_%A_%a.out
#SBATCH -e slurm_logs/ncv_full_%A_%a.err

# ============================================================================
# N-Convergence FULL pipeline — N=1000 only (128/340 train, no test)
# Full sim -> WSINDy pipeline.  24h generous time limit.
# ============================================================================

set -euo pipefail

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
export PYTHONPATH=src
mkdir -p slurm_logs oscar_output

MANIFEST="configs/rerun/nconv_full_manifest.txt"

CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
if [ -z "$CONFIG" ]; then
    echo "ERROR: No config for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

EXP_NAME_WS=$(basename "$CONFIG" .yaml)
EXP_NAME="${EXP_NAME_WS%_ws}"
OUTPUT_DIR="oscar_output/${EXP_NAME}"

echo "============================================"
echo "N-CONVERGENCE — FULL pipeline rerun"
echo "============================================"
echo "Array task:  $SLURM_ARRAY_TASK_ID"
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
