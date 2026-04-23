#!/bin/bash
#SBATCH --job-name=wso_patch
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=96G
#SBATCH --time=18:00:00
#SBATCH -p batch
#SBATCH --array=0-3
#SBATCH -o slurm_logs/wso_patch_%A_%a.out
#SBATCH -e slurm_logs/wso_patch_%A_%a.err

# ============================================================================
# WSINDy-ONLY PATCH — 4 jobs where training data exists but WSINDy timed out
# ============================================================================
# Index 0: NDYN04_gas_eta0p15_ws   (noise sweep, training 342 sims done)
# Index 1: NDYN05_blackhole_eta0p05_ws  (noise sweep, training 342 sims done)
# Index 2: NDYN08_pure_vicsek_N0500   (nconv, training 342 sims done)
# Index 3: NDYN08_pure_vicsek_N1000   (nconv, training 342 sims done)
#
# Each job skips sim generation, runs WSINDy directly on existing /train data.
# Time estimate: WSINDy alone takes ~1-2h; 6h is a generous budget.
#
# Usage:
#   sbatch slurm_scripts/wsonly_patch.slurm
# ============================================================================

set -euo pipefail

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
export PYTHONPATH=src
mkdir -p slurm_logs

MANIFEST="configs/wsonly_patch_manifest.txt"
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: $MANIFEST not found."
    exit 1
fi

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
if [ -z "$LINE" ]; then
    echo "ERROR: No entry for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

CONFIG=$(echo "$LINE" | awk '{print $1}')
OUTPUT_DIR=$(echo "$LINE" | awk '{print $2}')
# experiment_name is the basename of the output dir (carries _ws suffix where needed)
EXP_NAME=$(basename "$OUTPUT_DIR")

echo "============================================"
echo "WSINDy-ONLY PATCH"
echo "============================================"
echo "Manifest:    $MANIFEST"
echo "Array task:  $SLURM_ARRAY_TASK_ID / 3"
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

python -u ROM_WSINDY_pipeline.py \
    --config "$CONFIG" \
    --experiment_name "$EXP_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --wsindy-only

EXIT_CODE=$?
echo ""
echo "End time:  $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
