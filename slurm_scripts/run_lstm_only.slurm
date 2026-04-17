#!/bin/bash
#SBATCH --job-name=lstm_only
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH -p batch
#SBATCH --array=0-6
#SBATCH -o slurm_logs/lstm_only_%A_%a.out
#SBATCH -e slurm_logs/lstm_only_%A_%a.err

set -euo pipefail

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs oscar_output

MANIFEST="configs/lstm_only/manifest.txt"

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: $MANIFEST not found."
    exit 1
fi

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
EXP_NAME=$(echo "$LINE" | awk '{print $1}')
ROM_SOURCE=$(echo "$LINE" | awk '{print $2}')

if [ -z "$EXP_NAME" ] || [ -z "$ROM_SOURCE" ]; then
    echo "ERROR: No entry for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

CONFIG="configs/lstm_only/${EXP_NAME}.yaml"
OUTPUT_DIR="oscar_output/${EXP_NAME}"
ROM_SOURCE_DIR="oscar_output/${ROM_SOURCE}/rom_common"

echo "============================================"
echo "LSTM-ONLY TRAINING"
echo "============================================"
echo "Array task:    $SLURM_ARRAY_TASK_ID"
echo "Experiment:    $EXP_NAME"
echo "Config:        $CONFIG"
echo "Output dir:    $OUTPUT_DIR"
echo "ROM source:    $ROM_SOURCE_DIR"
echo "Start time:    $(date)"
echo "Node:          $(hostname)"
echo "Python:        $(which python)"
echo ""

# Create output dir and symlink rom_common from source experiment
mkdir -p "$OUTPUT_DIR"
if [ ! -e "$OUTPUT_DIR/rom_common" ]; then
    ln -s "$(cd "$ROM_SOURCE_DIR" && pwd)" "$OUTPUT_DIR/rom_common"
    echo "Symlinked rom_common -> $ROM_SOURCE_DIR"
else
    echo "rom_common already exists in $OUTPUT_DIR"
fi

# Verify required files exist
if [ ! -f "$OUTPUT_DIR/rom_common/latent_dataset.npz" ]; then
    echo "ERROR: latent_dataset.npz not found in rom_common!"
    exit 1
fi
if [ ! -f "$OUTPUT_DIR/rom_common/pod_basis.npz" ]; then
    echo "ERROR: pod_basis.npz not found in rom_common!"
    exit 1
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
