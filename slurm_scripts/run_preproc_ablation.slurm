#!/bin/bash
#SBATCH --job-name=preproc_abl
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH -p batch
#SBATCH --array=0-23
#SBATCH -o slurm_logs/preproc_abl_%A_%a.out
#SBATCH -e slurm_logs/preproc_abl_%A_%a.err

# ============================================================================
# PREPROCESSING ABLATION — 4 regimes × 2 transforms × 3 mass modes = 24 tasks
# ============================================================================
# LSTM-only (MVAR disabled, WSINDy disabled)
# max_epochs=150, patience=25  (light)
#
# Regimes: NDYN04_gas, NDYN04_gas_VS, NDYN05_blackhole, NDYN05_blackhole_VS
# density_transform: sqrt, raw
# mass_postprocess: none, simplex, scale
#
# Usage:
#   sbatch slurm_scripts/run_preproc_ablation.slurm
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs oscar_output

MANIFEST="configs/systematic/preproc_ablation/manifest.txt"
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
echo "PREPROCESSING ABLATION — LSTM only"
echo "============================================"
echo "Array task:  $SLURM_ARRAY_TASK_ID / 23"
echo "Config:      $CONFIG"
echo "Experiment:  $EXP_NAME"
echo "Start time:  $(date)"
echo "Node:        $(hostname)"
echo ""

python ROM_WSINDY_pipeline.py \
    --config "$CONFIG" \
    --experiment_name "$EXP_NAME"

EXIT_CODE=$?
echo ""
echo "End time:  $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
