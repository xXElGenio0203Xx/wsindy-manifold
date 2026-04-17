#!/bin/bash
#SBATCH --job-name=thesis_ndyn07
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH -p batch
#SBATCH --array=0-1
#SBATCH -o slurm_logs/thesis_ndyn07_%A_%a.out
#SBATCH -e slurm_logs/thesis_ndyn07_%A_%a.err

# ============================================================================
# THESIS-FINAL — NDYN07_crystal: Full pipeline (sim + POD + MVAR + LSTM + WSINDy)
# ============================================================================
# This regime was missing from Phase A/B manifests. Runs the complete pipeline
# from scratch for both constant-speed and variable-speed variants.
#
# Array: 0 = NDYN07_crystal (CS), 1 = NDYN07_crystal_VS
#
# Estimated runtime: ~3-4h per regime (sim ~1.5h, POD+MVAR ~5min, LSTM ~30min,
#                    WSINDy+bootstrap(B=200) ~1-2h)
#
# Usage:
#   sbatch slurm_scripts/run_thesis_ndyn07.slurm
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
export PYTHONPATH=src
mkdir -p slurm_logs oscar_output

CONFIGS=(
    "configs/systematic/thesis_final/phase_b/NDYN07_crystal.yaml"
    "configs/systematic/thesis_final/phase_b/NDYN07_crystal_VS.yaml"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
if [ -z "$CONFIG" ]; then
    echo "ERROR: No config for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

EXP_NAME=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['experiment_name'])")

echo "============================================"
echo "THESIS-FINAL — NDYN07 Crystal (full pipeline)"
echo "============================================"
echo "Array task:  $SLURM_ARRAY_TASK_ID"
echo "Config:      $CONFIG"
echo "Experiment:  $EXP_NAME"
echo "Start time:  $(date)"
echo "Node:        $(hostname)"
echo "Python:      $(which python)"
echo ""

python ROM_WSINDY_pipeline.py \
    --config "$CONFIG" \
    --experiment_name "$EXP_NAME"

EXIT_CODE=$?
echo ""
echo "End time:  $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
