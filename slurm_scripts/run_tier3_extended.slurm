#!/bin/bash
#SBATCH --job-name=tier3_ext
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -p batch
#SBATCH --array=0-18%7
#SBATCH -o slurm_logs/tier3_ext_%A_%a.out
#SBATCH -e slurm_logs/tier3_ext_%A_%a.err

# ============================================================================
# TIER 3 — Extended Catalogue: 5 residual NDYN + 14 DO = 19 tasks
# ============================================================================
# N=300, sqrt+none, MVAR+LSTM only (no WSINDy), lag=5
# cleanup_train_after_pod: true
# Lighter than Tier 1/2 (no WSINDy → faster per task)
#
# Pre-requisite: Generate configs first:
#   python generate_extended_catalogue.py --output_dir configs/extended_catalogue
#
# Usage:
#   sbatch slurm_scripts/run_tier3_extended.slurm
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
export PYTHONPATH=src
mkdir -p slurm_logs oscar_output

MANIFEST="configs/extended_catalogue/manifest.txt"
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: $MANIFEST not found. Run: python generate_extended_catalogue.py"
    exit 1
fi

N_CONFIGS=$(wc -l < "$MANIFEST" | tr -d ' ')
LAST_INDEX=$((N_CONFIGS - 1))

if [ "$SLURM_ARRAY_TASK_ID" -gt "$LAST_INDEX" ]; then
    echo "ERROR: Array index $SLURM_ARRAY_TASK_ID exceeds manifest ($N_CONFIGS configs)"
    exit 1
fi

CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
if [ -z "$CONFIG" ]; then
    echo "ERROR: No config for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

EXP_NAME=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['experiment_name'])")

echo "============================================"
echo "TIER 3 — Extended Catalogue"
echo "============================================"
echo "Manifest:    $MANIFEST"
echo "Array task:  $SLURM_ARRAY_TASK_ID / $LAST_INDEX"
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
