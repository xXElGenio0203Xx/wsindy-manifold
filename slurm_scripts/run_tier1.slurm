#!/bin/bash
#SBATCH --job-name=tier1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH -p batch
#SBATCH --array=0-17%7
#SBATCH -o slurm_logs/tier1_%A_%a.out
#SBATCH -e slurm_logs/tier1_%A_%a.err

# ============================================================================
# TIER 1 — Main Regimes: 9 regimes × 2 lag variants = 18 tasks
# ============================================================================
# N=300, sqrt+none, MVAR+LSTM+WSINDy, bootstrap B=200
# Lags: w=5 (baseline) and w=w*_BIC (data-driven)
# cleanup_train_after_pod: true (prevents disk-quota issues)
#
# Usage:
#   sbatch slurm_scripts/run_tier1.slurm
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
export PYTHONPATH=src
mkdir -p slurm_logs oscar_output

MANIFEST="configs/systematic/tier1/manifest.txt"
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: $MANIFEST not found."
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
echo "TIER 1 — Main Regimes"
echo "============================================"
echo "Manifest:    $MANIFEST"
echo "Array task:  $SLURM_ARRAY_TASK_ID / $LAST_INDEX"
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
