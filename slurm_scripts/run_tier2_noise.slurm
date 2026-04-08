#!/bin/bash
#SBATCH --job-name=tier2_noise
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH -p batch
#SBATCH --array=0-19%5
#SBATCH -o slurm_logs/tier2_noise_%A_%a.out
#SBATCH -e slurm_logs/tier2_noise_%A_%a.err

# ============================================================================
# TIER 2 — Noise Sweep: 4 regimes × 5 noise levels = 20 tasks
# ============================================================================
# N=300, sqrt+none, MVAR+LSTM+WSINDy, lag=5 only
# cleanup_train_after_pod: true
#
# Pre-requisite: Generate configs first:
#   python generate_noise_sweep.py --output_dir configs/noise_sweep
#
# Usage:
#   sbatch slurm_scripts/run_tier2_noise.slurm
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
export PYTHONPATH=src
mkdir -p slurm_logs oscar_output

MANIFEST="configs/noise_sweep/manifest.txt"
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: $MANIFEST not found. Run: python generate_noise_sweep.py"
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
echo "TIER 2 — Noise Sweep"
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
