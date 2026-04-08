#!/bin/bash
#SBATCH --job-name=tier4_nconv
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH -p batch
#SBATCH --array=0-5%4
#SBATCH -o slurm_logs/tier4_nconv_%A_%a.out
#SBATCH -e slurm_logs/tier4_nconv_%A_%a.err

# ============================================================================
# TIER 4 — N-Convergence: Pure Vicsek at N ∈ {50,100,200,300,500,1000}
# ============================================================================
# Tests Messenger & Bortz O(N^{-1/2}) convergence rate.
# N=300, sqrt+none, MVAR+LSTM+WSINDy, lag=5
# cleanup_train_after_pod: true
# N=1000 may need up to 12h (340 sims × 1000 particles)
#
# Pre-requisite: Generate configs first:
#   python generate_n_convergence.py --output_dir configs/n_convergence
#
# Usage:
#   sbatch slurm_scripts/run_tier4_convergence.slurm
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
export PYTHONPATH=src
mkdir -p slurm_logs oscar_output

MANIFEST="configs/n_convergence/manifest.txt"
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: $MANIFEST not found. Run: python generate_n_convergence.py"
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
echo "TIER 4 — N-Convergence"
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
