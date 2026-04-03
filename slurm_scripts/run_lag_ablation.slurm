#!/bin/bash
#SBATCH --job-name=lag_ablation
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH -p batch
#SBATCH --array=0-15
#SBATCH -o slurm_logs/lag_ablation_%A_%a.out
#SBATCH -e slurm_logs/lag_ablation_%A_%a.err

# ============================================================================
# LAG ABLATION — 4 regimes × 4 IC criteria (BIC/HQIC/AIC/FPE)
# ============================================================================
# Reuses Phase B simulation data (POD basis already in oscar_output/).
# Only MVAR + LSTM are trained (WSINDy disabled for speed).
#
# Lags from rom_hyperparameters/results_oscar_aligned/<regime>/summary.json:
#   gas:       BIC=2  HQIC=6   AIC=50  FPE=42
#   blackhole: BIC=2  HQIC=10  AIC=50  FPE=39
#   supernova: BIC=2  HQIC=4   AIC=50  FPE=49
#   vicsek:    BIC=2  HQIC=4   AIC=10  FPE=10
#
# Usage:
#   sbatch slurm_scripts/run_lag_ablation.slurm
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs oscar_output

MANIFEST="configs/systematic/thesis_final/lag_ablation/manifest.txt"
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: $MANIFEST not found."
    exit 1
fi

N_CONFIGS=$(wc -l < "$MANIFEST" | tr -d ' ')
LAST_INDEX=$((N_CONFIGS - 1))

CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
if [ -z "$CONFIG" ]; then
    echo "ERROR: No config for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

EXP_NAME=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['experiment_name'])")

echo "============================================"
echo "LAG ABLATION — MVAR + LSTM (no WSINDy)"
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
