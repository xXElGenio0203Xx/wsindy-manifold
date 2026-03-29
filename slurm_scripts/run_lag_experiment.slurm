#!/bin/bash
#SBATCH --job-name=lag_exp
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=48G
#SBATCH --time=15:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/lag_experiment_%j.out
#SBATCH -e slurm_logs/lag_experiment_%j.err

# ============================================================================
# LAG EXPERIMENT: DO_CS01 with MVAR lag=8, LSTM lag=20
# ============================================================================
# Tests whether giving each model its native lag improves forecast stability.
# Baseline: DO_CS01_swarm_C01_l05  (MVAR lag=5 R²=0.95, LSTM lag=5 R²=-1211)
# This run: MVAR lag=8, LSTM lag=20 (separate datasets)
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs oscar_output

CONFIG="configs/lag_experiment_CS01.yaml"
EXP_NAME="LAG_CS01_mvar8_lstm20"

echo "============================================"
echo "LAG EXPERIMENT: MVAR lag=8, LSTM lag=20"
echo "============================================"
echo "Config:      $CONFIG"
echo "Experiment:  $EXP_NAME"
echo "Start time:  $(date)"
echo "Node:        $(hostname)"
echo "Python:      $(which python)"
echo ""

# Apply the pipeline patch (idempotent — checks if already patched)
python scripts/patch_separate_lstm_lag.py || {
    echo "Patch may have already been applied. Continuing..."
}

# Run the pipeline
python ROM_WSINDY_pipeline.py \
    --config "$CONFIG" \
    --experiment_name "$EXP_NAME"

EXIT_CODE=$?
echo ""
echo "End time:  $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
