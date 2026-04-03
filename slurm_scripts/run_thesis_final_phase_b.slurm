#!/bin/bash
#SBATCH --job-name=thesis_phB
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH -p batch
#SBATCH --array=0-6
#SBATCH -o slurm_logs/thesis_phB_%A_%a.out
#SBATCH -e slurm_logs/thesis_phB_%A_%a.err

# ============================================================================
# THESIS-FINAL — Phase B: LSTM + WSINDy + BIC-optimal MVAR (7 regimes)
# ============================================================================
# Requires Phase A to have completed first (simulation data + POD basis already
# present in oscar_output/<EXP>_thesis_final/rom_common/).
#
# Changes vs Phase A:
#   - lstm.enabled = true  (hidden_units=128, num_layers=2)
#   - wsindy.enabled = true  (n_ell=20, bootstrap B=200)
#   - mvar.lag = BIC-optimal per regime (gas:5, gas_VS:4, blackhole:4,
#                blackhole_VS:5, supernova:4, supernova_VS:4, vicsek:6)
#
# Usage:
#   sbatch slurm_scripts/run_thesis_final_phase_b.slurm
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs oscar_output

MANIFEST="configs/systematic/thesis_final/phase_b/manifest.txt"
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
echo "THESIS-FINAL — Phase B (LSTM + WSINDy)"
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
