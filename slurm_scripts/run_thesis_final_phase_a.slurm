#!/bin/bash
#SBATCH --job-name=thesis_phA
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH -p batch
#SBATCH --array=0-6
#SBATCH -o slurm_logs/thesis_phA_%A_%a.out
#SBATCH -e slurm_logs/thesis_phA_%A_%a.err

# ============================================================================
# THESIS-FINAL — Phase A: Simulation + POD + MVAR (7 regimes)
# ============================================================================
# LSTM and WSINDy disabled.  MVAR enabled so the pipeline builds the POD basis
# and latent dataset needed for downstream lag selection.  Produces:
#   oscar_output/<EXP>_thesis_final/train/         (simulation data)
#   oscar_output/<EXP>_thesis_final/rom_common/     (POD basis, latent dataset)
#   oscar_output/<EXP>_thesis_final/MVAR/           (baseline MVAR fit)
#
# After completion, run alvarez_lag_selection.py locally to choose MVAR lags,
# then submit Phase B (full pipeline with upgraded models).
#
# Usage:
#   sbatch slurm_scripts/run_thesis_final_phase_a.slurm
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs oscar_output

MANIFEST="configs/systematic/thesis_final/phase_a/manifest.txt"
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
echo "THESIS-FINAL — Phase A (sim + POD + MVAR)"
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
