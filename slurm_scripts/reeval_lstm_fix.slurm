#!/bin/bash
#SBATCH --job-name=reeval_fix
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch
#SBATCH -o slurm_logs/reeval_fix_%j.out
#SBATCH -e slurm_logs/reeval_fix_%j.err

# ============================================================================
# Re-evaluate completed experiments with LSTM lag fix
# Only re-runs Step 6 (inference) — no retraining needed.
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
export PYTHONPATH=src:$PYTHONPATH
mkdir -p slurm_logs

echo "============================================"
echo "LSTM RE-EVALUATION WITH LAG FIX"
echo "============================================"
echo "Start:  $(date)"
echo "Node:   $(hostname)"
echo "Python: $(which python)"
echo ""

python rerun_evaluation.py oscar_output/ --all

EXIT_CODE=$?
echo ""
echo "============================================"
echo "End:    $(date)"
echo "Exit:   $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
