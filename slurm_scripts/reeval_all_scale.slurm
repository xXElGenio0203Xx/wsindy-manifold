#!/bin/bash
#SBATCH --job-name=reeval_all
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/reeval_all_%j.out
#SBATCH --error=slurm_logs/reeval_all_%j.err

# ============================================================================
# Re-evaluate ALL completed experiments: LSTM with mass_postprocess=scale
# ============================================================================
# This re-runs ONLY the evaluation step (no training, no simulation).
# Overwrites existing LSTM/test_results.csv with scale-corrected predictions.
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate
cd ~/wsindy-manifold
export PYTHONPATH="$PWD/src:$PYTHONPATH"
mkdir -p slurm_logs

echo "============================================"
echo "Re-evaluating ALL experiments with mass_postprocess=scale"
echo "============================================"
echo "Start time: $(date)"
echo

python rerun_evaluation.py oscar_output/ --all --lstm-only --mass-postprocess scale

echo
echo "End time: $(date)"
echo "Exit code: $?"
