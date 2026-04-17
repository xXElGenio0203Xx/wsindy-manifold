#!/bin/bash
#SBATCH --job-name=reeval_mvar
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/reeval_mvar_%j.out
#SBATCH --error=slurm_logs/reeval_mvar_%j.err

# ============================================================================
# Re-evaluate ALL completed experiments: MVAR (mass_postprocess=scale)
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate
cd ~/wsindy-manifold
export PYTHONPATH="$PWD/src:$PYTHONPATH"
mkdir -p slurm_logs

echo "============================================"
echo "Re-evaluating ALL experiments: MVAR with mass_postprocess=scale"
echo "============================================"
echo "Start time: $(date)"
echo

python rerun_evaluation.py oscar_output/ --all --mvar-only --mass-postprocess scale

echo
echo "End time: $(date)"
echo "Exit code: $?"
