#!/bin/bash
#SBATCH --job-name=reeval_scale
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/reeval_scale_%j.out

source ~/wsindy_env_new/bin/activate
cd ~/wsindy-manifold
export PYTHONPATH="$PWD/src:$PYTHONPATH"

echo "============================================"
echo "Re-evaluating with mass_postprocess=scale"
echo "============================================"
echo "Start time: $(date)"
echo

python rerun_evaluation.py oscar_output/ --all --lstm-only --mass-postprocess scale

echo
echo "End time: $(date)"
echo "Exit code: $?"
