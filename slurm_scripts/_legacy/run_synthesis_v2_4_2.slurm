#!/bin/bash
#SBATCH --job-name=synth_v2_4_2
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/synthesis_v2_4_2_%j.out
#SBATCH -e slurm_logs/synthesis_v2_4_2_%j.err

# ============================================================================
# SYNTHESIS V2.4.2 — V2.4 + Low Regularization (α=0.01)
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

echo "============================================"
echo "SYNTHESIS V2.4.2 — α=0.01, No Scaling, Clamping"
echo "Start: $(date)"
echo "============================================"

python ROM_pipeline.py \
    --config configs/synthesis_v2_4_2.yaml \
    --experiment_name synthesis_v2_4_2

echo "============================================"
echo "End: $(date)"
echo "============================================"
