#!/bin/bash
#SBATCH --job-name=synthesis_v3_2
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/synthesis_v3_2_%j.out
#SBATCH -e slurm_logs/synthesis_v3_2_%j.err

# ============================================================================
# SYNTHESIS V3.2 — Dynamic Physics (speed=3, Ca=2, R=4) + V2.4 ROM
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

echo "============================================"
echo "SYNTHESIS V3.2 — Dynamic Physics + V2.4 ROM"
echo "Start: $(date)"
echo "============================================"

python ROM_pipeline.py \
    --config configs/synthesis_v3_2.yaml \
    --experiment_name synthesis_v3_2

echo "============================================"
echo "End: $(date)"
echo "============================================"
