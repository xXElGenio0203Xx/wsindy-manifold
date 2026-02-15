#!/bin/bash
#SBATCH --job-name=synthesis_v2_8
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/synthesis_v2_8_%j.out
#SBATCH -e slurm_logs/synthesis_v2_8_%j.err

# ============================================================================
# SYNTHESIS V2.8 — V2.4 Champion + Higher Lag p=5
# ============================================================================
# Tests: does longer memory (0.60s vs 0.36s) improve predictions?
# Baseline: V2.4 R²=+0.272 with p=3
# V1 used p=5 and got R²=+0.28 — was lag a factor?
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

echo "============================================"
echo "SYNTHESIS V2.8 — p=5, No Scaling, Clamping"
echo "Start: $(date)"
echo "============================================"

python ROM_pipeline.py \
    --config configs/synthesis_v2_8.yaml \
    --experiment_name synthesis_v2_8

echo "============================================"
echo "End: $(date)"
echo "============================================"
