#!/bin/bash
#SBATCH --job-name=synthesis_v2_6
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/synthesis_v2_6_%j.out
#SBATCH -e slurm_logs/synthesis_v2_6_%j.err

# ============================================================================
# SYNTHESIS V2.6 — V2.4 Champion + d=20 (More POD Modes, No Scaling)
# ============================================================================
# Tests: does capturing 91% energy (d=20) beat 71% (d=10)?
# Baseline: V2.4 R²=+0.272 with d=10
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

echo "============================================"
echo "SYNTHESIS V2.6 — d=20, No Scaling, Clamping"
echo "Start: $(date)"
echo "============================================"

python ROM_pipeline.py \
    --config configs/synthesis_v2_6.yaml \
    --experiment_name synthesis_v2_6

echo "============================================"
echo "End: $(date)"
echo "============================================"
