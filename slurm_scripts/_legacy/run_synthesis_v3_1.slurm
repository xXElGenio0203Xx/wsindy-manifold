#!/bin/bash
#SBATCH --job-name=synthesis_v3_1
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/synthesis_v3_1_%j.out
#SBATCH -e slurm_logs/synthesis_v3_1_%j.err

# ============================================================================
# SYNTHESIS V3.1 — Alvarez-Caliber: 98% POD Energy, Long Simulations
# ============================================================================
# 400 Gaussian ICs × T=30s → 98K training samples
# d=35 (98% energy), p=5, α=1e-4
# NO eigenvalue scaling, NO clamping
# LSTM: 128 hidden, 3 layers, 5000 epochs
# Budget: ~7-8 hours (12 hour limit)
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

echo "============================================"
echo "SYNTHESIS V3.1 — 98% POD Energy"
echo "  400 Gaussian ICs × T=30s"
echo "  d=35, p=5, α=1e-4"
echo "  NO eigenvalue scaling"
echo "  NO clamping"
echo "  LSTM: 128h × 3L × 5000ep"
echo "  Start: $(date)"
echo "============================================"

python ROM_pipeline.py \
    --config configs/synthesis_v3_1.yaml \
    --experiment_name synthesis_v3_1

echo ""
echo "End time: $(date)"
echo "Exit code: $?"
echo "============================================"
