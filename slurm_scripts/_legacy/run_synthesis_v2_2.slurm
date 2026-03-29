#!/bin/bash
#SBATCH --job-name=synthesis_v2_2
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/synthesis_v2_2_%j.out
#SBATCH -e slurm_logs/synthesis_v2_2_%j.err

# ============================================================================
# SYNTHESIS V2.2 — Evidence-Based Minimal ROM
# ============================================================================
# Reverts Schur projection (broke dynamics in V2.1).
# d=10, p=3, α=10.0 (aim for natural stability before scaling).
# LSTM gets larger capacity (64 hidden, 3000 epochs).
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

echo "============================================"
echo "SYNTHESIS V2.2 — Evidence-Based Minimal ROM"
echo "============================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""
echo "Key parameters:"
echo "  fixed_modes: 10 (72% energy)"
echo "  MVAR: lag=3, ridge_alpha=10.0"
echo "  eigenvalue_threshold: 0.97 (uniform scaling, NOT Schur)"
echo "  LSTM: 64 hidden, 2 layers, 3000 epochs"
echo ""

python ROM_pipeline.py \
    --config configs/synthesis_v2_2.yaml \
    --experiment_name synthesis_v2_2

echo ""
echo "End time: $(date)"
echo "Exit code: $?"
