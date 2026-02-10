#!/bin/bash
#SBATCH --job-name=synthesis_v2
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/synthesis_v2_%j.out
#SBATCH -e slurm_logs/synthesis_v2_%j.err

# ============================================================================
# SYNTHESIS V2 — Stability-aware ROM with compact latent space
# ============================================================================
# KEY CHANGES FROM V1:
#   - fixed_modes=8 (down from 19 via pod_energy=0.90)
#   - eigenvalue_threshold=0.98 (FULL companion matrix stability check)
#   - ridge_alpha=1.0 (up from 1e-4)
#   - lag=3 (down from 5)
#   - 300 training runs (up from 200)
#   - LSTM hidden_units=32 (down from 64, matched to d=8)
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

echo "============================================"
echo "SYNTHESIS V2 — Stability-aware compact ROM"
echo "============================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo ""
echo "Key parameters:"
echo "  fixed_modes: 8"
echo "  MVAR lag: 3, ridge: 1.0"
echo "  eigenvalue_threshold: 0.98"
echo "  Training runs: 300"
echo "  LSTM hidden: 32, layers: 2"
echo ""

python ROM_pipeline.py \
    --config configs/synthesis_v2.yaml \
    --experiment_name synthesis_v2

echo ""
echo "End time: $(date)"
echo "Exit code: $?"
