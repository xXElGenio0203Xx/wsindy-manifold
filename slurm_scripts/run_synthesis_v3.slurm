#!/bin/bash
#SBATCH --job-name=synthesis_v3
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/synthesis_v3_%j.out
#SBATCH -e slurm_logs/synthesis_v3_%j.err

# ============================================================================
# SYNTHESIS V3 — Alvarez-Faithful ROM
# ============================================================================
# Mimics Alvarez et al. (2025) methodology with our Vicsek+Morse physics:
#   - 20 training ICs × T=60s → 500 ROM frames/IC → 10,000 total snapshots
#   - 10 test ICs × T=60s → ~496 autoregressive forecast steps
#   - POD: 99% energy threshold (data-driven, NOT fixed modes)
#   - MVAR: lag=4, ridge α=1e-6 (Alvarez exact values)
#   - LSTM: 1 layer, 16 hidden, lr=1e-3, batch=32, 100 epochs, patience=5
#   - Eigenvalue threshold: 0.999 (safety net only)
#
# See ALVAREZ_VS_OURS.md for full comparison.
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

echo "============================================"
echo "SYNTHESIS V3 — Alvarez-Faithful ROM"
echo "============================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo ""
echo "Key parameters (matched to Alvarez et al. 2025):"
echo "  T=60s, 20 train ICs, 10 test ICs"
echo "  POD: 99% energy (data-driven d)"
echo "  MVAR: lag=4, ridge_alpha=1e-6"
echo "  LSTM: 1 layer, 16 hidden, lr=1e-3"
echo "  Eigenvalue threshold: 0.999 (safety only)"
echo "  Expected sample/param ratio: ~14.4x"
echo ""

python ROM_pipeline.py \
    --config configs/synthesis_v3_alvarez.yaml \
    --experiment_name synthesis_v3_alvarez

echo ""
echo "End time: $(date)"
echo "Exit code: $?"
