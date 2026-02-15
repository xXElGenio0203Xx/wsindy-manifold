#!/bin/bash
#SBATCH --job-name=v4_mega
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/synthesis_v4_megaheavyweight_%j.out
#SBATCH -e slurm_logs/synthesis_v4_megaheavyweight_%j.err

# ============================================================================
# SYNTHESIS V4 — MEGA HEAVYWEIGHT
# ============================================================================
#
#   1000 Gaussian ICs × T=60s → 493,000 training samples
#   d=35 (98% energy), p=7, α=1e-4
#   MVAR: 8,610 params, ratio = 57.3×
#   LSTM: 256 hidden, 4 layers, 10,000 epochs
#   NO eigenvalue scaling, NO clamping
#   48 hour wall time, 64GB RAM, 8 cores
#
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

echo "============================================================"
echo "  SYNTHESIS V4 — MEGA HEAVYWEIGHT"
echo "============================================================"
echo "  1000 Gaussian ICs × T=60s"
echo "  d=35 (98% energy), p=7, α=1e-4"
echo "  NO eigenvalue scaling, NO clamping"
echo "  MVAR: 8,610 params, 493K samples (57×)"
echo "  LSTM: 256h × 4L × 10,000 epochs"
echo "  Start: $(date)"
echo "============================================================"
echo ""

python ROM_pipeline.py \
    --config configs/synthesis_v4_megaheavyweight.yaml \
    --experiment_name synthesis_v4_megaheavyweight

echo ""
echo "============================================================"
echo "  End: $(date)"
echo "  Exit code: $?"
echo "============================================================"
