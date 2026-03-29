#!/bin/bash
#SBATCH --job-name=synthesis_v2_1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/synthesis_v2_1_%j.out
#SBATCH -e slurm_logs/synthesis_v2_1_%j.err

# ============================================================================
# SYNTHESIS V2.1 — Selective Stabilization + More POD Energy
# ============================================================================
# KEY CHANGES FROM V2:
#   - fixed_modes=14 (up from 8) → 80% energy vs 64%
#   - ridge_alpha=0.1 (down from 1.0) → less over-regularized
#   - lag=5 (up from 3) → more temporal memory
#   - eigenvalue_threshold=0.995 → relaxed (selective projection handles it)
#   - SELECTIVE Schur stabilization (code fix in mvar_trainer.py)
#     → only unstable eigenvalues are projected, stable ones preserved
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

echo "============================================"
echo "SYNTHESIS V2.1 — Selective Schur Stabilization"
echo "============================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo ""
echo "Key changes from V2:"
echo "  fixed_modes: 14 (was 8, 80% vs 64% energy)"
echo "  MVAR lag: 5 (was 3)"
echo "  ridge_alpha: 0.1 (was 1.0)"
echo "  eigenvalue_threshold: 0.995 (was 0.98)"
echo "  Stabilization: SELECTIVE Schur projection (was uniform scaling)"
echo ""

python ROM_pipeline.py \
    --config configs/synthesis_v2_1.yaml \
    --experiment_name synthesis_v2_1

echo ""
echo "End time: $(date)"
echo "Exit code: $?"
