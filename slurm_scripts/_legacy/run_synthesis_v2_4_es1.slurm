#!/bin/bash
#SBATCH --job-name=v2_4_es1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/synthesis_v2_4_es1_%j.out
#SBATCH -e slurm_logs/synthesis_v2_4_es1_%j.err

# ============================================================================
# V2.4_ES1 — V2.4 Champion + Eigenvalue Scaling to 1.0
# ============================================================================
#   V2.4 identical setup (261 ICs, d=10, p=3, α=10, clamp ON)
#   ONLY CHANGE: eigenvalue_threshold=1.0 (normalize to unit circle)
#
#   COMPARISON:
#     V2.2: threshold=0.97 → R²=-0.015 (over-damped)
#     V2.4: no threshold   → R²=+0.272 (champion, ρ=1.208)
#     V2.4_es1: threshold=1.0 → ??? (stable, no damping)
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate
cd ~/wsindy-manifold
export PYTHONPATH=src:$PYTHONPATH
mkdir -p slurm_logs

echo "============================================================"
echo "  V2.4_ES1 — Eigenvalue Scaling to 1.0 (No Damping)"
echo "  Start: $(date)"
echo "============================================================"

python ROM_pipeline.py \
    --config configs/synthesis_v2_4_es1.yaml \
    --experiment_name synthesis_v2_4_es1

echo "  End: $(date), Exit: $?"
