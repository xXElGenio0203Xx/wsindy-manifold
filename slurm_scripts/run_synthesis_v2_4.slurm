#!/bin/bash
#SBATCH --job-name=synthesis_v2_4
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/synthesis_v2_4_%j.out
#SBATCH -e slurm_logs/synthesis_v2_4_%j.err

# ============================================================================
# SYNTHESIS V2.4 — No Eigenvalue Scaling, WITH Clamping
# ============================================================================
# Reproduces V1's strategy: no eigenvalue scaling, clamp+renorm.
# Compare with V2.3 to isolate clamping effect on R².
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

echo "============================================"
echo "SYNTHESIS V2.4 — No Scaling, WITH Clamping"
echo "Start: $(date)"
echo "============================================"

python ROM_pipeline.py \
    --config configs/synthesis_v2_4.yaml \
    --experiment_name synthesis_v2_4

echo "============================================"
echo "End: $(date)"
echo "============================================"
