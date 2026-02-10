#!/bin/bash
#SBATCH --job-name=synthesis_v1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/synthesis_v1_%j.out
#SBATCH -e slurm_logs/synthesis_v1_%j.err

# ============================================================================
# SYNTHESIS V1 — All lessons combined
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

echo "============================================"
echo "SYNTHESIS V1 — Combined lessons experiment"
echo "============================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo ""

python ROM_pipeline.py \
    --config configs/synthesis_v1.yaml \
    --experiment_name synthesis_v1

echo ""
echo "End time: $(date)"
echo "Exit code: $?"
