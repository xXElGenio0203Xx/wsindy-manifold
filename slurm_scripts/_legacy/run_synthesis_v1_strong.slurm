#!/bin/bash
#SBATCH --job-name=synth_v1_strong
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/synthesis_v1_strong_%j.out
#SBATCH -e slurm_logs/synthesis_v1_strong_%j.err

# ============================================================================
# SYNTHESIS V1 STRONG — V1 with 4× longer training sims
# ============================================================================
# Training: 200 runs × 20s each (V1 was 5s) → 4× more MVAR training data
# Test: 5s (same as V1) for fair evaluation
# ICs: gaussian + two_clusters only
# 8h time limit (training sims take ~4× longer than V1's 2h)

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

echo "============================================"
echo "SYNTHESIS V1 STRONG — 4× more training data"
echo "============================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo ""
echo "Key differences from V1:"
echo "  Training sim T: 20s (V1 was 5s)"
echo "  Test sim T: 5s (same as V1)"
echo "  ICs: gaussian + two_clusters only"
echo "  Expected MVAR samples: ~32,400 (V1 was ~7,400)"
echo ""

export PYTHONPATH=src

python ROM_pipeline.py \
    --config configs/synthesis_v1_strong.yaml \
    --experiment_name synthesis_v1_strong

echo ""
echo "End time: $(date)"
echo "Exit code: $?"
