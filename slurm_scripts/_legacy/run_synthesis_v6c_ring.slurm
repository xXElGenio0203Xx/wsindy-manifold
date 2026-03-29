#!/bin/bash
#SBATCH --job-name=v6c_ring
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/synthesis_v6c_ring_%j.out
#SBATCH -e slurm_logs/synthesis_v6c_ring_%j.err

# ============================================================================
# V6c — Single Ring IC, Long Coherent Trajectory
# ============================================================================
#   1 training sim: N=400, T=120s, ring IC, speed=5, Ca=3
#   POD: 90% energy | MVAR: p=5, α=10 | LSTM: 128h × 3L × 5000ep
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate
cd ~/wsindy-manifold
export PYTHONPATH=src:$PYTHONPATH
mkdir -p slurm_logs

echo "============================================================"
echo "  V6c — Single RING IC (T=120s, N=400, speed=5)"
echo "  Start: $(date)"
echo "============================================================"

python ROM_pipeline.py \
    --config configs/synthesis_v6c_ring.yaml \
    --experiment_name synthesis_v6c_ring

echo "  End: $(date), Exit: $?"
