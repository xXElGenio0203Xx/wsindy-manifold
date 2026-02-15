#!/bin/bash
#SBATCH --job-name=v3_4v1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/synthesis_v3_4v1_%j.out
#SBATCH -e slurm_logs/synthesis_v3_4v1_%j.err

# ============================================================================
# V3.4v1 — Extreme Dynamics, Single Long Gaussian IC
# ============================================================================
#   1 training sim: N=300, T=60s, gaussian IC, speed=5, Ca=3
#   POD: 95% energy | MVAR: p=7, α=10 | LSTM: 128h × 3L × 5000ep
#   
#   HYPOTHESIS: Single coherent trajectory + deeper lag + more POD modes
#   can beat V3.4's 261 mixed ICs (MVAR R²=+0.080)
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate
cd ~/wsindy-manifold
export PYTHONPATH=src:$PYTHONPATH
mkdir -p slurm_logs

echo "============================================================"
echo "  V3.4v1 — Single Gaussian IC (T=60s, N=300, speed=5)"
echo "  Start: $(date)"
echo "============================================================"

python ROM_pipeline.py \
    --config configs/synthesis_v3_4v1.yaml \
    --experiment_name synthesis_v3_4v1

echo "  End: $(date), Exit: $?"
