#!/bin/bash
#SBATCH --job-name=v5_mega
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/synthesis_v5_mega_%j.out
#SBATCH -e slurm_logs/synthesis_v5_mega_%j.err

# ============================================================================
# SYNTHESIS V5 — MEGA AMBITIOUS
# ============================================================================
#
#   V2.4 champion ROM + dynamic physics + massive training data
#
#   Physics:  speed=5, Ca=3, R=5, η=0.1, N=200 particles
#   ICs:      500 train (gauss+uniform+two_clusters+ring), 50 test
#   Time:     T=12s (particles cross domain ~4×)
#   ROM:      d=15, p=4, α=10, NO scaling, clamp ON
#   MVAR:     915 params, 48K samples (52× overdetermined)
#   LSTM:     128h × 3L × 5000 epochs
#
#   Memory:   96GB requested (peak ~15GB, safe 6× margin)
#             V4 MEGA died at 64GB because 1080×500×2304 matrix
#             V5 snapshot: 500×100×2304 = 115M floats = 0.9GB — no risk
#
#   Time:     10hr wall (expect ~5-6hr)
#             Sims: 500 × N=200 × T=12s ≈ 2hr
#             LSTM: ~2-3hr
#             Test: ~15min
#
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
export PYTHONPATH=src:$PYTHONPATH
mkdir -p slurm_logs

echo "============================================================"
echo "  SYNTHESIS V5 — MEGA AMBITIOUS"
echo "============================================================"
echo "  Physics: speed=5, Ca=3, R=5, η=0.1, N=200"
echo "  ICs: 500 train (4 types), 50 test (4 types)"
echo "  T=12s, dt=0.04, subsample=3 → 100 frames/IC"
echo "  ROM: d=15, p=4, α=10, NO scaling, clamp ON"
echo "  LSTM: 128h × 3L × 5000 epochs"
echo "  Memory: 96GB requested"
echo "  Start: $(date)"
echo "============================================================"
echo ""

# Print memory info for debugging
echo "System memory info:"
free -h 2>/dev/null || echo "(free not available)"
echo ""

python ROM_pipeline.py \
    --config configs/synthesis_v5_mega.yaml \
    --experiment_name synthesis_v5_mega

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "  End: $(date)"
echo "  Exit code: ${EXIT_CODE}"
echo "============================================================"

# Print peak memory usage
echo ""
echo "Peak memory usage:"
cat /proc/self/status 2>/dev/null | grep -i "VmPeak\|VmHWM" || echo "(not available)"

exit ${EXIT_CODE}
