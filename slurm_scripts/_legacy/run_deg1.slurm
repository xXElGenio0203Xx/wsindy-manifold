#!/bin/bash
#SBATCH --job-name=DEG1_degrad
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/DEG1_%j.out
#SBATCH -e slurm_logs/DEG1_%j.err

# ============================================================================
# DEG1 — R² Degradation: Very long horizon (T=200s, 10x training)
# ============================================================================
# Reuses pre-trained MVAR from ABL8_v2 (sqrt+simplex+align, R²≈0.97).
# Generates 4 test sims (one per IC type) at T=200s and evaluates.
# Expected runtime: ~60-120 min (mostly test sim generation)
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

export PYTHONPATH=src

echo "============================================================"
echo "R² DEGRADATION EXPERIMENT — DEG1"
echo "============================================================"
echo "Start time: $(date)"
echo "Node:       $(hostname)"
echo ""

python run_degradation_experiment.py \
    --base_experiment ABL8_N200_sqrt_simplex_align_H300_v2 \
    --config configs/DEG1_long_horizon_200s.yaml \
    --experiment_name DEG1_long_horizon_200s

EXIT_CODE=$?
echo ""
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"

if [ $EXIT_CODE -eq 0 ] && [ -f "oscar_output/DEG1_long_horizon_200s/summary.json" ]; then
    python3 -c "
import json
s = json.load(open('oscar_output/DEG1_long_horizon_200s/summary.json'))
print(f'Mean R²: {s[\"mean_r2_test\"]:.4f}')
print(f'Per-test R²:')
for k,v in s.get('per_test_r2',{}).items():
    print(f'  {k}: {v:.4f}')
" 2>/dev/null || true
fi

echo "Output: oscar_output/DEG1_long_horizon_200s/"
echo "Done."
