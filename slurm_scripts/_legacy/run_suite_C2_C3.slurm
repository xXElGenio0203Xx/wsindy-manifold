#!/bin/bash
#SBATCH --job-name=suite_C2_C3
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=24G
#SBATCH --time=16:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/suite_C2_C3_%j.out
#SBATCH -e slurm_logs/suite_C2_C3_%j.err

# ============================================================================
# BLOCKS C2 + C3: k-step MVAR training + Simplex projection
# ============================================================================
#
# C2: 4 experiments (k-step MVAR @ H150)
#   C2-1: raw_best, k=5
#   C2-2: raw_best, k=10
#   C2-3: sqrt_best, k=5
#   C2-4: sqrt_best, k=10
#
# C3: 2 experiments (simplex projection @ H150)
#   C3-1: raw_best
#   C3-2: sqrt_best
#
# k-step MVAR adds ~5 min training per run (Adam on tiny model).
# Estimated total: ~10-12h.
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

export PYTHONPATH=src

echo "============================================================"
echo "BLOCKS C2 + C3 — k-step MVAR + Simplex (6 experiments)"
echo "============================================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""

PASS=0
FAIL=0
TOTAL=6
RESULTS=""

run_experiment() {
    local CONFIG="$1"
    local NAME="$2"
    local NUM="$3"
    
    echo ""
    echo "============================================================"
    echo "[$NUM/$TOTAL] $NAME"
    echo "============================================================"
    echo "Config: $CONFIG"
    echo "Start: $(date)"
    
    python ROM_pipeline.py \
        --config "$CONFIG" \
        --experiment_name "$NAME"
    
    local EXIT_CODE=$?
    echo "Exit code: $EXIT_CODE"
    echo "End: $(date)"
    
    if [ $EXIT_CODE -eq 0 ]; then
        PASS=$((PASS + 1))
        RESULTS="$RESULTS\n  ✅ [$NUM] $NAME"
        if [ -f "oscar_output/$NAME/summary.json" ]; then
            python3 -c "
import json
d = json.load(open('oscar_output/$NAME/summary.json'))
m = d.get('mvar', d.get('lstm', {}))
r2 = m.get('mean_r2_test', '?')
r2_1s = m.get('mean_r2_1step_test', '?')
neg = m.get('mean_negativity_frac', '?')
kk = m.get('kstep_k', 0)
clamp = d.get('clamp_mode', '?')
if isinstance(r2, float):
    print(f'     R²_rollout={r2:.4f}  R²_1step={r2_1s:.4f}  neg={neg:.4f}  kstep_k={kk}  clamp={clamp}')
else:
    print(f'     R²={r2}')
" 2>/dev/null || true
        fi
    else
        FAIL=$((FAIL + 1))
        RESULTS="$RESULTS\n  ❌ [$NUM] $NAME (exit=$EXIT_CODE)"
    fi
    echo ""
}

# ============== C2: k-step MVAR ==============

run_experiment \
    "configs/suite_C2_mvar_kstepTrain_rawBest_k5_p5_H150.yaml" \
    "suite_C2_mvar_kstepTrain_rawBest_k5_p5_H150" \
    1

run_experiment \
    "configs/suite_C2_mvar_kstepTrain_rawBest_k10_p5_H150.yaml" \
    "suite_C2_mvar_kstepTrain_rawBest_k10_p5_H150" \
    2

run_experiment \
    "configs/suite_C2_mvar_kstepTrain_sqrtBest_k5_p5_H150.yaml" \
    "suite_C2_mvar_kstepTrain_sqrtBest_k5_p5_H150" \
    3

run_experiment \
    "configs/suite_C2_mvar_kstepTrain_sqrtBest_k10_p5_H150.yaml" \
    "suite_C2_mvar_kstepTrain_sqrtBest_k10_p5_H150" \
    4

# ============== C3: Simplex Projection ==============

run_experiment \
    "configs/suite_C3_projSimplex_rawBest_H150.yaml" \
    "suite_C3_projSimplex_rawBest_H150" \
    5

run_experiment \
    "configs/suite_C3_projSimplex_sqrtBest_H150.yaml" \
    "suite_C3_projSimplex_sqrtBest_H150" \
    6

echo ""
echo "============================================================"
echo "BLOCKS C2 + C3 COMPLETE"
echo "============================================================"
echo "Passed: $PASS / $TOTAL"
echo "Failed: $FAIL / $TOTAL"
echo ""
echo "Results:"
echo -e "$RESULTS"
echo "============================================================"
