#!/bin/bash
#SBATCH --job-name=fastest_path
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=24G
#SBATCH --time=16:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/fastest_path_%j.out
#SBATCH -e slurm_logs/fastest_path_%j.err

# ============================================================================
# FASTEST PATH: 4 experiments to answer the key questions
# ============================================================================
#
# 1) C0 (all 3)           → declare raw_best, sqrt_best
# 2) C2-3 sqrt_best k=5   → k-step MVAR on sqrt_best @ H150
# 3) C4-3 lstm sqrt_best   → LSTM w20 k5 alpha0.3 @ H150
# 4) C3-2 simplex sqrt     → simplex projection @ H150
#
# After this, expand to horizon plots (C1).
#
# NOTE: This runs C0 first, then uses sqrt d=11 as the assumed winner
#       for C2/C3/C4. If C0 results change the winner, re-run those.
#
# Estimated total: ~12-14h (C0 ~4h, each H150 run ~2.5h)
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

export PYTHONPATH=src

echo "============================================================"
echo "FASTEST PATH — 7 experiments (C0 + C2-3 + C4-3 + C3-2)"
echo "============================================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""

PASS=0
FAIL=0
TOTAL=7
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
pod = d.get('pod', {})
r_pod = pod.get('r_pod', '?')
transform = pod.get('density_transform', '?')
clamp = d.get('clamp_mode', '?')
kk = m.get('kstep_k', 0)
print(f'     d={r_pod} transform={transform} clamp={clamp} kstep_k={kk}')
if isinstance(r2, float):
    print(f'     R²_rollout={r2:.4f}  R²_1step={r2_1s:.4f}  neg={neg:.4f}')
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

# ============== Phase 1: C0 — Declare winners ==============

echo ">>> PHASE 1: Block C0 — Sqrt vs Dimension Isolation"
echo ""

run_experiment \
    "configs/suite_C0_raw_D11_p5_V1.yaml" \
    "suite_C0_raw_D11_p5_V1" \
    1

run_experiment \
    "configs/suite_C0_sqrt_D19_p5_V1.yaml" \
    "suite_C0_sqrt_D19_p5_V1" \
    2

run_experiment \
    "configs/suite_C0_sqrt_D11_p5_V1.yaml" \
    "suite_C0_sqrt_D11_p5_V1" \
    3

echo ""
echo ">>> C0 done. Proceeding with sqrt d=11 as assumed sqrt_best."
echo ""

# ============== Phase 2: Best method probes @ H150 ==============

echo ">>> PHASE 2: C2-3 + C4-3 + C3-2 on sqrt_best @ H150"
echo ""

run_experiment \
    "configs/suite_C2_mvar_kstepTrain_sqrtBest_k5_p5_H150.yaml" \
    "suite_C2_mvar_kstepTrain_sqrtBest_k5_p5_H150" \
    4

run_experiment \
    "configs/suite_C4_lstm_sqrtBest_w20_k5_alpha0p3_H150.yaml" \
    "suite_C4_lstm_sqrtBest_w20_k5_alpha0p3_H150" \
    5

run_experiment \
    "configs/suite_C3_projSimplex_sqrtBest_H150.yaml" \
    "suite_C3_projSimplex_sqrtBest_H150" \
    6

# Bonus: also run baseline MVAR sqrt_best @ H150 for direct comparison
run_experiment \
    "configs/suite_C1_sqrtBest_H150.yaml" \
    "suite_C1_sqrtBest_H150" \
    7

echo ""
echo "============================================================"
echo "FASTEST PATH COMPLETE"
echo "============================================================"
echo "Passed: $PASS / $TOTAL"
echo "Failed: $FAIL / $TOTAL"
echo ""
echo "Results:"
echo -e "$RESULTS"
echo ""
echo "NEXT STEPS:"
echo "  1. Compare C0 results → confirm raw_best and sqrt_best"
echo "  2. Compare C2-3 vs C1-5 → does k-step MVAR help?"
echo "  3. Compare C4-3 vs C1-5 → does LSTM close the gap?"
echo "  4. Compare C3-2 vs C1-5 → does simplex beat clampC2?"
echo "  5. If winners found, expand to C1 horizon plots + H300 follow-ups"
echo "============================================================"
