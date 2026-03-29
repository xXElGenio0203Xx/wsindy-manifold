#!/bin/bash
#SBATCH --job-name=suite_C1_horizon
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=24G
#SBATCH --time=18:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/suite_C1_horizon_%j.out
#SBATCH -e slurm_logs/suite_C1_horizon_%j.err

# ============================================================================
# BLOCK C1: Horizon curves (raw_best vs sqrt_best) — THESIS-GRADE PLOT
# ============================================================================
#
# C1-1: raw_best H37      (~60 min — short test sims)
# C1-2: raw_best H150     (~90 min — medium test sims)
# C1-3: raw_best H300     (~120 min — long test sims)
# C1-4: sqrt_best H37
# C1-5: sqrt_best H150
# C1-6: sqrt_best H300
#
# Run AFTER C0 declares winners. Update fixed_modes if needed.
# Estimated total: ~10-14h.
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

export PYTHONPATH=src

echo "============================================================"
echo "BLOCK C1 — Horizon Curves (6 experiments)"
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
pod = d.get('pod', {})
r_pod = pod.get('r_pod', '?')
transform = pod.get('density_transform', '?')
if isinstance(r2, float):
    print(f'     d={r_pod}, transform={transform}, R²_rollout={r2:.4f}, R²_1step={r2_1s:.4f}')
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

# ============== raw_best horizon ==============

run_experiment \
    "configs/suite_C1_rawBest_H37.yaml" \
    "suite_C1_rawBest_H37" \
    1

run_experiment \
    "configs/suite_C1_rawBest_H150.yaml" \
    "suite_C1_rawBest_H150" \
    2

run_experiment \
    "configs/suite_C1_rawBest_H300.yaml" \
    "suite_C1_rawBest_H300" \
    3

# ============== sqrt_best horizon ==============

run_experiment \
    "configs/suite_C1_sqrtBest_H37.yaml" \
    "suite_C1_sqrtBest_H37" \
    4

run_experiment \
    "configs/suite_C1_sqrtBest_H150.yaml" \
    "suite_C1_sqrtBest_H150" \
    5

run_experiment \
    "configs/suite_C1_sqrtBest_H300.yaml" \
    "suite_C1_sqrtBest_H300" \
    6

echo ""
echo "============================================================"
echo "BLOCK C1 COMPLETE — Horizon Curves"
echo "============================================================"
echo "Passed: $PASS / $TOTAL"
echo "Failed: $FAIL / $TOTAL"
echo ""
echo "Results:"
echo -e "$RESULTS"
echo ""
echo "Plot: R²_rollout vs H for raw_best and sqrt_best"
echo "============================================================"
