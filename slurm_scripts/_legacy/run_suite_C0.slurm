#!/bin/bash
#SBATCH --job-name=suite_C0
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/suite_C0_%j.out
#SBATCH -e slurm_logs/suite_C0_%j.err

# ============================================================================
# BLOCK C0: Isolate WHY sqrt helps (transform vs lower d)  [TOP PRIORITY]
# ============================================================================
#
# C0-1: raw rho, d=11     → dimension-only effect
# C0-2: sqrt(rho), d=19   → transform-only effect
# C0-3: sqrt(rho), d=11   → confirm sqrt benefit at matched d
#
# Each run: ~200 train sims + 30 test → ~60-90 min each
# Estimated total: ~4h. 8h limit gives buffer.
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

export PYTHONPATH=src

echo "============================================================"
echo "BLOCK C0 — Sqrt vs Dimension Isolation (3 experiments)"
echo "============================================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""

PASS=0
FAIL=0
RESULTS=""

run_experiment() {
    local CONFIG="$1"
    local NAME="$2"
    local NUM="$3"
    local TOTAL="$4"
    
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
print(f'     d={r_pod}, transform={transform}')
if isinstance(r2, float):
    print(f'     R²_rollout={r2:.4f}  R²_1step={r2_1s:.4f}  neg_frac={neg:.4f}')
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

run_experiment \
    "configs/suite_C0_raw_D11_p5_V1.yaml" \
    "suite_C0_raw_D11_p5_V1" \
    1 3

run_experiment \
    "configs/suite_C0_sqrt_D19_p5_V1.yaml" \
    "suite_C0_sqrt_D19_p5_V1" \
    2 3

run_experiment \
    "configs/suite_C0_sqrt_D11_p5_V1.yaml" \
    "suite_C0_sqrt_D11_p5_V1" \
    3 3

echo ""
echo "============================================================"
echo "BLOCK C0 COMPLETE"
echo "============================================================"
echo "Passed: $PASS / 3"
echo "Failed: $FAIL / 3"
echo ""
echo "Results:"
echo -e "$RESULTS"
echo ""
echo "NEXT: Declare winners (raw_best, sqrt_best) and update C1/C2/C3/C4 configs"
echo "============================================================"
