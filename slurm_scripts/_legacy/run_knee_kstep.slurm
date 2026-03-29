#!/bin/bash
#SBATCH --job-name=knee_kstep
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/knee_kstep_%j.out
#SBATCH -e slurm_logs/knee_kstep_%j.err

# ============================================================================
# SUITE KNEE: k-step MVAR Diagnostics (K15-K16)
# ============================================================================
#
# K15: sqrt d=19 p=5 k-step MVAR k=20 alpha=1.0 H162
# K16: sqrt d=19 p=5 k-step MVAR k=10 alpha=0.5 H162
#
# Submit in parallel with run_knee_horizon.slurm
# Expected: ~3h each → ~6h total
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

export PYTHONPATH=src

echo "============================================================"
echo "SUITE KNEE — k-step MVAR Diagnostics (K15-K16)"
echo "============================================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""

PASS=0
FAIL=0
TOTAL=2
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
m = d.get('mvar', {})
r2 = m.get('mean_r2_test', '?')
r2_1s = m.get('mean_r2_1step_test', '?')
neg = m.get('mean_negativity_frac', '?')
sr = m.get('spectral_radius', '?')
kk = m.get('kstep_k', 0)
clamp = d.get('clamp_mode', '?')
print(f'     clamp={clamp} kstep_k={kk} spectral_radius={sr}')
if isinstance(r2, float):
    print(f'     R²_rollout={r2:+.4f}  R²_1step={r2_1s:.4f}  neg={neg:.4f}')
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

# ============== K15: k-step k=20, alpha=1.0 ==============

run_experiment \
    "configs/K15_v1_sqrtD19_p5_kstep20_a1_H162.yaml" \
    "K15_v1_sqrtD19_p5_kstep20_a1_H162" 1

# ============== K16: k-step k=10, alpha=0.5 ==============

run_experiment \
    "configs/K16_v1_sqrtD19_p5_kstep10_a05_H162.yaml" \
    "K16_v1_sqrtD19_p5_kstep10_a05_H162" 2

# ============== Summary ==============

echo ""
echo "============================================================"
echo "SUITE KNEE — K-STEP DIAGNOSTICS COMPLETE"
echo "============================================================"
echo "End time: $(date)"
echo "Passed: $PASS / $TOTAL"
echo "Failed: $FAIL / $TOTAL"
echo ""
echo "Results:"
echo -e "$RESULTS"
echo ""
echo "Done."
