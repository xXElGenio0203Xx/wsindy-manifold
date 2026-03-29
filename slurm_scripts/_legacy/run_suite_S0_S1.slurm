#!/bin/bash
#SBATCH --job-name=suite_S0_S1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/suite_S0_S1_%j.out
#SBATCH -e slurm_logs/suite_S0_S1_%j.err

# ============================================================================
# SUITE S0 + S1: 10 experiments in sequence
# ============================================================================
#
# S0 (sanity checks):
#   1. Det advection, MVAR p=1, H37      (~5 min)
#   2. Det advection, MVAR p=3, H37      (~5 min)
#   3. Det advection, MVAR p=1, H150     (~8 min — longer test sims)
#   4. Det advection, MVAR p=3, H150     (~8 min)
#   5. Det advection, LSTM w=10, H37     (~15 min — LSTM training)
#   6. Det advection, LSTM w=20, H150    (~15 min)
#   7. Frozen identity, MVAR p=1         (~3 min)
#   8. Frozen identity, LSTM w=10        (~10 min)
#
# S1 (POD bottleneck probes on V1 regime):
#   9.  POD E=80%, MVAR p=5              (~90 min — 200 train runs)
#   10. POD E=90%, MVAR p=5              (~90 min — V1 reproduction)
#
# Estimated total: ~4h. 12h limit gives plenty of buffer.
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

export PYTHONPATH=src

echo "============================================================"
echo "SUITE S0 + S1 — 10 experiments"
echo "============================================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo ""

# Track successes and failures
PASS=0
FAIL=0
RESULTS=""

run_experiment() {
    local CONFIG="$1"
    local NAME="$2"
    local NUM="$3"
    
    echo ""
    echo "============================================================"
    echo "[$NUM/10] $NAME"
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
        # Extract R² from summary.json if available
        if [ -f "oscar_output/$NAME/summary.json" ]; then
            python3 -c "
import json
d = json.load(open('oscar_output/$NAME/summary.json'))
m = d.get('mvar', d.get('lstm', {}))
r2 = m.get('mean_r2_test', '?')
r2_1s = m.get('mean_r2_1step_test', '?')
neg = m.get('mean_negativity_frac', '?')
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

# ============== S0: Sanity Checks ==============

run_experiment \
    "configs/suite_S0_det_advect_H37_mvar_p1.yaml" \
    "suite_S0_det_advect_H37_mvar_p1" \
    1

run_experiment \
    "configs/suite_S0_det_advect_H37_mvar_p3.yaml" \
    "suite_S0_det_advect_H37_mvar_p3" \
    2

run_experiment \
    "configs/suite_S0_det_advect_H150_mvar_p1.yaml" \
    "suite_S0_det_advect_H150_mvar_p1" \
    3

run_experiment \
    "configs/suite_S0_det_advect_H150_mvar_p3.yaml" \
    "suite_S0_det_advect_H150_mvar_p3" \
    4

run_experiment \
    "configs/suite_S0_det_advect_H37_lstm_w10.yaml" \
    "suite_S0_det_advect_H37_lstm_w10" \
    5

run_experiment \
    "configs/suite_S0_det_advect_H150_lstm_w20.yaml" \
    "suite_S0_det_advect_H150_lstm_w20" \
    6

run_experiment \
    "configs/suite_S0_frozen_identity_mvar_p1.yaml" \
    "suite_S0_frozen_identity_mvar_p1" \
    7

run_experiment \
    "configs/suite_S0_frozen_identity_lstm_w10.yaml" \
    "suite_S0_frozen_identity_lstm_w10" \
    8

# ============== S1: POD Bottleneck Probes ==============

run_experiment \
    "configs/suite_S1_podE80_raw_mvar_p5.yaml" \
    "suite_S1_podE80_raw_mvar_p5" \
    9

run_experiment \
    "configs/suite_S1_podE90_raw_mvar_p5.yaml" \
    "suite_S1_podE90_raw_mvar_p5" \
    10

# ============== Final Summary ==============

echo ""
echo "============================================================"
echo "SUITE S0 + S1 COMPLETE"
echo "============================================================"
echo "Total time: started $(date)"
echo "Passed: $PASS / 10"
echo "Failed: $FAIL / 10"
echo ""
echo "Results:"
echo -e "$RESULTS"
echo ""
echo "============================================================"
