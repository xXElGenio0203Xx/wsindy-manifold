#!/bin/bash
#SBATCH --job-name=suite_C4_lstm
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=24G
#SBATCH --time=16:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/suite_C4_lstm_%j.out
#SBATCH -e slurm_logs/suite_C4_lstm_%j.err

# ============================================================================
# BLOCK C4: LSTM "fixed" re-entry
# ============================================================================
#
# C4-1: raw_best, w=20, k=5, alpha=0.3, H150
# C4-2: raw_best, w=40, k=10, alpha=0.5, H150
# C4-3: sqrt_best, w=20, k=5, alpha=0.3, H150
# C4-4: sqrt_best, w=40, k=10, alpha=0.5, H150
#
# LSTM training: ~15-30 min each (3000 epochs max, patience=150)
# Sim generation: ~90 min each (200 train, longer tests)
# Estimated total: ~10-14h.
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

export PYTHONPATH=src

echo "============================================================"
echo "BLOCK C4 — LSTM Fixed Re-entry (4 experiments)"
echo "============================================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""

PASS=0
FAIL=0
TOTAL=4
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
m = d.get('lstm', d.get('mvar', {}))
r2 = m.get('mean_r2_test', '?')
r2_1s = m.get('mean_r2_1step_test', '?')
neg = m.get('mean_negativity_frac', '?')
vloss = m.get('val_loss', '?')
if isinstance(r2, float):
    print(f'     R²_rollout={r2:.4f}  R²_1step={r2_1s:.4f}  neg={neg:.4f}  val_loss={vloss}')
else:
    print(f'     R²={r2}  val_loss={vloss}')
" 2>/dev/null || true
        fi
    else
        FAIL=$((FAIL + 1))
        RESULTS="$RESULTS\n  ❌ [$NUM] $NAME (exit=$EXIT_CODE)"
    fi
    echo ""
}

run_experiment \
    "configs/suite_C4_lstm_rawBest_w20_k5_alpha0p3_H150.yaml" \
    "suite_C4_lstm_rawBest_w20_k5_alpha0p3_H150" \
    1

run_experiment \
    "configs/suite_C4_lstm_rawBest_w40_k10_alpha0p5_H150.yaml" \
    "suite_C4_lstm_rawBest_w40_k10_alpha0p5_H150" \
    2

run_experiment \
    "configs/suite_C4_lstm_sqrtBest_w20_k5_alpha0p3_H150.yaml" \
    "suite_C4_lstm_sqrtBest_w20_k5_alpha0p3_H150" \
    3

run_experiment \
    "configs/suite_C4_lstm_sqrtBest_w40_k10_alpha0p5_H150.yaml" \
    "suite_C4_lstm_sqrtBest_w40_k10_alpha0p5_H150" \
    4

echo ""
echo "============================================================"
echo "BLOCK C4 COMPLETE"
echo "============================================================"
echo "Passed: $PASS / $TOTAL"
echo "Failed: $FAIL / $TOTAL"
echo ""
echo "Results:"
echo -e "$RESULTS"
echo ""
echo "Compare LSTM vs MVAR at same horizon (H150) to see if gap closed."
echo "============================================================"
