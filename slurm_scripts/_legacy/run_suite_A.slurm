#!/bin/bash
#SBATCH --job-name=suite_A
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/suite_A_%j.out
#SBATCH -e slurm_logs/suite_A_%j.err

# ============================================================================
# SUITE A: Targeted Diagnostics — Why does rollout fail on easy dynamics?
# ============================================================================
#
# A1. MVAR p3 + latent standardization    (~15 min)
# A2. MVAR p3 + d=20 (more POD modes)     (~15 min)
# A3. MVAR p5 (higher lag order)           (~15 min)
# A4. MVAR p3 + explicit C2 clamp         (~15 min, control)
# A5. LSTM w20 + latent standardization    (~60 min, LSTM training)
#
# Estimated: ~2h total. 8h limit gives ample buffer.
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

export PYTHONPATH=src

echo "============================================================"
echo "SUITE A — Targeted Rollout Diagnostics"
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
    local TAG="$3"
    
    echo ""
    echo "============================================================"
    echo "[$TAG] $NAME"
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
        RESULTS="$RESULTS\n  ✅ [$TAG] $NAME"
        if [ -f "oscar_output/$NAME/summary.json" ]; then
            python3 scripts/read_suite_results.py "oscar_output/$NAME/summary.json" 2>/dev/null || true
        fi
    else
        FAIL=$((FAIL + 1))
        RESULTS="$RESULTS\n  ❌ [$TAG] $NAME (exit=$EXIT_CODE)"
    fi
    
    echo ""
}

# ============== Suite A Experiments ==============

run_experiment \
    "configs/suite_A_det_advect_H150_mvar_p3_stdON.yaml" \
    "suite_A_det_advect_H150_mvar_p3_stdON" \
    "A1"

run_experiment \
    "configs/suite_A_det_advect_H150_mvar_p3_d20.yaml" \
    "suite_A_det_advect_H150_mvar_p3_d20" \
    "A2"

run_experiment \
    "configs/suite_A_det_advect_H150_mvar_p5.yaml" \
    "suite_A_det_advect_H150_mvar_p5" \
    "A3"

run_experiment \
    "configs/suite_A_det_advect_H150_mvar_p3_clampC2.yaml" \
    "suite_A_det_advect_H150_mvar_p3_clampC2" \
    "A4"

run_experiment \
    "configs/suite_A_det_advect_H150_lstm_w20_stdON.yaml" \
    "suite_A_det_advect_H150_lstm_w20_stdON" \
    "A5"

# ============== Final Summary ==============

echo ""
echo "============================================================"
echo "SUITE A COMPLETE — Rollout Diagnostics"
echo "============================================================"
echo "End time: $(date)"
echo "Passed: $PASS / 5"
echo "Failed: $FAIL / 5"
echo ""
echo "Results:"
echo -e "$RESULTS"
echo ""
echo "Comparison with baselines:"
echo "  Base MVAR p3:   R2_roll=+0.2838  R2_1step=+0.9993  neg=4.9%  d=12"
echo "  Base LSTM w20:  R2_roll=-0.3860  R2_1step=-0.1517  neg=6.8%  d=12"
echo "============================================================"
