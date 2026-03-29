#!/bin/bash
#SBATCH --job-name=suite_B
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=24G
#SBATCH --time=6:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/suite_B_%j.out
#SBATCH -e slurm_logs/suite_B_%j.err

# ============================================================================
# SUITE B: Isolate rollout failure + stabilization diagnostics
# ============================================================================
#
# B1–B4: Spectral scaling sweep (rhoFree / 1.00 / 0.99 / 0.97)   ~4×1min
# B5–B7: N_train sweep (54 / 176 / 400 sims)                     ~1+5+15min
# B8–B10: k-step teacher-forced (k=1 / k=10 / k=full)            ~3×1min
#
# Estimated total: ~30min.  6h limit gives ample buffer for B7.
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

export PYTHONPATH=src

echo "============================================================"
echo "SUITE B — Rollout Failure Isolation & Stabilization"
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

# ============== B1–B4: Spectral scaling ==============

run_experiment \
    "configs/suite_B_spec_detAdv_H150_rhoFree.yaml" \
    "suite_B_spec_detAdv_H150_rhoFree" \
    "B1"

run_experiment \
    "configs/suite_B_spec_detAdv_H150_rho1p00.yaml" \
    "suite_B_spec_detAdv_H150_rho1p00" \
    "B2"

run_experiment \
    "configs/suite_B_spec_detAdv_H150_rho0p99.yaml" \
    "suite_B_spec_detAdv_H150_rho0p99" \
    "B3"

run_experiment \
    "configs/suite_B_spec_detAdv_H150_rho0p97.yaml" \
    "suite_B_spec_detAdv_H150_rho0p97" \
    "B4"

# ============== B5–B7: N_train sweep ==============

run_experiment \
    "configs/suite_B_ntrain_detAdv_54sims.yaml" \
    "suite_B_ntrain_detAdv_54sims" \
    "B5"

run_experiment \
    "configs/suite_B_ntrain_detAdv_176sims.yaml" \
    "suite_B_ntrain_detAdv_176sims" \
    "B6"

run_experiment \
    "configs/suite_B_ntrain_detAdv_400sims.yaml" \
    "suite_B_ntrain_detAdv_400sims" \
    "B7"

# ============== B8–B10: k-step teacher forcing ==============

run_experiment \
    "configs/suite_B_tf_detAdv_kstep_1.yaml" \
    "suite_B_tf_detAdv_kstep_1" \
    "B8"

run_experiment \
    "configs/suite_B_tf_detAdv_kstep_10.yaml" \
    "suite_B_tf_detAdv_kstep_10" \
    "B9"

run_experiment \
    "configs/suite_B_tf_detAdv_kstep_full.yaml" \
    "suite_B_tf_detAdv_kstep_full" \
    "B10"

# ============== Final Summary ==============

echo ""
echo "============================================================"
echo "SUITE B COMPLETE"
echo "============================================================"
echo "End time: $(date)"
echo "Passed: $PASS / 10"
echo "Failed: $FAIL / 10"
echo ""
echo "Results:"
echo -e "$RESULTS"
echo ""
echo "─── Spectral scaling (B1–B4) ───"
echo "  B1 rhoFree:  baseline (ρ≈1.17)"
echo "  B2 rho1.00:  marginally stable"
echo "  B3 rho0.99:  mild contraction"
echo "  B4 rho0.97:  strong contraction"
echo ""
echo "─── N_train sweep (B5–B7) ───"
echo "  B5 54 sims:  baseline"
echo "  B6 176 sims: 3.3× more data"
echo "  B7 400 sims: 7.4× more data"
echo ""
echo "─── k-step TF (B8–B10) ───"
echo "  B8 k=1:    pure teacher-forced"
echo "  B9 k=10:   reset every 10 steps"
echo "  B10 k=full: autonomous rollout"
echo "============================================================"
