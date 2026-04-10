#!/bin/bash
# ============================================================================
# watch_tier1_and_submit.sh  — Run ON OSCAR (in screen/tmux)
# ============================================================================
# Monitors tier1 tasks 0-4.  When they ALL stop running, submits the
# WSINDy-only fallback (no bootstrap) for experiments that have training
# data but no WSINDy results yet.
#
# Usage:  screen -S watcher   # or tmux
#         bash scripts/watch_tier1_and_submit.sh
# ============================================================================
set -uo pipefail
cd ~/wsindy-manifold

JOB_BASE="1426337"       # tier1 array job ID
FALLBACK="slurm_scripts/run_tier1_wsindy_noboot.slurm"
CHECK_INTERVAL=300        # seconds between checks (5 min)
MAX_CHECKS=200            # safety: stop after ~16 hours

echo "================================================"
echo " Tier1 Watcher — started $(date)"
echo " Monitoring job ${JOB_BASE} tasks 0-4"
echo " Will submit ${FALLBACK} when they finish."
echo " Check interval: ${CHECK_INTERVAL}s"
echo "================================================"

for ((c=1; c<=MAX_CHECKS; c++)); do
    # Count running tier1 tasks (tasks 0-4 specifically)
    RUNNING=0
    for i in 0 1 2 3 4; do
        state=$(scontrol show job "${JOB_BASE}_${i}" 2>/dev/null | grep -o 'JobState=[A-Z]*' | cut -d= -f2 || echo "UNKNOWN")
        if [[ "$state" == "RUNNING" ]]; then
            RUNNING=$((RUNNING + 1))
        fi
    done

    echo "[$(date '+%H:%M:%S')] Check $c/$MAX_CHECKS: $RUNNING/5 tasks still running"

    if [[ $RUNNING -eq 0 ]]; then
        echo ""
        echo "*** All tasks 0-4 finished/timed out! ***"

        # Check which experiments need WSINDy
        NEED=0
        for exp in NDYN04_gas_tier1_w5 NDYN04_gas_tier1_bic NDYN04_gas_VS_tier1_w5 NDYN04_gas_VS_tier1_bic NDYN05_blackhole_tier1_w5; do
            d="oscar_output/${exp}"
            if [ -d "$d/train" ] && [ -f "$d/train/metadata.json" ] && \
               ! [ -f "$d/WSINDy/multifield_model.json" ]; then
                echo "  NEEDS WSINDy: $exp"
                ((NEED++)) || true
            else
                echo "  OK or no data: $exp"
            fi
        done

        if [[ $NEED -gt 0 ]]; then
            echo ""
            echo "Submitting fallback: sbatch ${FALLBACK}"
            sbatch "${FALLBACK}"
            echo "Submitted at $(date)"
        else
            echo "All 5 experiments already have WSINDy results. Nothing to submit."
        fi

        echo ""
        echo "Watcher exiting."
        exit 0
    fi

    sleep $CHECK_INTERVAL
done

echo ""
echo "WARNING: Max checks ($MAX_CHECKS) reached. Exiting without submitting."
echo "Check manually: squeue -u emaciaso"
exit 1
