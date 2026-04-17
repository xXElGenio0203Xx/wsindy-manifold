#!/bin/bash
# ============================================================================
# watch_tier1_and_submit.sh  — Run ON OSCAR (in screen/tmux/nohup)
# ============================================================================
# Monitors tier1 tasks 0-4.  When they ALL stop running, submits the
# WSINDy-only fallback (no bootstrap) for experiments that have training
# data but no WSINDy results yet.
#
# Usage:  nohup bash scripts/watch_tier1_and_submit.sh > /tmp/watcher.log 2>&1 &
# ============================================================================
cd ~/wsindy-manifold

FALLBACK="slurm_scripts/run_tier1_wsindy_noboot.slurm"
CHECK_INTERVAL=300        # seconds between checks (5 min)
MAX_CHECKS=200            # safety: stop after ~16 hours

echo "================================================"
echo " Tier1 Watcher — started $(date)"
echo " Will submit ${FALLBACK} when tasks 0-4 finish."
echo " Check interval: ${CHECK_INTERVAL}s"
echo "================================================"

for c in $(seq 1 $MAX_CHECKS); do
    # Count running tier1 tasks using a single squeue call
    RUNNING=$(squeue -u emaciaso -j 1426337 --format="%i %t" 2>/dev/null \
        | grep -E "1426337_[0-4] " | grep -c " R" || true)

    echo "[$(date '+%H:%M:%S')] Check $c/$MAX_CHECKS: $RUNNING/5 tasks running"

    if [ "$RUNNING" = "0" ]; then
        echo ""
        echo "*** All tasks 0-4 finished/timed out! ***"

        # Check which experiments need WSINDy
        NEED=0
        for exp in NDYN04_gas_tier1_w5 NDYN04_gas_tier1_bic NDYN04_gas_VS_tier1_w5 NDYN04_gas_VS_tier1_bic NDYN05_blackhole_tier1_w5; do
            d="oscar_output/${exp}"
            if [ -d "$d/train" ] && [ -f "$d/train/metadata.json" ] && \
               ! [ -f "$d/WSINDy/multifield_model.json" ]; then
                echo "  NEEDS WSINDy: $exp"
                NEED=$((NEED + 1))
            else
                echo "  OK or no data: $exp"
            fi
        done

        if [ "$NEED" -gt 0 ]; then
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
