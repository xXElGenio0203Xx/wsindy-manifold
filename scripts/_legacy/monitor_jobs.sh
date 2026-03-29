#!/bin/bash
# ============================================================================
# Auto-monitor all OSCAR jobs and collect results
# Run locally: bash scripts/monitor_jobs.sh
# ============================================================================

JOBS="342371 342372 342399 342410"
NAMES=("V2.3" "V2.4" "V3.1" "V4_MEGA")
LOG_FILE="job_monitor_results.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "Job Monitor Started: $(date)" | tee -a "$LOG_FILE"
echo "Monitoring jobs: $JOBS" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

while true; do
    echo "" | tee -a "$LOG_FILE"
    echo "--- Check at $(date) ---" | tee -a "$LOG_FILE"
    
    ALL_DONE=true
    
    for i in "${!NAMES[@]}"; do
        JOB_ID=$(echo $JOBS | cut -d' ' -f$((i+1)))
        NAME="${NAMES[$i]}"
        
        # Get job state
        STATE=$(ssh oscar "sacct -j $JOB_ID --format=State --noheader -P | head -1" 2>/dev/null)
        ELAPSED=$(ssh oscar "sacct -j $JOB_ID --format=Elapsed --noheader -P | head -1" 2>/dev/null)
        
        echo "  $NAME (Job $JOB_ID): $STATE  Elapsed: $ELAPSED" | tee -a "$LOG_FILE"
        
        if [[ "$STATE" == "COMPLETED" ]]; then
            # Try to grab summary
            SUMMARY=$(ssh oscar "cat ~/wsindy-manifold/oscar_output/synthesis_*/summary.json 2>/dev/null" | grep -A2 "\"experiment_name\": \"$NAME\"" 2>/dev/null)
            
            # Get the right experiment name for each job
            case $NAME in
                "V2.3") EXP="synthesis_v2_3" ;;
                "V2.4") EXP="synthesis_v2_4" ;;
                "V3.1") EXP="synthesis_v3_1" ;;
                "V4_MEGA") EXP="synthesis_v4_megaheavyweight" ;;
            esac
            
            MVAR_R2=$(ssh oscar "python3 -c \"import json; d=json.load(open('oscar_output/$EXP/summary.json')); print(f'MVAR R2={d[\\\"mvar\\\"][\\\"mean_r2_test\\\"]:.6f}')\" 2>/dev/null" 2>/dev/null)
            LSTM_R2=$(ssh oscar "python3 -c \"import json; d=json.load(open('oscar_output/$EXP/summary.json')); print(f'LSTM R2={d[\\\"lstm\\\"][\\\"mean_r2_test\\\"]:.6f}')\" 2>/dev/null" 2>/dev/null)
            
            if [[ -n "$MVAR_R2" ]]; then
                echo "    >>> $MVAR_R2" | tee -a "$LOG_FILE"
                echo "    >>> $LSTM_R2" | tee -a "$LOG_FILE"
            else
                echo "    >>> Summary not found yet" | tee -a "$LOG_FILE"
            fi
        elif [[ "$STATE" == "FAILED" || "$STATE" == "TIMEOUT" || "$STATE" == "CANCELLED" ]]; then
            echo "    >>> JOB $STATE — check logs" | tee -a "$LOG_FILE"
            # Grab last error
            case $NAME in
                "V2.3") LOGF="synthesis_v2_3_342371" ;;
                "V2.4") LOGF="synthesis_v2_4_342372" ;;
                "V3.1") LOGF="synthesis_v3_1_342399" ;;
                "V4_MEGA") LOGF="synthesis_v4_megaheavyweight_342410" ;;
            esac
            ssh oscar "tail -5 ~/wsindy-manifold/slurm_logs/${LOGF}.err 2>/dev/null" 2>/dev/null | tee -a "$LOG_FILE"
        elif [[ "$STATE" == "RUNNING" ]]; then
            ALL_DONE=false
        else
            ALL_DONE=false
        fi
    done
    
    if $ALL_DONE; then
        echo "" | tee -a "$LOG_FILE"
        echo "========================================" | tee -a "$LOG_FILE"
        echo "ALL JOBS FINISHED at $(date)" | tee -a "$LOG_FILE"
        echo "========================================" | tee -a "$LOG_FILE"
        
        # Final summary dump
        echo "" | tee -a "$LOG_FILE"
        echo "=== FINAL RESULTS ===" | tee -a "$LOG_FILE"
        for EXP in synthesis_v2_3 synthesis_v2_4 synthesis_v3_1 synthesis_v4_megaheavyweight; do
            echo "" | tee -a "$LOG_FILE"
            echo "--- $EXP ---" | tee -a "$LOG_FILE"
            ssh oscar "cd ~/wsindy-manifold && cat oscar_output/$EXP/summary.json 2>/dev/null | python3 -c \"
import sys,json
try:
    d=json.load(sys.stdin)
    print(f'  MVAR: R2_train={d[\\\"mvar\\\"][\\\"r2_train\\\"]:.4f}, R2_test={d[\\\"mvar\\\"][\\\"mean_r2_test\\\"]:.6f}')
    print(f'  LSTM: val_loss={d[\\\"lstm\\\"][\\\"val_loss\\\"]:.4f}, R2_test={d[\\\"lstm\\\"][\\\"mean_r2_test\\\"]:.6f}')
    print(f'  POD modes: {d[\\\"r_pod\\\"]}, Train ICs: {d[\\\"n_train\\\"]}, Test ICs: {d[\\\"n_test\\\"]}')
    print(f'  Total time: {d[\\\"total_time_minutes\\\"]:.1f} min')
except: print('  (no summary available)')
\"" 2>/dev/null | tee -a "$LOG_FILE"
        done
        
        break
    fi
    
    echo "  Sleeping 5 minutes..." | tee -a "$LOG_FILE"
    sleep 300
done

echo ""
echo "Results saved to: $LOG_FILE"
