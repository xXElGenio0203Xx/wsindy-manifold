#!/bin/bash
###############################################################################
# auto_fetch_and_visualize.sh
# ============================
# Polls OSCAR for job completion, downloads results (excluding bulky train data),
# runs run_visualizations.py locally, and logs summary R².
#
# Usage:
#   bash scripts/auto_fetch_and_visualize.sh           # interactive
#   nohup bash scripts/auto_fetch_and_visualize.sh &   # background / go to sleep
#
# Logs everything to: auto_fetch_results.log
#
# Compatible with macOS bash 3.x (no associative arrays).
###############################################################################

# ======================== CONFIGURATION ========================

OSCAR_HOST="oscar"
OSCAR_BASE="~/wsindy-manifold/oscar_output"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_BASE="$PROJECT_ROOT/oscar_output"
LOG_FILE="$PROJECT_ROOT/auto_fetch_results.log"
POLL_INTERVAL=300   # seconds between polls (5 min)

# Job arrays (parallel indexed arrays — bash 3.x compatible)
JOB_IDS=(342371 342372 342399 342410)
JOB_NAMES=("synthesis_v2_3" "synthesis_v2_4" "synthesis_v3_1" "synthesis_v4_megaheavyweight")
JOB_STATUS=("pending" "pending" "pending" "pending")

NUM_JOBS=${#JOB_IDS[@]}

# ======================== FUNCTIONS ========================

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

check_job_state() {
    local job_id=$1
    local state
    state=$(ssh "$OSCAR_HOST" "sacct -j $job_id --format=State --noheader -P 2>/dev/null | head -1 | tr -d ' '" 2>/dev/null || echo "UNKNOWN")
    echo "$state"
}

extract_r2_from_oscar() {
    local exp_name=$1
    ssh "$OSCAR_HOST" "python3 -c \"
import json, sys
try:
    with open('$OSCAR_BASE/$exp_name/summary.json') as f:
        s = json.load(f)
    mvar = s.get('mvar_test_r2', 'N/A')
    lstm = s.get('lstm_test_r2', 'N/A')
    if mvar == 'N/A':
        tm = s.get('test_metrics', {})
        if 'mvar' in tm:
            mvar = tm['mvar'].get('mean_r2', 'N/A')
        if 'lstm' in tm:
            lstm = tm['lstm'].get('mean_r2', 'N/A')
    print('MVAR_R2={}'.format(mvar))
    print('LSTM_R2={}'.format(lstm))
except Exception as e:
    print('ERROR={}'.format(e))
\"" 2>/dev/null || echo "ERROR=ssh_failed"
}

download_results() {
    local exp_name=$1
    local local_dir="$LOCAL_BASE/$exp_name"

    log "📥 Downloading $exp_name results..."
    mkdir -p "$local_dir"

    # 1. Top-level files (config, summary)
    log "   ↓ Top-level files..."
    rsync -az \
        "$OSCAR_HOST:$OSCAR_BASE/$exp_name/*.json" \
        "$OSCAR_HOST:$OSCAR_BASE/$exp_name/*.yaml" \
        "$local_dir/" 2>/dev/null || true

    # 2. rom_common/ (POD basis, ~4MB)
    log "   ↓ rom_common/..."
    rsync -az \
        "$OSCAR_HOST:$OSCAR_BASE/$exp_name/rom_common/" \
        "$local_dir/rom_common/" 2>/dev/null || true

    # 3. MVAR/ model (~13KB)
    log "   ↓ MVAR/..."
    rsync -az \
        "$OSCAR_HOST:$OSCAR_BASE/$exp_name/MVAR/" \
        "$local_dir/MVAR/" 2>/dev/null || true

    # 4. LSTM/ model (~235KB)
    log "   ↓ LSTM/..."
    rsync -az \
        "$OSCAR_HOST:$OSCAR_BASE/$exp_name/LSTM/" \
        "$local_dir/LSTM/" 2>/dev/null || true

    # 5. test/ (all test runs with density predictions)
    log "   ↓ test/ (predictions + metrics)..."
    rsync -az \
        "$OSCAR_HOST:$OSCAR_BASE/$exp_name/test/" \
        "$local_dir/test/" 2>/dev/null || true

    # 6. train/metadata.json (needed for viz pipeline)
    log "   ↓ train/metadata.json..."
    mkdir -p "$local_dir/train"
    rsync -az \
        "$OSCAR_HOST:$OSCAR_BASE/$exp_name/train/metadata.json" \
        "$local_dir/train/" 2>/dev/null || true

    # 7. First training run (needed for grid info: xgrid, ygrid)
    local first_train
    first_train=$(ssh "$OSCAR_HOST" "ls -d $OSCAR_BASE/$exp_name/train/train_000 2>/dev/null || echo ''" 2>/dev/null)
    if [ -n "$first_train" ]; then
        log "   ↓ train/train_000/ (grid info, ~2MB)..."
        rsync -az \
            "$OSCAR_HOST:$first_train/" \
            "$local_dir/train/train_000/" 2>/dev/null || true
    fi

    # Report download size
    local size
    size=$(du -sh "$local_dir" 2>/dev/null | cut -f1)
    log "   ✅ Download complete: $size total"
}

run_viz() {
    local exp_name=$1
    log "🎨 Running visualization pipeline for $exp_name..."

    cd "$PROJECT_ROOT"
    python run_visualizations.py --experiment_name "$exp_name" >> "$LOG_FILE" 2>&1
    local exit_code=$?

    if [ "$exit_code" -eq 0 ]; then
        log "   ✅ Visualizations complete! Output: predictions/$exp_name/"
    else
        log "   ⚠️  Visualization pipeline exit code: $exit_code"
    fi
    return $exit_code
}

print_results_table() {
    log ""
    log "╔═══════════════════════════════════════════════════════════════╗"
    log "║                    RESULTS SUMMARY                          ║"
    log "╠═══════════════════════════════════════════════════════════════╣"

    local i=0
    while [ $i -lt $NUM_JOBS ]; do
        local job_id="${JOB_IDS[$i]}"
        local exp="${JOB_NAMES[$i]}"
        local status="${JOB_STATUS[$i]}"

        if [ "$status" = "done" ]; then
            local summary="$LOCAL_BASE/$exp/summary.json"
            local mvar_r2="N/A"
            local lstm_r2="N/A"
            if [ -f "$summary" ]; then
                mvar_r2=$(python3 -c "
import json
with open('$summary') as f:
    s = json.load(f)
mvar = s.get('mvar_test_r2', s.get('test_metrics', {}).get('mvar', {}).get('mean_r2', 'N/A'))
print(mvar)
" 2>/dev/null || echo "N/A")
                lstm_r2=$(python3 -c "
import json
with open('$summary') as f:
    s = json.load(f)
lstm = s.get('lstm_test_r2', s.get('test_metrics', {}).get('lstm', {}).get('mean_r2', 'N/A'))
print(lstm)
" 2>/dev/null || echo "N/A")
            fi
            log "║  $job_id | $exp"
            log "║         ✅ MVAR R²=$mvar_r2  LSTM R²=$lstm_r2"
        elif [ "$status" = "failed" ]; then
            log "║  $job_id | $exp"
            log "║         ❌ FAILED"
        else
            log "║  $job_id | $exp"
            log "║         ⏳ $status"
        fi
        i=$((i + 1))
    done

    log "╚═══════════════════════════════════════════════════════════════╝"
    log ""
}

# ======================== MAIN LOOP ========================

log ""
log "════════════════════════════════════════════════════════════"
log "  AUTO FETCH & VISUALIZE — Started"
log "  Monitoring $NUM_JOBS jobs, polling every ${POLL_INTERVAL}s"
log "  Log: $LOG_FILE"
log "════════════════════════════════════════════════════════════"
log ""

i=0
while [ $i -lt $NUM_JOBS ]; do
    log "  📋 Job ${JOB_IDS[$i]} → ${JOB_NAMES[$i]}"
    i=$((i + 1))
done
log ""

while true; do
    n_done=0
    i=0

    while [ $i -lt $NUM_JOBS ]; do
        job_id="${JOB_IDS[$i]}"
        exp_name="${JOB_NAMES[$i]}"
        status="${JOB_STATUS[$i]}"

        # Skip already processed
        if [ "$status" = "done" ] || [ "$status" = "failed" ]; then
            n_done=$((n_done + 1))
            i=$((i + 1))
            continue
        fi

        # Check state
        state=$(check_job_state "$job_id")
        log "🔍 Job $job_id ($exp_name): $state"

        case "$state" in
            COMPLETED)
                log "🎉 Job $job_id COMPLETED!"

                # Quick R² peek before downloading
                log "   Checking R² on OSCAR..."
                r2_output=$(extract_r2_from_oscar "$exp_name")
                log "   $r2_output"

                # Download
                download_results "$exp_name"

                # Visualize
                run_viz "$exp_name"

                JOB_STATUS[$i]="done"
                n_done=$((n_done + 1))

                # Download slurm log
                rsync -az "$OSCAR_HOST:~/wsindy-manifold/slurm_logs/*${job_id}*" "$PROJECT_ROOT/slurm_logs/" 2>/dev/null || true

                print_results_table
                ;;

            FAILED|TIMEOUT|CANCELLED|OUT_OF_ME*)
                log "❌ Job $job_id $state!"
                JOB_STATUS[$i]="failed"
                n_done=$((n_done + 1))

                # Download slurm log for debugging
                rsync -az "$OSCAR_HOST:~/wsindy-manifold/slurm_logs/*${job_id}*" "$PROJECT_ROOT/slurm_logs/" 2>/dev/null || true

                # Try partial download
                log "   Attempting partial results download..."
                download_results "$exp_name" || true
                ;;

            RUNNING|PENDING)
                # Check sim progress
                progress=$(ssh "$OSCAR_HOST" "grep -c 'Progress: 100%' ~/wsindy-manifold/slurm_logs/*${job_id}*.out 2>/dev/null || echo 0" 2>/dev/null)
                log "   Sims completed so far: $progress"
                ;;

            *)
                log "   Unknown state: $state"
                ;;
        esac

        i=$((i + 1))
    done

    # Check if all done
    if [ "$n_done" -eq "$NUM_JOBS" ]; then
        log ""
        log "════════════════════════════════════════════════════════════"
        log "  ALL JOBS COMPLETE — Final Summary"
        log "════════════════════════════════════════════════════════════"
        print_results_table

        log "📂 Visualizations saved to:"
        i=0
        while [ $i -lt $NUM_JOBS ]; do
            exp="${JOB_NAMES[$i]}"
            if [ -d "$PROJECT_ROOT/predictions/$exp" ]; then
                log "   predictions/$exp/"
            fi
            i=$((i + 1))
        done

        log ""
        log "🏁 Auto-fetch pipeline finished. Exiting."
        break
    fi

    log "💤 Sleeping ${POLL_INTERVAL}s until next check... ($n_done/$NUM_JOBS done)"
    sleep "$POLL_INTERVAL"
done
