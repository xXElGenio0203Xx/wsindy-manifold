#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# quick_vfix_check.sh — Fast single-SSH check of all VFIX results
# ═══════════════════════════════════════════════════════════════════
# Runs ONE SSH session, executes everything remotely, prints summary.
# Usage:  bash scripts/quick_vfix_check.sh
# ═══════════════════════════════════════════════════════════════════

echo "Connecting to Oscar..."
ssh oscar bash -s <<'EOF'
echo "═══════════════════════════════════════════════════════"
echo "  VFIX Experiment Status — $(date)"
echo "═══════════════════════════════════════════════════════"
echo ""

# 1) SLURM job history
echo "━━━ SLURM Job History (VFIX) ━━━"
sacct -u emaciaso --starttime=2025-01-01 \
  --format=JobID,JobName%35,State%12,Elapsed,ExitCode 2>/dev/null \
  | head -1
sacct -u emaciaso --starttime=2025-01-01 \
  --format=JobID,JobName%35,State%12,Elapsed,ExitCode 2>/dev/null \
  | grep -i vfix || echo "  (no VFIX jobs in sacct — may have aged out)"
echo ""

# 2) Currently queued/running
echo "━━━ Currently Queued/Running ━━━"
squeue -u emaciaso --format="%.10i %.35j %.8T %.10M %.6D %R" 2>/dev/null \
  | grep -i vfix || echo "  (no VFIX jobs currently running)"
echo ""

# 3) Output directories
echo "━━━ VFIX Output Directories ━━━"
OUTDIR=~/wsindy-manifold/oscar_output
VFIX_DIRS=$(ls -d ${OUTDIR}/VFIX_* 2>/dev/null)

if [ -z "$VFIX_DIRS" ]; then
    echo "  No VFIX_* directories found in $OUTDIR"
    echo ""
    echo "  Searching elsewhere..."
    find ~/wsindy-manifold -maxdepth 4 -type d -name "VFIX_*" 2>/dev/null | head -20
    # Also check /oscar/scratch
    find /oscar/scratch/emaciaso -maxdepth 4 -type d -name "VFIX_*" 2>/dev/null | head -20
else
    TOTAL=$(echo "$VFIX_DIRS" | wc -l)
    echo "  Found ${TOTAL} directories"
    echo ""
    
    COMPLETED=0
    FAILED=0
    
    echo "━━━ Per-Experiment Status ━━━"
    printf "  %-40s %-10s %-15s %s\n" "EXPERIMENT" "STATUS" "R²_ROLLOUT" "NOTES"
    printf "  %-40s %-10s %-15s %s\n" "──────────" "──────" "──────────" "─────"
    
    for d in $VFIX_DIRS; do
        exp=$(basename "$d")
        
        if [ -f "$d/summary.json" ]; then
            STATUS="DONE"
            ((COMPLETED++))
            
            # Extract R² from summary.json
            R2=$(python3 -c "
import json, sys
try:
    with open('$d/summary.json') as f:
        s = json.load(f)
    # Try different possible key paths
    lstm = s.get('LSTM', s.get('lstm', {}))
    r2 = lstm.get('rollout_r2', lstm.get('r2_rollout', lstm.get('test_r2_mean', None)))
    if r2 is None:
        # Maybe it's at top level
        r2 = s.get('rollout_r2', s.get('r2_rollout', 'N/A'))
    if isinstance(r2, (int, float)):
        print(f'{r2:.4f}')
    else:
        print(r2)
except Exception as e:
    print(f'err:{e}')
" 2>/dev/null || echo "parse-err")
            
            # Check test results
            if [ -f "$d/LSTM/test_results.csv" ]; then
                NOTES="test_results OK"
            else
                NOTES="no test_results.csv"
            fi
            
            printf "  %-40s \033[0;32m%-10s\033[0m %-15s %s\n" "$exp" "$STATUS" "$R2" "$NOTES"
        else
            ((FAILED++))
            # Check if partially done
            if [ -d "$d/LSTM" ]; then
                NOTES="LSTM dir exists, no summary"
            elif [ -d "$d/rom_common" ]; then
                NOTES="rom_common exists, LSTM missing"
            elif [ -d "$d/train" ]; then
                NOTES="only train data"
            else
                NOTES="nearly empty"
            fi
            printf "  %-40s \033[0;31m%-10s\033[0m %-15s %s\n" "$exp" "FAILED" "—" "$NOTES"
        fi
    done
    
    echo ""
    echo "━━━ Summary ━━━"
    echo "  Completed: $COMPLETED / $TOTAL"
    echo "  Failed:    $FAILED / $TOTAL"
fi

echo ""

# 4) Disk usage
echo "━━━ Disk Usage ━━━"
du -sh ${OUTDIR}/VFIX_* 2>/dev/null | sort -h || echo "  N/A"
echo ""
echo "  Total VFIX:"
du -sh ${OUTDIR}/VFIX_* 2>/dev/null | awk '{sum+=$1} END {print "  "sum" (units vary)"}' || echo "  N/A"

echo ""

# 5) Check slurm logs for errors
echo "━━━ Recent VFIX Slurm Logs (last errors) ━━━"
for log in ~/wsindy-manifold/slurm_logs/*VFIX*.out ~/wsindy-manifold/slurm_logs/*vfix*.out; do
    if [ -f "$log" ]; then
        ERRS=$(grep -ci "error\|traceback\|exception\|killed\|oom" "$log" 2>/dev/null || echo "0")
        if [ "$ERRS" -gt 0 ]; then
            echo "  $(basename $log): $ERRS error lines"
            grep -i "error\|traceback\|exception" "$log" 2>/dev/null | tail -3 | sed 's/^/    /'
        fi
    fi
done
echo "  Done."

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Check complete."
echo "═══════════════════════════════════════════════════════"
EOF
