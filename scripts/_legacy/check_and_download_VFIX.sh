#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# check_and_download_VFIX.sh — Check VFIX status & download results
# ═══════════════════════════════════════════════════════════════════
# Usage:
#   bash scripts/check_and_download_VFIX.sh              # Check status only
#   bash scripts/check_and_download_VFIX.sh --download    # Check + download
# ═══════════════════════════════════════════════════════════════════
set -e

OSCAR_HOST="oscar"
REMOTE_DIR="~/wsindy-manifold/oscar_output"
LOCAL_DIR="./oscar_output"
DOWNLOAD=false

if [[ "$1" == "--download" ]]; then
    DOWNLOAD=true
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}═══════════════════════════════════════════${NC}"
echo -e "${CYAN}  VFIX Experiment Suite — Status Check     ${NC}"
echo -e "${CYAN}═══════════════════════════════════════════${NC}"
echo ""

# ─── Step 1: Check SLURM job status ─────────────────────────────
echo -e "${BLUE}[1/4] Checking SLURM job history...${NC}"
ssh "$OSCAR_HOST" "sacct -u emaciaso --starttime=2025-01-01 --format=JobID,JobName%30,State%12,Elapsed,ExitCode 2>/dev/null | grep -i vfix" 2>/dev/null || echo -e "${YELLOW}  No VFIX jobs found in sacct (may have aged out)${NC}"
echo ""

# ─── Step 2: Check which VFIX outputs exist on Oscar ────────────
echo -e "${BLUE}[2/4] Listing VFIX results on Oscar...${NC}"
REMOTE_EXPERIMENTS=$(ssh "$OSCAR_HOST" "ls -d ${REMOTE_DIR}/VFIX_* 2>/dev/null | xargs -I{} basename {}" 2>/dev/null || echo "")

if [[ -z "$REMOTE_EXPERIMENTS" ]]; then
    echo -e "${RED}  No VFIX experiment directories found on Oscar!${NC}"
    echo -e "  Expected at: ${REMOTE_DIR}/VFIX_*"
    echo ""
    echo -e "${YELLOW}Checking if results are in a different location...${NC}"
    ssh "$OSCAR_HOST" "find ~/wsindy-manifold -maxdepth 3 -type d -name 'VFIX_*' 2>/dev/null | head -20" 2>/dev/null || true
    echo ""
    exit 1
fi

TOTAL=$(echo "$REMOTE_EXPERIMENTS" | wc -l | tr -d ' ')
echo -e "  Found ${GREEN}${TOTAL}${NC} VFIX experiment directories on Oscar"
echo ""

# ─── Step 3: Check completeness (summary.json = finished) ──────
echo -e "${BLUE}[3/4] Checking experiment completeness...${NC}"
COMPLETED=0
FAILED=0
RUNNING=0

while IFS= read -r exp; do
    HAS_SUMMARY=$(ssh "$OSCAR_HOST" "test -f ${REMOTE_DIR}/${exp}/summary.json && echo yes || echo no" 2>/dev/null)
    HAS_LSTM=$(ssh "$OSCAR_HOST" "test -d ${REMOTE_DIR}/${exp}/LSTM && echo yes || echo no" 2>/dev/null)
    
    if [[ "$HAS_SUMMARY" == "yes" ]]; then
        # Check if LSTM test results exist
        HAS_TEST=$(ssh "$OSCAR_HOST" "test -f ${REMOTE_DIR}/${exp}/LSTM/test_results.csv && echo yes || echo no" 2>/dev/null)
        if [[ "$HAS_TEST" == "yes" ]]; then
            echo -e "  ${GREEN}✓${NC} ${exp} — COMPLETE"
            ((COMPLETED++))
        else
            echo -e "  ${YELLOW}~${NC} ${exp} — summary exists but no LSTM test results"
            ((FAILED++))
        fi
    else
        # Check if it's still running
        IS_RUNNING=$(ssh "$OSCAR_HOST" "squeue -u emaciaso --name=${exp} --noheader 2>/dev/null | wc -l | tr -d ' '" 2>/dev/null || echo "0")
        if [[ "$IS_RUNNING" -gt 0 ]]; then
            echo -e "  ${YELLOW}⏳${NC} ${exp} — STILL RUNNING"
            ((RUNNING++))
        else
            echo -e "  ${RED}✗${NC} ${exp} — FAILED or INCOMPLETE"
            ((FAILED++))
        fi
    fi
done <<< "$REMOTE_EXPERIMENTS"

echo ""
echo -e "  ${GREEN}Completed: $COMPLETED${NC} | ${YELLOW}Running: $RUNNING${NC} | ${RED}Failed: $FAILED${NC} | Total: $TOTAL"
echo ""

# ─── Step 4: Download if requested ─────────────────────────────
if [[ "$DOWNLOAD" == true && "$COMPLETED" -gt 0 ]]; then
    echo -e "${BLUE}[4/4] Downloading VFIX results...${NC}"
    mkdir -p "$LOCAL_DIR"
    
    while IFS= read -r exp; do
        HAS_SUMMARY=$(ssh "$OSCAR_HOST" "test -f ${REMOTE_DIR}/${exp}/summary.json && echo yes || echo no" 2>/dev/null)
        if [[ "$HAS_SUMMARY" == "yes" ]]; then
            echo -e "  Downloading ${CYAN}${exp}${NC}..."
            rsync -avz --progress \
                --exclude='train/' \
                --exclude='__pycache__/' \
                --exclude='*.pyc' \
                "${OSCAR_HOST}:${REMOTE_DIR}/${exp}/" \
                "${LOCAL_DIR}/${exp}/" 2>/dev/null
            echo -e "  ${GREEN}Done${NC}"
        fi
    done <<< "$REMOTE_EXPERIMENTS"
    
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Download complete! Results in: ${LOCAL_DIR}/${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════${NC}"
else
    if [[ "$DOWNLOAD" == false ]]; then
        echo -e "${YELLOW}To download results, run:${NC}"
        echo -e "  bash scripts/check_and_download_VFIX.sh --download"
    fi
fi

echo ""
echo -e "${BLUE}Quick-peek at R² scores (if available):${NC}"
ssh "$OSCAR_HOST" bash -s <<'REMOTE_SCRIPT'
for d in ~/wsindy-manifold/oscar_output/VFIX_*/; do
    exp=$(basename "$d")
    summary="$d/summary.json"
    if [ -f "$summary" ]; then
        r2=$(python3 -c "
import json
with open('$summary') as f:
    s = json.load(f)
lstm = s.get('LSTM', s.get('lstm', {}))
rollout = lstm.get('rollout_r2', lstm.get('r2_rollout', 'N/A'))
print(f'{rollout:.4f}' if isinstance(rollout, (int,float)) else rollout)
" 2>/dev/null || echo "parse-err")
        printf "  %-35s  R²=%s\n" "$exp" "$r2"
    fi
done
REMOTE_SCRIPT
