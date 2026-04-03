#!/bin/bash
# ============================================================================
# Upload + Submit LSTM Fix Experiments to Oscar
# ============================================================================
# Phase 1: Run 6 LFIX configs (LST4 architecture on hard regimes)
# Phase 2: (optional) Rerun all 54 systematic configs with fixed LSTM params
#
# Usage:
#   ./shell_scripts/submit_lstm_fix.sh                    # submit LFIX only
#   ./shell_scripts/submit_lstm_fix.sh --full              # also resubmit systematics
#   ./shell_scripts/submit_lstm_fix.sh --dry-run           # upload only
# ============================================================================

set -e

OSCAR_HOST="oscar"
OSCAR_DIR="~/wsindy-manifold"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

DRY_RUN=false
FULL_RERUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=true; shift ;;
        --full)      FULL_RERUN=true; shift ;;
        --help|-h)
            head -10 "$0" | tail -7
            exit 0 ;;
        *)  echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "======================================================================"
echo "LSTM FIX — OSCAR SUBMISSION"
echo "======================================================================"
echo ""
echo "  LFIX configs:      6 (LST4 arch on hard regimes)"
if [ "$FULL_RERUN" = true ]; then
    echo "  Systematic rerun:  54 (all configs with fixed LSTM params)"
fi
echo ""

# ---- Step 1: Sync everything to Oscar ----
echo -e "${BLUE}Step 1: Syncing code & configs to Oscar...${NC}"

rsync -avz --relative \
    configs/lstm_fix/ \
    configs/systematic/ \
    slurm_scripts/run_lstm_fix.slurm \
    slurm_scripts/run_systematic.slurm \
    ROM_WSINDY_pipeline.py \
    src/ \
    requirements.txt \
    "$OSCAR_HOST:$OSCAR_DIR/" 2>&1 | tail -5

echo -e "  ${GREEN}✓ Sync complete${NC}"
echo ""

# ---- Step 2: Show what changed in LSTM params ----
echo -e "${BLUE}LSTM config changes (vs. old systematic):${NC}"
echo "  learning_rate:  0.001   → 0.0007"
echo "  batch_size:     256     → 512"
echo "  gradient_clip:  1.0     → 5.0"
echo "  patience:       30      → 40"
echo "  lag:            5       → 20"
echo "  hidden_units:   128     → 64"
echo "  normalize_input: (default) → true (explicit)"
echo "  weight_decay:   (none)  → 1e-5"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}DRY RUN — files synced, not submitting.${NC}"
    echo ""
    echo "To submit manually on Oscar:"
    echo "  ssh $OSCAR_HOST"
    echo "  cd $OSCAR_DIR"
    echo "  sbatch slurm_scripts/run_lstm_fix.slurm          # 6 LFIX configs"
    if [ "$FULL_RERUN" = true ]; then
        echo "  sbatch slurm_scripts/run_systematic.slurm        # 54 systematic"
    fi
    exit 0
fi

# ---- Step 3: Submit LFIX ----
echo -e "${BLUE}Step 2: Submitting LFIX experiments (6 configs, sequential)...${NC}"

ssh "$OSCAR_HOST" << 'ENDSSH'
cd ~/wsindy-manifold || exit 1
# LOGIN NODE GUARD — do not add 'du', 'find -exec', or python training here
du() { echo "ERROR: 'du' is blocked on login nodes. Use 'myquota' or 'interact' first." >&2; return 1; }; export -f du
mkdir -p slurm_logs oscar_output

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

LFIX_JOB=$(sbatch slurm_scripts/run_lstm_fix.slurm | grep -oP '\d+')

if [ -n "$LFIX_JOB" ]; then
    echo "✓ LFIX job submitted: $LFIX_JOB"
    squeue -u $USER -j $LFIX_JOB --format="%.18i %.12j %.8T %.12M %.12l %.6D %R" 2>/dev/null | head -5
else
    echo "✗ LFIX submission failed"
    exit 1
fi
ENDSSH

echo -e "  ${GREEN}✓ LFIX job submitted${NC}"
echo ""

# ---- Step 4: Optionally resubmit full systematic suite ----
if [ "$FULL_RERUN" = true ]; then
    echo -e "${BLUE}Step 3: Submitting full systematic rerun (54 configs, array)...${NC}"
    echo -e "${YELLOW}  ⚠ This will rerun ALL experiments with fixed LSTM params${NC}"
    echo ""

    # Regenerate manifest on Oscar
    ssh "$OSCAR_HOST" << 'ENDSSH'
cd ~/wsindy-manifold || exit 1
# LOGIN NODE GUARD — do not add 'du', 'find -exec', or python training here
du() { echo "ERROR: 'du' is blocked on login nodes. Use 'myquota' or 'interact' first." >&2; return 1; }; export -f du

# Regenerate manifest
ls configs/systematic/*.yaml | sort > configs/systematic/manifest.txt
N=$(wc -l < configs/systematic/manifest.txt | tr -d ' ')
echo "Manifest has $N configs"

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

SYS_JOB=$(sbatch slurm_scripts/run_systematic.slurm | grep -oP '\d+')

if [ -n "$SYS_JOB" ]; then
    echo "✓ Systematic array job submitted: $SYS_JOB"
    squeue -u $USER -j $SYS_JOB --format="%.18i %.12j %.8T %.12M %.12l %.6D %R" 2>/dev/null | head -5
else
    echo "✗ Systematic submission failed"
    exit 1
fi
ENDSSH

    echo -e "  ${GREEN}✓ Systematic jobs submitted${NC}"
fi

echo ""
echo "======================================================================"
echo "DONE"
echo "======================================================================"
echo ""
echo "Monitor with:"
echo "  ssh $OSCAR_HOST 'squeue -u \$USER'"
echo ""
echo "Check results:"
echo "  rsync -avz $OSCAR_HOST:$OSCAR_DIR/oscar_output/LFIX_* oscar_output/"
