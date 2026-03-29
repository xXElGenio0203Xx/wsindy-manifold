#!/bin/bash
# ============================================================================
# Upload + Submit Main-Regime Thesis Sweep to Oscar
# ============================================================================
# Runs only configs/systematic/main_regimes/*.yaml as a dedicated SLURM array.
#
# Usage:
#   ./shell_scripts/submit_main_regimes.sh
#   ./shell_scripts/submit_main_regimes.sh --dry-run
#   ./shell_scripts/submit_main_regimes.sh --range 0-3
# ============================================================================

set -euo pipefail

OSCAR_HOST="oscar"
OSCAR_DIR="~/wsindy-manifold"
MANIFEST_LOCAL="configs/systematic/main_regimes/manifest.txt"
MANIFEST_REMOTE="configs/systematic/main_regimes/manifest.txt"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

DRY_RUN=false
ARRAY_RANGE=""
MEMORY="64G"
TIME_LIMIT="24:00:00"
ARRAY_MAX_CONCURRENT="4"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --range) ARRAY_RANGE="$2"; shift 2 ;;
        --mem) MEMORY="$2"; shift 2 ;;
        --time) TIME_LIMIT="$2"; shift 2 ;;
        --max-concurrent) ARRAY_MAX_CONCURRENT="$2"; shift 2 ;;
        --help|-h)
            head -12 "$0" | tail -9
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "======================================================================"
echo "MAIN REGIMES THESIS SWEEP — OSCAR SUBMISSION"
echo "======================================================================"
echo ""

echo -e "${BLUE}Step 1: Generating main-regime manifest...${NC}"
ls configs/systematic/main_regimes/*.yaml | sort > "$MANIFEST_LOCAL"
N_CONFIGS=$(wc -l < "$MANIFEST_LOCAL" | tr -d ' ')
LAST_INDEX=$((N_CONFIGS - 1))
echo -e "  ${GREEN}$N_CONFIGS configs in manifest${NC}"
echo ""

echo "  Configs:"
nl -ba "$MANIFEST_LOCAL" | while read -r idx cfg; do
    echo "    [$((idx - 1))] $(basename "$cfg" .yaml)"
done
echo ""

echo -e "${BLUE}Step 2: Syncing code & configs to Oscar...${NC}"
RSYNC_LOG=$(mktemp)
rsync -avz --relative \
    configs/systematic/main_regimes/ \
    slurm_scripts/run_systematic.slurm \
    ROM_WSINDY_pipeline.py \
    src/ \
    requirements.txt \
    "$OSCAR_HOST:$OSCAR_DIR/" > "$RSYNC_LOG" 2>&1
tail -3 "$RSYNC_LOG"
rm -f "$RSYNC_LOG"
echo -e "  ${GREEN}✓ Sync complete${NC}"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}DRY RUN — not submitting. Files are synced to Oscar.${NC}"
    echo ""
    echo "To submit manually:"
    echo "  ssh $OSCAR_HOST"
    echo "  cd ~/wsindy-manifold"
    echo "  sbatch --array=0-$LAST_INDEX --export=ALL,MANIFEST_PATH=$MANIFEST_REMOTE slurm_scripts/run_systematic.slurm"
    exit 0
fi

ARRAY_SPEC="$ARRAY_RANGE"
if [ -z "$ARRAY_SPEC" ]; then
    ARRAY_SPEC="0-$LAST_INDEX"
fi

echo -e "${BLUE}Step 3: Submitting SLURM array job...${NC}"
echo "  Manifest: $MANIFEST_REMOTE"
echo "  Array:    $ARRAY_SPEC%$ARRAY_MAX_CONCURRENT"
echo "  Memory:   $MEMORY"
echo "  Time:     $TIME_LIMIT"

ssh "$OSCAR_HOST" << ENDSSH
cd ~/wsindy-manifold || exit 1
mkdir -p slurm_logs oscar_output

N=\$(wc -l < "$MANIFEST_REMOTE" | tr -d ' ')
echo "Manifest has \$N configs"

JOB_ID=\$(sbatch \
    --array=${ARRAY_SPEC}%${ARRAY_MAX_CONCURRENT} \
    --job-name=mainregimes \
    --mem=$MEMORY \
    --time=$TIME_LIMIT \
    --output=slurm_logs/mainregimes_%A_%a.out \
    --error=slurm_logs/mainregimes_%A_%a.err \
    --export=ALL,MANIFEST_PATH=$MANIFEST_REMOTE \
    slurm_scripts/run_systematic.slurm | grep -oP '\d+')

if [ -n "\$JOB_ID" ]; then
    echo ""
    echo "✓ Job array submitted: \$JOB_ID"
    squeue -u \$USER -j \$JOB_ID --format="%.18i %.12j %.8T %.12M %.12l %.6D %R" 2>/dev/null | head -5
else
    echo "✗ Submission failed"
    exit 1
fi
ENDSSH

echo ""
echo -e "${GREEN}======================================================================"
echo "SUBMISSION COMPLETE!"
echo "======================================================================${NC}"
echo ""
echo "Monitor:"
echo "  ssh $OSCAR_HOST 'squeue -u \$USER'"
echo "  ssh $OSCAR_HOST 'tail -f ~/wsindy-manifold/slurm_logs/mainregimes_*.out'"
