#!/bin/bash
# ============================================================================
# Upload + Submit Main-Regime WSINDy-Only Sweep to Oscar
# ============================================================================
# Uses the existing 7 main_regimes configs, which are already at N=100.
# This submits one regime per array task and writes to oscar_output/*_wsindy_v2.
#
# Usage:
#   ./shell_scripts/submit_main_regimes_wsindy_v2.sh
#   ./shell_scripts/submit_main_regimes_wsindy_v2.sh --dry-run
#   ./shell_scripts/submit_main_regimes_wsindy_v2.sh --range 0-3
# ============================================================================

set -euo pipefail

OSCAR_HOST="oscar"
OSCAR_DIR="~/wsindy-manifold"
MANIFEST_LOCAL="configs/systematic/main_regimes/manifest.txt"
MANIFEST_REMOTE="configs/systematic/main_regimes/manifest.txt"
OUTPUT_SUFFIX="wsindy_v2"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

DRY_RUN=false
ARRAY_RANGE=""
MEMORY="96G"
TIME_LIMIT="06:00:00"
ARRAY_MAX_CONCURRENT="4"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --range) ARRAY_RANGE="$2"; shift 2 ;;
        --mem) MEMORY="$2"; shift 2 ;;
        --time) TIME_LIMIT="$2"; shift 2 ;;
        --max-concurrent) ARRAY_MAX_CONCURRENT="$2"; shift 2 ;;
        --suffix) OUTPUT_SUFFIX="$2"; shift 2 ;;
        --help|-h)
            head -12 "$0" | tail -9
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "======================================================================"
echo "MAIN REGIMES WSINDY-ONLY SWEEP — OSCAR SUBMISSION"
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
    shell_scripts/submit_main_regimes_wsindy_v2.sh \
    slurm_scripts/run_main_regimes_wsindy_only.slurm \
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
    echo "  sbatch --array=0-$LAST_INDEX --export=ALL,MANIFEST_PATH=$MANIFEST_REMOTE,OUTPUT_SUFFIX=$OUTPUT_SUFFIX slurm_scripts/run_main_regimes_wsindy_only.slurm"
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
echo "  Suffix:   $OUTPUT_SUFFIX"

ssh "$OSCAR_HOST" << ENDSSH
cd ~/wsindy-manifold || exit 1
# LOGIN NODE GUARD — do not add 'du', 'find -exec', or python training here
du() { echo "ERROR: 'du' is blocked on login nodes. Use 'myquota' or 'interact' first." >&2; return 1; }; export -f du
mkdir -p slurm_logs oscar_output

N=\$(wc -l < "$MANIFEST_REMOTE" | tr -d ' ')
echo "Manifest has \$N configs"

JOB_ID=\$(sbatch \
    --array=${ARRAY_SPEC}%${ARRAY_MAX_CONCURRENT} \
    --job-name=mainwsyv2 \
    --mem=$MEMORY \
    --time=$TIME_LIMIT \
    --output=slurm_logs/mainwsyv2_%A_%a.out \
    --error=slurm_logs/mainwsyv2_%A_%a.err \
    --export=ALL,MANIFEST_PATH=$MANIFEST_REMOTE,OUTPUT_SUFFIX=$OUTPUT_SUFFIX \
    slurm_scripts/run_main_regimes_wsindy_only.slurm | grep -oP '\d+')

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
echo "  ssh $OSCAR_HOST 'tail -f ~/wsindy-manifold/slurm_logs/mainwsyv2_*.out'"
