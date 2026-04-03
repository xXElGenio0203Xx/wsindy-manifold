#!/bin/bash
# ============================================================================
# Upload + Submit Thesis-Final Phase A to Oscar
# ============================================================================
# Syncs code + Phase A configs to Oscar, then submits the SLURM array job
# (7 tasks: sim + POD only, all models disabled).
#
# Usage:
#   ./shell_scripts/submit_thesis_final_phase_a.sh
#   ./shell_scripts/submit_thesis_final_phase_a.sh --dry-run
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
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=true; shift ;;
        --help|-h)   head -11 "$0" | tail -8; exit 0 ;;
        *)           echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "======================================================================"
echo "THESIS-FINAL PHASE A — OSCAR SUBMISSION"
echo "======================================================================"
echo ""

# ---- Step 1: Show what will be submitted ----
echo -e "${BLUE}Step 1: Phase A manifest${NC}"
MANIFEST="configs/systematic/thesis_final/phase_a/manifest.txt"
N_CONFIGS=$(wc -l < "$MANIFEST" | tr -d ' ')
echo -e "  ${GREEN}$N_CONFIGS configs (sim + POD only)${NC}"
cat "$MANIFEST" | while read -r cfg; do
    echo "    $(basename "$cfg" .yaml)"
done
echo ""

# ---- Step 2: Sync to Oscar ----
echo -e "${BLUE}Step 2: Syncing code & configs to Oscar...${NC}"

rsync -avz --relative \
    configs/systematic/thesis_final/ \
    slurm_scripts/run_thesis_final_phase_a.slurm \
    shell_scripts/verify_phase_a.sh \
    ROM_WSINDY_pipeline.py \
    src/ \
    requirements.txt \
    "$OSCAR_HOST:$OSCAR_DIR/" 2>&1 | tail -5

echo -e "  ${GREEN}✓ Sync complete${NC}"
echo ""

# ---- Step 3: Submit ----
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}DRY RUN — files synced but not submitted.${NC}"
    echo ""
    echo "To submit manually:"
    echo "  ssh $OSCAR_HOST"
    echo "  cd ~/wsindy-manifold"
    echo "  sbatch slurm_scripts/run_thesis_final_phase_a.slurm"
    exit 0
fi

echo -e "${BLUE}Step 3: Submitting SLURM array job (7 tasks, 4h, 96GB)...${NC}"

ssh "$OSCAR_HOST" << 'ENDSSH'
cd ~/wsindy-manifold || exit 1
# LOGIN NODE GUARD — do not add 'du', 'find -exec', or python training here
du() { echo "ERROR: 'du' is blocked on login nodes. Use 'myquota' or 'interact' first." >&2; return 1; }; export -f du
mkdir -p slurm_logs oscar_output

# Verify manifest
N=$(wc -l < configs/systematic/thesis_final/phase_a/manifest.txt | tr -d ' ')
echo "Manifest has $N configs"

# Verify environment
module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

# Submit
JOB_ID=$(sbatch slurm_scripts/run_thesis_final_phase_a.slurm | grep -oP '\d+')

if [ -n "$JOB_ID" ]; then
    echo ""
    echo "✓ Phase A job array submitted: $JOB_ID"
    squeue -u $USER -j $JOB_ID --format="%.18i %.12j %.8T %.12M %.12l %.6D %R" 2>/dev/null | head -10
else
    echo "✗ Submission failed"
    exit 1
fi
ENDSSH

echo ""
echo -e "${GREEN}======================================================================"
echo "PHASE A SUBMISSION COMPLETE!"
echo "======================================================================${NC}"
echo ""
echo "Monitor:"
echo "  ssh $OSCAR_HOST 'squeue -u \$USER'"
echo "  ssh $OSCAR_HOST 'tail -f ~/wsindy-manifold/slurm_logs/thesis_phA_*.out'"
echo ""
echo "After completion:"
echo "  ssh $OSCAR_HOST 'cd ~/wsindy-manifold && bash shell_scripts/verify_phase_a.sh'"
