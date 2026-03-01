#!/bin/bash
# ============================================================================
# Upload + Submit Systematic Regime Sweep to Oscar
# ============================================================================
# Syncs code + configs to Oscar, generates the manifest, and submits
# the SLURM array job.
#
# Usage:
#   ./shell_scripts/submit_systematic.sh              # upload + submit all 54
#   ./shell_scripts/submit_systematic.sh --dry-run    # upload only, show what would run
#   ./shell_scripts/submit_systematic.sh --range 0-25 # submit a subset
# ============================================================================

set -e

OSCAR_HOST="oscar"
OSCAR_DIR="~/wsindy-manifold"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

DRY_RUN=false
ARRAY_RANGE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=true; shift ;;
        --range)     ARRAY_RANGE="$2"; shift 2 ;;
        --help|-h)
            head -12 "$0" | tail -9
            exit 0 ;;
        *)           echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "======================================================================"
echo "SYSTEMATIC REGIME SWEEP — OSCAR SUBMISSION"
echo "======================================================================"
echo ""

# ---- Step 1: Generate manifest locally ----
echo -e "${BLUE}Step 1: Generating config manifest...${NC}"
MANIFEST_LOCAL="configs/systematic/manifest.txt"
ls configs/systematic/*.yaml | sort > "$MANIFEST_LOCAL"
N_CONFIGS=$(wc -l < "$MANIFEST_LOCAL" | tr -d ' ')
echo -e "  ${GREEN}$N_CONFIGS configs in manifest${NC}"
echo ""

# Show configs
echo "  Configs:"
cat "$MANIFEST_LOCAL" | while read -r cfg; do
    name=$(basename "$cfg" .yaml)
    echo "    [$(($(grep -n "$cfg" "$MANIFEST_LOCAL" | cut -d: -f1) - 1))] $name"
done
echo ""

# ---- Step 2: Upload code + configs to Oscar ----
echo -e "${BLUE}Step 2: Syncing code & configs to Oscar...${NC}"

rsync -avz --relative \
    configs/systematic/ \
    slurm_scripts/run_systematic.slurm \
    ROM_pipeline.py \
    src/ \
    requirements.txt \
    "$OSCAR_HOST:$OSCAR_DIR/" 2>&1 | tail -3

echo -e "  ${GREEN}✓ Sync complete${NC}"
echo ""

# ---- Step 3: Submit ----
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}DRY RUN — not submitting. Files are synced to Oscar.${NC}"
    echo ""
    echo "To submit manually:"
    echo "  ssh $OSCAR_HOST"
    echo "  cd ~/wsindy-manifold"
    echo "  sbatch slurm_scripts/run_systematic.slurm"
    exit 0
fi

echo -e "${BLUE}Step 3: Submitting SLURM array job...${NC}"

SBATCH_ARGS=""
if [ -n "$ARRAY_RANGE" ]; then
    SBATCH_ARGS="--array=$ARRAY_RANGE"
    echo -e "  Using custom array range: ${YELLOW}$ARRAY_RANGE${NC}"
fi

ssh "$OSCAR_HOST" << ENDSSH
cd ~/wsindy-manifold || exit 1
mkdir -p slurm_logs oscar_output

# Verify manifest
N=\$(wc -l < configs/systematic/manifest.txt | tr -d ' ')
echo "Manifest has \$N configs"

# Verify environment
module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate
python -c "import numpy, torch; print('numpy', numpy.__version__, 'torch', torch.__version__)"

# Submit
JOB_ID=\$(sbatch $SBATCH_ARGS slurm_scripts/run_systematic.slurm | grep -oP '\d+')

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
echo "  ssh $OSCAR_HOST 'tail -f ~/wsindy-manifold/slurm_logs/systematic_*.out'"
echo ""
echo "Download results when done:"
echo "  ./shell_scripts/download_from_oscar.sh --all"
echo ""
echo "Run analysis pipeline:"
echo "  python plot_pipeline.py --data_dir oscar_output"
