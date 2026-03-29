#!/bin/bash
# ============================================================================
# Upload + Submit VDYN Clamp-Mode Comparison to Oscar
# ============================================================================
# Syncs code + configs to Oscar and submits the SLURM array job (14 tasks).
#
# Usage:
#   ./shell_scripts/submit_vdyn_clamp.sh              # upload + submit all 14
#   ./shell_scripts/submit_vdyn_clamp.sh --dry-run    # upload only
#   ./shell_scripts/submit_vdyn_clamp.sh --range 0-6  # C0 only
#   ./shell_scripts/submit_vdyn_clamp.sh --range 7-13 # C2 only
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
ARRAY_RANGE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=true; shift ;;
        --range)     ARRAY_RANGE="$2"; shift 2 ;;
        --help|-h)
            head -12 "$0" | tail -7
            exit 0 ;;
        *)           echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "======================================================================"
echo "VDYN CLAMP-MODE COMPARISON — OSCAR SUBMISSION"
echo "======================================================================"
echo ""

# ---- Step 1: Generate manifest ----
echo -e "${BLUE}Step 1: Generating config manifest...${NC}"
MANIFEST_LOCAL="configs/vdyn_clamp/manifest.txt"
ls configs/vdyn_clamp/*.yaml | sort > "$MANIFEST_LOCAL"
N_CONFIGS=$(wc -l < "$MANIFEST_LOCAL" | tr -d ' ')
echo -e "  ${GREEN}$N_CONFIGS configs in manifest${NC}"
echo ""

echo "  Configs:"
IDX=0
while read -r cfg; do
    name=$(basename "$cfg" .yaml)
    printf "    [%2d] %s\n" "$IDX" "$name"
    IDX=$((IDX + 1))
done < "$MANIFEST_LOCAL"
echo ""

# ---- Step 2: Upload to Oscar ----
echo -e "${BLUE}Step 2: Syncing code & configs to Oscar...${NC}"

rsync -avz --relative \
    configs/vdyn_clamp/ \
    slurm_scripts/run_vdyn_clamp.slurm \
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
    echo "  sbatch slurm_scripts/run_vdyn_clamp.slurm"
    exit 0
fi

echo -e "${BLUE}Step 3: Submitting SLURM array job...${NC}"

SBATCH_ARGS=""
if [ -n "$ARRAY_RANGE" ]; then
    SBATCH_ARGS="--array=$ARRAY_RANGE"
    echo -e "  Custom array range: ${YELLOW}$ARRAY_RANGE${NC}"
fi

ssh "$OSCAR_HOST" << ENDSSH
cd ~/wsindy-manifold || exit 1
mkdir -p slurm_logs oscar_output

N=\$(wc -l < configs/vdyn_clamp/manifest.txt | tr -d ' ')
echo "Manifest has \$N configs"

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate
python -c "import numpy, torch; print('numpy', numpy.__version__, 'torch', torch.__version__)"

JOB_ID=\$(sbatch $SBATCH_ARGS slurm_scripts/run_vdyn_clamp.slurm | grep -oP '\d+')

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
echo "VDYN CLAMP COMPARISON SUBMITTED"
echo "======================================================================"
echo ""
echo "  14 experiments: 7 VDYN dynamics × {C0, C2}"
echo "  Monitor: ssh oscar 'squeue -u \$USER'"
echo "  Results: oscar:~/wsindy-manifold/oscar_output/VDYN*_C[02]/"
echo -e "======================================================================${NC}"
