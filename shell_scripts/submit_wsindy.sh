#!/bin/bash
# ============================================================================
# Submit ROM + WSINDy Pipeline to Oscar
# ============================================================================
# Syncs code + config to Oscar, then submits SLURM job.
#
# Usage:
#   ./shell_scripts/submit_wsindy.sh                     # Default DYN1_gentle_wsindy
#   ./shell_scripts/submit_wsindy.sh <config> <exp_name>  # Custom config
# ============================================================================

set -e

OSCAR_HOST="oscar"
OSCAR_DIR="~/wsindy-manifold"

CONFIG=${1:-configs/DYN1_gentle_wsindy.yaml}
EXPERIMENT=${2:-DYN1_gentle_wsindy}
SLURM_SCRIPT="slurm_scripts/run_wsindy_rom.slurm"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${CYAN}======================================================================"
echo "  ROM + WSINDy Pipeline — Oscar Submission"
echo -e "======================================================================${NC}"
echo ""
echo "  Config:     $CONFIG"
echo "  Experiment: $EXPERIMENT"
echo "  SLURM:      $SLURM_SCRIPT"
echo ""

# ── Verify local files ────────────────────────────────────────────
for f in "$CONFIG" "$SLURM_SCRIPT" "ROM_WSINDY_pipeline.py"; do
    if [ ! -f "$f" ]; then
        echo -e "${RED}✗ Missing: $f${NC}"
        exit 1
    fi
done
echo -e "${GREEN}✓ All local files present${NC}"
echo ""

# ── Sync to Oscar ─────────────────────────────────────────────────
echo -e "${CYAN}Syncing files to Oscar...${NC}"

FILES_TO_SYNC=(
    # Pipeline scripts
    "ROM_WSINDY_pipeline.py"
    "ROM_pipeline.py"
    "$CONFIG"
    "$SLURM_SCRIPT"
    # WSINDy package
    "src/wsindy/"
    # Existing dependencies
    "src/rectsim/"
    "src/rom/"
    # Requirements
    "requirements.txt"
    "pyproject.toml"
)

for file in "${FILES_TO_SYNC[@]}"; do
    echo "  → $file"
    rsync -avz --relative "$file" "$OSCAR_HOST:$OSCAR_DIR/" 2>/dev/null || {
        echo -e "  ${RED}✗ Failed to transfer $file${NC}"
        exit 1
    }
done

echo ""
echo -e "${GREEN}✓ File transfer complete${NC}"
echo ""

# ── Submit SLURM job ──────────────────────────────────────────────
echo -e "${CYAN}Submitting SLURM job...${NC}"

ssh "$OSCAR_HOST" << ENDSSH
cd ~/wsindy-manifold || exit 1
mkdir -p oscar_output slurm_logs

# Make sure wsindy package is importable
pip install -e . --quiet 2>/dev/null || true

echo "Submitting: $SLURM_SCRIPT"
JOB_ID=\$(sbatch "$SLURM_SCRIPT" "$CONFIG" "$EXPERIMENT" | grep -oP '\d+')

if [ -n "\$JOB_ID" ]; then
    echo ""
    echo "✓ Job submitted! ID: \$JOB_ID"
    echo ""
    echo "Monitor:"
    echo "  squeue -u \$USER"
    echo "  tail -f slurm_logs/wsindy_rom_\${JOB_ID}.out"
else
    echo "✗ Submission failed"
    exit 1
fi
ENDSSH

STATUS=$?

echo ""
if [ $STATUS -eq 0 ]; then
    echo -e "${GREEN}======================================================================"
    echo "  SUBMISSION COMPLETE"
    echo -e "======================================================================${NC}"
    echo ""
    echo "  Next steps:"
    echo "    1. Monitor:  ssh oscar 'tail -f ~/wsindy-manifold/slurm_logs/wsindy_rom_*.out'"
    echo "    2. Download: ./shell_scripts/download_from_oscar.sh $EXPERIMENT"
    echo "    3. Visualize locally"
    echo ""
else
    echo -e "${RED}======================================================================"
    echo "  SUBMISSION FAILED"
    echo -e "======================================================================${NC}"
    echo ""
    echo "  Manual submit:"
    echo "    ssh oscar"
    echo "    cd ~/wsindy-manifold"
    echo "    sbatch $SLURM_SCRIPT $CONFIG $EXPERIMENT"
    echo ""
    exit 1
fi
