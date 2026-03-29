#!/bin/bash
# ============================================================================
# Upload + Submit LSTM Size Ablation to Oscar
# ============================================================================
# Generates 30 configs (6 regimes x 5 size variants), syncs them to Oscar,
# and submits an array job.
#
# Usage:
#   ./shell_scripts/submit_lstm_size_ablation.sh
#   ./shell_scripts/submit_lstm_size_ablation.sh --dry-run
#   ./shell_scripts/submit_lstm_size_ablation.sh --range 1-10
# ============================================================================

set -euo pipefail

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
            head -12 "$0" | tail -9
            exit 0 ;;
        *)           echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "======================================================================"
echo "LSTM SIZE ABLATION — OSCAR SUBMISSION"
echo "======================================================================"
echo ""

echo -e "${BLUE}Step 1: Regenerating configs/LSTM_size_ablation...${NC}"
python scripts/create_lstm_size_ablation_configs.py
MANIFEST_LOCAL="configs/LSTM_size_ablation/manifest.txt"
N_CONFIGS=$(wc -l < "${MANIFEST_LOCAL}" | tr -d ' ')
echo -e "  ${GREEN}${N_CONFIGS} configs in manifest${NC}"
echo ""

echo -e "${BLUE}Step 2: Syncing code & configs to Oscar...${NC}"
rsync -avz --relative \
    configs/LSTM_size_ablation/ \
    slurm_scripts/run_lstm_size_ablation.slurm \
    scripts/create_lstm_size_ablation_configs.py \
    ROM_WSINDY_pipeline.py \
    src/ \
    requirements.txt \
    "${OSCAR_HOST}:${OSCAR_DIR}/" 2>&1 | tail -5
echo -e "  ${GREEN}✓ Sync complete${NC}"
echo ""

if [ "${DRY_RUN}" = true ]; then
    echo -e "${YELLOW}DRY RUN — files synced, not submitting.${NC}"
    echo ""
    echo "To submit manually on Oscar:"
    echo "  ssh ${OSCAR_HOST}"
    echo "  cd ${OSCAR_DIR}"
    echo "  sbatch slurm_scripts/run_lstm_size_ablation.slurm"
    exit 0
fi

echo -e "${BLUE}Step 3: Submitting SLURM array job...${NC}"
SBATCH_ARGS=""
if [ -n "${ARRAY_RANGE}" ]; then
    SBATCH_ARGS="--array=${ARRAY_RANGE}"
    echo -e "  Using custom array range: ${YELLOW}${ARRAY_RANGE}${NC}"
fi

ssh "${OSCAR_HOST}" << ENDSSH
cd ~/wsindy-manifold || exit 1
mkdir -p slurm_logs oscar_output

N=\$(wc -l < configs/LSTM_size_ablation/manifest.txt | tr -d ' ')
echo "Manifest has \$N configs"

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

JOB_ID=\$(sbatch ${SBATCH_ARGS} slurm_scripts/run_lstm_size_ablation.slurm | grep -oP '\d+')

if [ -n "\${JOB_ID}" ]; then
    echo ""
    echo "✓ Job array submitted: \${JOB_ID}"
    squeue -u \$USER -j \${JOB_ID} --format="%.18i %.12j %.8T %.12M %.12l %.6D %R" 2>/dev/null | head -5
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
echo "  ssh ${OSCAR_HOST} 'squeue -u \$USER'"
echo "  ssh ${OSCAR_HOST} 'tail -f ~/wsindy-manifold/slurm_logs/lsize_*.out'"
echo ""
echo "Download results when done:"
echo "  rsync -avz ${OSCAR_HOST}:${OSCAR_DIR}/oscar_output/LSIZE_* oscar_output/"
