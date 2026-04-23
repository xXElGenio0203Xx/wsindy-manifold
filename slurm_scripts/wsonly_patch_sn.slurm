#!/bin/bash
#SBATCH --job-name=wso_sn
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/wso_sn_%j.out
#SBATCH -e slurm_logs/wso_sn_%j.err

# ============================================================================
# WSINDy-ONLY PATCH — NDYN06_supernova_eta0p50
# ============================================================================
# ws_miss[3] (job 1799017_3) is running the full pipeline for supernova_eta0p50.
# It will complete training sims in ~8-9h then time out just like the others.
# Submit this job AFTER ws_miss[3] finishes (check with: squeue -u emaciaso).
#
# Usage (after ws_miss[3] is no longer in queue):
#   sbatch slurm_scripts/wsonly_patch_sn.slurm
# ============================================================================

set -euo pipefail

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
export PYTHONPATH=src
mkdir -p slurm_logs

CONFIG="configs/noise_sweep_ws/NDYN06_supernova_eta0p50.yaml"
OUTPUT_DIR="oscar_output/NDYN06_supernova_eta0p50_ws"

echo "============================================"
echo "WSINDy-ONLY PATCH — supernova_eta0p50"
echo "============================================"
echo "Config:      $CONFIG"
echo "Output dir:  $OUTPUT_DIR"
echo "Start time:  $(date)"
echo "Node:        $(hostname)"
echo ""

if [ ! -d "$OUTPUT_DIR/train" ]; then
    echo "ERROR: Training data not found at $OUTPUT_DIR/train"
    echo "ws_miss[3] may not have finished yet. Check squeue -u emaciaso"
    exit 1
fi

python ROM_WSINDY_pipeline.py \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --wsindy-only

EXIT_CODE=$?
echo ""
echo "End time:  $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
