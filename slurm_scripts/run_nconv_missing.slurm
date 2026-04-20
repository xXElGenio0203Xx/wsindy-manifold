#!/bin/bash
#SBATCH --job-name=nconv_miss
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH -p batch
#SBATCH --array=0-1
#SBATCH -o slurm_logs/nconv_missing_%A_%a.out
#SBATCH -e slurm_logs/nconv_missing_%A_%a.err

# ============================================================================
# MISSING N-CONVERGENCE — 2 full-pipeline jobs (N=500 & N=1000)
# ============================================================================
# Pure Vicsek regime, MVAR+LSTM+WSINDy, 340 training sims, lag=5.
#
# Time estimates (from particle-count scaling of completed N=300 jobs):
#   N=300:  completed in ~12h   (baseline)
#   N=500:  ~2× simulation cost → ~24h expected
#   N=1000: ~4× simulation cost → ~36–48h expected
#
# Both allocated 48h (max) so neither times out. The scheduler releases
# resources as soon as N=500 finishes.
#
# Memory: 128G (N=1000 has ~3.3× more particle data than N=300).
#
# Usage:
#   sbatch slurm_scripts/run_nconv_missing.slurm
# ============================================================================

set -euo pipefail

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
export PYTHONPATH=src
mkdir -p slurm_logs oscar_output

MANIFEST="configs/n_convergence/manifest_missing.txt"
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: $MANIFEST not found."
    exit 1
fi

CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
if [ -z "$CONFIG" ]; then
    echo "ERROR: No config for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

EXP_NAME=$(basename "$CONFIG" .yaml)
OUTPUT_DIR="oscar_output/${EXP_NAME}"

echo "============================================"
echo "N-CONVERGENCE — Missing full pipeline jobs"
echo "============================================"
echo "Manifest:    $MANIFEST"
echo "Array task:  $SLURM_ARRAY_TASK_ID / 1"
echo "Config:      $CONFIG"
echo "Experiment:  $EXP_NAME"
echo "Output dir:  $OUTPUT_DIR"
echo "Start time:  $(date)"
echo "Node:        $(hostname)"
echo ""

# Remove stale partial data for clean re-run
# N=500 has 3 partial files, N=1000 dir may exist but be empty
if [ -d "$OUTPUT_DIR" ]; then
    echo "Removing stale $OUTPUT_DIR for clean re-run"
    rm -rf "$OUTPUT_DIR"
fi

python ROM_WSINDY_pipeline.py \
    --config "$CONFIG" \
    --experiment_name "$EXP_NAME" \
    --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?
echo ""
echo "End time:  $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
