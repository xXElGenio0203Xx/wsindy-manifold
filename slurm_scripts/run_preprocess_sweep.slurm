#!/bin/bash
#SBATCH --job-name=PSW_sweep
#SBATCH --array=1-120%20
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/psw_%A_%a.out
#SBATCH --error=slurm_logs/psw_%A_%a.err

# ── Preprocessing Sweep: 120 jobs (20 combos × 6 regimes) ──────────────────
# Each array task runs ONE config through ROM_WSINDY_pipeline.py
# Reduced sim (N=100, Lx=15) for fast screening.

set -euo pipefail

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold

# Read the config filename from the manifest (line = array task ID)
MANIFEST="configs/preprocess_sweep/manifest.txt"
CONFIG_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")

if [[ -z "$CONFIG_FILE" ]]; then
    echo "ERROR: No config at line ${SLURM_ARRAY_TASK_ID} in manifest"
    exit 1
fi

CONFIG="configs/preprocess_sweep/${CONFIG_FILE}"
# Experiment name = filename without .yaml
EXP_NAME="${CONFIG_FILE%.yaml}"

echo "=========================================="
echo "Array task: ${SLURM_ARRAY_TASK_ID} / 120"
echo "Config:     ${CONFIG}"
echo "Experiment: ${EXP_NAME}"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "=========================================="

# Skip if already completed
OUTDIR="oscar_output/${EXP_NAME}"
if [[ -f "${OUTDIR}/summary.json" ]]; then
    echo "SKIP: ${EXP_NAME} already has summary.json"
    exit 0
fi

python ROM_WSINDY_pipeline.py \
    --config "$CONFIG" \
    --experiment_name "$EXP_NAME"

echo "=========================================="
echo "Finished: ${EXP_NAME}"
echo "End:      $(date)"
echo "=========================================="
