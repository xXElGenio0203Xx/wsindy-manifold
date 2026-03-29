#!/bin/bash
#SBATCH --job-name=extract_thesis
#SBATCH --array=0-53
#SBATCH --output=slurm_logs/extract_thesis_%A_%a.out
#SBATCH --error=slurm_logs/extract_thesis_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=1
#SBATCH --partition=batch

# ── Environment ──────────────────────────────────────────────────────────────
module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate
export PYTHONPATH=src

cd ~/wsindy-manifold
mkdir -p slurm_logs

# ── Derive experiment name from manifest ─────────────────────────────────────
MANIFEST=configs/systematic/manifest.txt
CONFIG_PATH=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
EXP_NAME=$(basename "$CONFIG_PATH" .yaml)
EXP_DIR="oscar_output/${EXP_NAME}"

echo "=== Task ${SLURM_ARRAY_TASK_ID}: ${EXP_NAME} ==="

if [ ! -d "${EXP_DIR}" ]; then
    echo "SKIP: ${EXP_DIR} does not exist"
    exit 0
fi

if [ ! -d "${EXP_DIR}/test/test_000" ]; then
    echo "SKIP: no test_000/ in ${EXP_DIR}"
    exit 0
fi

# ── Run unified thesis data extraction ───────────────────────────────────────
# Extracts: kde_snapshots.npz, mass_timeseries.npz, spatial_order.npz, lp_errors.npz
python scripts/extract_thesis_data.py "${EXP_DIR}" --force

echo "=== Done ===""
