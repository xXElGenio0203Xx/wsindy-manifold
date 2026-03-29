#!/bin/bash
#SBATCH --job-name=WSYRCH
#SBATCH --array=1-3%1
#SBATCH --time=1-12:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/wsyrich_%A_%a.out
#SBATCH --error=slurm_logs/wsyrich_%A_%a.err

set -euo pipefail

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold

MANIFEST="configs/wsindy_probe_rich/manifest.txt"
CONFIG_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")

if [[ -z "${CONFIG_FILE}" ]]; then
    echo "ERROR: No config at line ${SLURM_ARRAY_TASK_ID} in ${MANIFEST}"
    exit 1
fi

CONFIG="configs/wsindy_probe_rich/${CONFIG_FILE}"
EXP_NAME="${CONFIG_FILE%.yaml}"

echo "=========================================="
echo "WSINDy RICH PROBE"
echo "Array task: ${SLURM_ARRAY_TASK_ID} / 3"
echo "Config:     ${CONFIG}"
echo "Experiment: ${EXP_NAME}"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "=========================================="

OUTDIR="oscar_output/${EXP_NAME}"
if [[ -d "${OUTDIR}" ]]; then
    echo "FORCE RE-RUN: removing stale ${OUTDIR}"
    rm -rf "${OUTDIR}"
fi

python ROM_WSINDY_pipeline.py \
    --config "${CONFIG}" \
    --experiment_name "${EXP_NAME}"

echo "=========================================="
echo "Finished: ${EXP_NAME}"
echo "End:      $(date)"
echo "=========================================="
