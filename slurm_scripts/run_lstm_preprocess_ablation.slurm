#!/bin/bash
#SBATCH --job-name=LPREP
#SBATCH --array=1-120%10
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/lprep_%A_%a.out
#SBATCH --error=slurm_logs/lprep_%A_%a.err

# LSTM preprocessing ablation: 120 jobs (6 regimes x 20 preprocessing combos)
# Fixed architecture: h64_L2 with lag=5, sim.N=200.

set -euo pipefail

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold

MANIFEST="configs/LSTM_preprocess_ablation/manifest.txt"
CONFIG_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")

if [[ -z "${CONFIG_FILE}" ]]; then
    echo "ERROR: No config at line ${SLURM_ARRAY_TASK_ID} in ${MANIFEST}"
    exit 1
fi

CONFIG="configs/LSTM_preprocess_ablation/${CONFIG_FILE}"
EXP_NAME="${CONFIG_FILE%.yaml}"

echo "=========================================="
echo "LSTM PREPROCESS ABLATION"
echo "Array task: ${SLURM_ARRAY_TASK_ID} / 120"
echo "Config:     ${CONFIG}"
echo "Experiment: ${EXP_NAME}"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "=========================================="

OUTDIR="oscar_output/${EXP_NAME}"
if [[ -f "${OUTDIR}/summary.json" ]]; then
    echo "SKIP: ${EXP_NAME} already has summary.json"
    exit 0
fi
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
