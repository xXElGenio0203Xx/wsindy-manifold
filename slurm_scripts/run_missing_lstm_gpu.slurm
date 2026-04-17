#!/bin/bash
#SBATCH --job-name=miss_lstm
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-5%2
#SBATCH -o slurm_logs/miss_lstm_%A_%a.out
#SBATCH -e slurm_logs/miss_lstm_%A_%a.err

# ============================================================================
# FULL PIPELINE on GPU for 6 catalogue regimes missing LSTM results
# ============================================================================
# Runs sim -> POD -> MVAR -> LSTM -> eval.  Steps 1-3 are CPU-only but
# complete in <15 min each; LSTM training uses the GPU (~30-45 min).
# GPU QOS: max 2 GPUs + 192G total → %2 concurrency with 48G each.
# ============================================================================

set -euo pipefail

module load python/3.11.11-5e66
module load cuda/12.9.0-cinr
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
export PYTHONPATH=src
mkdir -p slurm_logs oscar_output configs/missing_lstm

MANIFEST="configs/missing_lstm/manifest.txt"

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: $MANIFEST not found."
    exit 1
fi

CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
if [ -z "$CONFIG" ]; then
    echo "ERROR: No config for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Derive experiment name from config
EXP_NAME=$(python -c "
import yaml, sys
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg['experiment_name'])
")

OUTPUT_DIR="oscar_output/${EXP_NAME}"

echo "============================================"
echo "MISSING LSTM — FULL PIPELINE (GPU)"
echo "============================================"
echo "Array task:    $SLURM_ARRAY_TASK_ID"
echo "Experiment:    $EXP_NAME"
echo "Config:        $CONFIG"
echo "Output dir:    $OUTPUT_DIR"
echo "Start time:    $(date)"
echo "Node:          $(hostname)"
echo "GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'not detected yet')"
echo ""

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

python ROM_WSINDY_pipeline.py \
    --config "$CONFIG" \
    --experiment_name "$EXP_NAME" \
    --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?
echo ""
echo "End time:  $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
