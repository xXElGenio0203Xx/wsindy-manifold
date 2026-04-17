#!/bin/bash
#SBATCH --job-name=lstm_gpu_cr
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -o slurm_logs/lstm_gpu_crystal_%j.out
#SBATCH -e slurm_logs/lstm_gpu_crystal_%j.err

# Single-task: crystal LSTM (array index 1 from manifest)
set -euo pipefail

module load python/3.11.11-5e66
module load cuda/12.9.0-cinr
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
export PYTHONPATH=src
mkdir -p slurm_logs oscar_output

EXP_NAME="NDYN07_crystal_lstm"
ROM_SOURCE="NDYN07_crystal_wsindy_v3"
CONFIG="configs/lstm_only/${EXP_NAME}.yaml"
OUTPUT_DIR="oscar_output/${EXP_NAME}"
ROM_SOURCE_DIR="oscar_output/${ROM_SOURCE}/rom_common"

echo "============================================"
echo "LSTM-ONLY TRAINING (GPU) — Crystal"
echo "============================================"
echo "Experiment:    $EXP_NAME"
echo "Config:        $CONFIG"
echo "Output dir:    $OUTPUT_DIR"
echo "ROM source:    $ROM_SOURCE_DIR"
echo "Start time:    $(date)"
echo "Node:          $(hostname)"
echo "GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

mkdir -p "$OUTPUT_DIR"
if [ ! -e "$OUTPUT_DIR/rom_common" ]; then
    ln -s "$(cd "$ROM_SOURCE_DIR" && pwd)" "$OUTPUT_DIR/rom_common"
fi

python ROM_WSINDY_pipeline.py \
    --config "$CONFIG" \
    --experiment_name "$EXP_NAME" \
    --output-dir "$OUTPUT_DIR"

echo "End time: $(date)"
