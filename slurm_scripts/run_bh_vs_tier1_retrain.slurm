#!/bin/bash
#SBATCH --job-name=bh_vs_t1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -o slurm_logs/bh_vs_tier1_retrain_%j.out
#SBATCH -e slurm_logs/bh_vs_tier1_retrain_%j.err

set -euo pipefail
module load python/3.11.11-5e66
module load cuda/12.9.0-cinr
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
export PYTHONPATH=src
mkdir -p slurm_logs

CONFIG="configs/missing_lstm/NDYN05_blackhole_VS_tier1_w5_lstm.yaml"
EXP_NAME="NDYN05_blackhole_VS_tier1_w5_lstm"
OUTPUT_DIR="oscar_output/${EXP_NAME}"
SOURCE_DIR="oscar_output/NDYN05_blackhole_VS_tier1_w5/rom_common"

echo "=== LSTM retrain: NDYN05_blackhole_VS_tier1_w5 ==="
echo "Improved: dropout=0.1, residual=true, mass_postprocess=scale"
echo "Start: $(date)  Node: $(hostname)"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

mkdir -p "$OUTPUT_DIR"
if [ ! -e "$OUTPUT_DIR/rom_common" ]; then
    ln -s "$(cd "$SOURCE_DIR" && pwd)" "$OUTPUT_DIR/rom_common"
    echo "Symlinked rom_common -> $SOURCE_DIR"
fi

# Also symlink test dir for evaluation
if [ ! -e "$OUTPUT_DIR/test" ]; then
    ln -s "$(cd oscar_output/NDYN05_blackhole_VS_tier1_w5/test && pwd)" "$OUTPUT_DIR/test"
    echo "Symlinked test dir"
fi

python ROM_WSINDY_pipeline.py \
    --config "$CONFIG" \
    --experiment_name "$EXP_NAME" \
    --output-dir "$OUTPUT_DIR"

echo "End: $(date)  Exit: $?"
