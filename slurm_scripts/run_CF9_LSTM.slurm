#!/bin/bash
#SBATCH --job-name=CF9_LSTM
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -o slurm_logs/CF9_LSTM_%j.out
#SBATCH -e slurm_logs/CF9_LSTM_%j.err

# ============================================================================
# CF9 — MVAR + LSTM Combined (√ρ + Simplex, d=19, H37)
# ============================================================================
# Previous attempt (job 396757) FAILED: killed while still generating
# training sims (87% after 2h 52m). The 200 × 77s sims overwhelmed the
# 3h time limit.
#
# Fix: 8h time limit (was ~3h before) and GPU partition for LSTM training.
#
# Config: Both MVAR (p=5, α=0.01) and LSTM (hidden=64, layers=2) enabled.
# Trains both models, evaluates both, compares.
#
# Expected runtime: ~6-7h
#   - 200 train sims × ~60s / 8 workers = ~25min
#   - 30 test sims = ~5min
#   - MVAR training: ~5min
#   - LSTM training: ~20-40min (GPU)
#   - Evaluation: ~30min per model
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

export PYTHONPATH=src

CONFIG="configs/CF9_longPrefix_LSTM_H37.yaml"
EXP_NAME="CF9_longPrefix_LSTM_H37"

echo "============================================================"
echo "CF9 MVAR + LSTM PIPELINE"
echo "============================================================"
echo "Config: $CONFIG"
echo "Experiment: $EXP_NAME"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_NTASKS"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo ""

# Check GPU availability
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
" 2>/dev/null || echo "Warning: PyTorch GPU check failed"

echo ""

# ── Step 1: Data Generation ──
echo "STEP 1/2: DATA GENERATION + MODEL TRAINING"
echo "============================================================"

python ROM_pipeline.py \
    --config "$CONFIG" \
    --experiment_name "$EXP_NAME"

DATA_EXIT=$?
echo "Data generation exit code: $DATA_EXIT"
echo "End: $(date)"

if [ $DATA_EXIT -ne 0 ]; then
    echo "❌ Pipeline failed! Aborting."
    exit $DATA_EXIT
fi

# ── Step 2: Visualization ──
echo ""
echo "============================================================"
echo "STEP 2/2: VISUALIZATION PIPELINE"
echo "============================================================"

python run_visualizations.py \
    --experiment_name "$EXP_NAME"

VIZ_EXIT=$?
echo "Visualization exit code: $VIZ_EXIT"

if [ $VIZ_EXIT -ne 0 ]; then
    echo "⚠️  Visualization failed (exit code: $VIZ_EXIT)"
    echo "   Data OK — re-run visualization locally if needed."
fi

# ── Summary ──
echo ""
echo "============================================================"
echo "CF9 PIPELINE COMPLETE"
echo "============================================================"
echo "End time: $(date)"

if [ -f "oscar_output/$EXP_NAME/summary.json" ]; then
    echo "Results:"
    python3 -c "
import json
d = json.load(open('oscar_output/$EXP_NAME/summary.json'))
for model_name in ['mvar', 'lstm']:
    m = d.get(model_name, {})
    r2 = m.get('mean_r2_test', None)
    if r2 is not None:
        r2_1s = m.get('mean_r2_1step_test', '?')
        neg = m.get('mean_negativity_frac', '?')
        sr = m.get('spectral_radius', 'N/A')
        print(f'  [{model_name.upper()}] R²_rollout = {r2:+.4f}, R²_1step = {r2_1s}, neg = {neg}')
" 2>/dev/null || echo "  (summary extraction failed)"
fi

echo ""
echo "Output: oscar_output/$EXP_NAME/"
echo "Done."
