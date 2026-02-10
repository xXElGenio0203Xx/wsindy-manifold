#!/bin/bash
#SBATCH --job-name=heavyweight
#SBATCH --output=slurm_logs/heavyweight_%j.out
#SBATCH --error=slurm_logs/heavyweight_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=batch

# ============================================================================
# HEAVYWEIGHT OSCAR — Max quality Gaussian-only MVAR vs LSTM
# ============================================================================
#
# N=100 particles, T=15s, bandwidth=3.0 → smooth density fields
# POD: 45% energy → ~5-8 modes (clean, forecastable dynamics)
# 100 train + 25 test (all Gaussian clusters)
# MVAR: lag=12, ridge=1e-5 → ~438 params
# LSTM: h=64, L=2, dropout=0.1, 2000 epochs → ~51K params
#
# Estimated runtime:
#   Simulations: 125 runs × ~10s each ≈ 20 min
#   POD: SVD on (37,600 × 2,304) ≈ 1 min
#   MVAR training: <5s
#   LSTM training: ~30 min (2000 epochs max, early stopping)
#   Evaluation: 25 tests × ~362 steps ≈ 5 min
#   Total: ~1 hour
#
# ============================================================================

set -e
set -u

CONFIG_FILE="configs/heavyweight_oscar.yaml"
EXPERIMENT_NAME="heavyweight_oscar"
REPO_DIR="/users/emaciaso/wsindy-manifold"

echo "======================================================================"
echo "JOB: HEAVYWEIGHT OSCAR — Gaussian-only, smooth, low-rank"
echo "======================================================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "CPUs:       $SLURM_CPUS_PER_TASK"
echo "Memory:     $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "Config:     $CONFIG_FILE"
echo "Experiment: $EXPERIMENT_NAME"
echo "======================================================================"
echo ""

# Environment Setup
echo "Setting up environment..."
module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate
echo "✓ Python: $(which python)"
echo "✓ Python version: $(python --version)"
echo ""

cd "$REPO_DIR" || exit 1
echo "✓ Working directory: $(pwd)"
echo ""

mkdir -p oscar_output slurm_logs

# ==============================================================================
# Pipeline Execution
# ==============================================================================
echo "======================================================================"
echo "STEP 1/2: ROM PIPELINE"
echo "======================================================================"
echo ""
echo "Training: 100 Gaussian runs (5x × 4y × 5var)"
echo "  N=100 particles, T=15.0s, dt=0.04, density 48×48, bandwidth=3.0"
echo "  POD: 45% energy (~5-8 modes)"
echo "  MVAR: lag=12, ridge=1e-5"
echo "  LSTM: h=64, L=2, dropout=0.1, 2000 epochs, LR scheduler"
echo ""
echo "Testing: 25 Gaussian runs (5x × 5y, interpolated)"
echo "  Forecast: ~362 steps (14.5s from t=0.5s)"
echo ""

python ROM_pipeline.py \
    --config "$CONFIG_FILE" \
    --experiment_name "$EXPERIMENT_NAME" \
    2>&1 | tee "oscar_output/${EXPERIMENT_NAME}_pipeline.log"

ROM_EXIT=$?

if [ $ROM_EXIT -ne 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✗ ROM PIPELINE FAILED (exit code: $ROM_EXIT)"
    echo "======================================================================"
    exit $ROM_EXIT
fi

echo ""
echo "✓ ROM pipeline completed successfully"
echo ""

# ==============================================================================
# Visualization Pipeline
# ==============================================================================
echo "======================================================================"
echo "STEP 2/2: VISUALIZATION PIPELINE"
echo "======================================================================"
echo ""

python run_visualizations.py \
    --experiment_name "$EXPERIMENT_NAME" \
    2>&1 | tee "oscar_output/${EXPERIMENT_NAME}_viz.log"

VIZ_EXIT=$?

if [ $VIZ_EXIT -ne 0 ]; then
    echo ""
    echo "⚠️  Visualization pipeline failed (exit code: $VIZ_EXIT)"
    echo "   ROM results are still valid — run visualizations locally."
fi

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "======================================================================"
echo "JOB COMPLETE"
echo "======================================================================"
echo "End time: $(date)"
echo ""

if [ -f "oscar_output/$EXPERIMENT_NAME/summary.json" ]; then
    echo "Pipeline Summary:"
    python3 -c "
import json
s = json.load(open('oscar_output/$EXPERIMENT_NAME/summary.json'))
print(f\"  Training runs:  {s.get('n_train', 'N/A')}\")
print(f\"  Test runs:      {s.get('n_test', 'N/A')}\")
print(f\"  POD modes:      {s.get('r_pod', 'N/A')}\")
print(f\"  Total time:     {s.get('total_time_minutes', 0):.1f} minutes\")
if 'mvar' in s:
    m = s['mvar']
    print(f\"  MVAR train R²:  {m.get('r2_train', 'N/A'):.4f}\")
    if 'mean_r2_test' in m:
        print(f\"  MVAR test R²:   {m['mean_r2_test']:.4f}\")
if 'lstm' in s:
    l = s['lstm']
    print(f\"  LSTM val loss:   {l.get('val_loss', 'N/A'):.6f}\")
    if 'mean_r2_test' in l:
        print(f\"  LSTM test R²:   {l['mean_r2_test']:.4f}\")
" 2>/dev/null || echo "  (summary file exists but could not be parsed)"
fi

echo ""
echo "Output: oscar_output/$EXPERIMENT_NAME/"
echo ""
echo "To download results locally:"
echo "  rsync -avz oscar:~/wsindy-manifold/oscar_output/$EXPERIMENT_NAME/ ./oscar_output/$EXPERIMENT_NAME/"
echo "  rsync -avz oscar:~/wsindy-manifold/predictions/$EXPERIMENT_NAME/ ./predictions/$EXPERIMENT_NAME/"
echo ""
echo "To re-run visualizations locally:"
echo "  python run_visualizations.py --experiment_name $EXPERIMENT_NAME"
echo "======================================================================"
