#!/bin/bash
#SBATCH --job-name=optimal_oscar
#SBATCH --output=slurm_logs/optimal_oscar_%j.out
#SBATCH --error=slurm_logs/optimal_oscar_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=batch

# ============================================================================
# OPTIMAL OSCAR — Production-quality MVAR vs LSTM comparison
# ============================================================================
#
# This runs the full ROM pipeline with:
#   - 60 training runs (4 IC families × 15 variants)
#   - 20 test runs (4 IC families × 5 variants, unseen parameters)
#   - N=30 particles, T=8s, dt=0.04, density 64×64
#   - POD: 95% energy threshold → auto-selects ~25-35 modes
#   - MVAR: lag=8, ridge=5e-4 → ~7,230 params at d≈30
#   - LSTM: h=32, L=1, lag=8 → ~9,054 params at d≈30
#   - Forecast: 190 steps (7.6s from t=0.4s onward)
#
# Estimated runtime:
#   - Simulations: 80 runs × ~3s each ≈ 4 min
#   - POD: SVD on (12,060 × 4,096) ≈ 2 min
#   - MVAR training: <5s
#   - LSTM training: ~5 min (500 epochs max, early stopping)
#   - Evaluation: 20 tests × 190-step forecasts ≈ 3 min
#   - Total: ~20 min (6h wall time is very generous safety margin)
#
# Usage:
#   sbatch slurm_scripts/run_optimal_oscar.slurm
#
# ============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# Configuration
CONFIG_FILE="configs/optimal_oscar.yaml"
EXPERIMENT_NAME="optimal_oscar"
REPO_DIR="/users/emaciaso/wsindy-manifold"

# ==============================================================================
# Job Info
# ==============================================================================
echo "======================================================================"
echo "JOB: OPTIMAL OSCAR — Production MVAR vs LSTM"
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

# ==============================================================================
# Environment Setup
# ==============================================================================
echo "Setting up environment..."

# Load python module and activate virtual environment
module load python/3.11.11-5e66

# Activate virtual environment
source ~/wsindy_env_new/bin/activate

echo "✓ Python: $(which python)"
echo "✓ Python version: $(python --version)"
echo ""

# Navigate to repository
cd "$REPO_DIR" || exit 1
echo "✓ Working directory: $(pwd)"
echo ""

# Create directories
mkdir -p oscar_output slurm_logs

# ==============================================================================
# Pipeline Execution
# ==============================================================================
echo "======================================================================"
echo "STEP 1/2: ROM PIPELINE (simulation → POD → training → evaluation)"
echo "======================================================================"
echo ""
echo "Training: 60 runs (4 families × 15 variants)"
echo "  N=30 particles, T=8.0s, dt=0.04, density 64×64"
echo "  POD: 95% energy threshold (auto-selects modes)"
echo "  MVAR: lag=8, ridge_alpha=5e-4 (~7,230 params at d≈30)"
echo "  LSTM: h=32, L=1, epochs≤500 (~9,054 params at d≈30)"
echo ""
echo "Testing: 20 runs (4 families × 5 unseen variants)"
echo "  Forecast: 190 steps (7.6s from t=0.4s onward)"
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
    echo "Check: oscar_output/${EXPERIMENT_NAME}_pipeline.log"
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
    echo "   ROM results are still valid — visualization can be re-run locally."
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

# Print pipeline summary if available
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
