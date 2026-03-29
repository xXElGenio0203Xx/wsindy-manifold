#!/bin/bash
#SBATCH --job-name=stable_v2_test
#SBATCH --output=slurm_logs/stable_mvar_v2_%j.out
#SBATCH --error=slurm_logs/stable_mvar_v2_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=batch

# ============================================================================
# Stable MVAR v2: Explicit Stability Enforcement
# ============================================================================
# Key changes from robust_v1:
#   1. Reduced complexity: POD 99% (vs 99.5%), lag=2 (vs 4)
#   2. Very strong regularization: α=0.5 (10× stronger)
#   3. Eigenvalue scaling: Scale down if max|λ| > 0.95
#   4. Same mixed distributions (400 training runs)
# ============================================================================

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# Load modules
module load python/3.11.0s-ixrhc3q
module load ffmpeg/7.0-xny2fb2

# Activate environment
source ~/wsindy_env/bin/activate

# Install scikit-learn if not present
pip install -q scikit-learn

# Navigate to project directory
cd ~/wsindy-manifold

# Set config file (default or from environment)
CONFIG_FILE="${CONFIG:-configs/stable_mvar_v2.yaml}"
EXPERIMENT_NAME=$(basename "$CONFIG_FILE" .yaml)

echo ""
echo "========================================"
echo "CONFIGURATION"
echo "========================================"
echo "Experiment: $EXPERIMENT_NAME"
echo "Config: $CONFIG_FILE"
echo ""

# Run the pipeline
echo "========================================"
echo "RUNNING PIPELINE"
echo "========================================"
python run_stable_mvar_pipeline.py \
    --config "$CONFIG_FILE" \
    --experiment_name "$EXPERIMENT_NAME"

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ PIPELINE COMPLETED SUCCESSFULLY"
    echo "========================================"
    echo ""
    echo "Results saved to:"
    echo "  - oscar_output/stable_mvar_v2/"
    echo "  - predictions/stable_mvar_v2/"
else
    echo "❌ PIPELINE FAILED"
    echo "========================================"
fi

echo ""
echo "Job ended: $(date)"

exit $EXIT_CODE
