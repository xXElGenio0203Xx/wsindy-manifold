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
module load python/3.11.0
module load gcc/12.1.0

# Navigate to project directory
cd ~/wsindy-manifold

# Install dependencies
echo "Installing dependencies..."
pip install -q numpy scipy matplotlib pyyaml tqdm scikit-learn pandas

echo ""
echo "========================================"
echo "CONFIGURATION"
echo "========================================"
echo "Experiment: stable_mvar_v2"
echo "Config: configs/stable_mvar_v2.yaml"
echo ""
echo "KEY PARAMETERS:"
echo "  - POD energy: 99% (reduced)"
echo "  - MVAR lag: 2 (reduced)"
echo "  - Ridge alpha: 0.5 (very strong)"
echo "  - Eigenvalue threshold: 0.95"
echo "  - Training runs: 400 (mixed)"
echo "  - Test runs: 40"
echo ""

# Run the pipeline
echo "========================================"
echo "RUNNING PIPELINE"
echo "========================================"
python run_stable_mvar_pipeline.py \
    --config configs/stable_mvar_v2.yaml \
    --experiment_name stable_mvar_v2

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
