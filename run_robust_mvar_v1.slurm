#!/bin/bash
#SBATCH --job-name=robust_mvar_v1
#SBATCH --output=slurm_logs/robust_mvar_v1_%j.out
#SBATCH --error=slurm_logs/robust_mvar_v1_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=batch

echo "=========================================="
echo "Robust ROM-MVAR v1 - Test Run (T=1s)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 64GB"
echo ""

# Load modules
module load python/3.11.0s-ixrhc3q
module load ffmpeg/7.0-xny2fb2

# Activate environment
source ~/wsindy_env/bin/activate

# Install scikit-learn if not present
pip install -q scikit-learn

# Navigate to workspace
cd ~/wsindy-manifold

echo "Python environment:"
which python
python --version
echo ""

# Display configuration info
echo "=========================================="
echo "Configuration"
echo "=========================================="
echo "Experiment: robust_mvar_v1"
echo "Config: configs/robust_mvar_v1.yaml"
echo ""
echo "Training strategy:"
echo "  - 400 runs total"
echo "  - 50% Gaussian (200 runs) - 5×5 spatial grid"
echo "  - 25% Uniform (100 runs)"
echo "  - 12.5% Ring (50 runs)"
echo "  - 12.5% Two-cluster (50 runs)"
echo ""
echo "Test run: T=1s (verification only)"
echo "Full run: Change T to 50s after this succeeds"
echo ""
echo "Optimizations:"
echo "  - N: 150 (reduced from 300)"
echo "  - Domain: 15×15 (reduced from 30×30)"
echo "  - Ridge α: 0.01 (increased from 1e-6 for stability)"
echo ""

# Run pipeline
echo "=========================================="
echo "Starting Pipeline"
echo "=========================================="
python run_robust_mvar_pipeline.py \
    --config configs/robust_mvar_v1.yaml \
    --experiment_name robust_mvar_v1

exit_code=$?

echo ""
echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "PIPELINE COMPLETED SUCCESSFULLY"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Check outputs in oscar_output/robust_mvar_v1/"
    echo "2. If everything looks good, update config:"
    echo "   - Change T: 1.0 → 50.0 in configs/robust_mvar_v1.yaml"
    echo "   - Increase time: 01:00:00 → 18:00:00 in slurm script"
    echo "3. Resubmit for full 50s simulation"
    echo ""
    echo "Expected full run time: ~10-12 hours"
    echo "  - 400 training runs × ~1.5 min each = ~10 hours"
    echo "  - POD/MVAR training = ~10 min"
    echo "  - 40 test runs × ~1.5 min each = ~1 hour"
    echo "  - Predictions = ~2 min"
else
    echo "PIPELINE FAILED"
    echo "=========================================="
    echo "Check error log for details:"
    echo "  slurm_logs/robust_mvar_v1_${SLURM_JOB_ID}.err"
fi
echo ""
echo "End time: $(date)"
