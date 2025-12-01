#!/bin/bash
#SBATCH --job-name=gaussians_oscar
#SBATCH --output=slurm_logs/gaussians_oscar_%j.out
#SBATCH --error=slurm_logs/gaussians_oscar_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=batch

# ============================================================================
# Gaussians Oscar Experiment - ROM-MVAR Generalization Test
# ============================================================================
# This job runs the complete gaussians pipeline on Oscar:
#   1. Generate 200 training runs (varying variance, same center)
#   2. Build global POD basis and train MVAR
#   3. Generate 50 test runs (varying center, fixed variance)
#   4. Generate ROM-MVAR predictions for test runs
#
# Expected runtime: ~2-3 hours
# ============================================================================

echo "=========================================="
echo "GAUSSIANS OSCAR EXPERIMENT"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo ""

# Load modules and activate environment
module load python/3.11.0
module load ffmpeg/4.2.2

# Source the conda environment
source ~/wsindy_env/bin/activate

echo "Python environment:"
which python
python --version
echo ""

# Create output directory
mkdir -p slurm_logs

# Navigate to workspace
cd ~/wsindy-manifold

echo "=========================================="
echo "Starting Gaussians Pipeline"
echo "=========================================="
echo "Configuration: configs/gaussians_oscar.yaml"
echo "Training runs: 200 (10 variances × 20 samples)"
echo "Test runs: 50 (50 different centers)"
echo ""

# Run the pipeline
python run_gaussians_pipeline.py \
    --config configs/gaussians_oscar.yaml \
    --experiment_name gaussians_oscar

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "PIPELINE COMPLETED SUCCESSFULLY"
    echo "=========================================="
    echo "End time: $(date)"
    echo ""
    echo "Output locations:"
    echo "  Training data: oscar_output/gaussians_oscar/train/"
    echo "  Test data: oscar_output/gaussians_oscar/test/"
    echo "  ROM models: oscar_output/gaussians_oscar/mvar/"
    echo "  Predictions: predictions/gaussians_oscar/"
    echo ""
    echo "Next steps:"
    echo "  1. Download results: rsync -avz oscar:~/wsindy-manifold/oscar_output/gaussians_oscar/ ."
    echo "  2. Download predictions: rsync -avz oscar:~/wsindy-manifold/predictions/gaussians_oscar/ ."
    echo "  3. Run visualizations: python run_visualizations.py --experiment_name gaussians_oscar"
else
    echo ""
    echo "=========================================="
    echo "PIPELINE FAILED"
    echo "=========================================="
    echo "Check error log: slurm_logs/gaussians_oscar_${SLURM_JOB_ID}.err"
    exit 1
fi
