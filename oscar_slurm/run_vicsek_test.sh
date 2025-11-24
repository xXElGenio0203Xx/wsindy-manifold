#!/bin/bash
#SBATCH --job-name=vicsek_test
#SBATCH --output=slurm_logs/vicsek_test_%j.out
#SBATCH --error=slurm_logs/vicsek_test_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --nodes=1

# Generate test data and predictions for vicsek_forces_oscar
# Previous job 14454605 completed training (200 runs, POD/MVAR models)
# This job generates 50 test runs + predictions using existing models

echo "========================================================================"
echo "VICSEK FORCES - TEST GENERATION AND PREDICTION"
echo "========================================================================"
echo "Starting at: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# Load Python environment
module load python/3.11
source ~/wsindy_env/bin/activate

# Navigate to project directory
cd ~/wsindy-manifold

# Pull latest code
echo "Pulling latest code from GitHub..."
git pull origin main
echo ""

# Set environment variables for parallel execution
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Run test generation and prediction pipeline
echo "Running test generation and prediction pipeline..."
echo "Experiment: vicsek_forces_oscar"
echo "Test runs: 50"
echo "CPUs: $SLURM_CPUS_PER_TASK workers"
echo ""

python run_test_generation.py \
    --experiment_name vicsek_forces_oscar \
    --n_test 50

echo ""
echo "========================================================================"
echo "Job completed at: $(date)"
echo "========================================================================"
