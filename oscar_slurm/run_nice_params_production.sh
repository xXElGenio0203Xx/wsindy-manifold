#!/bin/bash
#SBATCH --job-name=nice_params_prod
#SBATCH --output=slurm_logs/nice_params_%j.out
#SBATCH --error=slurm_logs/nice_params_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=batch

# Nice Parameters Production Run
# 200 training + 50 test simulations
# N=200 particles, T=30s, 64×64 density

echo "=========================================="
echo "NICE PARAMETERS PRODUCTION RUN"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo ""

# Load modules
module load python/3.11

# Activate virtual environment
source ~/wsindy_env/bin/activate

# Go to project directory
cd ~/wsindy-manifold

# Pull latest changes
echo "Pulling latest code from GitHub..."
git pull
echo ""

# Run data generation pipeline
echo "Starting data generation pipeline..."
echo "Config: configs/nice_params_production.yaml"
echo "Training: 200 simulations"
echo "Test: 50 simulations"
echo "Particles: 200, Duration: 30s"
echo "Density: 64×64, Workers: $SLURM_CPUS_PER_TASK"
echo ""

python run_data_generation.py \
    --config configs/nice_params_production.yaml \
    --experiment_name nice_params_production \
    --n_train 200 \
    --n_test 50

exit_code=$?

echo ""
echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "✓ Job completed successfully!"
    echo "Output: oscar_output/nice_params_production/"
else
    echo "✗ Job failed with exit code: $exit_code"
fi
echo "End time: $(date)"
echo "=========================================="

exit $exit_code
