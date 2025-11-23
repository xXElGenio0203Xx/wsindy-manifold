#!/bin/bash
#SBATCH --job-name=vicsek_production
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/production_%j.out
#SBATCH --error=slurm_logs/production_%j.err

# Oscar Production Run Script
# Vicsek + Forces with constant speed mode
# N=200, T=50s, large ensemble

echo "=========================================="
echo "Oscar Production Run"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Load conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wsindy

# Navigate to project directory
cd ~/wsindy-manifold

# Pull latest code
echo "Pulling latest code..."
git pull

# Create slurm_logs directory if it doesn't exist
mkdir -p slurm_logs

# Run data generation pipeline
echo ""
echo "Starting data generation pipeline..."
echo "Config: configs/oscar_production.yaml"
echo "Experiment: oscar_production"
echo "Training runs: 200"
echo "Test runs: 50"
echo ""

python run_data_generation.py \
  --config configs/oscar_production.yaml \
  --experiment_name oscar_production \
  --n_train 200 \
  --n_test 50 \
  --clean

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Production run completed successfully!"
    echo "End time: $(date)"
    echo "=========================================="
    echo ""
    echo "Results saved to: oscar_output/oscar_production/"
    echo ""
    echo "To download results to your local machine:"
    echo "scp -r USERNAME@ssh.ccv.brown.edu:~/wsindy-manifold/oscar_output/oscar_production ./"
    echo ""
    echo "Then run locally:"
    echo "python run_visualizations.py --experiment_name oscar_production"
else
    echo ""
    echo "ERROR: Pipeline failed with exit code $?"
    echo "Check slurm logs for details"
    exit 1
fi
