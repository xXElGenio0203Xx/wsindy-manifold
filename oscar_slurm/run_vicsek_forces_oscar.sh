#!/bin/bash
#SBATCH --job-name=vicsek_forces_oscar
#SBATCH --output=slurm_logs/vicsek_forces_%j.out
#SBATCH --error=slurm_logs/vicsek_forces_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=batch
#SBATCH --nodes=1

echo "==========================================
Starting Vicsek Forces Oscar Pipeline
=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo ""

# Pull latest code from GitHub
echo "Pulling latest code from GitHub..."
cd ~/wsindy-manifold
git pull origin main
echo ""

# Load modules and activate environment
echo "Loading Python environment..."
module load python/3.11
source ~/wsindy_env/bin/activate
echo "Python: $(which python)"
echo ""

# Create log directory
mkdir -p slurm_logs

# Run the pipeline
echo "Running complete pipeline..."
echo "  Config: configs/vicsek_forces_oscar.yaml"
echo "  Experiment: vicsek_forces_oscar"
echo "  Training: 200 runs"
echo "  Test: 50 runs"
echo "  CPUs: $SLURM_CPUS_PER_TASK"
echo ""

python run_data_generation.py \
    --config configs/vicsek_forces_oscar.yaml \
    --experiment_name vicsek_forces_oscar \
    --n_train 200 \
    --n_test 50

echo ""
echo "=========================================="
echo "✓ Job completed successfully!"
echo "Output: oscar_output/vicsek_forces_oscar/"
echo "End time: $(date)"
echo "=========================================="
