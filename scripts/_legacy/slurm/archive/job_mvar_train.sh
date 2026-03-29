#!/bin/bash
#SBATCH --job-name=rom_mvar_train
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/rom_train_%j.log
#SBATCH --error=logs/rom_train_%j.err

# ROM/MVAR Training on Oscar
#
# This job:
# 1. Generates M_train simulations with varied ICs
# 2. Computes global POD basis
# 3. Fits MVAR model
# 4. Saves model artifacts
#
# Usage:
#   sbatch scripts/slurm/job_mvar_train.sh configs/rom_mvar_example.yaml

set -e

# Check argument
if [ $# -lt 1 ]; then
    echo "Usage: sbatch job_mvar_train.sh <config.yaml> [overrides...]"
    exit 1
fi

CONFIG=$1
shift
OVERRIDES="$@"

echo "========================================="
echo "ROM/MVAR Training Job"
echo "========================================="
echo "Config: $CONFIG"
echo "Overrides: $OVERRIDES"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================="

# Load conda environment
module load miniconda3/23.11.0s
source /gpfs/runtime/opt/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate wsindy

# Create logs directory
mkdir -p logs

# Run training
echo ""
echo "Starting training..."
python scripts/rom_mvar_train.py --config $CONFIG $OVERRIDES

echo ""
echo "Training complete!"
echo "Check model/ directory for trained artifacts"
