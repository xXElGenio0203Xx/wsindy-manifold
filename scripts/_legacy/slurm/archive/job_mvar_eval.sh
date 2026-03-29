#!/bin/bash
#SBATCH --job-name=rom_mvar_eval
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/rom_eval_%j.log
#SBATCH --error=logs/rom_eval_%j.err

# ROM/MVAR Evaluation on Oscar
#
# This job:
# 1. Loads trained POD + MVAR model
# 2. Generates K_test unseen IC simulations
# 3. Computes MVAR forecasts
# 4. Evaluates metrics (R², RMSE, mass error, τ)
# 5. Saves per-IC and aggregate results (NO VIDEOS)
#
# Usage:
#   sbatch scripts/slurm/job_mvar_eval.sh <experiment_name> <config.yaml>

set -e

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: sbatch job_mvar_eval.sh <experiment_name> <config.yaml> [overrides...]"
    exit 1
fi

EXPERIMENT=$1
CONFIG=$2
shift 2
OVERRIDES="$@"

echo "========================================="
echo "ROM/MVAR Evaluation Job"
echo "========================================="
echo "Experiment: $EXPERIMENT"
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

# Run evaluation (videos and plots disabled by default)
echo ""
echo "Starting evaluation..."
python scripts/rom_mvar_eval.py \\
    --experiment $EXPERIMENT \\
    --config $CONFIG \\
    $OVERRIDES

echo ""
echo "Evaluation complete!"
echo "Results saved to rom_mvar/$EXPERIMENT/"
echo ""
echo "To generate visualizations locally:"
echo "  1. rsync rom_mvar/$EXPERIMENT/ to your laptop"
echo "  2. python scripts/rom_mvar_visualize.py --experiment $EXPERIMENT"
