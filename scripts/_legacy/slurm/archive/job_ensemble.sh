#!/bin/bash
#SBATCH --job-name=rom_ensemble
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem=64G
#SBATCH --output=logs/ensemble_%j.out
#SBATCH --error=logs/ensemble_%j.err

# ROM Pipeline - Stage 1: Generate Ensemble
#
# This script generates an ensemble of simulation runs with varied initial
# conditions but identical model parameters.
#
# Usage:
#   sbatch scripts/slurm/job_ensemble.sh
#
# Outputs:
#   simulations/<model_id>/runs/run_XXXX/density.npz

echo "=========================================="
echo "ROM Pipeline - Stage 1: Ensemble"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running on: $(hostname)"
echo ""

# Load modules
module load python/3.11
source ~/envs/wsindy/bin/activate

# Configuration
CONFIG_FILE="configs/rom_production.yaml"

# Check config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Config: $CONFIG_FILE"
echo ""

# Generate ensemble
rectsim ensemble --config "$CONFIG_FILE"

EXIT_CODE=$?

echo ""
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"

exit $EXIT_CODE
