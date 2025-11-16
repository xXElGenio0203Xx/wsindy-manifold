#!/bin/bash
#SBATCH --job-name=rectsim_single
#SBATCH --output=slurm_logs/rectsim_%j.out
#SBATCH --error=slurm_logs/rectsim_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=batch

# ==============================================================================
# SLURM Script for Running rectsim single
# ==============================================================================
# Usage:
#   sbatch run_rectsim_single.slurm
#
# To override config parameters, edit the RECTSIM_ARGS below or pass them
# as environment variables when submitting:
#   sbatch --export=CONFIG=configs/long_loose_N200_T400.yaml run_rectsim_single.slurm
# ==============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

# Default config file (can be overridden with --export=CONFIG=... at submit time)
CONFIG="${CONFIG:-configs/gentle_clustering.yaml}"

# Additional rectsim arguments (override config values)
# Format: --sim.N 200 --sim.T 100.0 --outputs.animate false
RECTSIM_ARGS="${RECTSIM_ARGS:---sim.N 200 --sim.T 100.0}"

# Repository path
REPO_DIR="/users/emaciaso/src/wsindy-manifold"

# Conda environment name
CONDA_ENV="wsindy"

# ------------------------------------------------------------------------------
# Job Info
# ------------------------------------------------------------------------------
echo "========================================"
echo "SLURM Job: rectsim single"
echo "========================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURMD_NODENAME"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "Memory:        $SLURM_MEM_PER_NODE MB"
echo "Start time:    $(date)"
echo "Config:        $CONFIG"
echo "Extra args:    $RECTSIM_ARGS"
echo "========================================"
echo ""

# ------------------------------------------------------------------------------
# Environment Setup
# ------------------------------------------------------------------------------
echo "Setting up environment..."

# Load miniconda module
module load miniconda3/23.11.0s

# Initialize conda
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

# Activate environment
conda activate "$CONDA_ENV"

# Verify environment
echo "✓ Python: $(which python)"
echo "✓ Conda env: $CONDA_ENVIRONMENT"
echo ""

# Change to repository directory
cd "$REPO_DIR"
echo "✓ Working directory: $(pwd)"
echo ""

# Create slurm_logs directory if it doesn't exist
mkdir -p slurm_logs

# ------------------------------------------------------------------------------
# Run Simulation
# ------------------------------------------------------------------------------
echo "Starting rectsim simulation..."
echo "Command: rectsim single --config $CONFIG $RECTSIM_ARGS"
echo ""

# Run the simulation
rectsim single --config "$CONFIG" $RECTSIM_ARGS

EXIT_CODE=$?

# ------------------------------------------------------------------------------
# Job Summary
# ------------------------------------------------------------------------------
echo ""
echo "========================================"
echo "Job completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "========================================"

exit $EXIT_CODE
