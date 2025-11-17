#!/bin/bash
#SBATCH --job-name=vicsek_ensemble
#SBATCH --partition=batch
#SBATCH --time=00:30:00
#SBATCH --array=0-49          # 50 simulations in parallel
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/vicsek_%A_%a.out
#SBATCH --error=logs/vicsek_%A_%a.err

# ==============================================================================
# Vicsek-Morse Ensemble Generation (Oscar Job Array)
# ==============================================================================
# This script runs 50 simulations in PARALLEL using Slurm arrays.
# Each task gets a unique seed and runs independently on different cores/nodes.
#
# Usage:
#   sbatch scripts/slurm/run_vicsek_morse_ensemble.slurm
#
# Output:
#   simulations/vicsek_morse_base/run_*/density.npz
#   simulations/vicsek_morse_base/run_*/traj.npz
#
# Runtime: ~1-2 minutes per simulation (T=300)
# Total wall time: ~1-2 minutes (all run in parallel!)
# ==============================================================================

set -e

# ------------------------------------------------------------------------------
# Environment Setup
# ------------------------------------------------------------------------------

# Load conda (Oscar CCV)
module --ignore_cache load miniconda3/23.11.0s-odstpk5
eval "$(conda shell.bash hook)"
conda activate wsindy

# Control threading (prevent BLAS oversubscription)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Change to repo directory
cd ~/src/wsindy-manifold

# Create logs directory
mkdir -p logs

# ------------------------------------------------------------------------------
# Job Info
# ------------------------------------------------------------------------------
echo "========================================="
echo "Vicsek-Morse Ensemble: Task ${SLURM_ARRAY_TASK_ID}"
echo "========================================="
echo "Job ID:        ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node:          ${SLURMD_NODENAME}"
echo "Start time:    $(date)"
echo "========================================="
echo ""

# ------------------------------------------------------------------------------
# Run Single Simulation
# ------------------------------------------------------------------------------

# Use array task ID as seed for this simulation
SEED=${SLURM_ARRAY_TASK_ID}
OUTPUT_DIR="simulations/vicsek_morse_base/run_${SEED}"

echo "Running simulation with seed=${SEED}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Run the simulation
# - Outputs only npz/csv (no videos, no plots)
# - Uses discrete Vicsek with Morse forces
# - T=300, N=200, 64x64 density grid
rectsim single \
  --config configs/vicsek_morse_base.yaml \
  --seed ${SEED} \
  --out-dir ${OUTPUT_DIR}

EXIT_CODE=$?

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
echo ""
echo "========================================="
echo "Task ${SLURM_ARRAY_TASK_ID} completed: exit code ${EXIT_CODE}"
echo "End time: $(date)"
echo "========================================="

exit ${EXIT_CODE}
