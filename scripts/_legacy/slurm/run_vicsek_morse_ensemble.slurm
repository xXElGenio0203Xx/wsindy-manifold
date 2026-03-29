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

# Get config from environment variable (set by submit script)
CONFIG=${CONFIG:-configs/vicsek_morse_base.yaml}

# Extract experiment name to build correct output directory
EXPERIMENT=$(basename ${CONFIG} .yaml | sed 's/vicsek_morse_//')
if [ "${EXPERIMENT}" = "base" ]; then
  OUTPUT_DIR="simulations/vicsek_morse_base/run_${SEED}"
else
  OUTPUT_DIR="simulations/vicsek_morse_${EXPERIMENT}/run_${SEED}"
fi

echo "Running simulation with seed=${SEED}"
echo "Config: ${CONFIG}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# DEBUG: Show the exact command being run
echo "DEBUG: Full command:"
echo "rectsim single --config ${CONFIG} --seed ${SEED} --out_dir ${OUTPUT_DIR}"
echo ""

# Run the simulation
# - Outputs density.npz for ROM (configs have animate_density: false already)
# - Uses discrete Vicsek with Morse forces
# - Override out_dir for this specific run
rectsim single \
  --config ${CONFIG} \
  --seed ${SEED} \
  --out_dir ${OUTPUT_DIR}

# DEBUG: Show where files actually ended up
echo ""
echo "DEBUG: Checking output locations..."
echo "Contents of ${OUTPUT_DIR}:"
ls -lh "${OUTPUT_DIR}" 2>&1 || echo "Directory does not exist or is empty"
echo ""
echo "Contents of simulations/vicsek_morse_base/:"
ls -lh simulations/vicsek_morse_base/*.npz 2>&1 | head -5 || echo "No NPZ files in base directory"

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
