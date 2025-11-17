#!/bin/bash
#SBATCH --job-name=vicsek_rom
#SBATCH --partition=batch
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4        # Use 4 cores for POD/MVAR linear algebra
#SBATCH --mem=16G
#SBATCH --output=logs/vicsek_rom_%j.out
#SBATCH --error=logs/vicsek_rom_%j.err

# ==============================================================================
# Vicsek-Morse ROM/MVAR Training (Oscar)
# ==============================================================================
# This script:
# 1. Loads density.npz from all ensemble runs
# 2. Computes global POD basis (SVD on stacked snapshots)
# 3. Projects all runs to latent space
# 4. Fits MVAR model on concatenated latent data
# 5. Saves model artifacts to rom_mvar/vicsek_morse_base/model/
#
# Usage (after ensemble completes):
#   sbatch scripts/slurm/run_vicsek_morse_rom.slurm
#
# Or with dependency (waits for ensemble):
#   jid=$(sbatch scripts/slurm/run_vicsek_morse_ensemble.slurm | awk '{print $4}')
#   sbatch --dependency=afterok:${jid} scripts/slurm/run_vicsek_morse_rom.slurm
#
# Runtime: ~5-10 minutes (POD on 50 runs × 3000 snapshots)
# ==============================================================================

set -e

# ------------------------------------------------------------------------------
# Environment Setup
# ------------------------------------------------------------------------------

# Load conda (Oscar CCV)
module --ignore_cache load miniconda3/23.11.0s-odstpk5
eval "$(conda shell.bash hook)"
conda activate wsindy

# Use 4 threads for BLAS (POD/MVAR linear algebra)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Change to repo directory
cd ~/src/wsindy-manifold

# Create logs directory
mkdir -p logs

# ------------------------------------------------------------------------------
# Job Info
# ------------------------------------------------------------------------------
echo "========================================="
echo "ROM/MVAR Training"
echo "========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          ${SLURMD_NODENAME}"
echo "CPUs:          ${SLURM_CPUS_PER_TASK}"
echo "Start time:    $(date)"
echo "========================================="
echo ""

# ------------------------------------------------------------------------------
# Run ROM/MVAR Training
# ------------------------------------------------------------------------------

# Get config from command line (default to base config if not provided)
CONFIG=${1:-configs/vicsek_morse_base.yaml}

# Extract experiment name from config
EXPERIMENT=$(grep "experiment_name:" ${CONFIG} | awk '{print $2}')

echo "Starting ROM/MVAR training pipeline..."
echo "Config: ${CONFIG}"
echo "Experiment: ${EXPERIMENT}"
echo ""

# Run training
# - Loads simulations/${EXPERIMENT}/run_*/density.npz
# - Computes global POD basis
# - Fits MVAR model
# - Saves to rom_mvar/${EXPERIMENT}/model/
python scripts/rom_mvar_train.py \
  --config ${CONFIG}

EXIT_CODE=$?

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
echo ""
echo "========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ ROM/MVAR training completed successfully!"
    echo ""
    echo "Model artifacts saved to:"
    echo "  rom_mvar/vicsek_morse_base/model/pod_basis.npz"
    echo "  rom_mvar/vicsek_morse_base/model/mvar_params.npz"
    echo "  rom_mvar/vicsek_morse_base/model/train_summary.json"
    echo ""
    echo "Next steps:"
    echo "  1. Run evaluation: sbatch scripts/slurm/run_vicsek_morse_eval.slurm"
    echo "  2. Or submit both with dependency: sbatch --dependency=afterok:${SLURM_JOB_ID} scripts/slurm/run_vicsek_morse_eval.slurm"
else
    echo "✗ ROM/MVAR training failed with exit code ${EXIT_CODE}"
fi
echo "End time: $(date)"
echo "========================================="

exit ${EXIT_CODE}
