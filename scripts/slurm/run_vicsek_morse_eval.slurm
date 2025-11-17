#!/bin/bash
#SBATCH --job-name=vicsek_eval
#SBATCH --partition=batch
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/vicsek_eval_%j.out
#SBATCH --error=logs/vicsek_eval_%j.err

# ==============================================================================
# Vicsek-Morse ROM/MVAR Evaluation (Oscar)
# ==============================================================================
# This script:
# 1. Loads trained ROM/MVAR model
# 2. Generates 2 unseen test IC simulations
# 3. Projects to latent space + forecasts with MVAR
# 4. Computes metrics (R², RMSE, mass error, τ)
# 5. Saves results to rom_mvar/vicsek_morse_base/test_ics/
#
# Usage (after training completes):
#   sbatch scripts/slurm/run_vicsek_morse_eval.slurm
#
# Or with dependency:
#   sbatch --dependency=afterok:<rom_job_id> scripts/slurm/run_vicsek_morse_eval.slurm
#
# Runtime: ~5 minutes (2 test sims + forecasting + metrics)
# ==============================================================================

set -e

# ------------------------------------------------------------------------------
# Environment Setup
# ------------------------------------------------------------------------------

# Load conda (Oscar CCV)
module --ignore_cache load miniconda3/23.11.0s-odstpk5
eval "$(conda shell.bash hook)"
conda activate wsindy

# Use 2 threads for light linear algebra
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# Change to repo directory
cd ~/src/wsindy-manifold

# Create logs directory
mkdir -p logs

# ------------------------------------------------------------------------------
# Job Info
# ------------------------------------------------------------------------------
echo "========================================="
echo "ROM/MVAR Evaluation"
echo "========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          ${SLURMD_NODENAME}"
echo "Start time:    $(date)"
echo "========================================="
echo ""

# ------------------------------------------------------------------------------
# Run ROM/MVAR Evaluation
# ------------------------------------------------------------------------------

echo "Starting ROM/MVAR evaluation..."
echo ""

# Run evaluation
# - Loads rom_mvar/vicsek_morse_base/model/*
# - Generates 2 test simulations (seeds 1000, 1001)
# - Forecasts with MVAR
# - Computes metrics
# - Saves to rom_mvar/vicsek_morse_base/test_ics/ic_*/
python scripts/rom_mvar_eval.py \
  --experiment vicsek_morse_base \
  --config configs/vicsek_morse_base.yaml

EXIT_CODE=$?

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
echo ""
echo "========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ ROM/MVAR evaluation completed successfully!"
    echo ""
    echo "Results saved to:"
    echo "  rom_mvar/vicsek_morse_base/test_ics/ic_1000/"
    echo "  rom_mvar/vicsek_morse_base/test_ics/ic_1001/"
    echo "  rom_mvar/vicsek_morse_base/aggregate_metrics/"
    echo ""
    echo "Next steps:"
    echo "  1. rsync results to local machine"
    echo "  2. Generate videos locally: python scripts/rom_mvar_visualize.py --experiment vicsek_morse_base"
else
    echo "✗ ROM/MVAR evaluation failed with exit code ${EXIT_CODE}"
fi
echo "End time: $(date)"
echo "========================================="

exit ${EXIT_CODE}
