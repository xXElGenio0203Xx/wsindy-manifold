#!/bin/bash
#SBATCH --job-name=rom_train
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ROM Pipeline - Stage 2 & 3: POD + MVAR Training
#
# This script builds the global POD basis and trains the MVAR model.
#
# Usage:
#   # Edit the parameters below, then:
#   sbatch scripts/slurm/job_train.sh
#
#   # Or submit with dependency on ensemble job:
#   sbatch --dependency=afterok:$ENSEMBLE_JOB_ID scripts/slurm/job_train.sh
#
# Outputs:
#   rom/<experiment>/pod/basis.npz
#   rom/<experiment>/latent/run_XXXX_latent.npz
#   rom/<experiment>/mvar/mvar_model.npz

echo "=========================================="
echo "ROM Pipeline - Stage 2 & 3: POD + MVAR"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running on: $(hostname)"
echo ""

# Load modules
module load python/3.11
source ~/envs/wsindy/bin/activate

# =============================================================================
# PARAMETERS - EDIT THESE
# =============================================================================

# Experiment configuration
EXP_NAME="production_run1"
SIM_ROOT="simulations/social_force_N200_T1000_alpha1.5_beta0.5_Cr2_Ca1_lr0.9_la1_v00.5_noise0.1/runs"

# Train/test split
TRAIN_RUNS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
TEST_RUNS=(16 17 18 19)

# POD parameters
ENERGY_THRESHOLD=0.995

# MVAR parameters
MVAR_ORDER=6
RIDGE=1e-6
TRAIN_FRAC=0.8

# =============================================================================

echo "Experiment:       $EXP_NAME"
echo "Simulation root:  $SIM_ROOT"
echo "Training runs:    ${TRAIN_RUNS[@]}"
echo "Test runs:        ${TEST_RUNS[@]}"
echo ""

# Check simulation directory exists
if [ ! -d "$SIM_ROOT" ]; then
    echo "ERROR: Simulation directory not found: $SIM_ROOT"
    exit 1
fi

# =============================================================================
# Stage 2: Build POD Basis
# =============================================================================

echo "[Stage 2/3] Building POD basis..."
echo ""

python scripts/rom_build_pod.py \
    --experiment_name "$EXP_NAME" \
    --sim_root "$SIM_ROOT" \
    --train_runs ${TRAIN_RUNS[@]} \
    --test_runs ${TEST_RUNS[@]} \
    --energy_threshold $ENERGY_THRESHOLD

POD_EXIT=$?

if [ $POD_EXIT -ne 0 ]; then
    echo "ERROR: POD stage failed with exit code $POD_EXIT"
    exit $POD_EXIT
fi

echo ""

# =============================================================================
# Stage 3: Train MVAR Model
# =============================================================================

echo "[Stage 3/3] Training MVAR model..."
echo ""

python scripts/rom_train_mvar.py \
    --experiment_name "$EXP_NAME" \
    --mvar_order $MVAR_ORDER \
    --ridge $RIDGE \
    --train_frac $TRAIN_FRAC

MVAR_EXIT=$?

if [ $MVAR_EXIT -ne 0 ]; then
    echo "ERROR: MVAR stage failed with exit code $MVAR_EXIT"
    exit $MVAR_EXIT
fi

echo ""
echo "=========================================="
echo "Training complete!"
echo "End time: $(date)"
echo "=========================================="

exit 0
