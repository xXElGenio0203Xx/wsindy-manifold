#!/bin/bash
#SBATCH --job-name=rom_eval
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# ROM Pipeline - Stage 4: Evaluation
#
# This script evaluates MVAR forecasts on test runs, generating comprehensive
# metrics, plots, and videos.
#
# Usage:
#   # Edit the parameters below, then:
#   sbatch scripts/slurm/job_evaluate.sh
#
#   # Or submit with dependency on training job:
#   sbatch --dependency=afterok:$TRAIN_JOB_ID scripts/slurm/job_evaluate.sh
#
# Outputs:
#   rom/<experiment>/mvar/forecast/*.npz
#   rom/<experiment>/mvar/forecast/*.json
#   rom/<experiment>/mvar/forecast/*.png
#   rom/<experiment>/mvar/forecast/*.mp4

echo "=========================================="
echo "ROM Pipeline - Stage 4: Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running on: $(hostname)"
echo ""

# Load modules
module load python/3.11
module load ffmpeg  # Required for video generation
source ~/envs/wsindy/bin/activate

# =============================================================================
# PARAMETERS - EDIT THESE (should match job_train.sh)
# =============================================================================

# Experiment configuration
EXP_NAME="production_run1"
SIM_ROOT="simulations/social_force_N200_T1000_alpha1.5_beta0.5_Cr2_Ca1_lr0.9_la1_v00.5_noise0.1/runs"

# Video generation
GENERATE_VIDEOS=true  # Set to false to skip videos (faster)

# =============================================================================

echo "Experiment:       $EXP_NAME"
echo "Simulation root:  $SIM_ROOT"
echo "Generate videos:  $GENERATE_VIDEOS"
echo ""

# Check directories exist
if [ ! -d "$SIM_ROOT" ]; then
    echo "ERROR: Simulation directory not found: $SIM_ROOT"
    exit 1
fi

if [ ! -d "rom/$EXP_NAME" ]; then
    echo "ERROR: ROM experiment directory not found: rom/$EXP_NAME"
    echo "Run job_train.sh first!"
    exit 1
fi

# Build command
CMD="python scripts/rom_evaluate.py \
    --experiment_name $EXP_NAME \
    --sim_root $SIM_ROOT"

# Add --no_videos flag if needed
if [ "$GENERATE_VIDEOS" = false ]; then
    CMD="$CMD --no_videos"
fi

# Run evaluation
echo "Running evaluation..."
echo ""

$CMD

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Evaluation failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo ""
echo "Review results:"
echo "  cd rom/$EXP_NAME/mvar/forecast"
echo "  cat aggregate_metrics.json"
echo ""
echo "End time: $(date)"
echo "=========================================="

exit 0
