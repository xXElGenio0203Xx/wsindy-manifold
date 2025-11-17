#!/bin/bash
# Master script to submit full ROM pipeline with job dependencies
#
# Usage:
#   bash scripts/slurm/submit_pipeline.sh

echo "=========================================="
echo "Submitting ROM Pipeline"
echo "=========================================="
echo ""

# Create logs directory
mkdir -p logs

# Submit Stage 1: Ensemble generation
echo "[1/3] Submitting ensemble generation..."
JOB1=$(sbatch scripts/slurm/job_ensemble.sh | awk '{print $4}')

if [ -z "$JOB1" ]; then
    echo "ERROR: Failed to submit ensemble job"
    exit 1
fi

echo "  Job ID: $JOB1"
echo ""

# Submit Stage 2 & 3: POD + MVAR training (depends on Stage 1)
echo "[2/3] Submitting POD + MVAR training (dependency: $JOB1)..."
JOB2=$(sbatch --dependency=afterok:$JOB1 scripts/slurm/job_train.sh | awk '{print $4}')

if [ -z "$JOB2" ]; then
    echo "ERROR: Failed to submit training job"
    exit 1
fi

echo "  Job ID: $JOB2"
echo ""

# Submit Stage 4: Evaluation (depends on Stage 2 & 3)
echo "[3/3] Submitting evaluation (dependency: $JOB2)..."
JOB3=$(sbatch --dependency=afterok:$JOB2 scripts/slurm/job_evaluate.sh | awk '{print $4}')

if [ -z "$JOB3" ]; then
    echo "ERROR: Failed to submit evaluation job"
    exit 1
fi

echo "  Job ID: $JOB3"
echo ""

echo "=========================================="
echo "Job chain submitted successfully!"
echo ""
echo "Job dependencies:"
echo "  $JOB1 (ensemble) -> $JOB2 (train) -> $JOB3 (evaluate)"
echo ""
echo "Monitor progress:"
echo "  squeue -u $USER"
echo "  tail -f logs/ensemble_$JOB1.out"
echo "  tail -f logs/train_$JOB2.out"
echo "  tail -f logs/eval_$JOB3.out"
echo ""
echo "Cancel all jobs:"
echo "  scancel $JOB1 $JOB2 $JOB3"
echo "=========================================="
