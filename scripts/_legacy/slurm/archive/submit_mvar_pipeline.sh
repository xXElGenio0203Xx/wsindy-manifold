#!/bin/bash
# Master script to submit complete ROM/MVAR pipeline with job dependencies
#
# Usage:
#   bash scripts/slurm/submit_mvar_pipeline.sh configs/rom_mvar_example.yaml

set -e

if [ $# -lt 1 ]; then
    echo "Usage: bash submit_mvar_pipeline.sh <config.yaml>"
    exit 1
fi

CONFIG=$1
EXPERIMENT=$(grep "experiment_name:" $CONFIG | awk '{print $2}')

echo "========================================="
echo "Submitting ROM/MVAR Pipeline"
echo "========================================="
echo "Config: $CONFIG"
echo "Experiment: $EXPERIMENT"
echo "========================================="

# Submit training job
echo ""
echo "Submitting training job..."
TRAIN_JOB=$(sbatch --parsable scripts/slurm/job_mvar_train.sh $CONFIG)
echo "  Train job ID: $TRAIN_JOB"

# Submit evaluation job (depends on training)
echo ""
echo "Submitting evaluation job (depends on $TRAIN_JOB)..."
EVAL_JOB=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB \\
    scripts/slurm/job_mvar_eval.sh $EXPERIMENT $CONFIG)
echo "  Eval job ID: $EVAL_JOB"

echo ""
echo "========================================="
echo "Pipeline submitted successfully!"
echo "========================================="
echo "Job chain: $TRAIN_JOB → $EVAL_JOB"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "After completion, visualize locally:"
echo "  rsync -avz oscar:/path/to/rom_mvar/$EXPERIMENT/ rom_mvar/$EXPERIMENT/"
echo "  python scripts/rom_mvar_visualize.py --experiment $EXPERIMENT"
echo ""
