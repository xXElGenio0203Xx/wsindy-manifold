#!/bin/bash
# Master script to run complete Oscar pipeline

echo "========================================"
echo "OSCAR PIPELINE - COMPLETE WORKFLOW"
echo "========================================"

# Create output and log directories
mkdir -p oscar_outputs/{training,test,models}
mkdir -p logs

# Step 1: Generate training data (100 simulations in parallel)
echo ""
echo "Step 1: Submitting training data generation..."
TRAIN_JOB=$(sbatch --parsable oscar_pipeline/01_generate_training_data.sh)
echo "  Job ID: ${TRAIN_JOB}"

# Step 2: Generate test data (20 simulations in parallel)
echo ""
echo "Step 2: Submitting test data generation..."
TEST_JOB=$(sbatch --parsable oscar_pipeline/02_generate_test_data.sh)
echo "  Job ID: ${TEST_JOB}"

# Step 3: Train models (depends on training data completion)
echo ""
echo "Step 3: Submitting model training (depends on training data)..."
MODEL_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} oscar_pipeline/03_train_pod_mvar.sh)
echo "  Job ID: ${MODEL_JOB}"

echo ""
echo "========================================"
echo "Jobs submitted successfully!"
echo "========================================"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f logs/train_*.out"
echo "  tail -f logs/test_*.out"
echo "  tail -f logs/training.out"
echo ""
echo "Check completion:"
echo "  sacct -j ${TRAIN_JOB},${TEST_JOB},${MODEL_JOB} --format=JobID,State,ExitCode"
echo ""
echo "When complete, sync results:"
echo "  rsync -avz oscar:/path/to/oscar_outputs/ ./oscar_outputs/"
echo ""
