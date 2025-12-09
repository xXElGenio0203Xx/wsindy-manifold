#!/bin/bash
#SBATCH --job-name=alvarez_test
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/alvarez_test_%j.out
#SBATCH --error=slurm_logs/alvarez_test_%j.err

# Alvarez-style ROM pipeline - Quick Test (T=1.0s)
# Verify well-conditioned MVAR setup works before full production run

echo "=========================================="
echo "Alvarez-Style ROM Pipeline - Quick Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running on: $(hostname)"
echo ""

# Load environment
source ~/wsindy-manifold/shell_scripts/setup_oscar_env.sh

# Go to project directory
cd ~/wsindy-manifold

# Run pipeline with test config
python run_unified_mvar_pipeline.py \
  --config configs/alvarez_style_test.yaml \
  --experiment_name alvarez_test

echo ""
echo "=========================================="
echo "Job Complete"
echo "End time: $(date)"
echo "=========================================="
