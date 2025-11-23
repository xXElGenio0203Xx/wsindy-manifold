#!/bin/bash
#SBATCH --job-name=gen_test
#SBATCH --array=0-19
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/test_%a.out
#SBATCH --error=logs/test_%a.err

# Oscar Pipeline - Step 2: Generate Test Data
# This array job generates 20 test simulations in parallel
# Includes order parameter computation

echo "=========================================="
echo "Test Simulation ${SLURM_ARRAY_TASK_ID}/19"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

# Load environment (adjust for your Oscar setup)
# module load python/3.9.0
# source ~/.virtualenvs/wsindy/bin/activate

# Run simulation with order parameters
python oscar_pipeline/generate_single_sim.py \
    --mode test \
    --sim_id ${SLURM_ARRAY_TASK_ID} \
    --output_dir oscar_outputs/test \
    --compute_order_params

echo "=========================================="
echo "Completed at: $(date)"
echo "=========================================="
