#!/bin/bash
#SBATCH --job-name=gen_train
#SBATCH --array=0-99
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/train_%a.out
#SBATCH --error=logs/train_%a.err

# Oscar Pipeline - Step 1: Generate Training Data
# This array job generates 100 training simulations in parallel

echo "=========================================="
echo "Training Simulation ${SLURM_ARRAY_TASK_ID}/99"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

# Load environment (adjust for your Oscar setup)
# module load python/3.9.0
# source ~/.virtualenvs/wsindy/bin/activate

# Run simulation
python oscar_pipeline/generate_single_sim.py \
    --mode train \
    --sim_id ${SLURM_ARRAY_TASK_ID} \
    --output_dir oscar_outputs/training

echo "=========================================="
echo "Completed at: $(date)"
echo "=========================================="
