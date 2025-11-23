#!/bin/bash
#SBATCH --job-name=train_pod_mvar
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/training.out
#SBATCH --error=logs/training.err

# Oscar Pipeline - Step 3: Train POD + MVAR Models
# This single job trains the models after all data is generated

echo "=========================================="
echo "Training POD + MVAR Models"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

# Load environment (adjust for your Oscar setup)
# module load python/3.9.0
# source ~/.virtualenvs/wsindy/bin/activate

# Train models
python oscar_pipeline/train_pod_mvar.py \
    --train_dir oscar_outputs/training \
    --output_dir oscar_outputs/models

echo "=========================================="
echo "Completed at: $(date)"
echo "=========================================="
