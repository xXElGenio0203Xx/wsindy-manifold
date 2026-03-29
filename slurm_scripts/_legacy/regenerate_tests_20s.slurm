#!/bin/bash
#SBATCH --job-name=alvarez_regen_tests
#SBATCH --output=slurm_logs/alvarez_regen_%j.out
#SBATCH --error=slurm_logs/alvarez_regen_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

# Activate conda environment
source ~/.bashrc
conda activate wsindy

# Run regeneration script
echo "=========================================="
echo "REGENERATING TEST DATA WITH T=20s"
echo "=========================================="
echo "Start time: $(date)"
echo ""

cd ~/wsindy-manifold
python regenerate_tests_20s.py

echo ""
echo "End time: $(date)"
echo "=========================================="
