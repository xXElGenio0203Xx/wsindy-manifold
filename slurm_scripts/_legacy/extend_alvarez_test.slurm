#!/bin/bash
#SBATCH --job-name=alvarez_extend
#SBATCH --output=slurm_logs/alvarez_extend_%j.out
#SBATCH --error=slurm_logs/alvarez_extend_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=batch

# Extend Alvarez test horizon to 1.5x training (12s)
# Only regenerates test runs, uses existing training data and ROM

module load anaconda/2024.02-1
source activate wsindy

cd ~/wsindy-manifold

echo "=========================================="
echo "Extending Alvarez Test Horizon"
echo "=========================================="
echo "Training: 8.0s (unchanged)"
echo "Test: 10.0s → 12.0s (1.5x)"
echo "Forecast: 2.0s → 4.0s (40 steps)"
echo ""

python extend_test_horizon.py

echo ""
echo "✓ Extension complete!"
echo "Download results with:"
echo "  scp -r emaciaso@ssh.ccv.brown.edu:~/wsindy-manifold/oscar_output/alvarez_production/test oscar_output/alvarez_production/"
