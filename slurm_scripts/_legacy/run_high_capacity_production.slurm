#!/bin/bash
#SBATCH --job-name=high_cap_prod
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/high_cap_prod_%j.out
#SBATCH --error=slurm_logs/high_cap_prod_%j.err

# High Capacity ROM pipeline - PRODUCTION RUN
# N=100 particles, 32×32 grid, bandwidth=4.0, POD energy=0.99, w=4
# Training: 10s, Test: 25s (15s forecast)

echo "=========================================="
echo "High Capacity ROM Pipeline - PRODUCTION"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running on: $(hostname)"
echo ""
echo "Configuration:"
echo "  Training: 1008 runs × 10.0s = 96,768 windows"
echo "  Test: 37 runs × 25.0s (15s forecast)"
echo "  Particles: N=100"
echo "  Density: 32×32 grid, bandwidth=4.0"
echo "  POD: energy_threshold=0.99 (adaptive)"
echo "  MVAR: w=4 lags, ridge=1e-4"
echo ""

# Load environment
source ~/wsindy-manifold/shell_scripts/setup_oscar_env.sh

# Go to project directory
cd ~/wsindy-manifold

# Run pipeline with production config
python run_unified_mvar_pipeline.py \
  --config configs/high_capacity_production.yaml \
  --experiment_name high_capacity_production

echo ""
echo "=========================================="
echo "Job Complete"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "To visualize results locally:"
echo "  scp -r oscar:~/wsindy-manifold/oscar_output/high_capacity_production ."
echo "  python run_visualizations.py --experiment_name high_capacity_production"
