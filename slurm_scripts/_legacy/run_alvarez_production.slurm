#!/bin/bash
#SBATCH --job-name=alvarez_prod
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/alvarez_prod_%j.out
#SBATCH --error=slurm_logs/alvarez_prod_%j.err

# Alvarez-style ROM pipeline - PRODUCTION RUN
# Well-conditioned MVAR: d=20, w=5, ridge=1e-4, bandwidth=3.0
# 400 training runs × 75 windows = 30,000 samples for 2,000 parameters (ρ=15)

echo "=========================================="
echo "Alvarez-Style ROM Pipeline - PRODUCTION"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running on: $(hostname)"
echo ""
echo "Configuration:"
echo "  Training: 400 runs × 8.0s = 30,000 windows"
echo "  Test: 40 runs × 10.0s (extrapolation)"
echo "  POD: d=20 fixed modes"
echo "  MVAR: w=5 lags, ridge=1e-4"
echo "  Identifiability: ρ ≈ 15"
echo ""

# Load environment
source ~/wsindy-manifold/shell_scripts/setup_oscar_env.sh

# Go to project directory
cd ~/wsindy-manifold

# Run pipeline with production config
python run_unified_mvar_pipeline.py \
  --config configs/alvarez_style_production.yaml \
  --experiment_name alvarez_production

echo ""
echo "=========================================="
echo "Job Complete"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "To visualize results locally:"
echo "  scp -r oscar:~/wsindy-manifold/oscar_output/alvarez_production ."
echo "  python run_visualizations.py --experiment_name alvarez_production"
