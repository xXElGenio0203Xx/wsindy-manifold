#!/bin/bash
#SBATCH --job-name=pod_mvar_test
#SBATCH --output=slurm_logs/pod_mvar_test_%j.out
#SBATCH --error=slurm_logs/pod_mvar_test_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=batch

echo "=========================================="
echo "POD/MVAR/Testing Pipeline"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Load modules
module load python/3.11.0s-ixrhc3q
module load ffmpeg/7.0-xny2fb2

# Activate environment
source ~/wsindy_env/bin/activate

# Navigate to workspace
cd ~/wsindy-manifold

# Run pipeline (will use existing training data)
python run_gaussians_pipeline.py --config configs/gaussians_local_continue.yaml --experiment_name gaussians_oscar

echo ""
echo "=========================================="
echo "COMPLETED"
echo "=========================================="
echo "End time: $(date)"
