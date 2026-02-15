#!/bin/bash
#SBATCH --job-name=synthesis_v2_5
#SBATCH --output=slurm_logs/synthesis_v2_5_%j.out
#SBATCH --error=slurm_logs/synthesis_v2_5_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

echo "============================================"
echo "SYNTHESIS V2.5 — V2.2 with d=25 modes"
echo "Start: $(date)"
echo "============================================"

cd ~/wsindy-manifold
source ~/wsindy_env_new/bin/activate
export PYTHONPATH=src

mkdir -p slurm_logs oscar_output

python ROM_pipeline.py \
    --config configs/synthesis_v2_5.yaml \
    --experiment_name synthesis_v2_5

echo "============================================"
echo "End: $(date)"
echo "============================================"
