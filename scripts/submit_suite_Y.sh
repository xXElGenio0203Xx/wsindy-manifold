#!/bin/bash
# Submit all 6 Suite Y experiments (H100 mid-horizon) on OSCAR
set -e
cd ~/wsindy-manifold
mkdir -p slurm_logs

WRAP_PREFIX="module load python/3.11.11-5e66 && source ~/wsindy_env_new/bin/activate && cd ~/wsindy-manifold && PYTHONPATH=src python ROM_pipeline.py"

for cfg in configs/Y1_V1_raw_H100.yaml configs/Y2_V1_sqrtSimplex_H100.yaml configs/Y3_V33_raw_H100.yaml configs/Y4_V33_sqrtSimplex_H100.yaml configs/Y5_V34_raw_H100.yaml configs/Y6_V34_sqrtSimplex_H100.yaml; do
    name=$(basename "$cfg" .yaml)
    echo "Submitting $name (H100, 3h)..."
    sbatch --job-name="$name" --time=03:00:00 -N1 -n4 --mem=24G -p batch \
        -o "slurm_logs/${name}_%j.out" -e "slurm_logs/${name}_%j.err" \
        --wrap="$WRAP_PREFIX --config $cfg --experiment_name $name"
done

echo ""
echo "=== All 6 Suite Y jobs submitted ==="
squeue -u "$USER"
