#!/bin/bash
# Submit all 12 Suite X experiments on OSCAR
set -e
cd ~/wsindy-manifold
mkdir -p slurm_logs

WRAP_PREFIX="module load python/3.11.11-5e66 && source ~/wsindy_env_new/bin/activate && cd ~/wsindy-manifold && PYTHONPATH=src python ROM_pipeline.py"

# H37 jobs (X1,X2,X5,X6,X9,X10) — 2h wall time
for cfg in configs/X1_V1_raw_H37.yaml configs/X2_V1_sqrtSimplex_H37.yaml configs/X5_V33_raw_H37.yaml configs/X6_V33_sqrtSimplex_H37.yaml configs/X9_V34_raw_H37.yaml configs/X10_V34_sqrtSimplex_H37.yaml; do
    name=$(basename "$cfg" .yaml)
    echo "Submitting $name (H37, 2h)..."
    sbatch --job-name="$name" --time=02:00:00 -N1 -n4 --mem=24G -p batch \
        -o "slurm_logs/${name}_%j.out" -e "slurm_logs/${name}_%j.err" \
        --wrap="$WRAP_PREFIX --config $cfg --experiment_name $name"
done

# H162 jobs (X3,X4,X7,X8,X11,X12) — 4h wall time
for cfg in configs/X3_V1_raw_H162.yaml configs/X4_V1_sqrtSimplex_H162.yaml configs/X7_V33_raw_H162.yaml configs/X8_V33_sqrtSimplex_H162.yaml configs/X11_V34_raw_H162.yaml configs/X12_V34_sqrtSimplex_H162.yaml; do
    name=$(basename "$cfg" .yaml)
    echo "Submitting $name (H162, 4h)..."
    sbatch --job-name="$name" --time=04:00:00 -N1 -n4 --mem=24G -p batch \
        -o "slurm_logs/${name}_%j.out" -e "slurm_logs/${name}_%j.err" \
        --wrap="$WRAP_PREFIX --config $cfg --experiment_name $name"
done

echo ""
echo "=== All 12 Suite X jobs submitted ==="
squeue -u "$USER"
