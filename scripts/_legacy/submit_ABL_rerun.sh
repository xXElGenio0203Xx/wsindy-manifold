#!/bin/bash
# Re-submit ABL simplex jobs (3,4,7,8) after mass_postprocess config fix.
# The original runs silently ignored the simplex setting because it was under
# rom: but the evaluator reads from eval:.  config_loader.py now forwards it.
set -e
cd ~/wsindy-manifold
mkdir -p slurm_logs

WRAP_PREFIX="module load python/3.11.11-5e66 && source ~/wsindy_env_new/bin/activate && cd ~/wsindy-manifold && PYTHONPATH=src python ROM_pipeline.py"

for cfg in \
  configs/ABL3_N200_raw_simplex_noAlign_H300.yaml \
  configs/ABL4_N200_raw_simplex_align_H300.yaml \
  configs/ABL7_N200_sqrt_simplex_noAlign_H300.yaml \
  configs/ABL8_N200_sqrt_simplex_align_H300.yaml; do

    name=$(basename "$cfg" .yaml)_v2
    echo "Submitting $name ..."
    sbatch --job-name="$name" --time=04:00:00 -N1 -n4 --mem=24G -p batch \
        -o "slurm_logs/${name}_%j.out" -e "slurm_logs/${name}_%j.err" \
        --wrap="$WRAP_PREFIX --config $cfg --experiment_name $name"
done

echo ""
echo "=== ABL simplex re-run jobs submitted ==="
squeue -u "$USER"
