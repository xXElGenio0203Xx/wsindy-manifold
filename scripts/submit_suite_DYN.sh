#!/bin/bash
# Submit all 7 DYN (dynamics regime) experiments on OSCAR.
# Each regime uses the winning ABL2 pipeline (raw+align) but with different
# physics parameters and 200s forecast horizons.
set -e
cd ~/wsindy-manifold
mkdir -p slurm_logs

WRAP_PREFIX="module load python/3.11.11-5e66 && source ~/wsindy_env_new/bin/activate && cd ~/wsindy-manifold && PYTHONPATH=src python ROM_pipeline.py"

for cfg in \
  configs/DYN1_gentle.yaml \
  configs/DYN2_hypervelocity.yaml \
  configs/DYN3_hypernoisy.yaml \
  configs/DYN4_blackhole.yaml \
  configs/DYN5_supernova.yaml \
  configs/DYN6_varspeed.yaml \
  configs/DYN7_pure_vicsek.yaml; do

    name=$(basename "$cfg" .yaml)
    echo "Submitting $name (8h) ..."
    sbatch --job-name="$name" --time=08:00:00 -N1 -n4 --mem=24G -p batch \
        -o "slurm_logs/${name}_%j.out" -e "slurm_logs/${name}_%j.err" \
        --wrap="$WRAP_PREFIX --config $cfg --experiment_name $name"
done

echo ""
echo "=== All 7 DYN jobs submitted ==="
squeue -u "$USER"
