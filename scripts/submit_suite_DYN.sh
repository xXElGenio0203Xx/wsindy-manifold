#!/bin/bash
# Submit DYN (dynamics regime) experiments on OSCAR.
# DYN5 and DYN7 already completed with 200s — only re-running 1-4,6 with 50s.
# DYN6 also had speed_mode bug (fixed: "variable" not "variable_with_forces").
set -e
cd ~/wsindy-manifold
mkdir -p slurm_logs

WRAP_PREFIX="module load python/3.11.11-5e66 && source ~/wsindy_env_new/bin/activate && cd ~/wsindy-manifold && PYTHONPATH=src python ROM_pipeline.py"

for cfg in \
  configs/DYN1_gentle.yaml \
  configs/DYN2_hypervelocity.yaml \
  configs/DYN3_hypernoisy.yaml \
  configs/DYN4_blackhole.yaml \
  configs/DYN6_varspeed.yaml; do

    name=$(basename "$cfg" .yaml)_v2
    echo "Submitting $name (12h) ..."
    sbatch --job-name="$name" --time=12:00:00 -N1 -n4 --mem=24G -p batch \
        -o "slurm_logs/${name}_%j.out" -e "slurm_logs/${name}_%j.err" \
        --wrap="$WRAP_PREFIX --config $cfg --experiment_name $name"
done

echo ""
echo "=== DYN v2 jobs submitted (DYN5,DYN7 already complete) ==="
squeue -u "$USER"
