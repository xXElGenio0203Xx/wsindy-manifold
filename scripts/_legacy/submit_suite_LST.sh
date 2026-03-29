#!/bin/bash
#
# Submit LSTM experiment suite (LST1-LST8) to OSCAR
# Each job: 144 train sims + MVAR + LSTM training + 20 test evals
# Expected runtime: ~4-6h (sims ~2h, MVAR <1min, LSTM ~30min, eval ~1h)
#

set -e

CONFIGS=(
    "LST1_raw_none_align_h32_L1"
    "LST2_raw_none_align_h64_L2"
    "LST3_raw_none_align_h64_L2_multistep"
    "LST4_sqrt_simplex_align_h64_L2"
    "LST5_raw_none_noAlign_h64_L2"
    "LST6_sqrt_none_noAlign_h64_L2"
    "LST7_raw_none_align_h128_L2"
    "LST8_raw_none_align_h64_L2_ss"
)

echo "============================================="
echo "LSTM Suite: Submitting ${#CONFIGS[@]} experiments"
echo "============================================="

for cfg in "${CONFIGS[@]}"; do
    echo ""
    echo "--- Submitting: ${cfg} ---"

    JOB_ID=$(sbatch \
        --job-name="${cfg}" \
        --output="slurm_logs/${cfg}_%j.out" \
        --error="slurm_logs/${cfg}_%j.err" \
        --time=10:00:00 \
        --mem=32G \
        --cpus-per-task=4 \
        --partition=batch \
        --wrap="
            module load python/3.11.11-5e66
            source ~/wsindy_env_new/bin/activate
            export PYTHONPATH=src
            cd ~/wsindy-manifold
            python ROM_pipeline.py --config configs/${cfg}.yaml --experiment_name ${cfg}
        " \
        --parsable)

    echo "   Job ID: ${JOB_ID}"
    echo "   Config: configs/${cfg}.yaml"
done

echo ""
echo "============================================="
echo "All ${#CONFIGS[@]} jobs submitted!"
echo "Monitor: squeue -u \$USER"
echo "============================================="
