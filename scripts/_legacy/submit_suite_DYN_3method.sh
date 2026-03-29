#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# submit_suite_DYN_3method.sh — Submit DYN1-7 with MVAR + LSTM + WSINDy
# ═══════════════════════════════════════════════════════════════════
# Uses ROM_WSINDY_pipeline.py (not ROM_pipeline.py) to run all 3 methods.
# Each job: ~12-24h depending on WSINDy bootstrap (B=50).
#
# Usage (on Oscar):
#   bash scripts/submit_suite_DYN_3method.sh
#   bash scripts/submit_suite_DYN_3method.sh --dry-run
# ═══════════════════════════════════════════════════════════════════
set -e

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN — will print commands but NOT submit ==="
fi

cd ~/wsindy-manifold
mkdir -p slurm_logs

WRAP_PREFIX="module load python/3.11.11-5e66 && source ~/wsindy_env_new/bin/activate && cd ~/wsindy-manifold && PYTHONPATH=src python ROM_WSINDY_pipeline.py"

declare -A CONFIGS
CONFIGS=(
    ["DYN1_gentle"]="configs/DYN1_gentle.yaml"
    ["DYN2_hypervelocity"]="configs/DYN2_hypervelocity.yaml"
    ["DYN3_hypernoisy"]="configs/DYN3_hypernoisy.yaml"
    ["DYN4_blackhole"]="configs/DYN4_blackhole.yaml"
    ["DYN5_supernova"]="configs/DYN5_supernova.yaml"
    ["DYN6_varspeed"]="configs/DYN6_varspeed.yaml"
    ["DYN7_pure_vicsek"]="configs/DYN7_pure_vicsek.yaml"
)

# WSINDy + LSTM adds compute time — request 24h, 32G
for name in DYN1_gentle DYN2_hypervelocity DYN3_hypernoisy DYN4_blackhole DYN5_supernova DYN6_varspeed DYN7_pure_vicsek; do
    cfg="${CONFIGS[$name]}"
    job_name="${name}_3method"

    CMD="sbatch --job-name=\"$job_name\" --time=24:00:00 -N1 -n4 --mem=32G -p batch \
        -o \"slurm_logs/${job_name}_%j.out\" -e \"slurm_logs/${job_name}_%j.err\" \
        --wrap=\"$WRAP_PREFIX --config $cfg --experiment_name $name\""

    echo "  $job_name → $cfg"

    if $DRY_RUN; then
        echo "    [DRY] $CMD"
    else
        eval "$CMD"
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  7 DYN experiments submitted (MVAR + LSTM + WSINDy)"
echo "  Pipeline: ROM_WSINDY_pipeline.py"
echo "  Monitor: squeue -u \$USER"
echo "  Logs:    slurm_logs/DYN*_3method_*.out"
echo "═══════════════════════════════════════════════════════════════"

if ! $DRY_RUN; then
    squeue -u "$USER"
fi
