#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# submit_VLST_lstm_ablation.sh — LSTM-only variable-speed ablation
# ═══════════════════════════════════════════════════════════════════
# Suite A (VLST_A1-7): sqrt + simplex + C0  — transform prevents negatives?
# Suite B (VLST_B1-7): raw + none + C2      — clamping alone sufficient?
#
# All are LSTM-only (MVAR disabled) to save compute.
# Uses ROM_pipeline.py (no WSINDy needed for this ablation).
#
# Usage:
#   bash scripts/submit_VLST_lstm_ablation.sh
#   bash scripts/submit_VLST_lstm_ablation.sh --dry-run
# ═══════════════════════════════════════════════════════════════════
set -e

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN ==="
fi

cd ~/wsindy-manifold
mkdir -p slurm_logs

WRAP_PREFIX="module load python/3.11.11-5e66 && source ~/wsindy_env_new/bin/activate && cd ~/wsindy-manifold && PYTHONPATH=src python ROM_pipeline.py"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Suite A: LSTM + sqrt + simplex + C0"
echo "═══════════════════════════════════════════════════════════"

SUITE_A_CONFIGS=(
    "VLST_A1_gentle_sqrtSimplex_C0"
    "VLST_A2_hypervelocity_sqrtSimplex_C0"
    "VLST_A3_hypernoisy_sqrtSimplex_C0"
    "VLST_A4_blackhole_sqrtSimplex_C0"
    "VLST_A5_supernova_sqrtSimplex_C0"
    "VLST_A6_baseline_sqrtSimplex_C0"
    "VLST_A7_pure_vicsek_sqrtSimplex_C0"
)

for name in "${SUITE_A_CONFIGS[@]}"; do
    cfg="configs/${name}.yaml"
    echo "  $name → $cfg"
    if $DRY_RUN; then
        echo "    [DRY] sbatch ... --wrap=\"$WRAP_PREFIX --config $cfg --experiment_name $name\""
    else
        sbatch --job-name="$name" --time=12:00:00 -N1 -n4 --mem=32G -p batch \
            -o "slurm_logs/${name}_%j.out" -e "slurm_logs/${name}_%j.err" \
            --wrap="$WRAP_PREFIX --config $cfg --experiment_name $name"
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Suite B: LSTM + raw + C2"
echo "═══════════════════════════════════════════════════════════"

SUITE_B_CONFIGS=(
    "VLST_B1_gentle_raw_C2"
    "VLST_B2_hypervelocity_raw_C2"
    "VLST_B3_hypernoisy_raw_C2"
    "VLST_B4_blackhole_raw_C2"
    "VLST_B5_supernova_raw_C2"
    "VLST_B6_baseline_raw_C2"
    "VLST_B7_pure_vicsek_raw_C2"
)

for name in "${SUITE_B_CONFIGS[@]}"; do
    cfg="configs/${name}.yaml"
    echo "  $name → $cfg"
    if $DRY_RUN; then
        echo "    [DRY] sbatch ... --wrap=\"$WRAP_PREFIX --config $cfg --experiment_name $name\""
    else
        sbatch --job-name="$name" --time=12:00:00 -N1 -n4 --mem=32G -p batch \
            -o "slurm_logs/${name}_%j.out" -e "slurm_logs/${name}_%j.err" \
            --wrap="$WRAP_PREFIX --config $cfg --experiment_name $name"
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  14 LSTM-only jobs submitted (7 Suite A + 7 Suite B)"
echo "  Pipeline: ROM_pipeline.py (LSTM only, no MVAR)"
echo "  Monitor: squeue -u \$USER"
echo "  Logs:    slurm_logs/VLST_*.out"
echo "═══════════════════════════════════════════════════════════"

if ! $DRY_RUN; then
    squeue -u "$USER"
fi
