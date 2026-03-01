#!/bin/bash
#SBATCH --job-name=XABL_suite
#SBATCH --array=1-8
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/XABL%a_%j.out
#SBATCH -e slurm_logs/XABL%a_%j.err

# ============================================================================
# EXTENDED ABLATION SUITE (XABL) — alignment=ON, 2³ factorial:
#   sqrt × simplex × spectral_scaling
# ============================================================================
# Tests whether spectral radius scaling to 1.0 helps/hurts when alignment is ON.
# 8 experiments, ~40-60 min each.
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

export PYTHONPATH=src

declare -a CONFIGS=(
    ""
    "configs/XABL1_align_raw_none_noScale.yaml"
    "configs/XABL2_align_raw_none_scale1.yaml"
    "configs/XABL3_align_raw_simplex_noScale.yaml"
    "configs/XABL4_align_raw_simplex_scale1.yaml"
    "configs/XABL5_align_sqrt_none_noScale.yaml"
    "configs/XABL6_align_sqrt_none_scale1.yaml"
    "configs/XABL7_align_sqrt_simplex_noScale.yaml"
    "configs/XABL8_align_sqrt_simplex_scale1.yaml"
)

declare -a EXPNAMES=(
    ""
    "XABL1_align_raw_none_noScale"
    "XABL2_align_raw_none_scale1"
    "XABL3_align_raw_simplex_noScale"
    "XABL4_align_raw_simplex_scale1"
    "XABL5_align_sqrt_none_noScale"
    "XABL6_align_sqrt_none_scale1"
    "XABL7_align_sqrt_simplex_noScale"
    "XABL8_align_sqrt_simplex_scale1"
)

declare -a LABELS=(
    ""
    "align / raw / none / no scaling"
    "align / raw / none / scale→1.0"
    "align / raw / simplex / no scaling"
    "align / raw / simplex / scale→1.0"
    "align / sqrt / none / no scaling"
    "align / sqrt / none / scale→1.0"
    "align / sqrt / simplex / no scaling"
    "align / sqrt / simplex / scale→1.0"
)

IDX=$SLURM_ARRAY_TASK_ID
CONFIG="${CONFIGS[$IDX]}"
EXP_NAME="${EXPNAMES[$IDX]}"
LABEL="${LABELS[$IDX]}"

echo "============================================================"
echo "EXTENDED ABLATION SUITE — XABL${IDX}"
echo "============================================================"
echo "Config:     $CONFIG"
echo "Experiment: $EXP_NAME"
echo "Settings:   $LABEL"
echo "Start time: $(date)"
echo "Node:       $(hostname)"
echo "CPUs:       $SLURM_NTASKS"
echo "Array task: $IDX / 8"
echo ""

# ── Run unified ROM pipeline ──
echo "Running unified ROM pipeline..."
echo "============================================================"

python run_unified_rom_pipeline.py \
    --config "$CONFIG" \
    --experiment_name "$EXP_NAME"

PIPE_EXIT=$?
echo "Pipeline exit code: $PIPE_EXIT"
echo "Time: $(date)"

if [ $PIPE_EXIT -ne 0 ]; then
    echo "❌ Pipeline failed for XABL${IDX}! Aborting."
    exit $PIPE_EXIT
fi

# ── Summary ──
echo ""
echo "============================================================"
echo "XABL${IDX} COMPLETE — $LABEL"
echo "============================================================"
echo "End time: $(date)"

if [ -f "oscar_output/$EXP_NAME/summary.json" ]; then
    python3 -c "
import json
d = json.load(open('oscar_output/$EXP_NAME/summary.json'))
m = d.get('mvar', {})
r2 = m.get('mean_r2_test', '?')
sr_before = m.get('spectral_radius', '?')
sr_after = m.get('spectral_radius_after', sr_before)
pod = d.get('pod', {})
print(f'  transform={pod.get(\"density_transform\",\"?\")}, postprocess={d.get(\"mass_postprocess\",\"?\")}')
if isinstance(r2, float):
    print(f'  R²_rollout = {r2:+.4f}')
    print(f'  ρ_before={sr_before}, ρ_after={sr_after}')
" 2>/dev/null || true
fi

echo "Output: oscar_output/$EXP_NAME/"
echo "Done."
