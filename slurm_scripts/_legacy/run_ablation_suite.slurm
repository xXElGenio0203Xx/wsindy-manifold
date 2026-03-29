#!/bin/bash
#SBATCH --job-name=ABL_suite
#SBATCH --array=1-8
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/ABL%a_%j.out
#SBATCH -e slurm_logs/ABL%a_%j.err

# ============================================================================
# ABLATION SUITE — 2³ factorial: sqrt × simplex × shift_align
# ============================================================================
# SLURM array job: task IDs 1-8, each runs one ablation config.
# N=200, Ca=1.5, Cr=0.5 — expected ~40-60 min each.
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

export PYTHONPATH=src

# Map array task ID to config and experiment name
declare -a CONFIGS=(
    ""  # placeholder for index 0
    "configs/ABL1_N200_raw_none_noAlign_H300.yaml"
    "configs/ABL2_N200_raw_none_align_H300.yaml"
    "configs/ABL3_N200_raw_simplex_noAlign_H300.yaml"
    "configs/ABL4_N200_raw_simplex_align_H300.yaml"
    "configs/ABL5_N200_sqrt_none_noAlign_H300.yaml"
    "configs/ABL6_N200_sqrt_none_align_H300.yaml"
    "configs/ABL7_N200_sqrt_simplex_noAlign_H300.yaml"
    "configs/ABL8_N200_sqrt_simplex_align_H300.yaml"
)

declare -a EXPNAMES=(
    ""
    "ABL1_N200_raw_none_noAlign_H300"
    "ABL2_N200_raw_none_align_H300"
    "ABL3_N200_raw_simplex_noAlign_H300"
    "ABL4_N200_raw_simplex_align_H300"
    "ABL5_N200_sqrt_none_noAlign_H300"
    "ABL6_N200_sqrt_none_align_H300"
    "ABL7_N200_sqrt_simplex_noAlign_H300"
    "ABL8_N200_sqrt_simplex_align_H300"
)

declare -a LABELS=(
    ""
    "raw / none / noAlign"
    "raw / none / align"
    "raw / simplex / noAlign"
    "raw / simplex / align"
    "sqrt / none / noAlign"
    "sqrt / none / align"
    "sqrt / simplex / noAlign"
    "sqrt / simplex / align"
)

IDX=$SLURM_ARRAY_TASK_ID
CONFIG="${CONFIGS[$IDX]}"
EXP_NAME="${EXPNAMES[$IDX]}"
LABEL="${LABELS[$IDX]}"

echo "============================================================"
echo "ABLATION SUITE — ABL${IDX}"
echo "============================================================"
echo "Config:     $CONFIG"
echo "Experiment: $EXP_NAME"
echo "Settings:   $LABEL"
echo "Start time: $(date)"
echo "Node:       $(hostname)"
echo "CPUs:       $SLURM_NTASKS"
echo "Array task: $IDX / 8"
echo ""

# ── Step 1: Full ROM Pipeline ──
echo "STEP 1/2: ROM PIPELINE"
echo "============================================================"

python ROM_pipeline.py \
    --config "$CONFIG" \
    --experiment_name "$EXP_NAME"

DATA_EXIT=$?
echo "ROM pipeline exit code: $DATA_EXIT"
echo "Time: $(date)"

if [ $DATA_EXIT -ne 0 ]; then
    echo "❌ ROM pipeline failed for ABL${IDX}! Aborting."
    exit $DATA_EXIT
fi

# ── Step 2: Visualization Pipeline ──
echo ""
echo "STEP 2/2: VISUALIZATION PIPELINE"
echo "============================================================"

python run_visualizations.py \
    --experiment_name "$EXP_NAME"

VIZ_EXIT=$?
echo "Visualization exit code: $VIZ_EXIT"

if [ $VIZ_EXIT -ne 0 ]; then
    echo "⚠️  Visualization failed (exit code: $VIZ_EXIT)"
    echo "   ROM pipeline succeeded — re-run visualization locally."
fi

# ── Summary ──
echo ""
echo "============================================================"
echo "ABL${IDX} COMPLETE — $LABEL"
echo "============================================================"
echo "End time: $(date)"

if [ -f "oscar_output/$EXP_NAME/summary.json" ]; then
    echo "Results:"
    python3 -c "
import json
d = json.load(open('oscar_output/$EXP_NAME/summary.json'))
m = d.get('mvar', {})
r2 = m.get('mean_r2_test', '?')
r2_1s = m.get('mean_r2_1step_test', '?')
neg = m.get('mean_negativity_frac', '?')
pod = d.get('pod', {})
r_pod = pod.get('r_pod', '?')
transform = pod.get('density_transform', '?')
sr = m.get('spectral_radius', '?')
print(f'  POD: d={r_pod}, transform={transform}')
if isinstance(r2, float):
    print(f'  R²_rollout = {r2:+.4f}')
    print(f'  R²_1step   = {r2_1s:.4f}')
    print(f'  neg_frac   = {neg:.4f}')
    print(f'  spec_rad   = {sr}')
else:
    print(f'  R² = {r2}')
" 2>/dev/null || echo "  (summary extraction failed)"
fi

echo ""
echo "Output: oscar_output/$EXP_NAME/"
echo "Done."
