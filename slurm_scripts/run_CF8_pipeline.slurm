#!/bin/bash
#SBATCH --job-name=CF8_pipeline
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/CF8_pipeline_%j.out
#SBATCH -e slurm_logs/CF8_pipeline_%j.err

# ============================================================================
# CF8 Long-Prefix Pipeline (Data Generation + Visualization)
# ============================================================================
#
# Runs the full standard pipeline with CF8 champion parameters:
#   - V1 regime (N=100, speed=1.5, Morse)
#   - 77s simulations (72s train prefix + 5s forecast window)
#   - √ρ transform, d=19, p=5, α=0.01
#   - Simplex mass projection
#   - 200 train ICs + 30 test ICs
#
# Expected runtime: ~6-8 hours
#   - 200 train sims × ~60s = ~3.3h
#   - 30 test sims × ~60s = ~30min
#   - POD/MVAR: ~10min
#   - Evaluation + visualization: ~30min
#
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

export PYTHONPATH=src

CONFIG="configs/CF8_longPrefix_sqrtSimplex_H37.yaml"
EXP_NAME="CF8_longPrefix_sqrtSimplex_H37"

echo "============================================================"
echo "CF8 LONG-PREFIX PIPELINE"
echo "============================================================"
echo "Config: $CONFIG"
echo "Experiment: $EXP_NAME"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""

# ── Step 1: Data Generation ──
echo "STEP 1/2: DATA GENERATION PIPELINE"
echo "============================================================"

python ROM_pipeline.py \
    --config "$CONFIG" \
    --experiment_name "$EXP_NAME"

DATA_EXIT=$?
echo "Data generation exit code: $DATA_EXIT"
echo "End: $(date)"

if [ $DATA_EXIT -ne 0 ]; then
    echo "❌ Data generation failed! Aborting."
    exit $DATA_EXIT
fi

# ── Step 2: Visualization Pipeline ──
echo ""
echo "============================================================"
echo "STEP 2/2: VISUALIZATION PIPELINE"
echo "============================================================"

python run_visualizations.py \
    --experiment_name "$EXP_NAME"

VIZ_EXIT=$?
echo "Visualization exit code: $VIZ_EXIT"

if [ $VIZ_EXIT -ne 0 ]; then
    echo "⚠️  Visualization pipeline failed (exit code: $VIZ_EXIT)"
    echo "   Data generation succeeded — you can re-run visualization locally."
fi

# ── Summary ──
echo ""
echo "============================================================"
echo "CF8 PIPELINE COMPLETE"
echo "============================================================"
echo "End time: $(date)"
echo ""

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
echo "Output data:    oscar_output/$EXP_NAME/"
echo "Visualizations: predictions/$EXP_NAME/"
echo ""
echo "Done."
