#!/bin/bash
#SBATCH --job-name=REP1_repulsive
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/REP1_%j.out
#SBATCH -e slurm_logs/REP1_%j.err

# ============================================================================
# REP1 — Repulsive regime: Ca=0.1 (weak attract), Cr=3.0 (strong repel), N=200
# ============================================================================
# Expected runtime: ~3-4h (N=200 vs N=500 in CUR → ~6x faster per sim)
# CPU-only (MVAR, no LSTM). Using batch partition.
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

export PYTHONPATH=src

CONFIG="configs/REP1_N200_repulsive_sqrtSimplex_H300.yaml"
EXP_NAME="REP1_N200_repulsive_sqrtSimplex_H300"

echo "============================================================"
echo "REP1 REPULSIVE REGIME PIPELINE"
echo "============================================================"
echo "Config: $CONFIG"
echo "Experiment: $EXP_NAME"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_NTASKS"
echo "Disk before:"
du -sh ~/wsindy-manifold/oscar_output/ 2>/dev/null
echo ""

# ── Step 1: Full ROM Pipeline (sim + POD + MVAR + eval) ──
echo "STEP 1/2: ROM PIPELINE"
echo "============================================================"

python ROM_pipeline.py \
    --config "$CONFIG" \
    --experiment_name "$EXP_NAME"

DATA_EXIT=$?
echo "ROM pipeline exit code: $DATA_EXIT"
echo "End: $(date)"

if [ $DATA_EXIT -ne 0 ]; then
    echo "❌ ROM pipeline failed! Aborting."
    echo "Disk after:"
    du -sh ~/wsindy-manifold/oscar_output/ 2>/dev/null
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
    echo "⚠️  Visualization failed (exit code: $VIZ_EXIT)"
    echo "   ROM pipeline succeeded — re-run visualization locally."
fi

# ── Summary ──
echo ""
echo "============================================================"
echo "REP1 PIPELINE COMPLETE"
echo "============================================================"
echo "End time: $(date)"
echo "Disk after:"
du -sh ~/wsindy-manifold/oscar_output/ 2>/dev/null

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
