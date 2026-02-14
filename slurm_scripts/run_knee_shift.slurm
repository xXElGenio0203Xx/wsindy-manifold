#!/bin/bash
#SBATCH --job-name=knee_shift
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/knee_shift_%j.out
#SBATCH -e slurm_logs/knee_shift_%j.err

# ============================================================================
# SUITE KNEE: Shift-Aligned Evaluation (K11-K14)
# ============================================================================
#
# Post-hoc evaluation on saved predictions from K4/K5/K9/K10.
# MUST be submitted AFTER run_knee_horizon.slurm completes.
#
# K11: shift-aligned eval of K4 (sqrt H162)
# K12: shift-aligned eval of K5 (sqrt H312)
# K13: shift-aligned eval of K9 (raw H162)
# K14: shift-aligned eval of K10 (raw H312)
#
# This is fast — just loads .npz files and does FFT cross-correlation.
# Expected: ~10-15 min total
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

export PYTHONPATH=src

echo "============================================================"
echo "SUITE KNEE — Shift-Aligned Evaluation (K11-K14)"
echo "============================================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""

PASS=0
FAIL=0
TOTAL=4
RESULTS=""

run_shift_eval() {
    local SOURCE="$1"
    local TAG="$2"
    local NUM="$3"
    
    echo ""
    echo "============================================================"
    echo "[$NUM/$TOTAL] $TAG — shift-aligned eval of $SOURCE"
    echo "============================================================"
    echo "Start: $(date)"
    
    # Check source exists
    if [ ! -d "oscar_output/$SOURCE/test" ]; then
        echo "  ❌ Source not found: oscar_output/$SOURCE/test"
        FAIL=$((FAIL + 1))
        RESULTS="$RESULTS\n  ❌ [$NUM] $TAG — source missing: $SOURCE"
        return 1
    fi
    
    python scripts/shift_aligned_eval.py \
        --source_exp "$SOURCE" \
        --tag "$TAG" \
        --base_dir oscar_output
    
    local EXIT_CODE=$?
    echo "Exit code: $EXIT_CODE"
    echo "End: $(date)"
    
    if [ $EXIT_CODE -eq 0 ]; then
        PASS=$((PASS + 1))
        RESULTS="$RESULTS\n  ✅ [$NUM] $TAG (source=$SOURCE)"
    else
        FAIL=$((FAIL + 1))
        RESULTS="$RESULTS\n  ❌ [$NUM] $TAG (exit=$EXIT_CODE)"
    fi
    echo ""
}

# K11: sqrt H162
run_shift_eval "K4_v1_sqrtD19_p5_H162" "K11" 1

# K12: sqrt H312
run_shift_eval "K5_v1_sqrtD19_p5_H312" "K12" 2

# K13: raw H162
run_shift_eval "K9_v1_rawD19_p5_H162" "K13" 3

# K14: raw H312
run_shift_eval "K10_v1_rawD19_p5_H312" "K14" 4

# ============== Summary ==============

echo ""
echo "============================================================"
echo "SUITE KNEE — SHIFT-ALIGNED EVALUATION COMPLETE"
echo "============================================================"
echo "End time: $(date)"
echo "Passed: $PASS / $TOTAL"
echo "Failed: $FAIL / $TOTAL"
echo ""
echo "Results:"
echo -e "$RESULTS"
echo ""

# ============== Combined table ==============
echo "============================================================"
echo "SHIFT-ALIGNED RESULTS TABLE"
echo "============================================================"
python3 -c "
import json
from pathlib import Path

evals = [
    ('K11', 'K4_v1_sqrtD19_p5_H162',  'sqrt', 'H162'),
    ('K12', 'K5_v1_sqrtD19_p5_H312',  'sqrt', 'H312'),
    ('K13', 'K9_v1_rawD19_p5_H162',   'raw',  'H162'),
    ('K14', 'K10_v1_rawD19_p5_H312',  'raw',  'H312'),
]

print(f'{'Tag':<6} {'Source':<35} {'Xform':<6} {'Horiz':<6} {'R²_orig':>10} {'R²_global':>10} {'R²_perfrm':>10} {'Δ_global':>10} {'shift_phys':>10}')
print('-' * 105)

for tag, source, xform, horiz in evals:
    sfile = Path(f'oscar_output/{source}/shift_aligned_summary_{tag}.json')
    if sfile.exists():
        s = json.load(open(sfile))
        r2o = s.get('r2_original_mean', '?')
        r2g = s.get('r2_global_shifted_mean', '?')
        r2p = s.get('r2_perframe_shifted_mean', '?')
        dg = s.get('r2_improvement_global', '?')
        sh = s.get('mean_global_shift_phys', '?')
        print(f'{tag:<6} {source:<35} {xform:<6} {horiz:<6} {r2o:>10.4f} {r2g:>10.4f} {r2p:>10.4f} {dg:>+10.4f} {sh:>10.3f}')
    else:
        print(f'{tag:<6} {source:<35} {xform:<6} {horiz:<6} {'MISSING':>10}')
" 2>/dev/null || echo "  (table generation failed)"

echo ""
echo "Done."
