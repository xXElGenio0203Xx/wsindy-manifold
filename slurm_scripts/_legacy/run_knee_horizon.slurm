#!/bin/bash
#SBATCH --job-name=knee_horizon
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/knee_horizon_%j.out
#SBATCH -e slurm_logs/knee_horizon_%j.err

# ============================================================================
# SUITE KNEE: Horizon Curve (K1-K10)
# ============================================================================
#
# A) K1-K5: sqrt d=19 p=5 clamp_C2 at H37/H60/H100/H162/H312
# B) K6-K10: raw d=19 p=5 clamp_C2 at same horizons
#
# These 10 experiments run sequentially within this job.
# Submit this job in parallel with run_knee_kstep.slurm
#
# Expected: ~2h each for short horizons, ~3h for long → ~20h total
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

export PYTHONPATH=src

echo "============================================================"
echo "SUITE KNEE — Horizon Curve (K1-K10)"
echo "============================================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""

PASS=0
FAIL=0
TOTAL=10
RESULTS=""

run_experiment() {
    local CONFIG="$1"
    local NAME="$2"
    local NUM="$3"
    
    echo ""
    echo "============================================================"
    echo "[$NUM/$TOTAL] $NAME"
    echo "============================================================"
    echo "Config: $CONFIG"
    echo "Start: $(date)"
    
    python ROM_pipeline.py \
        --config "$CONFIG" \
        --experiment_name "$NAME"
    
    local EXIT_CODE=$?
    echo "Exit code: $EXIT_CODE"
    echo "End: $(date)"
    
    if [ $EXIT_CODE -eq 0 ]; then
        PASS=$((PASS + 1))
        RESULTS="$RESULTS\n  ✅ [$NUM] $NAME"
        if [ -f "oscar_output/$NAME/summary.json" ]; then
            python3 -c "
import json
d = json.load(open('oscar_output/$NAME/summary.json'))
m = d.get('mvar', d.get('lstm', {}))
r2 = m.get('mean_r2_test', '?')
r2_1s = m.get('mean_r2_1step_test', '?')
neg = m.get('mean_negativity_frac', '?')
pod = d.get('pod', {})
r_pod = pod.get('r_pod', '?')
transform = pod.get('density_transform', '?')
clamp = d.get('clamp_mode', '?')
sr = m.get('spectral_radius', '?')
print(f'     d={r_pod} transform={transform} clamp={clamp}')
if isinstance(r2, float):
    print(f'     R²_rollout={r2:+.4f}  R²_1step={r2_1s:.4f}  neg={neg:.4f}  spectral_radius={sr}')
else:
    print(f'     R²={r2}')
" 2>/dev/null || true
        fi
    else
        FAIL=$((FAIL + 1))
        RESULTS="$RESULTS\n  ❌ [$NUM] $NAME (exit=$EXIT_CODE)"
    fi
    echo ""
}

# ============== Block A: sqrt d=19 horizon curve ==============

echo ">>> BLOCK A: sqrt d=19 horizon curve"

run_experiment \
    "configs/K1_v1_sqrtD19_p5_H37.yaml" \
    "K1_v1_sqrtD19_p5_H37" 1

run_experiment \
    "configs/K2_v1_sqrtD19_p5_H60.yaml" \
    "K2_v1_sqrtD19_p5_H60" 2

run_experiment \
    "configs/K3_v1_sqrtD19_p5_H100.yaml" \
    "K3_v1_sqrtD19_p5_H100" 3

run_experiment \
    "configs/K4_v1_sqrtD19_p5_H162.yaml" \
    "K4_v1_sqrtD19_p5_H162" 4

run_experiment \
    "configs/K5_v1_sqrtD19_p5_H312.yaml" \
    "K5_v1_sqrtD19_p5_H312" 5

# ============== Block B: raw d=19 controls ==============

echo ">>> BLOCK B: raw d=19 horizon curve (controls)"

run_experiment \
    "configs/K6_v1_rawD19_p5_H37.yaml" \
    "K6_v1_rawD19_p5_H37" 6

run_experiment \
    "configs/K7_v1_rawD19_p5_H60.yaml" \
    "K7_v1_rawD19_p5_H60" 7

run_experiment \
    "configs/K8_v1_rawD19_p5_H100.yaml" \
    "K8_v1_rawD19_p5_H100" 8

run_experiment \
    "configs/K9_v1_rawD19_p5_H162.yaml" \
    "K9_v1_rawD19_p5_H162" 9

run_experiment \
    "configs/K10_v1_rawD19_p5_H312.yaml" \
    "K10_v1_rawD19_p5_H312" 10

# ============== Summary ==============

echo ""
echo "============================================================"
echo "SUITE KNEE — HORIZON CURVE COMPLETE"
echo "============================================================"
echo "End time: $(date)"
echo "Passed: $PASS / $TOTAL"
echo "Failed: $FAIL / $TOTAL"
echo ""
echo "Results:"
echo -e "$RESULTS"
echo ""

# ============== Quick comparison table ==============
echo "============================================================"
echo "HORIZON CURVE TABLE"
echo "============================================================"
python3 -c "
import json, os
from pathlib import Path

experiments = [
    ('K1_v1_sqrtD19_p5_H37',  'sqrt', 'H37'),
    ('K2_v1_sqrtD19_p5_H60',  'sqrt', 'H60'),
    ('K3_v1_sqrtD19_p5_H100', 'sqrt', 'H100'),
    ('K4_v1_sqrtD19_p5_H162', 'sqrt', 'H162'),
    ('K5_v1_sqrtD19_p5_H312', 'sqrt', 'H312'),
    ('K6_v1_rawD19_p5_H37',   'raw',  'H37'),
    ('K7_v1_rawD19_p5_H60',   'raw',  'H60'),
    ('K8_v1_rawD19_p5_H100',  'raw',  'H100'),
    ('K9_v1_rawD19_p5_H162',  'raw',  'H162'),
    ('K10_v1_rawD19_p5_H312', 'raw',  'H312'),
]

print(f'{'Experiment':<35} {'Transform':<8} {'Horizon':<8} {'R²_roll':>10} {'R²_1step':>10} {'neg%':>8} {'spec_rad':>10}')
print('-' * 95)

for exp_name, transform, horizon in experiments:
    sfile = Path(f'oscar_output/{exp_name}/summary.json')
    if sfile.exists():
        d = json.load(open(sfile))
        m = d.get('mvar', {})
        r2 = m.get('mean_r2_test', '?')
        r2_1s = m.get('mean_r2_1step_test', '?')
        neg = m.get('mean_negativity_frac', '?')
        sr = m.get('spectral_radius', '?')
        r2_s = f'{r2:+.4f}' if isinstance(r2, float) else str(r2)
        r2_1s_s = f'{r2_1s:.4f}' if isinstance(r2_1s, float) else str(r2_1s)
        neg_s = f'{neg:.4f}' if isinstance(neg, float) else str(neg)
        sr_s = f'{sr:.4f}' if isinstance(sr, float) else str(sr)
        print(f'{exp_name:<35} {transform:<8} {horizon:<8} {r2_s:>10} {r2_1s_s:>10} {neg_s:>8} {sr_s:>10}')
    else:
        print(f'{exp_name:<35} {transform:<8} {horizon:<8} {'MISSING':>10}')
" 2>/dev/null || echo "  (table generation failed)"

echo ""
echo "Done."
