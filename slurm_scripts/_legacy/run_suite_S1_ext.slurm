#!/bin/bash
#SBATCH --job-name=suite_S1_ext
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=24G
#SBATCH --time=18:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/suite_S1_ext_%j.out
#SBATCH -e slurm_logs/suite_S1_ext_%j.err

# ============================================================================
# SUITE S1 EXTENDED: Experiments 11–20
# ============================================================================
#
# All experiments use V1 physics (200 train / 30 test), ~90 min each.
#
# 11. POD E=95% (sweet spot test)
# 12. Mean-subtracted density + POD E=90%
# 13. Log-density + POD E=90%
# 14. Sqrt-density + POD E=90%
# 15. Fixed d=10 POD
# 16. Fixed d=19 POD (V1 control)
# 17. Fixed d=30 POD (stress test)
# 18. Clamp C0 (no clamp, no renorm)
# 19. Clamp C1 (clamp only, no renorm)
# 20. Clamp C2 (clamp + renorm, V1 default)
#
# Estimated: 10 × ~90 min = 15h. 18h limit gives buffer.
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

export PYTHONPATH=src

echo "============================================================"
echo "SUITE S1 EXTENDED — Experiments 11–20"
echo "============================================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo ""

# Track successes and failures
PASS=0
FAIL=0
RESULTS=""

run_experiment() {
    local CONFIG="$1"
    local NAME="$2"
    local NUM="$3"
    
    echo ""
    echo "============================================================"
    echo "[$NUM/20] $NAME"
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
rho = m.get('spectral_radius', '?')
pod = d.get('pod', {})
pod_d = pod.get('r_pod', d.get('r_pod', '?'))
pod_e = pod.get('energy_captured', '?')
xform = pod.get('density_transform', 'raw')
clamp = d.get('clamp_mode', '?')
if isinstance(r2, float):
    line = f'     R2_roll={r2:.4f}'
    if isinstance(r2_1s, float): line += f'  R2_1step={r2_1s:.4f}'
    if isinstance(neg, float): line += f'  neg={neg:.4f}'
    if isinstance(rho, float): line += f'  rho={rho:.4f}'
    line += f'  d={pod_d}  E={pod_e}'
    line += f'  xform={xform}  clamp={clamp}'
    print(line)
else:
    print(f'     R2={r2}')
" 2>/dev/null || true
        fi
    else
        FAIL=$((FAIL + 1))
        RESULTS="$RESULTS\n  ❌ [$NUM] $NAME (exit=$EXIT_CODE)"
    fi
    
    echo ""
}

# ============== Experiments 11–20 ==============

run_experiment \
    "configs/suite_S1_podE95_raw_mvar_p5.yaml" \
    "suite_S1_podE95_raw_mvar_p5" \
    11

run_experiment \
    "configs/suite_S1_podE90_meansub_mvar_p5.yaml" \
    "suite_S1_podE90_meansub_mvar_p5" \
    12

run_experiment \
    "configs/suite_S1_podE90_logrho_mvar_p5.yaml" \
    "suite_S1_podE90_logrho_mvar_p5" \
    13

run_experiment \
    "configs/suite_S1_podE90_sqrtrho_mvar_p5.yaml" \
    "suite_S1_podE90_sqrtrho_mvar_p5" \
    14

run_experiment \
    "configs/suite_S1_podD10_raw_mvar_p5.yaml" \
    "suite_S1_podD10_raw_mvar_p5" \
    15

run_experiment \
    "configs/suite_S1_podD19_raw_mvar_p5.yaml" \
    "suite_S1_podD19_raw_mvar_p5" \
    16

run_experiment \
    "configs/suite_S1_podD30_raw_mvar_p5.yaml" \
    "suite_S1_podD30_raw_mvar_p5" \
    17

run_experiment \
    "configs/suite_S1_clamp_C0_V1.yaml" \
    "suite_S1_clamp_C0_V1" \
    18

run_experiment \
    "configs/suite_S1_clamp_C1_V1.yaml" \
    "suite_S1_clamp_C1_V1" \
    19

run_experiment \
    "configs/suite_S1_clamp_C2_V1.yaml" \
    "suite_S1_clamp_C2_V1" \
    20

# ============== Final Summary ==============

echo ""
echo "============================================================"
echo "SUITE S1 EXTENDED COMPLETE (Experiments 11–20)"
echo "============================================================"
echo "End time: $(date)"
echo "Passed: $PASS / 10"
echo "Failed: $FAIL / 10"
echo ""
echo "Results:"
echo -e "$RESULTS"
echo ""
echo "============================================================"
