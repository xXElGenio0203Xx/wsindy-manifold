#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# submit_VFIX_lstm_fix.sh — LSTM FIX suite (everything that works)
# ═══════════════════════════════════════════════════════════════════
# Key finding from 56 prior experiments:
#   residual=OFF improved rollout R² by +53 points on average.
#   Raw+C2 beat sqrt+simplex+C0 in 6/7 regimes.
#
# ALL variants: residual=OFF, LayerNorm=ON, dropout=0.0, multistep=OFF.
#
# Suite A — Width sweep (L=1, raw+C2):
#   F16:   Nh=16,  L=1  (2,723 params)   — Alvarez-scale
#   F32:   Nh=32,  L=1  (7,475 params)   — sweet spot ★ REF
#   F64:   Nh=64,  L=1  (23,123 params)  — wider
#   F128:  Nh=128, L=1  (78,995 params)  — wide
#
# Suite B — Depth sweep (raw+C2):
#   F32x2: Nh=32,  L=2  (15,923 params)  — deeper
#   F64x2: Nh=64,  L=2  (56,403 params)  — deep + wide
#
# Suite C — Trick / transform ablation at Nh=32, L=1:
#   FSS:   raw+C2  + scheduled_sampling=ON
#   FSQRT: sqrt+simplex+C0
#
# Total: 8 variants × 7 regimes = 56 jobs
#
# Usage:
#   bash scripts/submit_VFIX_lstm_fix.sh
#   bash scripts/submit_VFIX_lstm_fix.sh --dry-run
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

# ─── 7 regimes ──────────────────────────────────────────────────────
REGIMES=(gentle hypervelocity hypernoisy blackhole supernova baseline pure_vicsek)
REGIME_NUMS=(1 2 3 4 5 6 7)

submit_variant() {
    local VID="$1"   # e.g. F16, F32, F64, F128, F32x2, F64x2, FSS, FSQRT
    local DESC="$2"
    echo ""
    echo "───────────────────────────────────────────────────────"
    echo "  $VID: $DESC"
    echo "───────────────────────────────────────────────────────"

    for i in "${!REGIMES[@]}"; do
        local tag="${REGIMES[$i]}"
        local num="${REGIME_NUMS[$i]}"
        local name="VFIX_${VID}_${num}_${tag}"
        local cfg="configs/${name}.yaml"

        echo "  $name → $cfg"
        if $DRY_RUN; then
            echo "    [DRY] sbatch ... --wrap=\"$WRAP_PREFIX --config $cfg --experiment_name $name\""
        else
            sbatch --job-name="$name" --time=12:00:00 -N1 -n4 --mem=32G -p batch \
                -o "slurm_logs/${name}_%j.out" -e "slurm_logs/${name}_%j.err" \
                --wrap="$WRAP_PREFIX --config $cfg --experiment_name $name"
        fi
    done
}

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  VFIX — LSTM Fix Suite (everything that works)"
echo "  56 jobs total (8 variants × 7 regimes)"
echo "  ALL: residual=OFF, LayerNorm=ON, dropout=0, multistep=OFF"
echo "═══════════════════════════════════════════════════════════════════"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Suite A: Width Sweep (L=1, raw+C2)"
echo "═══════════════════════════════════════════════════════════════════"

submit_variant "F16"   "Nh=16, L=1 (2,723 params) — Alvarez-scale"
submit_variant "F32"   "Nh=32, L=1 (7,475 params) — sweet spot ★ REF"
submit_variant "F64"   "Nh=64, L=1 (23,123 params) — wider"
submit_variant "F128"  "Nh=128, L=1 (78,995 params) — wide"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Suite B: Depth Sweep (raw+C2)"
echo "═══════════════════════════════════════════════════════════════════"

submit_variant "F32x2" "Nh=32, L=2 (15,923 params) — deeper"
submit_variant "F64x2" "Nh=64, L=2 (56,403 params) — deep + wide"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Suite C: Trick / Transform Ablation (Nh=32, L=1)"
echo "═══════════════════════════════════════════════════════════════════"

submit_variant "FSS"   "Nh=32, L=1, raw+C2, scheduled_sampling=ON"
submit_variant "FSQRT" "Nh=32, L=1, sqrt+simplex+C0 (transform control)"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  DONE — submitted 56 VFIX jobs"
echo "═══════════════════════════════════════════════════════════════════"
