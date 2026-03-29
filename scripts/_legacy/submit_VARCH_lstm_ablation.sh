#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# submit_VARCH_lstm_ablation.sh — LSTM architecture ablation suite
# ═══════════════════════════════════════════════════════════════════
# Suite A — Capacity sweep (3 architectures × 7 regimes = 21 jobs)
#   A16:   Nh=16, L=1  (~2,723 params, Alvarez-spirit)
#   A32:   Nh=32, L=1  (~7,475 params, modest scale-up)
#   A32x2: Nh=32, L=2  (~15,923 params, reasonable depth)
#
# Suite B — Trick sweep at Nh=32, L=1 (3 configs × 7 regimes = 21 jobs)
#   B1: residual OFF     (is Δy formulation needed?)
#   B2: layer_norm OFF   (is LN needed for small models?)
#   B3: multistep ON     (does rollout loss help once capacity is sane?)
#
# All LSTM-only, sqrt + simplex + C0.
# Total: 42 jobs
#
# Usage:
#   bash scripts/submit_VARCH_lstm_ablation.sh
#   bash scripts/submit_VARCH_lstm_ablation.sh --dry-run
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
    local VID="$1"   # e.g. A16, A32, A32x2, B1, B2, B3
    local DESC="$2"
    echo ""
    echo "───────────────────────────────────────────────────────"
    echo "  $VID: $DESC"
    echo "───────────────────────────────────────────────────────"

    for i in "${!REGIMES[@]}"; do
        local tag="${REGIMES[$i]}"
        local num="${REGIME_NUMS[$i]}"
        local name="VARCH_${VID}_${num}_${tag}"
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
echo "═══════════════════════════════════════════════════════════"
echo "  VARCH — LSTM Architecture Ablation Suite"
echo "  42 jobs total (6 variants × 7 regimes)"
echo "═══════════════════════════════════════════════════════════"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Suite A: Capacity Sweep (residual + LN, no tricks)"
echo "═══════════════════════════════════════════════════════════"

submit_variant "A16"   "Nh=16, L=1 (~2,723 params, Alvarez-spirit)"
submit_variant "A32"   "Nh=32, L=1 (~7,475 params, modest)"
submit_variant "A32x2" "Nh=32, L=2 (~15,923 params, reasonable depth)"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Suite B: Training-Trick Sweep (all Nh=32, L=1)"
echo "═══════════════════════════════════════════════════════════"

submit_variant "B1" "No residual (residual=false, LN on)"
submit_variant "B2" "No LayerNorm (LN=false, residual on)"
submit_variant "B3" "Multistep loss (k=5, α=0.3, no SS)"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  42 LSTM-only jobs submitted"
echo "  Suite A: 21 (3 capacities × 7 regimes)"
echo "  Suite B: 21 (3 trick toggles × 7 regimes)"
echo "  Pipeline: ROM_pipeline.py (LSTM only, no MVAR)"
echo "  Monitor: squeue -u \$USER"
echo "  Logs:    slurm_logs/VARCH_*.out"
echo "═══════════════════════════════════════════════════════════"

if ! $DRY_RUN; then
    echo ""
    squeue -u "$USER"
fi
