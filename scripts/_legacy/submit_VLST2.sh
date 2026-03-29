#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# submit_VLST2.sh — Variable-speed LST re-run with richer training
# ═══════════════════════════════════════════════════════════════════
# Tests whether the LSTM success at constant speed (R²≈0.97) transfers
# to variable speed when we fix the training deficiencies:
#   - T_train: 20s → 30s  (50% more temporal data)
#   - All 4 IC types: uniform(60) + gaussian(100) + two_clusters(36) + ring(30)
#   - Same good hyperparams: epochs=300, batch=512, lr=0.0007
#   - Same test horizon: T_test=36.6s
#
# Variant 1 (VLST2_sqrt): sqrt/simplex/H=64/L=2  (matches LST4)
# Variant 2 (VLST2_raw):  raw/none/H=128/L=2     (matches LST7)
#
# Expected runtime: ~3-4h each (more ICs = longer sim generation)
#
# Usage:
#   bash scripts/submit_VLST2.sh
#   bash scripts/submit_VLST2.sh --dry-run
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
echo "═══════════════════════════════════════════════════════════════════"
echo "  VLST2 — Variable-Speed LSTM with Rich Training"
echo "  2 jobs: sqrt/H64/L2 + raw/H128/L2"
echo "  variable speed, T_train=30s, 4 IC types (~226 training sims)"
echo "═══════════════════════════════════════════════════════════════════"

# ─── Job 1: sqrt/simplex (matches LST4) ─────────────────────────
NAME1="VLST2_sqrt_h64_L2"
CFG1="configs/VLST2_sqrt_h64_L2.yaml"
echo ""
echo "  [1/2] $NAME1"
echo "         sqrt/simplex, H=64, L=2, clamp=C0"
if $DRY_RUN; then
    echo "    [DRY] sbatch ... --wrap=\"$WRAP_PREFIX --config $CFG1 --experiment_name $NAME1\""
else
    JOB1=$(sbatch --job-name="$NAME1" \
        --time=10:00:00 \
        -N1 -n8 --mem=48G \
        -p batch \
        -o "slurm_logs/${NAME1}_%j.out" \
        -e "slurm_logs/${NAME1}_%j.err" \
        --wrap="$WRAP_PREFIX --config $CFG1 --experiment_name $NAME1" \
        | awk '{print $4}')
    echo "    Submitted: Job $JOB1"
fi

# ─── Job 2: raw/none (matches LST7) ─────────────────────────────
NAME2="VLST2_raw_h128_L2"
CFG2="configs/VLST2_raw_h128_L2.yaml"
echo ""
echo "  [2/2] $NAME2"
echo "         raw/none, H=128, L=2, clamp=C0"
if $DRY_RUN; then
    echo "    [DRY] sbatch ... --wrap=\"$WRAP_PREFIX --config $CFG2 --experiment_name $NAME2\""
else
    JOB2=$(sbatch --job-name="$NAME2" \
        --time=10:00:00 \
        -N1 -n8 --mem=48G \
        -p batch \
        -o "slurm_logs/${NAME2}_%j.out" \
        -e "slurm_logs/${NAME2}_%j.err" \
        --wrap="$WRAP_PREFIX --config $CFG2 --experiment_name $NAME2" \
        | awk '{print $4}')
    echo "    Submitted: Job $JOB2"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Done. Monitor with: squeue -u \$USER | grep VLST2"
echo "═══════════════════════════════════════════════════════════════════"
